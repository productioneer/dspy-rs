//! RLM — Recursive Language Model module.
//! Python equivalent: dspy/predict/rlm.py
//!
//! Uses a sandboxed REPL to let the LLM programmatically explore large contexts
//! through code execution. The LLM writes Python code to examine data, call
//! sub-LMs for semantic analysis, and build up answers iteratively.

use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::interpreter::{CodeInterpreter, ExecutionResult, FinalOutput, InterpreterTool};
use crate::module_trait::Module;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::repl_types::{create_repl_variable, format_repl_variable, REPLHistory, REPLVariable};
use crate::settings::get_settings;
use crate::signature::{input_field, output_field, Signature};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

const ACTION_INSTRUCTIONS_TEMPLATE: &str = r#"You are tasked with producing the following outputs given the inputs {inputs}:
{output_fields}

You have access to a Python REPL environment. Write Python code and it will be executed. You will see the output, then write more code based on what you learned. This is an iterative process.

Available:
- Variables: {inputs} (your input data)
- `llm_query(prompt)` - query a sub-LLM (~500K char capacity) for semantic analysis
- `llm_query_batched(prompts)` - query multiple prompts concurrently (much faster for multiple queries)
- `print()` - ALWAYS print to see results
- `SUBMIT({final_output_names})` - submit final output when done
- Standard libraries: re, json, collections, math, etc.

IMPORTANT: This is ITERATIVE. Each code block you write will execute, you'll see the output, then you decide what to do next. Do NOT try to solve everything in one step.

1. EXPLORE FIRST - Look at your data before processing it. Print samples, check types/lengths, understand the structure.
2. ITERATE - Write small code snippets, observe outputs, then decide next steps. State persists between iterations.
3. VERIFY BEFORE SUBMITTING - If results seem wrong (zeros, empty, unexpected), reconsider your approach.
4. USE llm_query FOR SEMANTICS - String matching finds WHERE things are; llm_query understands WHAT things mean.
5. MINIMIZE RETYPING (INPUTS & OUTPUTS) - When values are long, precise, or error-prone (IDs, numbers, code, quotes), re-access them via variables and parse/compute in code instead of retyping. Use small, targeted prints to sanity-check, but avoid manual copying when variables can carry the exact value.
6. SUBMIT ONLY AFTER SEEING OUTPUTS - SUBMIT ends the current run immediately. If you need to inspect printed output, run it in one step, review the result, then call SUBMIT in a later step.

You have max {max_llm_calls} sub-LLM calls. When done, call SUBMIT() with your output."#;

/// Reserved tool names that conflict with built-in sandbox functions.
#[allow(dead_code)]
const RESERVED_TOOL_NAMES: &[&str] = &["llm_query", "llm_query_batched", "SUBMIT", "print"];

pub struct RLM {
    signature: Signature,
    max_iterations: usize,
    max_llm_calls: usize,
    max_output_chars: usize,
    verbose: bool,
    sub_lm: Option<Arc<dyn crate::lm::LM>>,
    external_interpreter: Option<Arc<Mutex<Box<dyn CodeInterpreter>>>>,
    generate_action: Predict,
    extract: Predict,
}

impl RLM {
    pub fn new(
        signature: Signature,
        max_iterations: Option<usize>,
        max_llm_calls: Option<usize>,
        max_output_chars: Option<usize>,
        verbose: bool,
        sub_lm: Option<Arc<dyn crate::lm::LM>>,
        interpreter: Option<Box<dyn CodeInterpreter>>,
    ) -> Result<Self> {
        let (action_sig, extract_sig) = build_signatures(&signature, max_llm_calls.unwrap_or(50));

        Ok(Self {
            signature,
            max_iterations: max_iterations.unwrap_or(20),
            max_llm_calls: max_llm_calls.unwrap_or(50),
            max_output_chars: max_output_chars.unwrap_or(100_000),
            verbose,
            sub_lm,
            external_interpreter: interpreter.map(|i| Arc::new(Mutex::new(i))),
            generate_action: Predict::new(action_sig),
            extract: Predict::new(extract_sig),
        })
    }

    fn validate_inputs(&self, args: &Example) -> Result<()> {
        let missing: Vec<String> = self
            .signature
            .input_fields()
            .filter_map(|(name, _)| {
                if args.get(name).is_none() {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        if !missing.is_empty() {
            let mut sorted = missing;
            sorted.sort();
            return Err(DspyError::Other(format!(
                "Missing required inputs: {}",
                sorted.join(", ")
            )));
        }
        Ok(())
    }

    fn build_variables(&self, args: &Example) -> Vec<REPLVariable> {
        let map = args.to_map();
        map.iter()
            .map(|(name, value)| {
                let field_info = self.signature.get_field(name);
                let desc = field_info.and_then(|f| f.description.as_deref());
                let json_value = serde_json::Value::from(value.clone());
                create_repl_variable(name, &json_value, desc, None, None)
            })
            .collect()
    }

    fn format_output(&self, output: &str) -> String {
        if output.is_empty() {
            return "(no output - did you forget to print?)".to_string();
        }
        if output.len() > self.max_output_chars {
            format!("{}\n... (truncated)", &output[..self.max_output_chars])
        } else {
            output.to_string()
        }
    }

    fn prepare_execution_tools(&self) -> HashMap<String, InterpreterTool> {
        let mut tools: HashMap<String, InterpreterTool> = HashMap::new();

        let call_count = Arc::new(Mutex::new(0usize));
        let max_calls = self.max_llm_calls;
        let sub_lm = self.sub_lm.clone();

        // llm_query tool
        let call_count_q = call_count.clone();
        let sub_lm_q = sub_lm.clone();
        tools.insert(
            "llm_query".to_string(),
            Box::new(move |kwargs: HashMap<String, serde_json::Value>| {
                let call_count = call_count_q.clone();
                let sub_lm = sub_lm_q.clone();
                Box::pin(async move {
                    let prompt = kwargs
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if prompt.is_empty() {
                        return Err("prompt cannot be empty".to_string());
                    }

                    let mut count = call_count.lock().await;
                    if *count + 1 > max_calls {
                        return Err(format!(
                            "LLM call limit exceeded: {} + 1 > {}. Use Python code for aggregation instead of making more LLM calls.",
                            *count, max_calls
                        ));
                    }
                    *count += 1;
                    drop(count);

                    let lm = if let Some(ref lm) = sub_lm {
                        lm.clone()
                    } else {
                        let settings = get_settings();
                        settings.lm.ok_or_else(|| {
                            "No LM configured. Use configure() or pass sub_lm to RLM.".to_string()
                        })?
                    };

                    let messages = vec![crate::lm::Message {
                        role: "user".to_string(),
                        content: prompt.to_string(),
                    }];
                    let config = lm.config().clone();
                    let responses = lm
                        .call(&messages, &config)
                        .await
                        .map_err(|e| format!("{}", e))?;

                    let text = responses
                        .first()
                        .map(|r| r.text.clone())
                        .unwrap_or_default();
                    Ok(serde_json::json!(text))
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<serde_json::Value, String>> + Send>>
            }),
        );

        // llm_query_batched tool
        let call_count_b = call_count.clone();
        let sub_lm_b = sub_lm.clone();
        tools.insert(
            "llm_query_batched".to_string(),
            Box::new(move |kwargs: HashMap<String, serde_json::Value>| {
                let call_count = call_count_b.clone();
                let sub_lm = sub_lm_b.clone();
                Box::pin(async move {
                    let prompts: Vec<String> = kwargs
                        .get("prompts")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();

                    if prompts.is_empty() {
                        return Ok(serde_json::json!([]));
                    }

                    let mut count = call_count.lock().await;
                    if *count + prompts.len() > max_calls {
                        return Err(format!(
                            "LLM call limit exceeded: {} + {} > {}. Use Python code for aggregation instead of making more LLM calls.",
                            *count, prompts.len(), max_calls
                        ));
                    }
                    *count += prompts.len();
                    drop(count);

                    let lm = if let Some(ref lm) = sub_lm {
                        lm.clone()
                    } else {
                        let settings = get_settings();
                        settings.lm.ok_or_else(|| {
                            "No LM configured. Use configure() or pass sub_lm to RLM.".to_string()
                        })?
                    };

                    let config = lm.config().clone();
                    let mut results = Vec::new();
                    for prompt in &prompts {
                        let messages = vec![crate::lm::Message {
                            role: "user".to_string(),
                            content: prompt.clone(),
                        }];
                        let responses = lm
                            .call(&messages, &config)
                            .await
                            .map_err(|e| format!("{}", e))?;
                        let text = responses
                            .first()
                            .map(|r| r.text.clone())
                            .unwrap_or_default();
                        results.push(text);
                    }

                    Ok(serde_json::json!(results))
                }) as std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<serde_json::Value, String>> + Send>>
            }),
        );

        tools
    }

    async fn execute_iteration(
        &self,
        repl: &Mutex<Box<dyn CodeInterpreter>>,
        variables: &[REPLVariable],
        history: &REPLHistory,
        iteration: usize,
        input_args: &Example,
        output_field_names: &[String],
    ) -> Result<IterationResult> {
        let variables_info = variables
            .iter()
            .map(format_repl_variable)
            .collect::<Vec<_>>()
            .join("\n\n");

        let repl_history_str = history.format(None);
        let iteration_str = format!("{}/{}", iteration + 1, self.max_iterations);
        let action_args = Example::new()
            .field("variables_info", variables_info)
            .field("repl_history", repl_history_str)
            .field("iteration", iteration_str);

        let action = self.generate_action.call(&action_args).await?;

        if self.verbose {
            eprintln!(
                "RLM iteration {}/{}\nReasoning: {}\nCode:\n{}",
                iteration + 1,
                self.max_iterations,
                action.get_str("reasoning").unwrap_or(""),
                action.get_str("code").unwrap_or("")
            );
        }

        let code = strip_code_fences(action.get_str("code").unwrap_or(""));
        let reasoning = action.get_str("reasoning").unwrap_or("").to_string();

        let result = {
            let mut interp = repl.lock().await;
            // Build variables map from input args
            let vars: HashMap<String, serde_json::Value> = input_args
                .to_map()
                .iter()
                .map(|(k, v)| (k.clone(), serde_json::Value::from(v.clone())))
                .collect();
            match interp.execute(&code, Some(&vars)).await {
                Ok(result) => result,
                Err(e) => {
                    // Treat interpreter errors as string errors in history
                    let error_output = format!("[Error] {}", e.message);
                    return Ok(IterationResult::Continue(history.append(
                        &reasoning,
                        &code,
                        &self.format_output(&error_output),
                    )));
                }
            }
        };

        self.process_execution_result(
            &action,
            result,
            history,
            output_field_names,
            &code,
            &reasoning,
        )
    }

    fn process_execution_result(
        &self,
        _pred: &Prediction,
        result: ExecutionResult,
        history: &REPLHistory,
        output_field_names: &[String],
        code: &str,
        reasoning: &str,
    ) -> Result<IterationResult> {
        match result {
            ExecutionResult::Output(output) => {
                let output_str = output.unwrap_or_default();
                // Check for error strings
                if output_str.starts_with("[Error]") {
                    return Ok(IterationResult::Continue(history.append(
                        reasoning,
                        code,
                        &self.format_output(&output_str),
                    )));
                }
                Ok(IterationResult::Continue(history.append(
                    reasoning,
                    code,
                    &self.format_output(&output_str),
                )))
            }
            ExecutionResult::Final(final_output) => {
                let (parsed, error) = process_final_output(&final_output, output_field_names);

                if let Some(err) = error {
                    return Ok(IterationResult::Continue(
                        history.append(reasoning, code, &err),
                    ));
                }

                let final_history = history.append(
                    reasoning,
                    code,
                    &format!(
                        "FINAL: {}",
                        serde_json::to_string(&parsed).unwrap_or_default()
                    ),
                );

                let parsed_map = parsed.unwrap_or_default();
                let mut fields: HashMap<String, crate::value::Value> = HashMap::new();
                for (k, v) in parsed_map {
                    fields.insert(k, crate::value::Value::from(v));
                }
                fields.insert(
                    "trajectory".to_string(),
                    crate::value::Value::from(final_history.to_json()),
                );
                fields.insert(
                    "final_reasoning".to_string(),
                    crate::value::Value::String(reasoning.to_string()),
                );

                Ok(IterationResult::Done(Prediction::from_example(
                    Example::from_map(fields),
                )))
            }
        }
    }

    async fn extract_fallback(
        &self,
        variables: &[REPLVariable],
        history: &REPLHistory,
        output_field_names: &[String],
    ) -> Result<Prediction> {
        let variables_info = variables
            .iter()
            .map(format_repl_variable)
            .collect::<Vec<_>>()
            .join("\n\n");

        let repl_history_str = history.format(None);
        let extract_args = Example::new()
            .field("variables_info", variables_info)
            .field("repl_history", repl_history_str);

        let extract_pred = self.extract.call(&extract_args).await?;

        let mut outputs: HashMap<String, crate::value::Value> = HashMap::new();
        for name in output_field_names {
            if let Some(val) = extract_pred.get(name) {
                outputs.insert(name.clone(), val.clone());
            }
        }
        outputs.insert(
            "trajectory".to_string(),
            crate::value::Value::from(history.to_json()),
        );
        outputs.insert(
            "final_reasoning".to_string(),
            crate::value::Value::String("Extract forced final output".to_string()),
        );

        Ok(Prediction::from_example(Example::from_map(outputs)))
    }
}

enum IterationResult {
    Continue(REPLHistory),
    Done(Prediction),
}

#[async_trait]
impl Module for RLM {
    fn module_type_name(&self) -> &str {
        "RLM"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        self.validate_inputs(args)?;

        let output_field_names: Vec<String> = self
            .signature
            .output_fields()
            .map(|(n, _)| n.clone())
            .collect();

        let execution_tools = self.prepare_execution_tools();
        let variables = self.build_variables(args);

        // Create or use existing interpreter
        let (repl, should_shutdown) = if let Some(ref ext) = self.external_interpreter {
            // Inject tools into external interpreter
            {
                let mut interp = ext.lock().await;
                let tools = interp.tools_mut();
                for (name, tool) in execution_tools {
                    tools.insert(name, tool);
                }
            }
            (ext.clone(), false)
        } else {
            let mut interp: Box<dyn CodeInterpreter> =
                Box::new(crate::mock_interpreter::MockInterpreter::new(vec![]));
            let tools = interp.tools_mut();
            for (name, tool) in execution_tools {
                tools.insert(name, tool);
            }
            (Arc::new(Mutex::new(interp)), true)
        };

        let mut history = REPLHistory::new();

        for iteration in 0..self.max_iterations {
            match self
                .execute_iteration(
                    &repl,
                    &variables,
                    &history,
                    iteration,
                    args,
                    &output_field_names,
                )
                .await?
            {
                IterationResult::Done(prediction) => {
                    if should_shutdown {
                        let mut interp = repl.lock().await;
                        let _ = interp.shutdown().await;
                    }
                    return Ok(prediction);
                }
                IterationResult::Continue(new_history) => {
                    history = new_history;
                }
            }
        }

        // Max iterations reached — use extract fallback
        let result = self
            .extract_fallback(&variables, &history, &output_field_names)
            .await;

        if should_shutdown {
            let mut interp = repl.lock().await;
            let _ = interp.shutdown().await;
        }

        result
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![
            ("generate_action", &self.generate_action),
            ("extract", &self.extract),
        ]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![
            ("generate_action", &mut self.generate_action),
            ("extract", &mut self.extract),
        ]
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(Self {
            signature: self.signature.clone(),
            max_iterations: self.max_iterations,
            max_llm_calls: self.max_llm_calls,
            max_output_chars: self.max_output_chars,
            verbose: self.verbose,
            sub_lm: self.sub_lm.clone(),
            external_interpreter: self.external_interpreter.clone(),
            generate_action: self.generate_action.clone(),
            extract: self.extract.clone(),
        })
    }
}

// ========================================================================
// Helpers
// ========================================================================

fn strip_code_fences(code: &str) -> String {
    let code = code.trim();
    let fence_re = regex::Regex::new(r"^```(?:python|py)?\s*\n(.*)\n```\s*$").unwrap();
    if let Some(caps) = fence_re.captures(code) {
        caps.get(1)
            .map(|m| m.as_str().to_string())
            .unwrap_or_else(|| code.to_string())
    } else {
        code.to_string()
    }
}

fn build_signatures(signature: &Signature, max_llm_calls: usize) -> (Signature, Signature) {
    let inputs_str = signature
        .input_fields()
        .map(|(n, _)| format!("`{}`", n))
        .collect::<Vec<_>>()
        .join(", ");
    let final_output_names = signature
        .output_fields()
        .map(|(n, _)| n.clone())
        .collect::<Vec<_>>()
        .join(", ");
    let output_fields_str = signature
        .output_fields()
        .map(|(n, _)| format!("- {}", n))
        .collect::<Vec<_>>()
        .join("\n");

    let task_instructions = if !signature.instructions().is_empty() {
        format!("{}\n\n", signature.instructions())
    } else {
        String::new()
    };

    let instructions = format!(
        "{}{}",
        task_instructions,
        ACTION_INSTRUCTIONS_TEMPLATE
            .replace("{inputs}", &inputs_str)
            .replace("{final_output_names}", &final_output_names)
            .replace("{output_fields}", &output_fields_str)
            .replace("{max_llm_calls}", &max_llm_calls.to_string())
    );

    // Action signature
    let action_fields = vec![
        input_field("variables_info")
            .with_desc("Metadata about the variables available in the REPL"),
        input_field("repl_history").with_desc("Previous REPL code executions and their outputs"),
        input_field("iteration")
            .with_desc("Current iteration number (1-indexed) out of max_iterations"),
        output_field("reasoning").with_desc(
            "Think step-by-step: what do you know? What remains? Plan your next action.",
        ),
        output_field("code").with_desc(
            "Python code to execute. Use markdown code block format: ```python\n<code>\n```",
        ),
    ];
    let action_sig = Signature::new(action_fields, &instructions);

    // Extract signature
    let extract_instructions = format!(
        "{}Based on the REPL trajectory, extract the final outputs now.\n\nReview your trajectory to see what information you gathered and what values you computed, then provide the final outputs.",
        if !task_instructions.is_empty() {
            format!("The trajectory was generated with the following objective: \n{}\n", task_instructions)
        } else {
            String::new()
        }
    );

    let mut extract_fields = vec![
        input_field("variables_info")
            .with_desc("Metadata about the variables available in the REPL"),
        input_field("repl_history").with_desc("Your REPL interactions so far"),
    ];
    for (_, field) in signature.output_fields() {
        extract_fields.push(field.clone());
    }
    let extract_sig = Signature::new(extract_fields, &extract_instructions);

    (action_sig, extract_sig)
}

fn process_final_output(
    result: &FinalOutput,
    output_field_names: &[String],
) -> (Option<HashMap<String, serde_json::Value>>, Option<String>) {
    let raw = &result.output;

    if !raw.is_object() {
        return (
            None,
            Some(format!(
                "[Error] FINAL returned {}, expected dict with fields: {}",
                if raw.is_array() {
                    "array"
                } else if raw.is_string() {
                    "string"
                } else {
                    "non-dict"
                },
                output_field_names.join(", ")
            )),
        );
    }

    let raw_obj = raw.as_object().unwrap();
    let missing: Vec<&str> = output_field_names
        .iter()
        .filter(|n| !raw_obj.contains_key(n.as_str()))
        .map(|n| n.as_str())
        .collect();

    if !missing.is_empty() {
        let mut sorted = missing;
        sorted.sort();
        return (
            None,
            Some(format!(
                "[Error] Missing output fields: {}. Use SUBMIT({})",
                sorted.join(", "),
                output_field_names.join(", ")
            )),
        );
    }

    let mut parsed = HashMap::new();
    for name in output_field_names {
        if let Some(val) = raw_obj.get(name.as_str()) {
            parsed.insert(name.clone(), val.clone());
        }
    }

    (Some(parsed), None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message, LM};
    use crate::mock_interpreter::{MockInterpreter, MockResponse};
    use crate::settings;
    use std::sync::Mutex as StdMutex;

    struct MockLM {
        responses: Vec<String>,
        call_index: StdMutex<usize>,
        config: LMConfig,
    }

    impl MockLM {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.iter().map(|s| s.to_string()).collect(),
                call_index: StdMutex::new(0),
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(&self, _messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            let mut idx = self.call_index.lock().unwrap();
            let response = self.responses[*idx % self.responses.len()].clone();
            *idx += 1;
            Ok(vec![LMResponse::new(response, None)])
        }
        fn model(&self) -> &str {
            "mock"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[test]
    fn test_constructor() {
        let rlm = RLM::new(
            Signature::from_string("context, query -> answer").unwrap(),
            None,
            None,
            None,
            false,
            None,
            Some(Box::new(MockInterpreter::new(vec![]))),
        )
        .unwrap();
        assert_eq!(rlm.max_iterations, 20);
        assert_eq!(rlm.max_llm_calls, 50);
    }

    #[test]
    fn test_validates_required_inputs() {
        let rlm = RLM::new(
            Signature::from_string("context, query -> answer").unwrap(),
            None,
            None,
            None,
            false,
            None,
            Some(Box::new(MockInterpreter::new(vec![]))),
        )
        .unwrap();

        let result = rlm.validate_inputs(&Example::new().field("context", "test"));
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Missing required inputs"));
        assert!(err.contains("query"));
    }

    #[tokio::test]
    async fn test_forward_with_submit() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![MockResponse::final_output(
            serde_json::json!({"answer": "42"}),
        )]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nLet me compute this.\n\n[[ ## code ## ]]\n```python\nSUBMIT(answer=\"42\")\n```\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(5),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "What is 6*7?"))
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), Some("42"));
        assert!(result.get("trajectory").is_some());
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_iterates_until_submit() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::output("42\n"),
            MockResponse::final_output(serde_json::json!({"answer": "42"})),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nLet me check.\n\n[[ ## code ## ]]\n```python\nprint(question)\n```\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nFound it.\n\n[[ ## code ## ]]\n```python\nSUBMIT(answer=\"42\")\n```\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(10),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "What is 6*7?"))
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), Some("42"));
        let trajectory = result.get("trajectory").unwrap();
        let trajectory_json = serde_json::Value::from(trajectory.clone());
        assert_eq!(trajectory_json.as_array().unwrap().len(), 2);
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_extract_fallback() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::output("data explored\n"),
            MockResponse::output("more data\n"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nExploring\n\n[[ ## code ## ]]\nprint('data explored')\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nMore exploring\n\n[[ ## code ## ]]\nprint('more data')\n\n[[ ## completed ## ]]",
            // Extract fallback
            "[[ ## answer ## ]]\nbest guess 42\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(2),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();

        assert_eq!(
            result.get_str("final_reasoning"),
            Some("Extract forced final output")
        );
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_handles_execution_errors() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::error("NameError: 'undefined_var'"),
            MockResponse::final_output(serde_json::json!({"answer": "recovered"})),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nTry this\n\n[[ ## code ## ]]\nprint(undefined_var)\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nFixed it\n\n[[ ## code ## ]]\nSUBMIT(answer=\"recovered\")\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(5),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), Some("recovered"));
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_rejects_invalid_submit() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::final_output(serde_json::json!("not a dict")),
            MockResponse::final_output(serde_json::json!({"answer": "correct"})),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nTry\n\n[[ ## code ## ]]\nSUBMIT(\"not a dict\")\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nFix\n\n[[ ## code ## ]]\nSUBMIT(answer=\"correct\")\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(5),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), Some("correct"));
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_rejects_missing_fields() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::final_output(serde_json::json!({})),
            MockResponse::final_output(serde_json::json!({"answer": "got it"})),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\nTry\n\n[[ ## code ## ]]\nSUBMIT()\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nFix\n\n[[ ## code ## ]]\nSUBMIT(answer=\"got it\")\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let rlm = RLM::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(5),
            None,
            None,
            false,
            None,
            Some(Box::new(mock_interp)),
        )
        .unwrap();

        let result = rlm
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();

        assert_eq!(result.get_str("answer"), Some("got it"));
        settings::reset_settings();
    }

    #[test]
    fn test_strip_code_fences() {
        assert_eq!(strip_code_fences("```python\nprint(1)\n```"), "print(1)");
        assert_eq!(strip_code_fences("print(1)"), "print(1)");
        assert_eq!(strip_code_fences("```py\nx = 1\n```"), "x = 1");
    }

    #[test]
    fn test_process_final_output_valid() {
        let fo = FinalOutput::new(serde_json::json!({"answer": "42"}));
        let (parsed, error) = process_final_output(&fo, &["answer".to_string()]);
        assert!(error.is_none());
        let parsed = parsed.unwrap();
        assert_eq!(parsed["answer"], serde_json::json!("42"));
    }

    #[test]
    fn test_process_final_output_not_dict() {
        let fo = FinalOutput::new(serde_json::json!("not a dict"));
        let (parsed, error) = process_final_output(&fo, &["answer".to_string()]);
        assert!(parsed.is_none());
        assert!(error.unwrap().contains("expected dict"));
    }

    #[test]
    fn test_process_final_output_missing_fields() {
        let fo = FinalOutput::new(serde_json::json!({}));
        let (parsed, error) = process_final_output(&fo, &["answer".to_string()]);
        assert!(parsed.is_none());
        assert!(error.unwrap().contains("Missing output fields"));
    }
}
