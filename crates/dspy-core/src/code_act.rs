//! CodeAct — Code-based agent that uses a code interpreter + tools.
//! Python equivalent: dspy/predict/code_act.py
//!
//! Combines ReAct-style iteration with code execution. The LLM generates
//! Python code at each step, which is executed in an interpreter with
//! access to predefined tools.

use crate::chain_of_thought::ChainOfThought;
use crate::error::Result;
use crate::example::Example;
use crate::interpreter::{CodeInterpreter, ExecutionResult, FinalOutput};
use crate::module_trait::Module;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::signature::{input_field, output_field, Signature};
use crate::tool::Tool;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct CodeAct {
    tools: Vec<Tool>,
    max_iters: usize,
    interpreter: Arc<Mutex<Box<dyn CodeInterpreter>>>,
    original_signature: Signature,
    code_act_predict: Predict,
    extractor: ChainOfThought,
}

impl CodeAct {
    pub fn new(
        signature: Signature,
        tools: Vec<Tool>,
        max_iters: Option<usize>,
        interpreter: Box<dyn CodeInterpreter>,
    ) -> Self {
        let instructions = build_instructions(&signature, &tools);

        // Build code act signature
        let mut fields = Vec::new();
        for (_, field) in signature.input_fields() {
            fields.push(field.clone());
        }
        fields.push(
            input_field("trajectory").with_desc("Previous code execution trajectory"),
        );
        fields.push(output_field("generated_code").with_desc(
            "Python code that when executed, produces output relevant to answering the question",
        ));
        fields.push(output_field("finished").with_desc(
            "a boolean flag to determine if the process is done",
        ));

        let code_act_sig = Signature::new(fields, &instructions);
        let code_act_predict = Predict::new(code_act_sig);

        // Build extract signature
        let mut extract_fields = Vec::new();
        for (_, field) in signature.input_fields() {
            extract_fields.push(field.clone());
        }
        extract_fields.push(
            input_field("trajectory").with_desc("Previous code execution trajectory"),
        );
        for (_, field) in signature.output_fields() {
            extract_fields.push(field.clone());
        }

        let extract_sig = Signature::new(
            extract_fields,
            signature.instructions(),
        );
        let extractor = ChainOfThought::new(extract_sig);

        Self {
            tools,
            max_iters: max_iters.unwrap_or(5),
            interpreter: Arc::new(Mutex::new(interpreter)),
            original_signature: signature,
            code_act_predict,
            extractor,
        }
    }

    /// Parse generated code from LLM output.
    fn parse_code(code_data: &Prediction) -> (String, Option<String>) {
        let raw_code_full = code_data.get_str("generated_code").unwrap_or("");
        let raw_code = raw_code_full
            .split("---")
            .next()
            .unwrap_or("")
            .split("\n\n\n")
            .next()
            .unwrap_or("");

        let fence_re = regex::Regex::new(r"```(?:python|py)?[ \n](.*?)[ \n]```?").unwrap();
        let code_block = if let Some(caps) = fence_re.captures(raw_code) {
            caps.get(1).map(|m| m.as_str()).unwrap_or(raw_code)
        } else {
            raw_code
        };
        let code_block = code_block.replace("\\n", "\n");

        if code_block.trim().is_empty() {
            return (
                raw_code.to_string(),
                Some("Error: Empty code after parsing.".to_string()),
            );
        }

        (code_block, None)
    }

    /// Execute code in the interpreter.
    async fn execute_code(
        &self,
        code: &str,
    ) -> (Option<String>, Option<String>) {
        if code.trim().is_empty() {
            return (None, Some("Error: Empty code before execution.".to_string()));
        }

        let mut interp = self.interpreter.lock().await;
        match interp.execute(code, None).await {
            Ok(ExecutionResult::Final(FinalOutput { output })) => {
                let s = if output.is_object() || output.is_array() {
                    serde_json::to_string(&output).unwrap_or_default()
                } else if let Some(s) = output.as_str() {
                    s.to_string()
                } else {
                    output.to_string()
                };
                (Some(s), None)
            }
            Ok(ExecutionResult::Output(Some(s))) => (Some(s), None),
            Ok(ExecutionResult::Output(None)) => (Some(String::new()), None),
            Err(e) => (None, Some(e.message)),
        }
    }

    /// Format trajectory dict for the extractor.
    fn format_trajectory(trajectory: &HashMap<String, String>) -> String {
        let max_chars = 50000;
        let result = serde_json::to_string_pretty(trajectory).unwrap_or_default();
        if result.len() > max_chars {
            format!("{}\n... (truncated)", &result[..max_chars])
        } else {
            result
        }
    }
}

#[async_trait]
impl Module for CodeAct {
    async fn forward(&self, args: &Example) -> Result<Prediction> {
        // Inject tool source code into interpreter
        {
            let mut interp = self.interpreter.lock().await;
            for tool in &self.tools {
                // Tools in Rust don't have sourceCode like TS — skip source injection
                // The tool functions are called via the interpreter's tool mechanism
                let _ = tool;
                let _ = &mut interp;
            }
        }

        let mut input_kwargs = Example::new();
        for (name, _) in self.original_signature.input_fields() {
            if let Some(val) = args.get(name) {
                input_kwargs = input_kwargs.field(name, val.to_string());
            }
        }

        let mut trajectory: HashMap<String, String> = HashMap::new();
        let max_iters = args
            .get("max_iters")
            .and_then(|v| match v {
                crate::value::Value::Integer(n) => Some(*n as usize),
                crate::value::Value::String(s) => s.parse::<usize>().ok(),
                _ => None,
            })
            .unwrap_or(self.max_iters);

        for idx in 0..max_iters {
            let mut predict_args = input_kwargs.clone();
            predict_args = predict_args.field(
                "trajectory",
                serde_json::to_string(&trajectory).unwrap_or_default(),
            );

            let code_data = self.code_act_predict.forward(&predict_args).await?;
            let (code, parse_error) = Self::parse_code(&code_data);

            if let Some(err) = parse_error {
                trajectory.insert(
                    format!("observation_{}", idx),
                    format!("Failed to parse the generated code: {}", err),
                );
                continue;
            }

            trajectory.insert(format!("generated_code_{}", idx), code.clone());
            let (output, exec_error) = self.execute_code(&code).await;

            if exec_error.is_none() {
                trajectory.insert(
                    format!("code_output_{}", idx),
                    output.unwrap_or_default(),
                );
            } else {
                trajectory.insert(
                    format!("observation_{}", idx),
                    format!(
                        "Failed to execute the generated code: {}",
                        exec_error.unwrap_or_default()
                    ),
                );
            }

            // Check if the agent says it's finished (handle both string and bool)
            let is_finished = if let Some(val) = code_data.get("finished") {
                match val {
                    crate::value::Value::Bool(b) => *b,
                    crate::value::Value::String(s) => {
                        s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("yes")
                    }
                    _ => false,
                }
            } else {
                false
            };
            if is_finished {
                break;
            }
        }

        // Extract final answer
        let mut extract_args = input_kwargs.clone();
        extract_args =
            extract_args.field("trajectory", Self::format_trajectory(&trajectory));

        let extract_result = self.extractor.forward(&extract_args).await?;

        {
            let mut interp = self.interpreter.lock().await;
            let _ = interp.shutdown().await;
        }

        // Build final prediction with trajectory
        let mut final_fields: HashMap<String, crate::value::Value> = HashMap::new();
        let extract_map = extract_result.example.to_map();
        for (k, v) in extract_map {
            final_fields.insert(k.clone(), v.clone());
        }
        let trajectory_json = serde_json::to_value(&trajectory).unwrap_or_default();
        final_fields.insert(
            "trajectory".to_string(),
            crate::value::Value::from(trajectory_json),
        );

        Ok(Prediction::from_example(
            Example::from_map(final_fields),
        ))
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![
            ("code_act_predict", &self.code_act_predict),
            ("extractor", self.extractor.predict()),
        ]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![
            ("code_act_predict", &mut self.code_act_predict),
            ("extractor", self.extractor.predict_mut()),
        ]
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(Self {
            tools: Vec::new(), // Tools can't be cloned (contain closures)
            max_iters: self.max_iters,
            interpreter: self.interpreter.clone(),
            original_signature: self.original_signature.clone(),
            code_act_predict: self.code_act_predict.clone(),
            extractor: ChainOfThought::new(self.extractor.predict().signature.clone()),
        })
    }
}

// ========================================================================
// Instruction building
// ========================================================================

fn build_instructions(sig: &Signature, tools: &[Tool]) -> String {
    let mut lines = Vec::new();
    if !sig.instructions().is_empty() {
        lines.push(format!("{}\n", sig.instructions()));
    }

    let inputs = sig
        .input_fields()
        .map(|(k, _)| format!("`{}`", k))
        .collect::<Vec<_>>()
        .join(", ");
    let outputs = sig
        .output_fields()
        .map(|(k, _)| format!("`{}`", k))
        .collect::<Vec<_>>()
        .join(", ");

    lines.push(format!(
        "You are an intelligent agent. For each episode, you will receive the fields {} as input.\n\
         Your goal is to generate executable Python code that collects any necessary information for producing {}.\n\
         For each iteration, you will generate a code snippet that either solves the task or progresses towards the solution.\n\
         Ensure any output you wish to extract from the code is printed to the console. The code should be enclosed in a fenced code block.\n\
         When all information for producing the outputs ({}) are available to be extracted, mark `finished=True` besides the final Python code.\n\
         You have access to the Python Standard Library and the following functions:",
        inputs, outputs, outputs
    ));

    for (idx, tool) in tools.iter().enumerate() {
        lines.push(format!(
            "({}) {}: {}",
            idx + 1,
            tool.name,
            if tool.desc.is_empty() {
                "No description"
            } else {
                &tool.desc
            }
        ));
    }

    lines.join("\n")
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
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> Result<Vec<LMResponse>> {
            let mut idx = self.call_index.lock().unwrap();
            let response = self.responses[*idx % self.responses.len()].clone();
            *idx += 1;
            Ok(vec![LMResponse {
                text: response,
                usage: None,
            }])
        }
        fn model(&self) -> &str { "mock" }
        fn config(&self) -> &LMConfig { &self.config }
        fn dump_state(&self) -> serde_json::Value { serde_json::json!({}) }
    }

    #[test]
    fn test_constructor() {
        let mock = MockInterpreter::new(vec![]);
        let tool = Tool::new(
            "factorial",
            "Compute factorial",
            HashMap::new(),
            |_| async move { Ok(serde_json::json!(120)) },
        );
        let codeact = CodeAct::new(
            Signature::from_string("n -> factorial_result").unwrap(),
            vec![tool],
            None,
            Box::new(mock),
        );
        assert_eq!(codeact.max_iters, 5);
    }

    #[tokio::test]
    async fn test_forward_runs_and_extracts() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::output("120"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            // codeact predict (Predict with generated_code + finished)
            "[[ ## generated_code ## ]]\n```python\nresult = factorial(5)\nprint(result)\n```\n\n[[ ## finished ## ]]\ntrue\n\n[[ ## completed ## ]]",
            // extractor (ChainOfThought with reasoning + factorial_result)
            "[[ ## reasoning ## ]]\nFactorial of 5 is 120\n\n[[ ## factorial_result ## ]]\n120\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let tool = Tool::new(
            "factorial",
            "Compute factorial",
            HashMap::new(),
            |_| async move { Ok(serde_json::json!(120)) },
        );

        let codeact = CodeAct::new(
            Signature::from_string("n -> factorial_result").unwrap(),
            vec![tool],
            Some(3),
            Box::new(mock_interp),
        );

        let result = codeact
            .forward(&Example::new().field("n", "5"))
            .await
            .unwrap();
        assert!(result.get_str("factorial_result").is_some());
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_iterates_on_errors() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::error("NameError"),
            MockResponse::output("42"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            // First iteration - bad code
            "[[ ## generated_code ## ]]\nbad_code()\n\n[[ ## finished ## ]]\nfalse\n\n[[ ## completed ## ]]",
            // Second iteration - good code
            "[[ ## generated_code ## ]]\nprint(42)\n\n[[ ## finished ## ]]\ntrue\n\n[[ ## completed ## ]]",
            // Extractor
            "[[ ## reasoning ## ]]\nGot 42\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let codeact = CodeAct::new(
            Signature::from_string("question -> answer").unwrap(),
            vec![],
            Some(5),
            Box::new(mock_interp),
        );

        let result = codeact
            .forward(&Example::new().field("question", "what is 42?"))
            .await
            .unwrap();
        assert!(result.get("trajectory").is_some());
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_respects_max_iters() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::output("a"),
            MockResponse::output("b"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## generated_code ## ]]\nprint('a')\n\n[[ ## finished ## ]]\nfalse\n\n[[ ## completed ## ]]",
            "[[ ## generated_code ## ]]\nprint('b')\n\n[[ ## finished ## ]]\nfalse\n\n[[ ## completed ## ]]",
            // extractor
            "[[ ## reasoning ## ]]\npartial\n\n[[ ## answer ## ]]\npartial\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let codeact = CodeAct::new(
            Signature::from_string("question -> answer").unwrap(),
            vec![],
            Some(2),
            Box::new(mock_interp),
        );

        let result = codeact
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();
        assert!(result.get_str("answer").is_some());
        settings::reset_settings();
    }
}
