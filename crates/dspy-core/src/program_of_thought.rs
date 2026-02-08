//! ProgramOfThought — LLM generates Python code to solve a problem.
//! Python equivalent: dspy/predict/program_of_thought.py
//!
//! Uses ChainOfThought for code generation, executes in a code interpreter,
//! and retries on errors up to max_iters.

use crate::chain_of_thought::ChainOfThought;
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::interpreter::{CodeInterpreter, ExecutionResult, FinalOutput};
use crate::module_trait::Module;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::signature::{input_field, output_field, Signature};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct ProgramOfThought {
    max_iters: usize,
    interpreter: Arc<Mutex<Box<dyn CodeInterpreter>>>,
    input_field_names: Vec<String>,
    output_field_names: Vec<String>,
    code_generate: ChainOfThought,
    code_regenerate: ChainOfThought,
    generate_output: ChainOfThought,
}

impl ProgramOfThought {
    pub fn new(
        signature: Signature,
        max_iters: Option<usize>,
        interpreter: Box<dyn CodeInterpreter>,
    ) -> Self {
        let input_names: Vec<String> = signature
            .input_fields()
            .map(|(name, _)| name.clone())
            .collect();
        let output_names: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| name.clone())
            .collect();

        let code_generate = ChainOfThought::new(build_signature(&signature, "generate"));
        let code_regenerate = ChainOfThought::new(build_signature(&signature, "regenerate"));
        let generate_output = ChainOfThought::new(build_signature(&signature, "answer"));

        Self {
            max_iters: max_iters.unwrap_or(3),
            interpreter: Arc::new(Mutex::new(interpreter)),
            input_field_names: input_names,
            output_field_names: output_names,
            code_generate,
            code_regenerate,
            generate_output,
        }
    }

    /// Parse generated code from LLM output.
    pub fn parse_code(code_data: &Prediction) -> (String, Option<String>) {
        let raw_code_full = code_data.get_str("generated_code").unwrap_or("");
        // Strip after --- or triple newlines
        let raw_code = raw_code_full
            .split("---")
            .next()
            .unwrap_or("")
            .split("\n\n\n")
            .next()
            .unwrap_or("");

        // Extract from markdown code fence
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

        if !code_block.contains('\n') && code_block.matches('=').count() > 1 {
            return (
                raw_code.to_string(),
                Some("Error: Code format is not correct.".to_string()),
            );
        }

        // If last line is an assignment, append the variable name as a bare expression
        let lines: Vec<&str> = code_block.lines().collect();
        let last_line = lines.last().unwrap_or(&"").trim();
        let assign_re = regex::Regex::new(r"^(\w+)\s*=").unwrap();
        let mut final_code = code_block.clone();
        if lines.len() > 1 {
            if let Some(caps) = assign_re.captures(last_line) {
                if let Some(var_name) = caps.get(1) {
                    final_code = format!("{}\n{}", code_block, var_name.as_str());
                }
            }
        }

        (final_code, None)
    }

    /// Execute code in the interpreter and handle errors.
    async fn execute_code(&self, code: &str) -> (Option<String>, Option<String>) {
        if code.trim().is_empty() {
            return (
                None,
                Some("Error: Empty code before execution.".to_string()),
            );
        }

        let mut interp = self.interpreter.lock().await;
        match interp.execute(code, None).await {
            Ok(ExecutionResult::Final(FinalOutput { output })) => {
                let output_str = if output.is_object() || output.is_array() {
                    serde_json::to_string(&output).unwrap_or_default()
                } else if let Some(s) = output.as_str() {
                    s.to_string()
                } else {
                    output.to_string()
                };
                (Some(output_str), None)
            }
            Ok(ExecutionResult::Output(Some(s))) => (Some(s), None),
            Ok(ExecutionResult::Output(None)) => (Some(String::new()), None),
            Err(e) => (None, Some(e.message)),
        }
    }
}

#[async_trait]
impl Module for ProgramOfThought {
    fn module_type_name(&self) -> &str {
        "ProgramOfThought"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        let mut input_kwargs = Example::new();
        for name in &self.input_field_names {
            if let Some(val) = args.get(name) {
                input_kwargs = input_kwargs.field(name, val.to_string());
            }
        }

        // Initial code generation
        let code_data = self.code_generate.call(&input_kwargs).await?;
        let (mut code, mut error) = Self::parse_code(&code_data);

        let mut output: Option<String> = None;
        if error.is_none() {
            let (exec_output, exec_error) = self.execute_code(&code).await;
            output = exec_output;
            error = exec_error;
        }

        // Retry loop
        let mut hop = 1;
        while error.is_some() {
            if hop >= self.max_iters {
                let mut interp = self.interpreter.lock().await;
                let _ = interp.shutdown().await;
                return Err(DspyError::Other(format!(
                    "Max hops reached. Failed to run ProgramOfThought: {}",
                    error.unwrap_or_default()
                )));
            }

            let mut regen_args = input_kwargs.clone();
            regen_args = regen_args.field("previous_code", code.clone());
            regen_args = regen_args.field("error", error.unwrap_or_default());

            let regen_data = self.code_regenerate.call(&regen_args).await?;
            let (new_code, new_error) = Self::parse_code(&regen_data);
            code = new_code;
            error = new_error;

            if error.is_none() {
                let (exec_output, exec_error) = self.execute_code(&code).await;
                output = exec_output;
                error = exec_error;
            }
            hop += 1;
        }

        // Extract final answer
        let mut answer_args = input_kwargs.clone();
        answer_args = answer_args.field("final_generated_code", code);
        answer_args = answer_args.field("code_output", output.unwrap_or_default());

        let output_result = self.generate_output.call(&answer_args).await?;
        let mut interp = self.interpreter.lock().await;
        let _ = interp.shutdown().await;
        Ok(output_result)
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        let mut preds = Vec::new();
        preds.extend(
            self.code_generate
                .named_predictors()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("code_generate.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds.extend(
            self.code_regenerate
                .named_predictors()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("code_regenerate.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds.extend(
            self.generate_output
                .named_predictors()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("generate_output.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        let mut preds = Vec::new();
        preds.extend(
            self.code_generate
                .named_predictors_mut()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("code_generate.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds.extend(
            self.code_regenerate
                .named_predictors_mut()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("code_regenerate.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds.extend(
            self.generate_output
                .named_predictors_mut()
                .into_iter()
                .map(|(n, p)| {
                    let name: &str = Box::leak(format!("generate_output.{}", n).into_boxed_str());
                    (name, p)
                }),
        );
        preds
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        // Cannot easily deep-copy interpreter, return a clone of the predictors
        Box::new(Self {
            max_iters: self.max_iters,
            interpreter: self.interpreter.clone(),
            input_field_names: self.input_field_names.clone(),
            output_field_names: self.output_field_names.clone(),
            code_generate: ChainOfThought::new(self.code_generate.predict().signature.clone()),
            code_regenerate: ChainOfThought::new(self.code_regenerate.predict().signature.clone()),
            generate_output: ChainOfThought::new(self.generate_output.predict().signature.clone()),
        })
    }
}

// ========================================================================
// Signature building helpers
// ========================================================================

fn build_signature(original: &Signature, mode: &str) -> Signature {
    let mut fields = Vec::new();

    // Copy input fields
    for (_, field) in original.input_fields() {
        fields.push(field.clone());
    }

    let output_names: Vec<String> = original
        .output_fields()
        .map(|(n, _)| format!("`{}`", n))
        .collect();
    let _output_names_str = output_names.join(", ");

    match mode {
        "generate" => {
            fields.push(
                output_field("generated_code")
                    .with_prefix("Code:")
                    .with_desc("python code that answers the question"),
            );
        }
        "regenerate" => {
            fields.push(
                input_field("previous_code")
                    .with_prefix("Previous Code:")
                    .with_desc("previously-generated python code that errored"),
            );
            fields.push(
                input_field("error")
                    .with_prefix("Error:")
                    .with_desc("error message from previously-generated python code"),
            );
            fields.push(
                output_field("generated_code")
                    .with_prefix("Code:")
                    .with_desc("python code that answers the question"),
            );
        }
        "answer" => {
            fields.push(
                input_field("final_generated_code")
                    .with_prefix("Code:")
                    .with_desc("python code that answers the question"),
            );
            fields.push(
                input_field("code_output")
                    .with_prefix("Code Output:")
                    .with_desc("output of previously-generated python code"),
            );
            for (_, field) in original.output_fields() {
                fields.push(field.clone());
            }
        }
        _ => {}
    }

    let instructions = build_instruction(original, mode);
    Signature::new(fields, &instructions)
}

fn build_instruction(original: &Signature, mode: &str) -> String {
    let field_names = build_signature_field_names(original, mode);
    let mode_inputs = field_names
        .0
        .iter()
        .map(|n| format!("`{}`", n))
        .collect::<Vec<_>>()
        .join(", ");
    let mode_outputs = field_names
        .1
        .iter()
        .map(|n| format!("`{}`", n))
        .collect::<Vec<_>>()
        .join(", ");
    let final_outputs = original
        .output_fields()
        .map(|(n, _)| format!("`{}`", n))
        .collect::<Vec<_>>()
        .join(", ");

    match mode {
        "generate" => {
            format!(
                "You will be given {} and you will respond with {}.\n\
                 Generating executable Python code that programmatically computes the correct {}.\n\
                 After you're done with the computation and think you have the final output, make sure to submit your output by calling the preloaded function `SUBMIT()`.\n\
                 You must structure your output in a dict, like {{\"field_a\": value_a, ...}}, with the correct value mapping for the field(s): {}.",
                mode_inputs, mode_outputs, mode_outputs, final_outputs
            )
        }
        "regenerate" => {
            format!(
                "You are given {} due to an error in previous code.\n\
                 Your task is to correct the error and provide the new `generated_code`.",
                mode_inputs
            )
        }
        "answer" => {
            format!(
                "Given the final code {}, provide the final {}.",
                mode_inputs, mode_outputs
            )
        }
        _ => String::new(),
    }
}

fn build_signature_field_names(original: &Signature, mode: &str) -> (Vec<String>, Vec<String>) {
    let mut inputs: Vec<String> = original.input_fields().map(|(n, _)| n.clone()).collect();

    match mode {
        "regenerate" => {
            inputs.push("previous_code".to_string());
            inputs.push("error".to_string());
        }
        "answer" => {
            inputs.push("final_generated_code".to_string());
            inputs.push("code_output".to_string());
        }
        _ => {}
    }

    let outputs = if mode == "answer" {
        original.output_fields().map(|(n, _)| n.clone()).collect()
    } else {
        vec!["generated_code".to_string()]
    };

    (inputs, outputs)
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
            Ok(vec![LMResponse {
                text: response,
                usage: None,
            }])
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
    fn test_constructor_from_string() {
        let mock = MockInterpreter::new(vec![MockResponse::output("42")]);
        let pot = ProgramOfThought::new(
            Signature::from_string("question -> answer").unwrap(),
            None,
            Box::new(mock),
        );
        // Just verify construction doesn't panic
        assert_eq!(pot.max_iters, 3);
    }

    #[test]
    fn test_parse_code_from_fence() {
        let pred = Prediction::from_example(
            Example::new().field("generated_code", "```python\nprint(\"hello\")\n```"),
        );
        let (code, error) = ProgramOfThought::parse_code(&pred);
        assert!(error.is_none());
        assert_eq!(code, "print(\"hello\")");
    }

    #[test]
    fn test_parse_code_raw() {
        let pred = Prediction::from_example(
            Example::new().field("generated_code", "result = 1 + 2\nresult"),
        );
        let (code, error) = ProgramOfThought::parse_code(&pred);
        assert!(error.is_none());
        assert!(code.contains("result = 1 + 2"));
    }

    #[test]
    fn test_parse_code_empty() {
        let pred = Prediction::from_example(Example::new().field("generated_code", ""));
        let (_, error) = ProgramOfThought::parse_code(&pred);
        assert!(error.is_some());
    }

    #[tokio::test]
    async fn test_forward_executes_and_extracts() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![MockResponse::output("{\"answer\": \"42\"}")]);

        let mock_lm = Arc::new(MockLM::new(vec![
            // code_generate (CoT: reasoning + generated_code)
            "[[ ## reasoning ## ]]\nLet me compute this.\n\n[[ ## generated_code ## ]]\n```python\nprint(json.dumps({\"answer\": \"42\"}))\n```\n\n[[ ## completed ## ]]",
            // generate_output (CoT: reasoning + answer)
            "[[ ## reasoning ## ]]\nThe answer is 42.\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let pot = ProgramOfThought::new(
            Signature::from_string("question -> answer").unwrap(),
            None,
            Box::new(mock_interp),
        );

        let result = pot
            .forward(&Example::new().field("question", "What is 6*7?"))
            .await
            .unwrap();
        // Should get an answer field
        assert!(result.get_str("answer").is_some());
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_retries_on_error() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::error("NameError: name 'foo' is not defined"),
            MockResponse::output("42"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            // code_generate
            "[[ ## reasoning ## ]]\nTry this\n\n[[ ## generated_code ## ]]\nprint(foo)\n\n[[ ## completed ## ]]",
            // code_regenerate (retry)
            "[[ ## reasoning ## ]]\nFixed it\n\n[[ ## generated_code ## ]]\nprint(42)\n\n[[ ## completed ## ]]",
            // generate_output
            "[[ ## reasoning ## ]]\nResult is 42\n\n[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let pot = ProgramOfThought::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(3),
            Box::new(mock_interp),
        );

        let result = pot
            .forward(&Example::new().field("question", "test"))
            .await
            .unwrap();
        assert!(result.get_str("answer").is_some());
        settings::reset_settings();
    }

    #[tokio::test]
    async fn test_forward_throws_after_max_iters() {
        settings::reset_settings();
        let mock_interp = MockInterpreter::new(vec![
            MockResponse::error("error1"),
            MockResponse::error("error2"),
            MockResponse::error("error3"),
        ]);

        let mock_lm = Arc::new(MockLM::new(vec![
            "[[ ## reasoning ## ]]\na\n\n[[ ## generated_code ## ]]\nbad_code()\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nb\n\n[[ ## generated_code ## ]]\nstill_bad()\n\n[[ ## completed ## ]]",
            "[[ ## reasoning ## ]]\nc\n\n[[ ## generated_code ## ]]\nstill_bad()\n\n[[ ## completed ## ]]",
        ]));
        settings::configure(settings::Settings::new().with_lm(mock_lm));

        let pot = ProgramOfThought::new(
            Signature::from_string("question -> answer").unwrap(),
            Some(3),
            Box::new(mock_interp),
        );

        let result = pot.forward(&Example::new().field("question", "test")).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Max hops reached"));
        settings::reset_settings();
    }
}
