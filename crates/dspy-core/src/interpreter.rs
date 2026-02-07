//! Code interpreter — abstract trait and supporting types.
//! Python equivalent: dspy/primitives/code_interpreter.py

use crate::error::DspyError;
use async_trait::async_trait;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Error raised during code interpretation.
#[derive(Debug, Clone)]
pub struct CodeInterpreterError {
    pub message: String,
}

impl CodeInterpreterError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CodeInterpreterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CodeInterpreterError {}

impl From<CodeInterpreterError> for DspyError {
    fn from(e: CodeInterpreterError) -> Self {
        DspyError::Other(e.message)
    }
}

/// Returned by interpreter.execute() when SUBMIT() is called in the code.
/// Signals that the execution loop should terminate and return the contained output.
#[derive(Debug, Clone)]
pub struct FinalOutput {
    pub output: serde_json::Value,
}

impl FinalOutput {
    pub fn new(output: serde_json::Value) -> Self {
        Self { output }
    }
}

impl std::fmt::Display for FinalOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FinalOutput({})", self.output)
    }
}

/// Result of code execution.
#[derive(Debug, Clone)]
pub enum ExecutionResult {
    /// Normal output (stdout text or null)
    Output(Option<String>),
    /// SUBMIT() was called with the final output
    Final(FinalOutput),
}

/// A host-side tool callable from interpreter code.
pub type InterpreterTool = Box<
    dyn Fn(
            HashMap<String, serde_json::Value>,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, String>> + Send>>
        + Send
        + Sync,
>;

/// Output field definition for typed SUBMIT signature.
#[derive(Debug, Clone)]
pub struct OutputFieldDef {
    pub name: String,
}

/// Protocol for code execution environments.
///
/// Implementations must provide:
/// - start(): Initialize (optional, can be lazy)
/// - execute(): Run code and return results
/// - shutdown(): Clean up resources
///
/// State persists across execute() calls within a session.
#[async_trait]
pub trait CodeInterpreter: Send + Sync {
    /// Initialize the interpreter. Idempotent.
    async fn start(&mut self) -> Result<(), CodeInterpreterError>;

    /// Execute Python code and return the result.
    async fn execute(
        &mut self,
        code: &str,
        variables: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<ExecutionResult, CodeInterpreterError>;

    /// Release resources and terminate the session.
    async fn shutdown(&mut self) -> Result<(), CodeInterpreterError>;

    /// Get mutable access to tools map for injection.
    fn tools_mut(&mut self) -> &mut HashMap<String, InterpreterTool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_interpreter_error() {
        let err = CodeInterpreterError::new("test error");
        assert_eq!(err.message, "test error");
        assert_eq!(format!("{}", err), "test error");
    }

    #[test]
    fn test_final_output_display() {
        let fo = FinalOutput::new(serde_json::json!({"answer": "42"}));
        let display = format!("{}", fo);
        assert!(display.contains("42"));
    }

    #[test]
    fn test_execution_result_variants() {
        let output = ExecutionResult::Output(Some("hello".to_string()));
        match output {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "hello"),
            _ => panic!("Expected Output"),
        }

        let final_out = ExecutionResult::Final(FinalOutput::new(serde_json::json!(42)));
        match final_out {
            ExecutionResult::Final(fo) => assert_eq!(fo.output, serde_json::json!(42)),
            _ => panic!("Expected Final"),
        }
    }
}
