//! Mock interpreter for testing — returns scripted responses.
//! Python equivalent: Uses MockInterpreter pattern from DSPy tests.

use crate::interpreter::{
    CodeInterpreter, CodeInterpreterError, ExecutionResult, FinalOutput, InterpreterTool,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// A scripted response for the mock interpreter.
#[derive(Debug, Clone)]
pub struct MockResponse {
    /// String output (simulates stdout)
    pub output: Option<String>,
    /// Final output (simulates SUBMIT() call)
    pub final_output: Option<serde_json::Value>,
    /// Error to throw
    pub error: Option<String>,
    /// Error type (default: RuntimeError)
    pub error_type: MockErrorType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MockErrorType {
    RuntimeError,
    SyntaxError,
    CodeInterpreterError,
}

impl Default for MockErrorType {
    fn default() -> Self {
        Self::RuntimeError
    }
}

impl MockResponse {
    /// Create a response with stdout output.
    pub fn output(s: impl Into<String>) -> Self {
        Self {
            output: Some(s.into()),
            final_output: None,
            error: None,
            error_type: MockErrorType::RuntimeError,
        }
    }

    /// Create a response with final output (SUBMIT).
    pub fn final_output(value: serde_json::Value) -> Self {
        Self {
            output: None,
            final_output: Some(value),
            error: None,
            error_type: MockErrorType::RuntimeError,
        }
    }

    /// Create an error response.
    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            output: None,
            final_output: None,
            error: Some(msg.into()),
            error_type: MockErrorType::RuntimeError,
        }
    }

    /// Create an error with specific type.
    pub fn error_with_type(msg: impl Into<String>, error_type: MockErrorType) -> Self {
        Self {
            output: None,
            final_output: None,
            error: Some(msg.into()),
            error_type,
        }
    }
}

pub struct MockInterpreter {
    responses: Vec<MockResponse>,
    call_index: usize,
    total_calls: usize,
    started: bool,
    tools: HashMap<String, InterpreterTool>,
}

impl MockInterpreter {
    pub fn new(responses: Vec<MockResponse>) -> Self {
        Self {
            responses,
            call_index: 0,
            total_calls: 0,
            started: false,
            tools: HashMap::new(),
        }
    }

    /// Get number of execute() calls made so far (persists across shutdown).
    pub fn call_count(&self) -> usize {
        self.total_calls
    }

    /// Reset the call index to replay responses.
    pub fn reset(&mut self) {
        self.call_index = 0;
    }
}

#[async_trait]
impl CodeInterpreter for MockInterpreter {
    async fn start(&mut self) -> Result<(), CodeInterpreterError> {
        self.started = true;
        Ok(())
    }

    async fn execute(
        &mut self,
        _code: &str,
        _variables: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<ExecutionResult, CodeInterpreterError> {
        if !self.started {
            self.start().await?;
        }

        if self.call_index >= self.responses.len() {
            return Ok(ExecutionResult::Output(None));
        }

        let response = self.responses[self.call_index].clone();
        self.call_index += 1;
        self.total_calls += 1;

        if let Some(error_msg) = response.error {
            let formatted = match response.error_type {
                MockErrorType::SyntaxError => format!("SyntaxError: {}", error_msg),
                MockErrorType::CodeInterpreterError => error_msg.clone(),
                MockErrorType::RuntimeError => format!("RuntimeError: {}", error_msg),
            };
            return Err(CodeInterpreterError::new(formatted));
        }

        if let Some(final_val) = response.final_output {
            return Ok(ExecutionResult::Final(FinalOutput::new(final_val)));
        }

        Ok(ExecutionResult::Output(response.output))
    }

    async fn shutdown(&mut self) -> Result<(), CodeInterpreterError> {
        self.started = false;
        self.call_index = 0;
        Ok(())
    }

    fn tools_mut(&mut self) -> &mut HashMap<String, InterpreterTool> {
        &mut self.tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_output() {
        let mut mock = MockInterpreter::new(vec![MockResponse::output("hello")]);
        let result = mock.execute("print('hello')", None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "hello"),
            _ => panic!("Expected output"),
        }
        assert_eq!(mock.call_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_final_output() {
        let mut mock = MockInterpreter::new(vec![MockResponse::final_output(
            serde_json::json!({"answer": "42"}),
        )]);
        let result = mock.execute("SUBMIT(answer='42')", None).await.unwrap();
        match result {
            ExecutionResult::Final(fo) => {
                assert_eq!(fo.output, serde_json::json!({"answer": "42"}));
            }
            _ => panic!("Expected FinalOutput"),
        }
    }

    #[tokio::test]
    async fn test_mock_error() {
        let mut mock = MockInterpreter::new(vec![MockResponse::error("name 'x' not defined")]);
        let result = mock.execute("print(x)", None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("name 'x' not defined"));
    }

    #[tokio::test]
    async fn test_mock_exhausted_returns_none() {
        let mut mock = MockInterpreter::new(vec![]);
        let result = mock.execute("anything", None).await.unwrap();
        match result {
            ExecutionResult::Output(None) => {}
            _ => panic!("Expected None output when exhausted"),
        }
    }

    #[tokio::test]
    async fn test_mock_multiple_calls() {
        let mut mock = MockInterpreter::new(vec![
            MockResponse::output("first"),
            MockResponse::error("oops"),
            MockResponse::output("third"),
        ]);

        // First call: output
        let r1 = mock.execute("a", None).await.unwrap();
        match r1 {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "first"),
            _ => panic!("Expected first output"),
        }

        // Second call: error
        let r2 = mock.execute("b", None).await;
        assert!(r2.is_err());

        // Third call: output
        let r3 = mock.execute("c", None).await.unwrap();
        match r3 {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "third"),
            _ => panic!("Expected third output"),
        }

        assert_eq!(mock.call_count(), 3);
    }

    #[tokio::test]
    async fn test_mock_shutdown_resets_index() {
        let mut mock = MockInterpreter::new(vec![MockResponse::output("hello")]);
        let _ = mock.execute("a", None).await;
        assert_eq!(mock.call_count(), 1);

        mock.shutdown().await.unwrap();
        // call_index reset, but total_calls persists
        assert_eq!(mock.call_count(), 1);

        // Can replay
        let result = mock.execute("a", None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "hello"),
            _ => panic!("Expected replay"),
        }
        assert_eq!(mock.call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_auto_starts() {
        let mut mock = MockInterpreter::new(vec![MockResponse::output("auto")]);
        // Don't call start() — should auto-start
        let result = mock.execute("x", None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "auto"),
            _ => panic!("Expected auto-start output"),
        }
    }
}
