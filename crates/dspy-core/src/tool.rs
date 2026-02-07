//! Tool — wraps a function for tool calling / function calling in LLMs.
//! Python equivalent: dspy/adapters/types/tool.py
//!
//! In Rust, tools use explicit schemas rather than runtime introspection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::error::{DspyError, Result};

/// JSON schema for a tool argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolArg {
    #[serde(rename = "type")]
    pub arg_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
}

/// A tool definition for LLM function calling.
pub struct Tool {
    pub name: String,
    pub desc: String,
    pub args: HashMap<String, ToolArg>,
    func: Box<dyn Fn(HashMap<String, serde_json::Value>) -> Pin<Box<dyn Future<Output = Result<serde_json::Value>> + Send>> + Send + Sync>,
}

impl Tool {
    pub fn new<F, Fut>(
        name: impl Into<String>,
        desc: impl Into<String>,
        args: HashMap<String, ToolArg>,
        func: F,
    ) -> Self
    where
        F: Fn(HashMap<String, serde_json::Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            desc: desc.into(),
            args,
            func: Box::new(move |kwargs| Box::pin(func(kwargs))),
        }
    }

    /// Call the tool with the given arguments.
    pub async fn call(&self, kwargs: HashMap<String, serde_json::Value>) -> Result<serde_json::Value> {
        self.validate_args(&kwargs)?;
        (self.func)(kwargs).await
    }

    /// Validate arguments against the schema.
    fn validate_args(&self, kwargs: &HashMap<String, serde_json::Value>) -> Result<()> {
        for key in kwargs.keys() {
            if !self.args.contains_key(key) {
                return Err(DspyError::Other(format!(
                    "Arg '{}' is not in the tool's args",
                    key
                )));
            }
        }
        Ok(())
    }

    /// Format as OpenAI/LiteLLM function calling schema.
    pub fn format_as_function(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {
                    "type": "object",
                    "properties": self.args,
                    "required": self.args.keys().collect::<Vec<_>>(),
                },
            },
        })
    }
}

impl std::fmt::Display for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.desc.is_empty() {
            write!(f, "{}. It takes arguments {:?}.", self.name, self.args.keys().collect::<Vec<_>>())
        } else {
            write!(
                f,
                "{}, whose description is <desc>{}</desc>. It takes arguments {:?}.",
                self.name,
                self.desc,
                self.args.keys().collect::<Vec<_>>()
            )
        }
    }
}

/// A single tool call from an LLM response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, serde_json::Value>,
}

impl ToolCall {
    pub fn new(name: impl Into<String>, args: HashMap<String, serde_json::Value>) -> Self {
        Self {
            name: name.into(),
            args,
        }
    }

    /// Execute this tool call against a set of available tools.
    pub async fn execute(&self, tools: &[&Tool]) -> Result<serde_json::Value> {
        let tool = tools
            .iter()
            .find(|t| t.name == self.name)
            .ok_or_else(|| DspyError::Other(format!("Tool function '{}' not found", self.name)))?;
        tool.call(self.args.clone()).await
    }

    pub fn format(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.args,
            },
        })
    }
}

/// A collection of tool calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCalls {
    pub tool_calls: Vec<ToolCall>,
}

impl ToolCalls {
    pub fn new(tool_calls: Vec<ToolCall>) -> Self {
        Self { tool_calls }
    }

    pub fn format(&self) -> serde_json::Value {
        serde_json::json!({
            "tool_calls": self.tool_calls.iter().map(|tc| tc.format()).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_call() {
        let mut args = HashMap::new();
        args.insert(
            "x".to_string(),
            ToolArg {
                arg_type: "number".to_string(),
                description: Some("First number".to_string()),
                default: None,
            },
        );
        args.insert(
            "y".to_string(),
            ToolArg {
                arg_type: "number".to_string(),
                description: Some("Second number".to_string()),
                default: None,
            },
        );

        let tool = Tool::new("add", "Add two numbers", args, |kwargs| async move {
            let x = kwargs.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let y = kwargs.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(serde_json::json!(x + y))
        });

        let mut call_args = HashMap::new();
        call_args.insert("x".to_string(), serde_json::json!(3));
        call_args.insert("y".to_string(), serde_json::json!(4));

        let result = tool.call(call_args).await.unwrap();
        assert_eq!(result, serde_json::json!(7.0));
    }

    #[tokio::test]
    async fn test_tool_invalid_arg() {
        let args = HashMap::new();
        let tool = Tool::new("noop", "Does nothing", args, |_| async move {
            Ok(serde_json::json!(null))
        });

        let mut call_args = HashMap::new();
        call_args.insert("unknown".to_string(), serde_json::json!("val"));

        let result = tool.call(call_args).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_format_as_function() {
        let mut args = HashMap::new();
        args.insert(
            "query".to_string(),
            ToolArg {
                arg_type: "string".to_string(),
                description: Some("Search query".to_string()),
                default: None,
            },
        );

        let tool = Tool::new("search", "Search the web", args, |_| async move {
            Ok(serde_json::json!("result"))
        });

        let schema = tool.format_as_function();
        assert_eq!(schema["function"]["name"], "search");
        assert_eq!(schema["function"]["description"], "Search the web");
    }

    #[test]
    fn test_tool_call_format() {
        let mut args = HashMap::new();
        args.insert("q".to_string(), serde_json::json!("hello"));

        let tc = ToolCall::new("search", args);
        let formatted = tc.format();
        assert_eq!(formatted["function"]["name"], "search");
    }

    #[tokio::test]
    async fn test_tool_call_execute() {
        let mut tool_args = HashMap::new();
        tool_args.insert(
            "msg".to_string(),
            ToolArg {
                arg_type: "string".to_string(),
                description: None,
                default: None,
            },
        );

        let tool = Tool::new("echo", "Echo back", tool_args, |kwargs| async move {
            let msg = kwargs.get("msg").and_then(|v| v.as_str()).unwrap_or("");
            Ok(serde_json::json!(msg))
        });

        let mut call_args = HashMap::new();
        call_args.insert("msg".to_string(), serde_json::json!("hello"));

        let tc = ToolCall::new("echo", call_args);
        let result = tc.execute(&[&tool]).await.unwrap();
        assert_eq!(result, serde_json::json!("hello"));
    }

    #[tokio::test]
    async fn test_tool_call_not_found() {
        let tc = ToolCall::new("nonexistent", HashMap::new());
        let result = tc.execute(&[]).await;
        assert!(result.is_err());
    }
}
