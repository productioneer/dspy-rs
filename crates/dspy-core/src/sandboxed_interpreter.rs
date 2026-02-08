//! Sandboxed Python interpreter using Deno + Pyodide (WebAssembly).
//!
//! Spawns a Deno subprocess running sandbox-runner.ts which loads Pyodide
//! (Python compiled to WASM). Python code executes inside the WASM sandbox
//! with no access to the host filesystem, network, or processes.
//!
//! Uses the same JSON-RPC 2.0 protocol as the TS SandboxedInterpreter.

use crate::interpreter::{
    CodeInterpreter, CodeInterpreterError, ExecutionResult, FinalOutput, InterpreterTool,
};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

/// Options for creating a sandboxed interpreter.
pub struct SandboxedInterpreterOptions {
    /// Path to Deno executable (default: "deno")
    pub deno_command: String,
    /// Tools callable from interpreter code
    pub tools: HashMap<String, InterpreterTool>,
}

impl Default for SandboxedInterpreterOptions {
    fn default() -> Self {
        Self {
            deno_command: "deno".to_string(),
            tools: HashMap::new(),
        }
    }
}

pub struct SandboxedInterpreter {
    deno_command: String,
    tools: HashMap<String, InterpreterTool>,
    process: Option<Child>,
    reader: Option<BufReader<tokio::process::ChildStdout>>,
    stdin: Option<tokio::process::ChildStdin>,
    request_id: u64,
    registered: bool,
}

impl SandboxedInterpreter {
    pub fn new(options: SandboxedInterpreterOptions) -> Self {
        Self {
            deno_command: options.deno_command,
            tools: options.tools,
            process: None,
            reader: None,
            stdin: None,
            request_id: 0,
            registered: false,
        }
    }

    /// Create with default options.
    pub fn default_new() -> Self {
        Self::new(SandboxedInterpreterOptions::default())
    }

    /// Get the path to the sandbox runner script.
    fn runner_path() -> PathBuf {
        // Navigate from the dspy-core crate to the repo root's shared/interpreter/
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir)
            .join("..")
            .join("..")
            .join("..")
            .join("shared")
            .join("interpreter")
            .join("sandbox-runner.ts")
    }

    /// Get the directory containing the sandbox runner.
    fn runner_dir() -> PathBuf {
        Self::runner_path().parent().unwrap().to_path_buf()
    }

    async fn ensure_process(&mut self) -> Result<(), CodeInterpreterError> {
        if self.process.is_some() {
            return Ok(());
        }
        self.registered = false;
        self.start().await
    }

    async fn register_if_needed(&mut self) -> Result<(), CodeInterpreterError> {
        if self.registered {
            return Ok(());
        }

        let tool_names: Vec<Value> = self
            .tools
            .keys()
            .map(|name| json!({"name": name}))
            .collect();

        if !tool_names.is_empty() {
            let params = json!({"tools": tool_names});
            self.send_request("register", params).await?;
        }
        self.registered = true;
        Ok(())
    }

    async fn send_request(
        &mut self,
        method: &str,
        params: Value,
    ) -> Result<Value, CodeInterpreterError> {
        self.request_id += 1;
        let id = self.request_id;
        let msg = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": id,
        });

        self.write_message(&msg).await?;
        let response = self.read_line().await?;
        let resp: Value = serde_json::from_str(&response)
            .map_err(|e| CodeInterpreterError::new(format!("Parse error: {e}")))?;

        if let Some(error) = resp.get("error") {
            return Err(CodeInterpreterError::new(
                error
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("Unknown error")
                    .to_string(),
            ));
        }
        Ok(resp)
    }

    async fn write_message(&mut self, msg: &Value) -> Result<(), CodeInterpreterError> {
        let stdin = self
            .stdin
            .as_mut()
            .ok_or_else(|| CodeInterpreterError::new("Sandbox process not started"))?;

        let line = serde_json::to_string(msg)
            .map_err(|e| CodeInterpreterError::new(format!("Serialize error: {e}")))?;

        stdin
            .write_all(format!("{line}\n").as_bytes())
            .await
            .map_err(|e| CodeInterpreterError::new(format!("Write error: {e}")))?;

        stdin
            .flush()
            .await
            .map_err(|e| CodeInterpreterError::new(format!("Flush error: {e}")))?;

        Ok(())
    }

    async fn read_line(&mut self) -> Result<String, CodeInterpreterError> {
        let reader = self
            .reader
            .as_mut()
            .ok_or_else(|| CodeInterpreterError::new("Sandbox process not started"))?;

        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .map_err(|e| CodeInterpreterError::new(format!("Read error: {e}")))?;

        if line.is_empty() {
            return Err(CodeInterpreterError::new(
                "No output from sandbox subprocess",
            ));
        }

        Ok(line.trim().to_string())
    }

    async fn handle_tool_call(&mut self, msg: &Value) -> Result<(), CodeInterpreterError> {
        let params = msg.get("params").cloned().unwrap_or(json!({}));
        let tool_name = params
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        let kwargs_val = params.get("kwargs").cloned().unwrap_or(json!({}));
        let request_id = msg.get("id").cloned().unwrap_or(Value::Null);

        let kwargs: HashMap<String, Value> = match kwargs_val.as_object() {
            Some(obj) => obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            None => HashMap::new(),
        };

        let tool = self.tools.get(&tool_name);
        match tool {
            Some(tool_fn) => match tool_fn(kwargs).await {
                Ok(result) => {
                    let is_json = result.is_object() || result.is_array();
                    let response = json!({
                        "jsonrpc": "2.0",
                        "result": {
                            "value": if is_json {
                                serde_json::to_string(&result).unwrap_or_default()
                            } else {
                                result.as_str().unwrap_or("").to_string()
                            },
                            "type": if is_json { "json" } else { "string" },
                        },
                        "id": request_id,
                    });
                    self.write_message(&response).await?;
                }
                Err(err_msg) => {
                    let response = json!({
                        "jsonrpc": "2.0",
                        "error": {"code": -32099, "message": err_msg},
                        "id": request_id,
                    });
                    self.write_message(&response).await?;
                }
            },
            None => {
                let response = json!({
                    "jsonrpc": "2.0",
                    "error": {"code": -32099, "message": format!("Unknown tool: {tool_name}")},
                    "id": request_id,
                });
                self.write_message(&response).await?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl CodeInterpreter for SandboxedInterpreter {
    async fn start(&mut self) -> Result<(), CodeInterpreterError> {
        if self.process.is_some() {
            return Ok(());
        }

        let runner_path = Self::runner_path();
        let runner_dir = Self::runner_dir();

        if !runner_path.exists() {
            return Err(CodeInterpreterError::new(format!(
                "Sandbox runner not found: {}",
                runner_path.display()
            )));
        }

        let allow_read = format!("--allow-read={}", runner_dir.display());
        let mut child = Command::new(&self.deno_command)
            .arg("run")
            .arg(&allow_read)
            .arg("--node-modules-dir=auto")
            .arg(runner_path.to_str().unwrap())
            .current_dir(&runner_dir)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| CodeInterpreterError::new(format!("Failed to start Deno sandbox: {e}")))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| CodeInterpreterError::new("Failed to capture sandbox stdout"))?;
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| CodeInterpreterError::new("Failed to capture sandbox stdin"))?;

        self.reader = Some(BufReader::new(stdout));
        self.stdin = Some(stdin);
        self.process = Some(child);

        // Health check
        self.send_request("ping", json!({})).await?;

        Ok(())
    }

    async fn execute(
        &mut self,
        code: &str,
        variables: Option<&HashMap<String, Value>>,
    ) -> Result<ExecutionResult, CodeInterpreterError> {
        self.ensure_process().await?;
        self.register_if_needed().await?;

        self.request_id += 1;
        let id = self.request_id;

        let mut params = json!({"code": code});
        if let Some(vars) = variables {
            if !vars.is_empty() {
                params
                    .as_object_mut()
                    .unwrap()
                    .insert("variables".to_string(), serde_json::to_value(vars).unwrap());
            }
        }

        let msg = json!({
            "jsonrpc": "2.0",
            "method": "execute",
            "params": params,
            "id": id,
        });
        self.write_message(&msg).await?;

        // Read responses — handle tool calls until final result
        loop {
            let line = self.read_line().await?;
            let resp: Value = serde_json::from_str(&line)
                .map_err(|_| CodeInterpreterError::new("Invalid JSON from sandbox"))?;

            // Handle tool call requests
            if resp.get("method").and_then(|m| m.as_str()) == Some("tool_call") {
                self.handle_tool_call(&resp).await?;
                continue;
            }

            // Handle success response
            if let Some(result) = resp.get("result") {
                if resp.get("id").and_then(|i| i.as_u64()) == Some(id) {
                    if let Some(final_val) = result.get("final") {
                        return Ok(ExecutionResult::Final(FinalOutput::new(final_val.clone())));
                    }
                    let output = result.get("output").and_then(|o| o.as_str());
                    return Ok(ExecutionResult::Output(output.map(String::from)));
                }
            }

            // Handle error response
            if let Some(error) = resp.get("error") {
                let error_type = error
                    .get("data")
                    .and_then(|d| d.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("Unknown");
                let error_msg = error
                    .get("data")
                    .and_then(|d| d.get("args"))
                    .and_then(|a| a.as_str())
                    .or_else(|| error.get("message").and_then(|m| m.as_str()))
                    .unwrap_or("Unknown error");

                return Err(CodeInterpreterError::new(format!(
                    "{error_type}: {error_msg}"
                )));
            }
        }
    }

    async fn shutdown(&mut self) -> Result<(), CodeInterpreterError> {
        if let Some(ref mut _process) = self.process {
            // Send shutdown message
            let msg = json!({
                "jsonrpc": "2.0",
                "method": "shutdown",
                "params": {},
                "id": Value::Null,
            });
            let _ = self.write_message(&msg).await;

            // Close stdin
            self.stdin = None;

            // Wait briefly for clean exit
            if let Some(ref mut child) = self.process {
                let _ = tokio::time::timeout(std::time::Duration::from_secs(5), child.wait()).await;
            }
        }

        self.process = None;
        self.reader = None;
        self.stdin = None;
        self.registered = false;
        Ok(())
    }

    fn tools_mut(&mut self) -> &mut HashMap<String, InterpreterTool> {
        &mut self.tools
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deno_available() -> bool {
        std::process::Command::new("deno")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn test_ping_and_simple_execute() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("print('hello world')", None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "hello world\n"),
            other => panic!("Expected output, got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_state_persistence() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        interp.execute("x = 42", None).await.unwrap();
        let result = interp.execute("print(x * 2)", None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "84\n"),
            other => panic!("Expected output '84\\n', got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_variable_injection() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let mut vars = HashMap::new();
        vars.insert("msg".to_string(), json!("injected value"));

        let result = interp.execute("print(msg)", Some(&vars)).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "injected value\n"),
            other => panic!("Expected injected value, got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_submit_kwargs() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp
            .execute("SUBMIT(answer='yes', confidence=0.9)", None)
            .await
            .unwrap();
        match result {
            ExecutionResult::Final(fo) => {
                assert_eq!(fo.output["answer"], "yes");
                assert_eq!(fo.output["confidence"], 0.9);
            }
            other => panic!("Expected FinalOutput, got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_submit_positional() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("SUBMIT('done')", None).await.unwrap();
        match result {
            ExecutionResult::Final(fo) => {
                assert_eq!(fo.output, json!("done"));
            }
            other => panic!("Expected FinalOutput, got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_syntax_error() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("def foo(", None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("SyntaxError"),
            "Expected SyntaxError, got: {}",
            err.message
        );

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_runtime_error() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("print(undefined_var)", None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("NameError"),
            "Expected NameError, got: {}",
            err.message
        );

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_no_output() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("y = 1 + 1", None).await.unwrap();
        match result {
            ExecutionResult::Output(None) => {}
            other => panic!("Expected Output(None), got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_multi_line_code() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let code = "result = 0\nfor i in range(5):\n    result += i\nprint(result)";
        let result = interp.execute(code, None).await.unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "10\n"),
            other => panic!("Expected '10\\n', got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_import_js_blocked() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut interp = SandboxedInterpreter::default_new();
        interp.start().await.unwrap();

        let result = interp.execute("import js", None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.to_lowercase().contains("blocked"),
            "Expected 'blocked' error, got: {}",
            err.message
        );

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_tool_call() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut tools: HashMap<String, InterpreterTool> = HashMap::new();
        tools.insert(
            "add_numbers".to_string(),
            Box::new(|kwargs: HashMap<String, Value>| {
                Box::pin(async move {
                    let a = kwargs.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let b = kwargs.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    Ok(json!((a + b).to_string()))
                })
            }),
        );

        let mut interp = SandboxedInterpreter::new(SandboxedInterpreterOptions {
            deno_command: "deno".to_string(),
            tools,
        });
        interp.start().await.unwrap();

        let result = interp
            .execute("result = add_numbers(a=3, b=4)\nprint(result)", None)
            .await
            .unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "7\n"),
            other => panic!("Expected '7\\n', got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn test_tool_returning_json() {
        if !deno_available() {
            eprintln!("Skipping: deno not available");
            return;
        }

        let mut tools: HashMap<String, InterpreterTool> = HashMap::new();
        tools.insert(
            "get_data".to_string(),
            Box::new(|_kwargs: HashMap<String, Value>| {
                Box::pin(async move { Ok(json!({"items": [1, 2, 3]})) })
            }),
        );

        let mut interp = SandboxedInterpreter::new(SandboxedInterpreterOptions {
            deno_command: "deno".to_string(),
            tools,
        });
        interp.start().await.unwrap();

        let result = interp
            .execute(
                "data = get_data()\nprint(type(data).__name__, len(data['items']))",
                None,
            )
            .await
            .unwrap();
        match result {
            ExecutionResult::Output(Some(s)) => assert_eq!(s, "dict 3\n"),
            other => panic!("Expected 'dict 3\\n', got: {other:?}"),
        }

        interp.shutdown().await.unwrap();
    }
}
