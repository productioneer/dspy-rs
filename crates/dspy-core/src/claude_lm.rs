//! ClaudeLM — Routes DSPy calls through the Claude CLI.
//!
//! Uses `claude -p --print --output-format json` to invoke the Claude API
//! via an authenticated Claude Max subscription. Temperature is controlled
//! via the CLAUDE_CODE_EXTRA_BODY environment variable.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::error::{DspyError, Result};
use crate::lm::{LMConfig, LMResponse, LM, Message, Usage};

/// Configuration for ClaudeLM.
#[derive(Debug, Clone)]
pub struct ClaudeLMConfig {
    /// Claude model name (e.g. "opus", "sonnet", "haiku").
    pub model: String,
    /// Default temperature. Passed via CLAUDE_CODE_EXTRA_BODY env var.
    pub temperature: Option<f64>,
    /// Default max tokens.
    pub max_tokens: Option<u32>,
    /// System prompt to prepend.
    pub system_prompt: Option<String>,
    /// Disable tools in the Claude CLI (default: true for safety).
    pub disable_tools: bool,
    /// Timeout in seconds per CLI invocation (default: 60).
    pub timeout_secs: u64,
    /// Max retry attempts on failure (default: 2).
    pub retries: u32,
    /// Max concurrent CLI processes (default: 4).
    pub max_concurrent: usize,
}

impl Default for ClaudeLMConfig {
    fn default() -> Self {
        Self {
            model: "sonnet".to_string(),
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            disable_tools: true,
            timeout_secs: 60,
            retries: 2,
            max_concurrent: 4,
        }
    }
}

/// Claude CLI-backed LM implementation.
pub struct ClaudeLM {
    lm_config: LMConfig,
    cli_config: ClaudeLMConfig,
    semaphore: Arc<Semaphore>,
}

impl ClaudeLM {
    pub fn new(config: ClaudeLMConfig) -> Self {
        let lm_config = LMConfig {
            model: config.model.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: None,
            n: None,
        };
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self { lm_config, cli_config: config, semaphore }
    }

    /// Create a ClaudeLM with default configuration.
    pub fn with_model(model: &str) -> Self {
        Self::new(ClaudeLMConfig {
            model: model.to_string(),
            ..Default::default()
        })
    }

    async fn invoke_once(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> Result<String> {
        let (system_prompt, formatted_prompt) =
            format_messages(messages, self.cli_config.system_prompt.as_deref());

        let mut args = vec![
            "-p".to_string(),
            "--print".to_string(),
            "--output-format".to_string(),
            "json".to_string(),
            "--no-session-persistence".to_string(),
            "--model".to_string(),
            config.model.clone(),
        ];

        if self.cli_config.disable_tools {
            args.push("--tools".to_string());
            args.push(String::new());
        }

        if let Some(ref sp) = system_prompt {
            args.push("--system-prompt".to_string());
            args.push(sp.clone());
        }

        // Build environment with temperature/top_p
        let mut extra_body = serde_json::Map::new();
        if let Some(temp) = config.temperature {
            extra_body.insert("temperature".to_string(), Value::from(temp));
        }
        if let Some(top_p) = config.top_p {
            extra_body.insert("top_p".to_string(), Value::from(top_p));
        }

        let mut last_err: Option<DspyError> = None;

        for attempt in 0..=self.cli_config.retries {
            let _permit = self.semaphore.acquire().await
                .map_err(|e| DspyError::Other(format!("Semaphore error: {e}")))?;

            let mut cmd = Command::new("claude");
            cmd.args(&args);
            cmd.stdin(std::process::Stdio::piped());
            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());

            if !extra_body.is_empty() {
                cmd.env(
                    "CLAUDE_CODE_EXTRA_BODY",
                    serde_json::to_string(&extra_body)
                        .unwrap_or_default(),
                );
            }

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(self.cli_config.timeout_secs),
                async {
                    let mut child = cmd.spawn()
                        .map_err(|e| DspyError::Other(format!("Failed to spawn claude CLI: {e}")))?;

                    if let Some(mut stdin) = child.stdin.take() {
                        use tokio::io::AsyncWriteExt;
                        let _ = stdin.write_all(formatted_prompt.as_bytes()).await;
                        drop(stdin);
                    }

                    let output = child.wait_with_output().await
                        .map_err(|e| DspyError::Other(format!("Claude CLI error: {e}")))?;

                    Ok::<_, DspyError>(output)
                },
            )
            .await;

            match result {
                Ok(Ok(output)) if output.status.success() => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    return parse_claude_output(&stdout);
                }
                Ok(Ok(output)) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    last_err = Some(DspyError::Other(format!(
                        "Claude CLI exited with code {:?}: {}",
                        output.status.code(),
                        stderr.trim()
                    )));
                }
                Ok(Err(e)) => {
                    last_err = Some(e);
                }
                Err(_) => {
                    last_err = Some(DspyError::Other(format!(
                        "Claude CLI timed out after {}s",
                        self.cli_config.timeout_secs
                    )));
                }
            }

            if attempt == self.cli_config.retries {
                break;
            }
        }

        Err(last_err.unwrap_or_else(|| DspyError::Other("CLI execution failed".into())))
    }
}

#[async_trait]
impl LM for ClaudeLM {
    async fn call(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> Result<Vec<LMResponse>> {
        let n = config.n.unwrap_or(1) as usize;
        let mut results = Vec::with_capacity(n);

        for _ in 0..n {
            let text = self.invoke_once(messages, config).await?;
            results.push(LMResponse { text, usage: None });
        }

        Ok(results)
    }

    fn model(&self) -> &str {
        &self.lm_config.model
    }

    fn config(&self) -> &LMConfig {
        &self.lm_config
    }

    fn dump_state(&self) -> Value {
        serde_json::json!({
            "type": "ClaudeLM",
            "model": self.lm_config.model,
            "temperature": self.lm_config.temperature,
            "disable_tools": self.cli_config.disable_tools,
            "system_prompt": self.cli_config.system_prompt,
        })
    }
}

/// Extract system prompt from first system message, format rest as labeled prompt.
fn format_messages(
    messages: &[Message],
    default_system_prompt: Option<&str>,
) -> (Option<String>, String) {
    let mut system_prompt: Option<String> = None;
    let mut parts = Vec::new();

    for msg in messages {
        if msg.role == "system" && system_prompt.is_none() {
            system_prompt = Some(msg.content.clone());
            continue;
        }
        parts.push(format!("[{}]: {}", msg.role.to_uppercase(), msg.content));
    }

    let system = system_prompt.or_else(|| default_system_prompt.map(|s| s.to_string()));
    (system, parts.join("\n\n"))
}

/// Parse Claude CLI JSON output, extracting the result field.
fn parse_claude_output(stdout: &str) -> Result<String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(DspyError::Other("Empty response from Claude CLI".into()));
    }

    let data: Value = serde_json::from_str(trimmed)
        .map_err(|e| DspyError::Other(format!("Invalid JSON from Claude CLI: {e}")))?;

    // structured_output takes priority
    if let Some(so) = data.get("structured_output") {
        if !so.is_null() {
            return Ok(match so.as_str() {
                Some(s) => s.to_string(),
                None => so.to_string(),
            });
        }
    }

    // result field
    match data.get("result").and_then(|v| v.as_str()) {
        Some(r) if !r.is_empty() => Ok(r.to_string()),
        _ => Err(DspyError::Other("Empty result in Claude CLI response".into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ClaudeLMConfig::default();
        assert_eq!(config.model, "sonnet");
        assert!(config.disable_tools);
        assert_eq!(config.timeout_secs, 60);
        assert_eq!(config.retries, 2);
        assert_eq!(config.max_concurrent, 4);
    }

    #[test]
    fn test_with_model() {
        let lm = ClaudeLM::with_model("opus");
        assert_eq!(lm.model(), "opus");
    }

    #[test]
    fn test_dump_state() {
        let lm = ClaudeLM::new(ClaudeLMConfig {
            model: "haiku".into(),
            system_prompt: Some("Be helpful".into()),
            disable_tools: false,
            ..Default::default()
        });
        let state = lm.dump_state();
        assert_eq!(state["type"], "ClaudeLM");
        assert_eq!(state["model"], "haiku");
        assert_eq!(state["system_prompt"], "Be helpful");
        assert_eq!(state["disable_tools"], false);
    }

    #[test]
    fn test_format_messages_extracts_system() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi there"),
        ];
        let (sys, prompt) = format_messages(&messages, None);
        assert_eq!(sys, Some("You are helpful".to_string()));
        assert!(prompt.contains("[USER]: Hello"));
        assert!(prompt.contains("[ASSISTANT]: Hi there"));
        assert!(!prompt.contains("You are helpful"));
    }

    #[test]
    fn test_format_messages_default_system() {
        let messages = vec![Message::user("Hello")];
        let (sys, _) = format_messages(&messages, Some("default sys"));
        assert_eq!(sys, Some("default sys".to_string()));
    }

    #[test]
    fn test_format_messages_no_system() {
        let messages = vec![Message::user("Hello")];
        let (sys, prompt) = format_messages(&messages, None);
        assert_eq!(sys, None);
        assert_eq!(prompt, "[USER]: Hello");
    }

    #[test]
    fn test_parse_result() {
        let json = r#"{"result": "Hello world", "usage": {}}"#;
        let result = parse_claude_output(json).unwrap();
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_parse_structured_output_object() {
        let json = r#"{"structured_output": {"answer": 42}, "result": "fallback"}"#;
        let result = parse_claude_output(json).unwrap();
        assert!(result.contains("42"));
    }

    #[test]
    fn test_parse_structured_output_string() {
        let json = r#"{"structured_output": "direct string"}"#;
        let result = parse_claude_output(json).unwrap();
        assert_eq!(result, "direct string");
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(parse_claude_output("").is_err());
        assert!(parse_claude_output("  ").is_err());
    }

    #[test]
    fn test_parse_empty_result_fails() {
        let json = r#"{"result": ""}"#;
        assert!(parse_claude_output(json).is_err());
    }

    #[test]
    fn test_parse_missing_result_fails() {
        let json = r#"{"other": "value"}"#;
        assert!(parse_claude_output(json).is_err());
    }
}
