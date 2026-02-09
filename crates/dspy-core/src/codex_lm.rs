//! CodexLM — Routes DSPy calls through the Codex CLI.
//!
//! Uses `codex exec --json` to invoke the OpenAI API via an authenticated
//! ChatGPT Pro subscription. Temperature is NOT supported by the Codex CLI;
//! a one-time warning is logged when requested.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::process::Command;
use tokio::sync::Semaphore;

use crate::error::{DspyError, Result};
use crate::lm::{LMConfig, LMResponse, Message, LM};

/// Configuration for CodexLM.
#[derive(Debug, Clone)]
pub struct CodexLMConfig {
    /// Model name (default: "gpt-5.2-codex").
    pub model: String,
    /// Default temperature (note: Codex CLI ignores this).
    pub temperature: Option<f64>,
    /// Default max tokens.
    pub max_tokens: Option<u32>,
    /// System prompt / developer instructions.
    pub system_prompt: Option<String>,
    /// Skip git repo check (default: true).
    pub skip_git_check: bool,
    /// Sandbox mode (default: "read-only").
    pub sandbox: String,
    /// Reasoning effort for reasoning models.
    pub reasoning_effort: Option<String>,
    /// Timeout in seconds per CLI invocation (default: 120).
    pub timeout_secs: u64,
    /// Max retry attempts on failure (default: 2).
    pub retries: u32,
    /// Max concurrent CLI processes (default: 4).
    pub max_concurrent: usize,
}

impl Default for CodexLMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-5.2-codex".to_string(),
            temperature: None,
            max_tokens: None,
            system_prompt: None,
            skip_git_check: true,
            sandbox: "read-only".to_string(),
            reasoning_effort: None,
            timeout_secs: 120,
            retries: 2,
            max_concurrent: 4,
        }
    }
}

/// Codex CLI-backed LM implementation.
pub struct CodexLM {
    lm_config: LMConfig,
    cli_config: CodexLMConfig,
    semaphore: Arc<Semaphore>,
    warned_temperature: AtomicBool,
}

impl CodexLM {
    pub fn new(config: CodexLMConfig) -> Self {
        let lm_config = LMConfig {
            model: config.model.clone(),
            temperature: config.temperature,
            max_tokens: config.max_tokens,
            top_p: None,
            n: None,
        };
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        Self {
            lm_config,
            cli_config: config,
            semaphore,
            warned_temperature: AtomicBool::new(false),
        }
    }

    /// Create a CodexLM with default configuration.
    pub fn with_model(model: &str) -> Self {
        Self::new(CodexLMConfig {
            model: model.to_string(),
            ..Default::default()
        })
    }

    async fn invoke_once(&self, messages: &[Message], _config: &LMConfig) -> Result<String> {
        let (system_prompt, formatted_prompt) =
            format_messages(messages, self.cli_config.system_prompt.as_deref());

        let mut args = vec!["exec".to_string()];

        if self.cli_config.skip_git_check {
            args.push("--skip-git-repo-check".to_string());
        }
        args.push("--sandbox".to_string());
        args.push(self.cli_config.sandbox.clone());
        args.push("--json".to_string());

        if let Some(ref sp) = system_prompt {
            args.push("-c".to_string());
            args.push(format!("developer_instructions={sp}"));
        }
        if let Some(ref re) = self.cli_config.reasoning_effort {
            args.push("-c".to_string());
            args.push(format!("reasoning_effort={re}"));
        }

        args.push("--".to_string());
        args.push(formatted_prompt);

        let mut last_err: Option<DspyError> = None;

        for attempt in 0..=self.cli_config.retries {
            let _permit = self
                .semaphore
                .acquire()
                .await
                .map_err(|e| DspyError::Other(format!("Semaphore error: {e}")))?;

            let result = tokio::time::timeout(
                std::time::Duration::from_secs(self.cli_config.timeout_secs),
                async {
                    let output = Command::new("codex")
                        .args(&args)
                        .stdout(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .output()
                        .await
                        .map_err(|e| DspyError::Other(format!("Failed to spawn codex CLI: {e}")))?;

                    Ok::<_, DspyError>(output)
                },
            )
            .await;

            match result {
                Ok(Ok(output)) if output.status.success() => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    return parse_codex_output(&stdout);
                }
                Ok(Ok(output)) => {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    last_err = Some(DspyError::Other(format!(
                        "Codex CLI exited with code {:?}: {}",
                        output.status.code(),
                        stderr.trim()
                    )));
                }
                Ok(Err(e)) => {
                    last_err = Some(e);
                }
                Err(_) => {
                    last_err = Some(DspyError::Other(format!(
                        "Codex CLI timed out after {}s",
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
impl LM for CodexLM {
    async fn call(&self, messages: &[Message], config: &LMConfig) -> Result<Vec<LMResponse>> {
        let n = config.n.unwrap_or(1) as usize;

        // Temperature warning
        if let Some(temp) = config.temperature {
            if temp != 0.0 && !self.warned_temperature.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "CodexLM: Codex CLI does not support temperature control. Temperature setting is ignored."
                );
            }
        }

        let mut results = Vec::with_capacity(n);
        for _ in 0..n {
            let text = self.invoke_once(messages, config).await?;
            results.push(LMResponse::new(text, None));
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
            "type": "CodexLM",
            "model": self.lm_config.model,
            "sandbox": self.cli_config.sandbox,
            "reasoning_effort": self.cli_config.reasoning_effort,
        })
    }
}

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

/// Parse Codex CLI output — tries single JSON, then JSONL, then raw text.
fn parse_codex_output(stdout: &str) -> Result<String> {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return Err(DspyError::Other("Empty response from Codex CLI".into()));
    }

    // Single JSON blob
    if let Ok(data) = serde_json::from_str::<Value>(trimmed) {
        if let Some(text) = extract_text(&data) {
            return Ok(text);
        }
    }

    // JSONL — try each line, last-to-first
    let lines: Vec<&str> = trimmed.lines().collect();
    for line in lines.iter().rev() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(data) = serde_json::from_str::<Value>(line) {
            if let Some(text) = extract_text(&data) {
                return Ok(text);
            }
        }
    }

    // Raw text fallback
    Ok(trimmed.to_string())
}

fn extract_text(data: &Value) -> Option<String> {
    if let Value::Array(arr) = data {
        for item in arr.iter().rev() {
            if let Some(t) = extract_text(item) {
                return Some(t);
            }
        }
        return None;
    }

    if let Value::Object(obj) = data {
        for key in &["output", "result", "text", "completion", "content"] {
            if let Some(val) = obj.get(*key) {
                if !val.is_null() {
                    return Some(match val.as_str() {
                        Some(s) => s.to_string(),
                        None => val.to_string(),
                    });
                }
            }
        }

        // Nested message.content
        if let Some(msg) = obj.get("message") {
            if let Some(content) = msg.get("content") {
                if !content.is_null() {
                    return Some(match content.as_str() {
                        Some(s) => s.to_string(),
                        None => content.to_string(),
                    });
                }
            }
        }

        // OpenAI choices format
        if let Some(Value::Array(choices)) = obj.get("choices") {
            if let Some(choice) = choices.first() {
                if let Some(msg) = choice.get("message") {
                    if let Some(content) = msg.get("content") {
                        if !content.is_null() {
                            return Some(match content.as_str() {
                                Some(s) => s.to_string(),
                                None => content.to_string(),
                            });
                        }
                    }
                }
                if let Some(text) = choice.get("text") {
                    if !text.is_null() {
                        return Some(text.to_string());
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CodexLMConfig::default();
        assert_eq!(config.model, "gpt-5.2-codex");
        assert!(config.skip_git_check);
        assert_eq!(config.sandbox, "read-only");
        assert_eq!(config.timeout_secs, 120);
        assert_eq!(config.retries, 2);
        assert_eq!(config.max_concurrent, 4);
    }

    #[test]
    fn test_with_model() {
        let lm = CodexLM::with_model("gpt-4o");
        assert_eq!(lm.model(), "gpt-4o");
    }

    #[test]
    fn test_dump_state() {
        let lm = CodexLM::new(CodexLMConfig {
            model: "gpt-5.2-codex".into(),
            sandbox: "workspace-write".into(),
            reasoning_effort: Some("high".into()),
            ..Default::default()
        });
        let state = lm.dump_state();
        assert_eq!(state["type"], "CodexLM");
        assert_eq!(state["model"], "gpt-5.2-codex");
        assert_eq!(state["sandbox"], "workspace-write");
        assert_eq!(state["reasoning_effort"], "high");
    }

    #[test]
    fn test_format_messages_extracts_system() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello"),
            Message::assistant("Hi"),
        ];
        let (sys, prompt) = format_messages(&messages, None);
        assert_eq!(sys, Some("You are helpful".to_string()));
        assert!(prompt.contains("[USER]: Hello"));
        assert!(prompt.contains("[ASSISTANT]: Hi"));
    }

    #[test]
    fn test_format_messages_default_system() {
        let messages = vec![Message::user("Hello")];
        let (sys, _) = format_messages(&messages, Some("default"));
        assert_eq!(sys, Some("default".to_string()));
    }

    #[test]
    fn test_parse_output_key() {
        let json = r#"{"output": "response text"}"#;
        let result = parse_codex_output(json).unwrap();
        assert_eq!(result, "response text");
    }

    #[test]
    fn test_parse_result_key() {
        let json = r#"{"result": "answer"}"#;
        let result = parse_codex_output(json).unwrap();
        assert_eq!(result, "answer");
    }

    #[test]
    fn test_parse_jsonl() {
        let jsonl = "{\"type\": \"event\"}\n{\"output\": \"final answer\"}";
        let result = parse_codex_output(jsonl).unwrap();
        assert_eq!(result, "final answer");
    }

    #[test]
    fn test_parse_choices_format() {
        let json = r#"{"choices": [{"message": {"content": "choice text"}}]}"#;
        let result = parse_codex_output(json).unwrap();
        assert_eq!(result, "choice text");
    }

    #[test]
    fn test_parse_raw_text_fallback() {
        let raw = "just plain text";
        let result = parse_codex_output(raw).unwrap();
        assert_eq!(result, "just plain text");
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(parse_codex_output("").is_err());
        assert!(parse_codex_output("  ").is_err());
    }
}
