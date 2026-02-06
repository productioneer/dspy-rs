//! LM — language model trait and types.
//! Python equivalent: dspy/clients/lm.py
//!
//! TODO: Full implementation

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMConfig {
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub n: Option<u32>,
}

impl LMConfig {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            n: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LMResponse {
    pub text: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self { role: "system".to_string(), content: content.to_string() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".to_string(), content: content.to_string() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".to_string(), content: content.to_string() }
    }
}

#[async_trait]
pub trait LM: Send + Sync {
    async fn call(&self, messages: &[Message], config: &LMConfig) -> crate::error::Result<Vec<LMResponse>>;
    fn model(&self) -> &str;
    fn config(&self) -> &LMConfig;
    fn dump_state(&self) -> serde_json::Value;
}
