//! Finetune types and utilities for weight optimization.
//! Python equivalent: dspy/clients/utils_finetune.py

use serde::{Deserialize, Serialize};

/// Training status enum (serializes to lowercase to match Python: "not_started", "pending", etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingStatus {
    NotStarted,
    Pending,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

/// Training data format (serializes to lowercase to match Python: "chat", "completion", "grpo_chat")
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainDataFormat {
    Chat,
    Completion,
    GrpoChat,
}

/// Message for training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMessage {
    pub role: String,
    pub content: String,
}

impl TrainingMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    pub fn system(content: &str) -> Self {
        Self::new("system", content)
    }

    pub fn user(content: &str) -> Self {
        Self::new("user", content)
    }

    pub fn assistant(content: &str) -> Self {
        Self::new("assistant", content)
    }
}

/// GRPO chat data — a single rollout with messages, completion, and reward
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRPOChatData {
    pub messages: Vec<TrainingMessage>,
    pub completion: TrainingMessage,
    pub reward: f64,
}

/// GRPO group — a group of rollouts with an optional batch ID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRPOGroup {
    pub batch_id: Option<usize>,
    pub group: Vec<GRPOChatData>,
}

/// GRPO status returned by a reinforce job
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRPOStatus {
    pub job_id: String,
    pub status: Option<String>,
    pub current_model: String,
    pub checkpoints: std::collections::HashMap<String, String>,
    pub last_checkpoint: Option<String>,
    pub pending_batch_ids: Vec<usize>,
}

/// Infer training data format from an adapter type name.
pub fn infer_data_format(adapter_name: &str) -> crate::Result<TrainDataFormat> {
    match adapter_name {
        "ChatAdapter" | "XMLAdapter" => Ok(TrainDataFormat::Chat),
        _ => Err(crate::DspyError::Other(format!(
            "Could not infer the data format for adapter: {adapter_name}"
        ))),
    }
}

/// Validate training data against a specified format.
pub fn validate_data_format(
    data: &[serde_json::Value],
    data_format: TrainDataFormat,
) -> crate::Result<()> {
    for (i, item) in data.iter().enumerate() {
        let err = match data_format {
            TrainDataFormat::Chat => find_data_error_chat(item),
            TrainDataFormat::Completion => find_data_error_completion(item),
            TrainDataFormat::GrpoChat => None, // No validation for GRPO chat format
        };
        if let Some(error) = err {
            return Err(crate::DspyError::Other(format!(
                "Data format error at index {i}: {error}"
            )));
        }
    }
    Ok(())
}

fn find_data_error_chat(data: &serde_json::Value) -> Option<String> {
    let obj = data.as_object()?;
    if !obj.contains_key("messages") || obj.len() != 1 {
        let keys: Vec<&String> = obj.keys().collect();
        return Some(format!(
            "Expected Keys: [\"messages\"]; Found Keys: {keys:?}"
        ));
    }

    let messages = match obj.get("messages").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return Some("The value of 'messages' should be an array".to_string()),
    };

    for (i, msg) in messages.iter().enumerate() {
        if let Some(err) = find_data_error_chat_message(msg) {
            return Some(format!("Error in message at index {i}: {err}"));
        }
    }

    None
}

fn find_data_error_chat_message(message: &serde_json::Value) -> Option<String> {
    let obj = match message.as_object() {
        Some(o) => o,
        None => return Some(format!("Not a dictionary -- found: {message}")),
    };

    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    let expected_keys = vec!["content", "role"];
    let key_strs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    if key_strs != expected_keys {
        return Some(format!(
            "Expected Keys: {expected_keys:?}; Found Keys: {key_strs:?}"
        ));
    }

    let role = obj.get("role").and_then(|v| v.as_str()).unwrap_or("");
    let valid_roles = ["assistant", "system", "user"];
    if !valid_roles.contains(&role) {
        return Some(format!(
            "Expected Roles: {valid_roles:?}; Found Role: {role}"
        ));
    }

    if !obj.get("content").map_or(false, |v| v.is_string()) {
        return Some("Expected Content Type: string".to_string());
    }

    None
}

fn find_data_error_completion(data: &serde_json::Value) -> Option<String> {
    let obj = match data.as_object() {
        Some(o) => o,
        None => return Some(format!("Not a dictionary -- found: {data}")),
    };

    let mut keys: Vec<&String> = obj.keys().collect();
    keys.sort();
    let expected_keys = vec!["completion", "prompt"];
    let key_strs: Vec<&str> = keys.iter().map(|k| k.as_str()).collect();
    if key_strs != expected_keys {
        return Some(format!(
            "Expected Keys: {expected_keys:?}; Found Keys: {key_strs:?}"
        ));
    }

    for key in &["prompt", "completion"] {
        if !obj.get(*key).map_or(false, |v| v.is_string()) {
            return Some(format!("Expected '{key}' to be of type string"));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_status() {
        assert_ne!(TrainingStatus::Running, TrainingStatus::Failed);
        assert_eq!(TrainingStatus::Succeeded, TrainingStatus::Succeeded);
    }

    #[test]
    fn test_train_data_format() {
        assert_ne!(TrainDataFormat::Chat, TrainDataFormat::Completion);
        assert_eq!(TrainDataFormat::GrpoChat, TrainDataFormat::GrpoChat);
    }

    #[test]
    fn test_infer_data_format() {
        assert_eq!(
            infer_data_format("ChatAdapter").unwrap(),
            TrainDataFormat::Chat
        );
        assert_eq!(
            infer_data_format("XMLAdapter").unwrap(),
            TrainDataFormat::Chat
        );
        assert!(infer_data_format("Unknown").is_err());
    }

    #[test]
    fn test_validate_chat_format_valid() {
        let data = vec![serde_json::json!({
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"}
            ]
        })];
        assert!(validate_data_format(&data, TrainDataFormat::Chat).is_ok());
    }

    #[test]
    fn test_validate_chat_format_invalid_keys() {
        let data = vec![serde_json::json!({
            "messages": [],
            "extra": "bad"
        })];
        assert!(validate_data_format(&data, TrainDataFormat::Chat).is_err());
    }

    #[test]
    fn test_validate_chat_format_invalid_role() {
        let data = vec![serde_json::json!({
            "messages": [
                {"role": "invalid", "content": "hello"}
            ]
        })];
        assert!(validate_data_format(&data, TrainDataFormat::Chat).is_err());
    }

    #[test]
    fn test_validate_completion_format_valid() {
        let data = vec![serde_json::json!({
            "prompt": "hello",
            "completion": "world"
        })];
        assert!(validate_data_format(&data, TrainDataFormat::Completion).is_ok());
    }

    #[test]
    fn test_validate_completion_format_invalid() {
        let data = vec![serde_json::json!({
            "prompt": "hello"
        })];
        assert!(validate_data_format(&data, TrainDataFormat::Completion).is_err());
    }

    #[test]
    fn test_grpo_chat_data() {
        let data = GRPOChatData {
            messages: vec![TrainingMessage::user("hello")],
            completion: TrainingMessage::assistant("hi"),
            reward: 1.0,
        };
        assert_eq!(data.messages.len(), 1);
        assert_eq!(data.reward, 1.0);
    }

    #[test]
    fn test_grpo_group() {
        let group = GRPOGroup {
            batch_id: Some(0),
            group: vec![],
        };
        assert_eq!(group.batch_id, Some(0));
    }
}
