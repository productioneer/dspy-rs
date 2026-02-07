//! JSONAdapter — LLM message formatting with JSON output format.
//! Python equivalent: dspy/adapters/json_adapter.py
//!
//! Key differences from ChatAdapter:
//! - Output fields use JSON format instead of `[[ ## field_name ## ]]` delimiters
//! - Requests JSON mode via response_format config (downstream LM handles it)
//! - Parses LM output as JSON instead of marker extraction

use crate::adapter::{Adapter, ChatAdapter};
use crate::callback::{with_callbacks_async, with_callbacks_sync, ComponentType};
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::{LMConfig, Message, LM};
use crate::signature::Signature;
use crate::value::Value;
use async_trait::async_trait;
use std::collections::HashMap;

pub struct JSONAdapter {
    chat_adapter: ChatAdapter,
}

impl JSONAdapter {
    pub fn new() -> Self {
        Self {
            chat_adapter: ChatAdapter::new(),
        }
    }

    /// Format system message with JSON output format description.
    pub fn format_system_message(&self, signature: &Signature) -> String {
        let mut parts = Vec::new();

        // Input fields section
        parts.push("Your input fields are:".to_string());
        parts.push(self.chat_adapter.format_field_description_string(signature.input_fields()));

        // Output fields section
        parts.push("Your output fields are:".to_string());
        parts.push(self.chat_adapter.format_field_description_string(signature.output_fields()));

        // Interaction structure
        parts.push(
            "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
        );
        parts.push(String::new());

        // Input format (still uses markers)
        parts.push("Inputs will have the following structure:".to_string());
        for (name, _) in signature.input_fields() {
            parts.push(format!("[[ ## {} ## ]]", name));
            parts.push(format!("{{{}}}", name));
            parts.push(String::new());
        }

        // Output format (JSON)
        parts.push("Outputs will be a JSON object with the following fields.".to_string());
        let output_fields: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| format!("  \"{}\": \"{{{}}}\"", name, name))
            .collect();
        parts.push(format!("{{\n{}\n}}", output_fields.join(",\n")));

        // Objective
        let instructions = signature.instructions();
        if !instructions.is_empty() {
            parts.push(format!(
                "In adhering to this structure, your objective is: \n        {}",
                instructions
            ));
        } else {
            let input_names: Vec<String> = signature
                .input_fields()
                .map(|(k, _)| format!("`{}`", k))
                .collect();
            let output_names: Vec<String> = signature
                .output_fields()
                .map(|(k, _)| format!("`{}`", k))
                .collect();
            parts.push(format!(
                "In adhering to this structure, your objective is: \n        Given the fields {}, produce the fields {}.",
                input_names.join(", "),
                output_names.join(", ")
            ));
        }

        parts.join("\n")
    }

    /// Format messages with JSON-formatted demo outputs.
    pub fn format_messages(
        &self,
        signature: &Signature,
        inputs: &Example,
        demos: &[Example],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // System message
        messages.push(Message::system(&self.format_system_message(signature)));

        // Demo messages
        for demo in demos {
            // User: input fields with markers (same as ChatAdapter)
            let mut user_parts = Vec::new();
            for (name, _) in signature.input_fields() {
                if let Some(val) = demo.get(&name) {
                    user_parts.push(format!("[[ ## {} ## ]]", name));
                    user_parts.push(val.to_string());
                }
            }
            messages.push(Message::user(&user_parts.join("\n")));

            // Assistant: output fields as JSON
            let mut output_map = serde_json::Map::new();
            for (name, _) in signature.output_fields() {
                if let Some(val) = demo.get(&name) {
                    output_map.insert(
                        name.clone(),
                        serde_json::Value::String(val.to_string()),
                    );
                }
            }
            let json_str = serde_json::to_string_pretty(&serde_json::Value::Object(output_map))
                .unwrap_or_default();
            messages.push(Message::assistant(&json_str));
        }

        // Final user message
        messages.push(Message::user(&self.format_user_message(signature, inputs)));

        messages
    }

    /// Format user message with JSON output requirements.
    pub fn format_user_message(&self, signature: &Signature, inputs: &Example) -> String {
        let mut parts = Vec::new();

        // Input fields with markers
        for (name, _) in signature.input_fields() {
            if let Some(val) = inputs.get(&name) {
                parts.push(format!("[[ ## {} ## ]]", name));
                parts.push(val.to_string());
                parts.push(String::new());
            }
        }

        // JSON output requirements
        let output_names: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| format!("`{}`", name))
            .collect();
        parts.push(format!(
            "Respond with a JSON object in the following order of fields: {}.",
            output_names.join(", then ")
        ));

        parts.join("\n")
    }

    /// Parse JSON output from LM response.
    pub fn parse_output(
        &self,
        output: &str,
        signature: &Signature,
    ) -> Result<HashMap<String, Value>> {
        let output_field_names: Vec<String> = signature
            .output_fields()
            .map(|(k, _)| k.clone())
            .collect();

        // Try direct JSON parse
        let parsed = self.try_parse_json(output)
            .or_else(|| {
                // Try to extract JSON object from text
                Self::extract_json_object(output)
                    .and_then(|s| self.try_parse_json(&s))
            });

        match parsed {
            Some(map) => {
                // Filter to only output fields
                let mut result = HashMap::new();
                for name in &output_field_names {
                    if let Some(val) = map.get(name) {
                        result.insert(name.clone(), val.clone());
                    }
                }
                if result.is_empty() && !output_field_names.is_empty() {
                    result.insert(
                        output_field_names[0].clone(),
                        Value::from(output.trim().to_string()),
                    );
                }
                Ok(result)
            }
            None => {
                // Fallback: treat entire output as first field
                let mut result = HashMap::new();
                if !output_field_names.is_empty() {
                    result.insert(
                        output_field_names[0].clone(),
                        Value::from(output.trim().to_string()),
                    );
                }
                Ok(result)
            }
        }
    }

    fn try_parse_json(&self, text: &str) -> Option<HashMap<String, Value>> {
        let trimmed = text.trim();
        let json_val: serde_json::Value = serde_json::from_str(trimmed).ok()?;
        let obj = json_val.as_object()?;
        let mut result = HashMap::new();
        for (k, v) in obj {
            result.insert(k.clone(), json_value_to_dspy_value(v));
        }
        Some(result)
    }

    fn extract_json_object(text: &str) -> Option<String> {
        // Find the first { and last matching }
        let start = text.find('{')?;
        let mut depth = 0i32;
        let mut end = None;
        for (i, ch) in text[start..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(start + i + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        end.map(|e| text[start..e].to_string())
    }
}

impl Default for JSONAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert serde_json::Value to DSPy Value.
fn json_value_to_dspy_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::String(s) => Value::from(s.clone()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i.to_string())
            } else if let Some(f) = n.as_f64() {
                Value::from(f.to_string())
            } else {
                Value::from(n.to_string())
            }
        }
        serde_json::Value::Bool(b) => Value::from(b.to_string()),
        serde_json::Value::Null => Value::from("null".to_string()),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            Value::from(v.to_string())
        }
    }
}

#[async_trait]
impl Adapter for JSONAdapter {
    async fn call(
        &self,
        lm: &dyn LM,
        signature: &Signature,
        demos: &[Example],
        inputs: &Example,
        config: &LMConfig,
    ) -> Result<Vec<HashMap<String, Value>>> {
        let format_inputs = serde_json::json!({
            "signature": signature.instructions(),
            "demos": demos.len(),
        });
        let messages: Vec<Message> = with_callbacks_sync(
            ComponentType::AdapterFormat,
            "JSONAdapter",
            &format_inputs,
            || Ok::<_, DspyError>(self.format_messages(signature, inputs, demos)),
        )?;

        let lm_inputs = serde_json::json!({
            "messages": messages.len(),
            "model": lm.model(),
        });
        let responses = with_callbacks_async(
            ComponentType::Lm,
            lm.model(),
            &lm_inputs,
            || lm.call(&messages, config),
        )
        .await?;

        let mut results = Vec::new();
        for resp in &responses {
            let parse_inputs = serde_json::json!({
                "output": resp.text,
                "signature": signature.instructions(),
            });
            let parsed = with_callbacks_sync(
                ComponentType::AdapterParse,
                "JSONAdapter",
                &parse_inputs,
                || self.parse_output(&resp.text, signature),
            )?;
            results.push(parsed);
        }

        if results.is_empty() {
            return Err(DspyError::LMError("No responses from LM".into()));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_system_message_describes_json_output() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("Outputs will be a JSON object"));
        assert!(msg.contains("\"answer\""));
    }

    #[test]
    fn test_system_message_uses_markers_for_inputs() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("[[ ## question ## ]]"));
        assert!(msg.contains("Inputs will have the following structure"));
    }

    #[test]
    fn test_system_message_no_completed_marker() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);
        assert!(!msg.contains("[[ ## completed ## ]]"));
    }

    #[test]
    fn test_user_message_json_requirements() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let msg = adapter.format_user_message(&sig, &inputs);
        assert!(msg.contains("[[ ## question ## ]]"));
        assert!(msg.contains("What is 2+2?"));
        assert!(msg.contains("Respond with a JSON object"));
        assert!(msg.contains("`answer`"));
    }

    #[test]
    fn test_format_messages_with_json_demos() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let demo = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");
        let messages = adapter.format_messages(&sig, &inputs, &[demo]);

        // [system, demo_user, demo_assistant, current_user]
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[2].role, "assistant");
        // Demo assistant should be JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&messages[2].content).unwrap();
        assert_eq!(parsed["answer"], "2");
    }

    #[test]
    fn test_parse_valid_json() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter.parse_output(r#"{"answer": "42"}"#, &sig).unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
    }

    #[test]
    fn test_parse_json_filters_extra_fields() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output(r#"{"answer": "42", "extra": "ignored"}"#, &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
        assert!(!result.contains_key("extra"));
    }

    #[test]
    fn test_parse_json_from_surrounding_text() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("Here is my answer:\n{\"answer\": \"42\"}\nDone.", &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
    }

    #[test]
    fn test_parse_multi_field_json() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> reasoning, answer").unwrap();
        let result = adapter
            .parse_output(r#"{"reasoning": "2+2=4", "answer": "4"}"#, &sig)
            .unwrap();
        assert_eq!(result["reasoning"].as_str(), Some("2+2=4"));
        assert_eq!(result["answer"].as_str(), Some("4"));
    }

    #[test]
    fn test_parse_fallback_to_raw_text() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("The answer is 42", &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("The answer is 42"));
    }

    #[test]
    fn test_format_messages_no_demos() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "test");
        let messages = adapter.format_messages(&sig, &inputs, &[]);
        assert_eq!(messages.len(), 2); // system + user
    }

    #[test]
    fn test_system_message_with_instructions() {
        let adapter = JSONAdapter::new();
        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Answer concisely");
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("Answer concisely"));
    }
}
