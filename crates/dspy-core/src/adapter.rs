//! Adapter — LLM message formatting and response parsing.
//! Python equivalent: dspy/adapters/chat_adapter.py
//!
//! The ChatAdapter formats prompts using `[[ ## field_name ## ]]` delimiters
//! (matching Python DSPy exactly) and parses LLM responses back into field values.

use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::{LMConfig, Message, LM};
use crate::signature::Signature;
use crate::value::Value;
use async_trait::async_trait;
use std::collections::HashMap;

#[async_trait]
pub trait Adapter: Send + Sync {
    async fn call(
        &self,
        lm: &dyn LM,
        signature: &Signature,
        demos: &[Example],
        inputs: &Example,
        config: &LMConfig,
    ) -> Result<Vec<HashMap<String, Value>>>;
}

pub struct ChatAdapter;

impl ChatAdapter {
    pub fn new() -> Self {
        Self
    }

    /// Format system message describing the task and field schema
    pub fn format_system_message(&self, signature: &Signature) -> String {
        let mut parts = Vec::new();

        parts.push("Your input fields are:".to_string());
        for (name, field) in signature.input_fields() {
            let desc = field
                .description
                .as_deref()
                .unwrap_or("N/A");
            parts.push(format!("- `{name}` ({desc})"));
        }

        parts.push(String::new());
        parts.push("Your output fields are:".to_string());
        for (name, field) in signature.output_fields() {
            let desc = field
                .description
                .as_deref()
                .unwrap_or("N/A");
            parts.push(format!("- `{name}` ({desc})"));
        }

        parts.push(String::new());
        parts.push("All interactions will be structured in the following way, with the appropriate values filled in.".to_string());

        // Show field template
        parts.push(String::new());
        for (name, _) in signature.fields() {
            parts.push(format!("[[ ## {name} ## ]]"));
            parts.push(format!("{{{name}}}"));
            parts.push(String::new());
        }

        if !signature.instructions().is_empty() {
            parts.push(format!(
                "In adhering to this structure, your objective is: {}",
                signature.instructions()
            ));
        } else {
            parts.push(
                "In adhering to this structure, your objective is to complete the task."
                    .to_string(),
            );
        }

        parts.join("\n")
    }

    /// Format user message with demos and current inputs
    pub fn format_user_message(
        &self,
        signature: &Signature,
        inputs: &Example,
        demos: &[Example],
    ) -> String {
        let mut parts = Vec::new();

        // Format demos
        for (i, demo) in demos.iter().enumerate() {
            parts.push(format!("---\n\nExample {}:", i + 1));
            parts.push(String::new());

            // Demo inputs
            for (name, _) in signature.input_fields() {
                parts.push(format!("[[ ## {name} ## ]]"));
                let val = demo
                    .get(name)
                    .map(|v| v.to_string())
                    .unwrap_or_default();
                parts.push(val);
                parts.push(String::new());
            }

            // Demo outputs (if present)
            for (name, _) in signature.output_fields() {
                if demo.has(name) {
                    parts.push(format!("[[ ## {name} ## ]]"));
                    let val = demo
                        .get(name)
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    parts.push(val);
                    parts.push(String::new());
                }
            }
        }

        // Format current inputs
        if !demos.is_empty() {
            parts.push("---\n".to_string());
        }

        for (name, _) in signature.input_fields() {
            parts.push(format!("[[ ## {name} ## ]]"));
            let val = inputs
                .get(name)
                .map(|v| v.to_string())
                .unwrap_or_default();
            parts.push(val);
            parts.push(String::new());
        }

        // Prompt for output fields
        parts.push("Respond with the corresponding output fields, starting with the field markers.".to_string());
        parts.push(String::new());

        for (name, _) in signature.output_fields() {
            parts.push(format!("[[ ## {name} ## ]]"));
        }

        parts.join("\n")
    }

    /// Parse LLM response text into field values using `[[ ## field ## ]]` delimiters
    pub fn parse_output(
        &self,
        output: &str,
        signature: &Signature,
    ) -> Result<HashMap<String, Value>> {
        let mut result = HashMap::new();
        let output_field_names: Vec<String> = signature
            .output_fields()
            .map(|(k, _)| k.clone())
            .collect();

        for (i, name) in output_field_names.iter().enumerate() {
            let marker = format!("[[ ## {name} ## ]]");
            if let Some(start_idx) = output.find(&marker) {
                let content_start = start_idx + marker.len();
                // Find the next marker or end of string
                let content_end = if i + 1 < output_field_names.len() {
                    let next_marker = format!("[[ ## {} ## ]]", output_field_names[i + 1]);
                    output[content_start..]
                        .find(&next_marker)
                        .map(|idx| content_start + idx)
                        .unwrap_or(output.len())
                } else {
                    output.len()
                };

                let value = output[content_start..content_end].trim().to_string();
                result.insert(name.clone(), Value::from(value));
            }
        }

        // If no markers found, treat entire output as the first output field
        if result.is_empty() && !output_field_names.is_empty() {
            result.insert(
                output_field_names[0].clone(),
                Value::from(output.trim().to_string()),
            );
        }

        Ok(result)
    }
}

impl Default for ChatAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Adapter for ChatAdapter {
    async fn call(
        &self,
        lm: &dyn LM,
        signature: &Signature,
        demos: &[Example],
        inputs: &Example,
        config: &LMConfig,
    ) -> Result<Vec<HashMap<String, Value>>> {
        let system_msg = self.format_system_message(signature);
        let user_msg = self.format_user_message(signature, inputs, demos);

        let messages = vec![
            Message::system(&system_msg),
            Message::user(&user_msg),
        ];

        let n = config.n.unwrap_or(1) as usize;
        let responses = lm.call(&messages, config).await?;

        let mut results = Vec::with_capacity(n);
        for resp in &responses {
            let parsed = self.parse_output(&resp.text, signature)?;
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

    fn test_sig() -> Signature {
        Signature::from_string("question -> answer").unwrap()
    }

    #[test]
    fn test_system_message_format() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("Your input fields are:"));
        assert!(msg.contains("`question`"));
        assert!(msg.contains("Your output fields are:"));
        assert!(msg.contains("`answer`"));
        assert!(msg.contains("[[ ## question ## ]]"));
        assert!(msg.contains("[[ ## answer ## ]]"));
    }

    #[test]
    fn test_user_message_no_demos() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let inputs = Example::new().field("question", "What is 2+2?");
        let msg = adapter.format_user_message(&sig, &inputs, &[]);
        assert!(msg.contains("[[ ## question ## ]]"));
        assert!(msg.contains("What is 2+2?"));
        assert!(msg.contains("[[ ## answer ## ]]"));
    }

    #[test]
    fn test_user_message_with_demos() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let inputs = Example::new().field("question", "What is 3+3?");
        let demo = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");
        let msg = adapter.format_user_message(&sig, &inputs, &[demo]);
        assert!(msg.contains("Example 1:"));
        assert!(msg.contains("What is 1+1?"));
        assert!(msg.contains("2")); // demo answer
        assert!(msg.contains("What is 3+3?")); // actual input
    }

    #[test]
    fn test_parse_output_with_markers() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer, confidence").unwrap();
        let output = "[[ ## answer ## ]]\n42\n\n[[ ## confidence ## ]]\nhigh";
        let parsed = adapter.parse_output(output, &sig).unwrap();
        assert_eq!(parsed["answer"].as_str(), Some("42"));
        assert_eq!(parsed["confidence"].as_str(), Some("high"));
    }

    #[test]
    fn test_parse_output_without_markers() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let output = "The answer is 42";
        let parsed = adapter.parse_output(output, &sig).unwrap();
        assert_eq!(parsed["answer"].as_str(), Some("The answer is 42"));
    }

    #[test]
    fn test_parse_output_multiline_values() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("q -> reasoning, answer").unwrap();
        let output = "[[ ## reasoning ## ]]\nFirst, I need to think.\nThen, I compute.\n\n[[ ## answer ## ]]\n42";
        let parsed = adapter.parse_output(output, &sig).unwrap();
        assert!(parsed["reasoning"]
            .as_str()
            .unwrap()
            .contains("First, I need to think."));
        assert_eq!(parsed["answer"].as_str(), Some("42"));
    }

    #[test]
    fn test_system_message_with_instructions() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Answer concisely");
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("Answer concisely"));
    }

    #[test]
    fn test_system_message_with_field_descriptions() {
        let adapter = ChatAdapter::new();
        let sig = Signature::define()
            .input_with_desc("question", "the question to answer")
            .output_with_desc("answer", "the final answer")
            .build();
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("the question to answer"));
        assert!(msg.contains("the final answer"));
    }
}
