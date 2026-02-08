//! Adapter — LLM message formatting and response parsing.
//! Python equivalent: dspy/adapters/chat_adapter.py
//!
//! The ChatAdapter formats prompts using `[[ ## field_name ## ]]` delimiters
//! (matching Python DSPy 3.1.2 exactly) and parses LLM responses back into field values.

use crate::callback::{with_callbacks_async, with_callbacks_sync, ComponentType};
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

    /// Format field description string matching Python's get_field_description_string().
    /// Each field: `N. \`name\` (type): desc`
    /// The desc is empty for simple str fields. Final result is trimmed (matching Python's .strip()).
    pub fn format_field_description_string<'a>(
        &self,
        fields: impl Iterator<Item = (&'a String, &'a crate::signature::FieldDef)>,
    ) -> String {
        let descriptions: Vec<String> = fields
            .enumerate()
            .map(|(idx, (name, field))| {
                let type_str = "str";
                let desc = field.description.as_deref().unwrap_or("");
                format!("{}. `{}` ({}): {}", idx + 1, name, type_str, desc)
            })
            .collect();
        // Python joins with \n then strips trailing whitespace (matching .strip() behavior)
        let joined = descriptions.join("\n");
        // trimEnd equivalent: trim trailing whitespace but preserve leading
        joined.trim_end().to_string()
    }

    /// Format output requirements string matching Python DSPy's user_message_output_requirements().
    fn format_output_requirements(&self, signature: &Signature) -> String {
        let output_names: Vec<String> = signature.output_fields().map(|(k, _)| k.clone()).collect();
        if output_names.len() == 1 {
            return format!(
                "Respond with the corresponding output fields, starting with the field `[[ ## {} ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.",
                output_names[0]
            );
        }

        // Multiple output fields
        let mut parts = Vec::new();
        parts.push(format!(
            "Respond with the corresponding output fields, starting with the field `[[ ## {} ## ]]`",
            output_names[0]
        ));
        for name in &output_names[1..] {
            parts.push(format!(", then `[[ ## {} ## ]]`", name));
        }
        parts.push(", and then ending with the marker for `[[ ## completed ## ]]`.".to_string());
        parts.join("")
    }

    /// Format system message describing the task and field schema.
    /// Matches Python DSPy 3.1.2 ChatAdapter.format_system_message exactly.
    pub fn format_system_message(&self, signature: &Signature) -> String {
        let mut parts = Vec::new();

        // Input fields section
        parts.push("Your input fields are:".to_string());
        parts.push(self.format_field_description_string(signature.input_fields()));

        // Output fields section
        parts.push("Your output fields are:".to_string());
        parts.push(self.format_field_description_string(signature.output_fields()));

        // Interaction structure template
        parts.push(
            "All interactions will be structured in the following way, with the appropriate values filled in.".to_string(),
        );
        parts.push(String::new());

        // Field template (all fields including inputs and outputs)
        for (name, _) in signature.fields() {
            parts.push(format!("[[ ## {} ## ]]", name));
            parts.push(format!("{{{}}}", name));
            parts.push(String::new());
        }

        // Completed marker
        parts.push("[[ ## completed ## ]]".to_string());

        // Objective
        let instructions = signature.instructions();
        if !instructions.is_empty() {
            parts.push(format!(
                "In adhering to this structure, your objective is: \n        {}",
                instructions
            ));
        } else {
            // Auto-generate from field names
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

    /// Format messages with demos as separate user/assistant pairs (matching Python DSPy).
    pub fn format_messages(
        &self,
        signature: &Signature,
        inputs: &Example,
        demos: &[Example],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // System message
        messages.push(Message::system(&self.format_system_message(signature)));

        // Demo messages as separate user/assistant pairs
        for demo in demos {
            // User message: demo input fields
            let mut user_parts = Vec::new();
            for (name, _) in signature.input_fields() {
                if let Some(val) = demo.get(&name) {
                    user_parts.push(format!("[[ ## {} ## ]]", name));
                    user_parts.push(val.to_string());
                }
            }
            messages.push(Message::user(&user_parts.join("\n")));

            // Assistant message: demo output fields + completed marker
            let mut assistant_parts = Vec::new();
            for (name, _) in signature.output_fields() {
                if demo.has(&name) {
                    assistant_parts.push(format!("[[ ## {} ## ]]", name));
                    let val = demo.get(&name).map(|v| v.to_string()).unwrap_or_default();
                    assistant_parts.push(val);
                }
            }
            assistant_parts.push(String::new());
            assistant_parts.push("[[ ## completed ## ]]".to_string());
            assistant_parts.push(String::new());
            messages.push(Message::assistant(&assistant_parts.join("\n")));
        }

        // Final user message: current inputs + prompt
        messages.push(Message::user(&self.format_user_message(signature, inputs)));

        messages
    }

    /// Format user message with current inputs and output prompt.
    /// Only for the final (non-demo) user message.
    pub fn format_user_message(&self, signature: &Signature, inputs: &Example) -> String {
        let mut parts = Vec::new();

        // Current input fields with blank line between them
        for (name, _) in signature.input_fields() {
            if let Some(val) = inputs.get(&name) {
                parts.push(format!("[[ ## {} ## ]]", name));
                parts.push(val.to_string());
                parts.push(String::new()); // blank line
            }
        }

        // Output requirements prompt
        parts.push(self.format_output_requirements(signature));

        parts.join("\n")
    }

    /// Parse LLM response text into field values using `[[ ## field ## ]]` delimiters.
    pub fn parse_output(
        &self,
        output: &str,
        signature: &Signature,
    ) -> Result<HashMap<String, Value>> {
        let mut result = HashMap::new();
        let output_field_names: Vec<String> =
            signature.output_fields().map(|(k, _)| k.clone()).collect();

        for name in &output_field_names {
            let marker = format!("[[ ## {} ## ]]", name);
            if let Some(start_idx) = output.find(&marker) {
                let content_start = start_idx + marker.len();

                // Find next marker (any [[ ## ... ## ]]) or end of string
                let rest = &output[content_start..];
                let content_end = if let Some(next_pos) = find_next_marker(rest) {
                    content_start + next_pos
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

/// Find the position of the next `[[ ## ... ## ]]` marker in a string.
fn find_next_marker(s: &str) -> Option<usize> {
    let marker_start = "[[ ## ";
    let marker_end = " ## ]]";
    if let Some(pos) = s.find(marker_start) {
        // Verify this is a complete marker
        if let Some(_end_pos) = s[pos..].find(marker_end) {
            return Some(pos);
        }
    }
    None
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
        let format_inputs = serde_json::json!({
            "signature": signature.instructions(),
            "demos": demos.len(),
        });
        let messages: Vec<Message> = with_callbacks_sync(
            ComponentType::AdapterFormat,
            "ChatAdapter",
            &format_inputs,
            || Ok::<_, DspyError>(self.format_messages(signature, inputs, demos)),
        )?;

        let lm_inputs = serde_json::json!({
            "messages": messages.len(),
            "model": lm.model(),
        });
        let responses = with_callbacks_async(ComponentType::Lm, lm.model(), &lm_inputs, || {
            lm.call(&messages, config)
        })
        .await?;

        let mut results = Vec::new();
        for resp in &responses {
            let parse_inputs = serde_json::json!({
                "output": resp.text,
                "signature": signature.instructions(),
            });
            let parsed = with_callbacks_sync(
                ComponentType::AdapterParse,
                "ChatAdapter",
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
        assert!(msg.contains("[[ ## completed ## ]]"));
        assert!(msg.contains("All interactions will be structured"));
    }

    #[test]
    fn test_format_messages_no_demos() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let inputs = Example::new().field("question", "What is 2+2?");
        let messages = adapter.format_messages(&sig, &inputs, &[]);
        // Should be [system, user]
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert!(messages[1].content.contains("[[ ## question ## ]]"));
        assert!(messages[1].content.contains("What is 2+2?"));
    }

    #[test]
    fn test_format_messages_with_demos() {
        let adapter = ChatAdapter::new();
        let sig = test_sig();
        let inputs = Example::new().field("question", "What is 3+3?");
        let demo = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");
        let messages = adapter.format_messages(&sig, &inputs, &[demo]);
        // Should be [system, user(demo_q), assistant(demo_a), user(current)]
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert!(messages[1].content.contains("What is 1+1?"));
        assert_eq!(messages[2].role, "assistant");
        assert!(messages[2].content.contains("2"));
        assert!(messages[2].content.contains("[[ ## completed ## ]]"));
        assert_eq!(messages[3].role, "user");
        assert!(messages[3].content.contains("What is 3+3?"));
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

    // --- Cross-validation tests against Python DSPy 3.1.2 golden outputs ---

    #[test]
    fn test_cross_validation_system_message_simple() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);

        // Exact match against Python DSPy 3.1.2
        let expected = "Your input fields are:\n\
            1. `question` (str):\n\
            Your output fields are:\n\
            1. `answer` (str):\n\
            All interactions will be structured in the following way, with the appropriate values filled in.\n\
            \n\
            [[ ## question ## ]]\n\
            {question}\n\
            \n\
            [[ ## answer ## ]]\n\
            {answer}\n\
            \n\
            [[ ## completed ## ]]\n\
            In adhering to this structure, your objective is: \n\
            \x20       Given the fields `question`, produce the fields `answer`.";
        assert_eq!(msg, expected);
    }

    #[test]
    fn test_cross_validation_system_message_multi_field() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question, context -> reasoning, answer").unwrap();
        let msg = adapter.format_system_message(&sig);

        // Multi-field: non-last fields have trailing space after ":"
        assert!(msg.contains("1. `question` (str): \n2. `context` (str):"));
        assert!(msg.contains("1. `reasoning` (str): \n2. `answer` (str):"));
    }

    #[test]
    fn test_cross_validation_user_message_simple() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let msg = adapter.format_user_message(&sig, &inputs);

        let expected = "[[ ## question ## ]]\n\
            What is 2+2?\n\
            \n\
            Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.";
        assert_eq!(msg, expected);
    }

    #[test]
    fn test_cross_validation_user_message_multi_field() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question, context -> reasoning, answer").unwrap();
        let inputs = Example::new()
            .field("question", "What color?")
            .field("context", "The sky is blue.");
        let msg = adapter.format_user_message(&sig, &inputs);

        let expected = "[[ ## question ## ]]\n\
            What color?\n\
            \n\
            [[ ## context ## ]]\n\
            The sky is blue.\n\
            \n\
            Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.";
        assert_eq!(msg, expected);
    }

    #[test]
    fn test_cross_validation_demo_messages() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let demo1 = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");
        let demo2 = Example::new()
            .field("question", "What is 3+3?")
            .field("answer", "6");
        let messages = adapter.format_messages(&sig, &inputs, &[demo1, demo2]);

        // Python DSPy format: [system, user(demo1_q), assistant(demo1_a), user(demo2_q), assistant(demo2_a), user(current)]
        assert_eq!(messages.len(), 6);
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "[[ ## question ## ]]\nWhat is 1+1?");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(
            messages[2].content,
            "[[ ## answer ## ]]\n2\n\n[[ ## completed ## ]]\n"
        );
        assert_eq!(messages[3].role, "user");
        assert_eq!(messages[3].content, "[[ ## question ## ]]\nWhat is 3+3?");
        assert_eq!(messages[4].role, "assistant");
        assert_eq!(
            messages[4].content,
            "[[ ## answer ## ]]\n6\n\n[[ ## completed ## ]]\n"
        );
        assert_eq!(messages[5].role, "user");
    }

    #[test]
    fn test_cross_validation_parse_simple() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let response = "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]";
        let parsed = adapter.parse_output(response, &sig).unwrap();
        assert_eq!(parsed["answer"].as_str(), Some("4"));
    }

    #[test]
    fn test_cross_validation_parse_multi_field() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> reasoning, answer").unwrap();
        let response =
            "[[ ## reasoning ## ]]\n2+2 equals 4\n\n[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]";
        let parsed = adapter.parse_output(response, &sig).unwrap();
        assert_eq!(parsed["reasoning"].as_str(), Some("2+2 equals 4"));
        assert_eq!(parsed["answer"].as_str(), Some("4"));
    }

    #[test]
    fn test_cross_validation_parse_multiline() {
        let adapter = ChatAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let response = "[[ ## answer ## ]]\nLine 1\nLine 2\nLine 3\n\n[[ ## completed ## ]]";
        let parsed = adapter.parse_output(response, &sig).unwrap();
        assert_eq!(parsed["answer"].as_str(), Some("Line 1\nLine 2\nLine 3"));
    }
}
