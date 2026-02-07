//! XMLAdapter — LLM message formatting with XML tags.
//! Python equivalent: dspy/adapters/xml_adapter.py
//!
//! Key differences from ChatAdapter:
//! - Uses <field>content</field> XML tags instead of `[[ ## field_name ## ]]` delimiters
//! - Parses LM output using regex to extract XML tag content

use crate::adapter::{Adapter, ChatAdapter};
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::{LMConfig, Message, LM};
use crate::signature::Signature;
use crate::value::Value;
use async_trait::async_trait;
use regex::Regex;
use std::collections::HashMap;

pub struct XMLAdapter {
    chat_adapter: ChatAdapter,
    /// Regex to find opening XML tags: <word_chars>
    open_tag_pattern: Regex,
}

impl XMLAdapter {
    pub fn new() -> Self {
        Self {
            chat_adapter: ChatAdapter::new(),
            open_tag_pattern: Regex::new(r"<(\w+)>").unwrap(),
        }
    }

    /// Extract all XML-tagged fields from text.
    /// Finds <name>content</name> patterns and returns (name, content) pairs.
    /// Only returns the first match per field name.
    fn extract_xml_fields(&self, text: &str) -> Vec<(String, String)> {
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for caps in self.open_tag_pattern.captures_iter(text) {
            let full_match = caps.get(0).unwrap();
            let tag_name = caps.get(1).unwrap().as_str();

            if seen.contains(tag_name) {
                continue;
            }

            // Find the matching closing tag
            let closing_tag = format!("</{}>", tag_name);
            let content_start = full_match.end();
            if let Some(close_pos) = text[content_start..].find(&closing_tag) {
                let content = text[content_start..content_start + close_pos].trim();
                results.push((tag_name.to_string(), content.to_string()));
                seen.insert(tag_name.to_string());
            }
        }

        results
    }

    /// Format system message with XML tag format description.
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

        // All fields in XML format
        for (name, _) in signature.fields() {
            parts.push(format!("<{}>", name));
            parts.push(format!("{{{}}}", name));
            parts.push(format!("</{}>", name));
            parts.push(String::new());
        }

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

    /// Format messages with XML-tagged fields.
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
            // User: input fields in XML
            let mut user_parts = Vec::new();
            for (name, _) in signature.input_fields() {
                if let Some(val) = demo.get(&name) {
                    user_parts.push(format!("<{}>", name));
                    user_parts.push(val.to_string());
                    user_parts.push(format!("</{}>", name));
                }
            }
            messages.push(Message::user(&user_parts.join("\n")));

            // Assistant: output fields in XML
            let mut assistant_parts = Vec::new();
            for (name, _) in signature.output_fields() {
                if demo.has(&name) {
                    assistant_parts.push(format!("<{}>", name));
                    let val = demo.get(&name).map(|v| v.to_string()).unwrap_or_default();
                    assistant_parts.push(val);
                    assistant_parts.push(format!("</{}>", name));
                }
            }
            messages.push(Message::assistant(&assistant_parts.join("\n")));
        }

        // Final user message
        messages.push(Message::user(&self.format_user_message(signature, inputs)));

        messages
    }

    /// Format user message with XML-tagged inputs and XML output requirements.
    pub fn format_user_message(&self, signature: &Signature, inputs: &Example) -> String {
        let mut parts = Vec::new();

        // Input fields in XML
        for (name, _) in signature.input_fields() {
            if let Some(val) = inputs.get(&name) {
                parts.push(format!("<{}>", name));
                parts.push(val.to_string());
                parts.push(format!("</{}>", name));
                parts.push(String::new());
            }
        }

        // XML output requirements
        let output_names: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| format!("`<{}>`", name))
            .collect();
        parts.push(format!(
            "Respond with the corresponding output fields wrapped in XML tags {}.",
            output_names.join(", then ")
        ));

        parts.join("\n")
    }

    /// Parse XML-tagged output from LM response.
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
        let output_set: std::collections::HashSet<&str> =
            output_field_names.iter().map(|s| s.as_str()).collect();

        for (name, content) in self.extract_xml_fields(output) {
            // Only extract output fields, take first match for each field
            if output_set.contains(name.as_str()) && !result.contains_key(&name) {
                result.insert(name, Value::from(content));
            }
        }

        // If no tags found, treat entire output as first output field
        if result.is_empty() && !output_field_names.is_empty() {
            result.insert(
                output_field_names[0].clone(),
                Value::from(output.trim().to_string()),
            );
        }

        Ok(result)
    }
}

impl Default for XMLAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Adapter for XMLAdapter {
    async fn call(
        &self,
        lm: &dyn LM,
        signature: &Signature,
        demos: &[Example],
        inputs: &Example,
        config: &LMConfig,
    ) -> Result<Vec<HashMap<String, Value>>> {
        let messages = self.format_messages(signature, inputs, demos);

        let responses = lm.call(&messages, config).await?;

        let mut results = Vec::new();
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

    #[test]
    fn test_system_message_uses_xml_tags() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("<question>"));
        assert!(msg.contains("</question>"));
        assert!(msg.contains("<answer>"));
        assert!(msg.contains("</answer>"));
    }

    #[test]
    fn test_system_message_no_bracket_markers() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let msg = adapter.format_system_message(&sig);
        assert!(!msg.contains("[[ ## question ## ]]"));
        assert!(!msg.contains("[[ ## answer ## ]]"));
        assert!(!msg.contains("[[ ## completed ## ]]"));
    }

    #[test]
    fn test_user_message_xml_format() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let msg = adapter.format_user_message(&sig, &inputs);
        assert!(msg.contains("<question>"));
        assert!(msg.contains("What is 2+2?"));
        assert!(msg.contains("</question>"));
        assert!(msg.contains("wrapped in XML tags"));
        assert!(msg.contains("`<answer>`"));
    }

    #[test]
    fn test_format_messages_with_xml_demos() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let demo = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");
        let messages = adapter.format_messages(&sig, &inputs, &[demo]);

        // [system, demo_user, demo_assistant, current_user]
        assert_eq!(messages.len(), 4);
        // Demo user: XML
        assert!(messages[1].content.contains("<question>"));
        assert!(messages[1].content.contains("What is 1+1?"));
        assert!(messages[1].content.contains("</question>"));
        // Demo assistant: XML
        assert!(messages[2].content.contains("<answer>"));
        assert!(messages[2].content.contains("2"));
        assert!(messages[2].content.contains("</answer>"));
    }

    #[test]
    fn test_parse_xml_tags() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter.parse_output("<answer>42</answer>", &sig).unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
    }

    #[test]
    fn test_parse_xml_from_surrounding_text() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("Here is my answer:\n<answer>42</answer>\nDone.", &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
    }

    #[test]
    fn test_parse_multi_field_xml() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> reasoning, answer").unwrap();
        let result = adapter
            .parse_output(
                "<reasoning>2+2 equals 4</reasoning>\n<answer>4</answer>",
                &sig,
            )
            .unwrap();
        assert_eq!(result["reasoning"].as_str(), Some("2+2 equals 4"));
        assert_eq!(result["answer"].as_str(), Some("4"));
    }

    #[test]
    fn test_parse_multiline_xml() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("<answer>Line 1\nLine 2\nLine 3</answer>", &sig)
            .unwrap();
        assert_eq!(
            result["answer"].as_str(),
            Some("Line 1\nLine 2\nLine 3")
        );
    }

    #[test]
    fn test_parse_ignores_non_output_fields() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output(
                "<question>ignored</question>\n<answer>42</answer>",
                &sig,
            )
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("42"));
        assert!(!result.contains_key("question"));
    }

    #[test]
    fn test_parse_takes_first_match() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("<answer>first</answer>\n<answer>second</answer>", &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("first"));
    }

    #[test]
    fn test_parse_fallback_to_raw_text() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer").unwrap();
        let result = adapter
            .parse_output("The answer is 42", &sig)
            .unwrap();
        assert_eq!(result["answer"].as_str(), Some("The answer is 42"));
    }

    #[test]
    fn test_system_message_with_instructions() {
        let adapter = XMLAdapter::new();
        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Be concise");
        let msg = adapter.format_system_message(&sig);
        assert!(msg.contains("Be concise"));
    }
}
