//! TwoStepAdapter — two-pass adapter for reasoning models.
//! Python equivalent: dspy/adapters/two_step_adapter.py
//!
//! Strategy:
//! 1. First pass: simple natural-language prompt sent to main LM (no strict format)
//! 2. Second pass: extraction LM with ChatAdapter parses the free-form response
//!
//! Useful with reasoning models (for instance, o3-mini) that struggle with structured outputs.

use crate::adapter::{Adapter, ChatAdapter};
use crate::callback::{with_callbacks_sync, ComponentType};
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::{LMConfig, Message, LM};
use crate::signature::Signature;
use crate::value::Value;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

pub struct TwoStepAdapter {
    extraction_lm: Arc<dyn LM>,
    chat_adapter: ChatAdapter,
}

impl TwoStepAdapter {
    pub fn new(extraction_lm: Arc<dyn LM>) -> Self {
        Self {
            extraction_lm,
            chat_adapter: ChatAdapter::new(),
        }
    }

    /// Format natural-language messages for the first pass (no strict format).
    fn format_first_pass_messages(
        &self,
        signature: &Signature,
        inputs: &Example,
        demos: &[Example],
    ) -> Vec<Message> {
        let mut messages = Vec::new();

        // System: task description
        messages.push(Message::system(&self.format_task_description(signature)));

        // Demos as simple user/assistant pairs
        for demo in demos {
            let user_content = self.format_simple_inputs(signature, demo);
            messages.push(Message::user(&user_content));

            let assistant_content = self.format_simple_outputs(signature, demo);
            messages.push(Message::assistant(&assistant_content));
        }

        // Current input
        messages.push(Message::user(&self.format_simple_inputs(signature, inputs)));

        messages
    }

    /// Create a natural-language task description from the signature.
    fn format_task_description(&self, signature: &Signature) -> String {
        let mut parts = Vec::new();

        parts.push(
            "You are a helpful assistant that can solve tasks based on user input.".to_string(),
        );

        // Input field descriptions
        let input_desc = self.format_field_list(signature.input_fields());
        parts.push(format!("As input, you will be provided with:\n{}", input_desc));

        // Output field descriptions
        let output_desc = self.format_field_list(signature.output_fields());
        parts.push(format!("Your outputs must contain:\n{}", output_desc));

        parts.push(
            "You should lay out your outputs in detail so that your answer can be understood by another agent".to_string(),
        );

        let instructions = signature.instructions();
        if !instructions.is_empty() {
            parts.push(format!("Specific instructions: {}", instructions));
        }

        parts.join("\n")
    }

    /// Format field list as numbered descriptions.
    fn format_field_list<'a>(
        &self,
        fields: impl Iterator<Item = (&'a String, &'a crate::signature::FieldDef)>,
    ) -> String {
        let items: Vec<String> = fields
            .enumerate()
            .map(|(idx, (name, field))| {
                let desc = field.description.as_deref().unwrap_or("");
                format!("{}. `{}` (str): {}", idx + 1, name, desc)
            })
            .collect();
        items.join("\n").trim_end().to_string()
    }

    /// Format inputs as simple "name: value" pairs.
    fn format_simple_inputs(&self, signature: &Signature, data: &Example) -> String {
        let mut parts = Vec::new();
        for (name, _) in signature.input_fields() {
            if let Some(val) = data.get(&name) {
                parts.push(format!("{}: {}", name, val));
            }
        }
        parts.join("\n\n").trim_end().to_string()
    }

    /// Format outputs as simple "name: value" pairs.
    fn format_simple_outputs(&self, signature: &Signature, data: &Example) -> String {
        let mut parts = Vec::new();
        for (name, _) in signature.output_fields() {
            if let Some(val) = data.get(&name) {
                parts.push(format!("{}: {}", name, val));
            }
        }
        parts.join("\n\n").trim_end().to_string()
    }

    /// Create an extractor signature: "text -> {original output fields}"
    fn create_extractor_signature(&self, original_signature: &Signature) -> Result<Signature> {
        let output_field_names: Vec<String> = original_signature
            .output_fields()
            .map(|(k, _)| k.clone())
            .collect();
        let outputs_str = output_field_names
            .iter()
            .map(|f| format!("`{}`", f))
            .collect::<Vec<_>>()
            .join(", ");
        let instructions = format!(
            "The input is a text that should contain all the necessary information to produce the fields {}. \
            Your job is to extract the fields from the text verbatim. Extract precisely the appropriate value (content) for each field.",
            outputs_str
        );

        let sig_spec = format!("text -> {}", output_field_names.join(", "));
        Signature::from_string(&sig_spec)
            .map(|s| s.with_instructions(&instructions))
    }
}

#[async_trait]
impl Adapter for TwoStepAdapter {
    async fn call(
        &self,
        lm: &dyn LM,
        signature: &Signature,
        demos: &[Example],
        inputs: &Example,
        config: &LMConfig,
    ) -> Result<Vec<HashMap<String, Value>>> {
        // Phase 1: format natural-language messages and call main LM
        let format_inputs = serde_json::json!({
            "signature": signature.instructions(),
            "demos": demos.len(),
        });
        let messages: Vec<Message> = with_callbacks_sync(
            ComponentType::AdapterFormat,
            "TwoStepAdapter",
            &format_inputs,
            || Ok::<_, DspyError>(self.format_first_pass_messages(signature, inputs, demos)),
        )?;
        let responses = lm.call(&messages, config).await?;

        if responses.is_empty() {
            return Err(DspyError::LMError("No responses from LM".into()));
        }

        // Phase 2: extract structured fields from each response
        let extractor_sig = self.create_extractor_signature(signature)?;
        let extraction_config = LMConfig::new(self.extraction_lm.model());

        let mut results = Vec::new();
        for resp in &responses {
            let extract_inputs = Example::new().field("text", resp.text.as_str());
            let extracted = self.chat_adapter.call(
                self.extraction_lm.as_ref(),
                &extractor_sig,
                &[],
                &extract_inputs,
                &extraction_config,
            ).await?;
            if let Some(first) = extracted.into_iter().next() {
                results.push(first);
            }
        }

        if results.is_empty() {
            return Err(DspyError::LMError("Extraction failed for all responses".into()));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::LMResponse;
    use std::sync::Mutex;

    struct MockLM {
        response: String,
        config: LMConfig,
        captured_messages: Mutex<Vec<Vec<Message>>>,
    }

    impl MockLM {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                config: LMConfig::new("mock"),
                captured_messages: Mutex::new(Vec::new()),
            }
        }

        fn last_messages(&self) -> Vec<Message> {
            let guard = self.captured_messages.lock().unwrap();
            guard.last().cloned().unwrap_or_default()
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(&self, messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            self.captured_messages.lock().unwrap().push(messages.to_vec());
            Ok(vec![LMResponse {
                text: self.response.clone(),
                usage: None,
            }])
        }
        fn model(&self) -> &str { "mock" }
        fn config(&self) -> &LMConfig { &self.config }
        fn dump_state(&self) -> serde_json::Value { serde_json::json!({}) }
    }

    #[tokio::test]
    async fn test_first_pass_natural_language() {
        let main_lm = Arc::new(MockLM::new("The answer is 4."));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm);
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");

        adapter.call(main_lm.as_ref(), &sig, &[], &inputs, &LMConfig::new("mock")).await.unwrap();

        // Main LM should receive natural-language messages
        let msgs = main_lm.last_messages();
        let system = &msgs[0];
        assert!(system.content.contains("helpful assistant"));
        assert!(!system.content.contains("[[ ##"));

        let user = &msgs[1];
        assert!(user.content.contains("What is 2+2?"));
        assert!(!user.content.contains("[[ ##"));
    }

    #[tokio::test]
    async fn test_second_pass_extraction() {
        let main_lm = Arc::new(MockLM::new("The answer is 4."));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\n4\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm.clone());
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");

        let results = adapter
            .call(main_lm.as_ref(), &sig, &[], &inputs, &LMConfig::new("mock"))
            .await
            .unwrap();

        // Extraction LM should be called with "text -> answer" signature
        let msgs = extract_lm.last_messages();
        let system = &msgs[0];
        assert!(system.content.contains("[[ ## text ## ]]"));
        assert!(system.content.contains("[[ ## answer ## ]]"));

        // Should get extracted result
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["answer"].as_str(), Some("4"));
    }

    #[tokio::test]
    async fn test_task_description_includes_fields() {
        let main_lm = Arc::new(MockLM::new("text"));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\nresult\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm);
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "test");

        adapter.call(main_lm.as_ref(), &sig, &[], &inputs, &LMConfig::new("mock")).await.unwrap();

        let msgs = main_lm.last_messages();
        let system = &msgs[0];
        assert!(system.content.contains("`question`"));
        assert!(system.content.contains("`answer`"));
        assert!(system.content.contains("outputs must contain"));
    }

    #[tokio::test]
    async fn test_includes_instructions() {
        let main_lm = Arc::new(MockLM::new("Four"));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\nFour\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm);
        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Answer in one word.");
        let inputs = Example::new().field("question", "What is 2+2?");

        adapter.call(main_lm.as_ref(), &sig, &[], &inputs, &LMConfig::new("mock")).await.unwrap();

        let msgs = main_lm.last_messages();
        let system = &msgs[0];
        assert!(system.content.contains("Answer in one word."));
    }

    #[tokio::test]
    async fn test_demos_as_simple_pairs() {
        let main_lm = Arc::new(MockLM::new("Four"));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\nFour\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm);
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "What is 2+2?");
        let demo = Example::new()
            .field("question", "What is 1+1?")
            .field("answer", "2");

        adapter.call(main_lm.as_ref(), &sig, &[demo], &inputs, &LMConfig::new("mock")).await.unwrap();

        // Should have: system, demo_user, demo_assistant, current_user
        let msgs = main_lm.last_messages();
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[1].role, "user");
        assert!(msgs[1].content.contains("What is 1+1?"));
        assert_eq!(msgs[2].role, "assistant");
        assert!(msgs[2].content.contains("2"));
    }

    #[tokio::test]
    async fn test_extraction_receives_main_lm_response() {
        let main_lm = Arc::new(MockLM::new("The answer is 42."));
        let extract_lm = Arc::new(MockLM::new(
            "[[ ## answer ## ]]\n42\n\n[[ ## completed ## ]]",
        ));
        let adapter = TwoStepAdapter::new(extract_lm.clone());
        let sig = Signature::from_string("question -> answer").unwrap();
        let inputs = Example::new().field("question", "test");

        adapter.call(main_lm.as_ref(), &sig, &[], &inputs, &LMConfig::new("mock")).await.unwrap();

        // Extraction user message should contain the main LM's response
        let msgs = extract_lm.last_messages();
        let user = &msgs[1]; // [system, user]
        assert!(user.content.contains("The answer is 42."));
    }
}
