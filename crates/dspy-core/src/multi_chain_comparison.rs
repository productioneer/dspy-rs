//! MultiChainComparison — generate M CoT attempts, then compare and select best.
//! Python equivalent: dspy/predict/multi_chain_comparison.py

use crate::error::Result;
use crate::example::Example;
use crate::lm::LM;
use crate::module_trait::Module;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::signature::{input_field, output_field, Signature};
use async_trait::async_trait;
use std::sync::Arc;

pub struct MultiChainComparison {
    m: usize,
    predict: Predict,
    last_key: String,
}

impl MultiChainComparison {
    pub fn new(signature: Signature, m: Option<usize>, _temperature: Option<f64>) -> Self {
        let m = m.unwrap_or(3);

        // Identify the last output key
        let output_keys: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| name.to_string())
            .collect();
        let last_key = output_keys
            .last()
            .expect("Signature must have at least one output field")
            .clone();

        // Build extended signature: add M reasoning_attempt input fields
        let mut sig = signature;
        for idx in 0..m {
            sig = sig.append(
                input_field(&format!("reasoning_attempt_{}", idx + 1))
                    .with_prefix(&format!("Student Attempt #{}:", idx + 1))
                    .with_desc("${reasoning attempt}"),
            );
        }

        // Prepend a "rationale" output field before existing outputs
        sig = sig.prepend(
            output_field("rationale")
                .with_prefix("Accurate Reasoning: Thank you everyone. Let's now holistically")
                .with_desc("${corrected reasoning}"),
        );

        let predict = Predict::new(sig);
        // Note: temperature is handled at LM config level, not stored on Predict

        Self {
            m,
            predict,
            last_key,
        }
    }

    /// Forward: takes a list of completions (reasoning attempts) + additional kwargs.
    pub async fn forward_with_completions(
        &self,
        completions: &[Example],
        kwargs: &Example,
    ) -> Result<Prediction> {
        assert_eq!(
            completions.len(),
            self.m,
            "Number of attempts ({}) doesn't match M ({})",
            completions.len(),
            self.m
        );

        let mut inputs = kwargs.clone();
        for (idx, c) in completions.iter().enumerate() {
            let rationale = c
                .get("rationale")
                .or_else(|| c.get("reasoning"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let rationale_first_line = rationale.trim().lines().next().unwrap_or("").trim();

            let answer = c
                .get(&self.last_key)
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let answer_first_line = answer.trim().lines().next().unwrap_or("").trim();

            inputs.set(
                format!("reasoning_attempt_{}", idx + 1),
                format!(
                    "\u{00ab}I'm trying to {} I'm not sure but my prediction is {}\u{00bb}",
                    rationale_first_line, answer_first_line
                ),
            );
        }

        self.predict.forward(&inputs).await
    }

    pub fn predict(&self) -> &Predict {
        &self.predict
    }

    pub fn predict_mut(&mut self) -> &mut Predict {
        &mut self.predict
    }
}

#[async_trait]
impl crate::module_trait::Module for MultiChainComparison {
    async fn forward(&self, _args: &Example) -> Result<Prediction> {
        // When called via Module trait, expect completions in args
        // This is a simplified path — the main API is forward_with_completions
        Err(crate::error::DspyError::Other(
            "Use forward_with_completions for MultiChainComparison".into(),
        ))
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![("predict", &self.predict)]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![("predict", &mut self.predict)]
    }

    fn set_lm(&mut self, lm: Arc<dyn LM>) {
        self.predict.set_lm(lm);
    }

    fn deep_copy(&self) -> Box<dyn crate::module_trait::Module> {
        Box::new(Self {
            m: self.m,
            predict: self.predict.clone(),
            last_key: self.last_key.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcc_signature_structure() {
        let sig = Signature::from_string("question -> answer").unwrap();
        let mcc = MultiChainComparison::new(sig, Some(3), None);
        let fields: Vec<_> = mcc.predict().signature.fields().keys().collect();
        // question, reasoning_attempt_1, reasoning_attempt_2, reasoning_attempt_3, rationale, answer
        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0], "question");
        assert!(fields[1].starts_with("reasoning_attempt_"));
        assert!(fields[2].starts_with("reasoning_attempt_"));
        assert!(fields[3].starts_with("reasoning_attempt_"));
        assert_eq!(fields[4], "rationale");
        assert_eq!(fields[5], "answer");
    }

    #[test]
    fn test_mcc_custom_m() {
        let sig = Signature::from_string("q -> a").unwrap();
        let mcc = MultiChainComparison::new(sig, Some(5), None);
        let input_count = mcc.predict().signature.input_fields().count();
        // q + 5 reasoning_attempts = 6 input fields
        assert_eq!(input_count, 6);
    }

    #[test]
    fn test_mcc_output_fields() {
        let sig = Signature::from_string("q -> a").unwrap();
        let mcc = MultiChainComparison::new(sig, Some(3), None);
        let output_keys: Vec<_> = mcc
            .predict()
            .signature
            .output_fields()
            .map(|(name, _)| name.to_string())
            .collect();
        assert_eq!(output_keys, vec!["rationale", "a"]);
    }
}
