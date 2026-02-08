//! ChainOfThought — prepends reasoning field to signature.
//! Python equivalent: dspy/predict/chain_of_thought.py

use crate::error::Result;
use crate::example::Example;
use crate::lm::LM;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::signature::{output_field, Signature};
use async_trait::async_trait;
use std::sync::Arc;

pub struct ChainOfThought {
    predict: Predict,
}

impl ChainOfThought {
    pub fn new(signature: Signature) -> Self {
        // Prepend a "reasoning" output field before existing output fields
        // Python DSPy uses desc="${reasoning}" which renders as empty in the adapter.
        // The prefix carries the actual prompt text. Match this for parity.
        let extended = signature.prepend(output_field("reasoning"));
        Self {
            predict: Predict::new(extended),
        }
    }

    pub fn predict(&self) -> &Predict {
        &self.predict
    }

    pub fn predict_mut(&mut self) -> &mut Predict {
        &mut self.predict
    }
}

#[async_trait]
impl crate::module_trait::Module for ChainOfThought {
    fn module_type_name(&self) -> &str {
        "ChainOfThought"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        self.predict.call(args).await
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
            predict: self.predict.clone(),
        })
    }

    fn dump_state(&self) -> serde_json::Value {
        self.predict.dump_state()
    }

    fn load_state(&mut self, state: &serde_json::Value) -> Result<()> {
        self.predict.load_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module_trait::Module;

    #[test]
    fn test_cot_prepends_reasoning() {
        let sig = Signature::from_string("question -> answer").unwrap();
        let cot = ChainOfThought::new(sig);

        let fields: Vec<_> = cot.predict().signature.fields().keys().collect();
        // Should be: question, reasoning, answer
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], "question");
        assert_eq!(fields[1], "reasoning");
        assert_eq!(fields[2], "answer");
    }

    #[test]
    fn test_cot_reasoning_is_output() {
        let sig = Signature::from_string("q -> a").unwrap();
        let cot = ChainOfThought::new(sig);
        assert_eq!(cot.predict().signature.output_fields().count(), 2);
        assert_eq!(cot.predict().signature.input_fields().count(), 1);
    }

    #[test]
    fn test_cot_named_predictors() {
        let sig = Signature::from_string("q -> a").unwrap();
        let cot = ChainOfThought::new(sig);
        let preds = cot.named_predictors();
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, "predict");
    }

    #[test]
    fn test_cot_deep_copy() {
        let sig = Signature::from_string("q -> a").unwrap();
        let cot = ChainOfThought::new(sig);
        let copied = cot.deep_copy();
        let preds = copied.named_predictors();
        assert_eq!(preds.len(), 1);
    }
}
