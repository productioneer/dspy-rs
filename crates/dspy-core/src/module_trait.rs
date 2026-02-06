//! Module trait — composable program unit.
//! Python equivalent: dspy/primitives/module.py

use crate::error::Result;
use crate::example::Example;
use crate::lm::LM;
use crate::predict::Predict;
use crate::prediction::Prediction;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait Module: Send + Sync {
    async fn forward(&self, args: &Example) -> Result<Prediction>;

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![]
    }

    fn set_lm(&mut self, lm: Arc<dyn LM>) {
        for (_, pred) in self.named_predictors_mut() {
            pred.set_lm(lm.clone());
        }
    }

    fn deep_copy(&self) -> Box<dyn Module>;

    fn dump_state(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn load_state(&mut self, _state: &serde_json::Value) -> Result<()> {
        Ok(())
    }
}
