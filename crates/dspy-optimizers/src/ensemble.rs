//! Ensemble — reduces a set of candidate programs into one via majority vote.
//! Python equivalent: dspy/teleprompt/ensemble.py

use dspy_core::{Example, Module, Prediction};
use std::collections::HashMap;
use std::sync::Arc;

/// EnsembleConfig controls ensemble behavior.
pub struct EnsembleConfig {
    /// How to aggregate multiple programs' outputs.
    pub reduce_fn: ReduceFn,
    /// Maximum number of programs to include in the ensemble.
    pub size: Option<usize>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            reduce_fn: ReduceFn::MajorityVote,
            size: None,
        }
    }
}

/// Aggregation strategy for combining multiple program outputs.
pub enum ReduceFn {
    /// Most common answer wins (per output field).
    MajorityVote,
    /// Custom aggregation function.
    Custom(Arc<dyn Fn(Vec<Prediction>) -> Prediction + Send + Sync>),
}

/// An ensemble module that runs multiple programs and aggregates their outputs.
pub struct EnsembleModule {
    programs: Vec<Box<dyn Module>>,
    reduce_fn: Arc<dyn Fn(Vec<Prediction>) -> Prediction + Send + Sync>,
}

impl EnsembleModule {
    pub fn new(programs: Vec<Box<dyn Module>>, config: EnsembleConfig) -> Self {
        let reduce_fn: Arc<dyn Fn(Vec<Prediction>) -> Prediction + Send + Sync> = match config.reduce_fn {
            ReduceFn::MajorityVote => Arc::new(majority_vote),
            ReduceFn::Custom(f) => f,
        };

        let programs = match config.size {
            Some(size) => programs.into_iter().take(size).collect(),
            None => programs,
        };

        Self { programs, reduce_fn }
    }

    pub fn programs(&self) -> &[Box<dyn Module>] {
        &self.programs
    }
}

#[async_trait::async_trait]
impl Module for EnsembleModule {
    async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
        let mut predictions = Vec::new();
        for program in &self.programs {
            match program.forward(args).await {
                Ok(pred) => predictions.push(pred),
                Err(_) => continue, // Skip failed programs
            }
        }

        if predictions.is_empty() {
            return Err(dspy_core::DspyError::ModuleError(
                "All ensemble programs failed".into(),
            ));
        }

        Ok((self.reduce_fn)(predictions))
    }

    fn named_predictors(&self) -> Vec<(&str, &dspy_core::Predict)> {
        // Ensemble doesn't expose individual predictors
        vec![]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut dspy_core::Predict)> {
        vec![]
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(EnsembleModule {
            programs: self.programs.iter().map(|p| p.deep_copy()).collect(),
            reduce_fn: self.reduce_fn.clone(),
        })
    }
}

/// Default majority vote: for each output field, pick the most common value.
fn majority_vote(predictions: Vec<Prediction>) -> Prediction {
    if predictions.is_empty() {
        return Prediction::new(HashMap::new());
    }
    if predictions.len() == 1 {
        return predictions.into_iter().next().unwrap();
    }

    // Collect all unique field names from all predictions
    let mut all_keys: Vec<String> = Vec::new();
    for pred in &predictions {
        for key in pred.keys() {
            if !all_keys.contains(key) {
                all_keys.push(key.clone());
            }
        }
    }

    // For each field, do majority vote
    let mut result_map = HashMap::new();
    for key in &all_keys {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for pred in &predictions {
            if let Some(val) = pred.get(key) {
                let val_str = format!("{val}");
                *counts.entry(val_str).or_insert(0) += 1;
            }
        }

        // Pick the value with highest count
        if let Some((best_str, _)) = counts.into_iter().max_by_key(|(_, count)| *count) {
            // Find the original Value for this string representation
            for pred in &predictions {
                if let Some(val) = pred.get(key) {
                    if format!("{val}") == best_str {
                        result_map.insert(key.clone(), val.clone());
                        break;
                    }
                }
            }
        }
    }

    Prediction::new(result_map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{
        Example, LM, LMConfig, LMResponse, Message, Predict, Signature,
    };
    use async_trait::async_trait;

    struct ConstLM {
        answer: String,
        config: LMConfig,
    }

    impl ConstLM {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                config: LMConfig::new("const"),
            }
        }
    }

    #[async_trait]
    impl LM for ConstLM {
        async fn call(&self, _: &[Message], _: &LMConfig) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse {
                text: format!("[[ ## answer ## ]]\n{}", self.answer),
                usage: None,
            }])
        }
        fn model(&self) -> &str { "const" }
        fn config(&self) -> &LMConfig { &self.config }
        fn dump_state(&self) -> serde_json::Value { serde_json::json!({}) }
    }

    struct SimpleModule {
        predict: Predict,
    }

    impl SimpleModule {
        fn new(lm: Arc<dyn LM>) -> Self {
            let mut predict = Predict::new(Signature::from_string("q -> answer").unwrap());
            predict.set_lm(lm);
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for SimpleModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
            self.predict.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(SimpleModule { predict: self.predict.clone() })
        }
    }

    #[tokio::test]
    async fn test_ensemble_majority_vote() {
        dspy_core::reset_settings();
        // 3 programs: 2 say "yes", 1 says "no" → majority is "yes"
        let programs: Vec<Box<dyn Module>> = vec![
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("yes")))),
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("yes")))),
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("no")))),
        ];

        let ensemble = EnsembleModule::new(programs, EnsembleConfig::default());
        let input = Example::new().field("q", "test");
        let result = ensemble.forward(&input).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("yes"));
    }

    #[tokio::test]
    async fn test_ensemble_single_program() {
        dspy_core::reset_settings();
        let programs: Vec<Box<dyn Module>> = vec![
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("only")))),
        ];

        let ensemble = EnsembleModule::new(programs, EnsembleConfig::default());
        let input = Example::new().field("q", "test");
        let result = ensemble.forward(&input).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("only"));
    }

    #[tokio::test]
    async fn test_ensemble_deep_copy() {
        dspy_core::reset_settings();
        let programs: Vec<Box<dyn Module>> = vec![
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("a")))),
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("b")))),
        ];

        let ensemble = EnsembleModule::new(programs, EnsembleConfig::default());
        let copy = ensemble.deep_copy();

        let input = Example::new().field("q", "test");
        let result = copy.forward(&input).await.unwrap();
        // With a tie (a vs b), first one seen wins
        assert!(result.get_str("answer").is_some());
    }

    #[tokio::test]
    async fn test_ensemble_with_size_limit() {
        dspy_core::reset_settings();
        let programs: Vec<Box<dyn Module>> = vec![
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("a")))),
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("b")))),
            Box::new(SimpleModule::new(Arc::new(ConstLM::new("c")))),
        ];

        let ensemble = EnsembleModule::new(programs, EnsembleConfig {
            reduce_fn: ReduceFn::MajorityVote,
            size: Some(2),
        });

        assert_eq!(ensemble.programs().len(), 2);
    }
}
