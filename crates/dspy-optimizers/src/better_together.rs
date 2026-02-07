//! BetterTogether: composes prompt optimization and weight optimization strategies.
//! Strategy format: "p -> w -> p" where p=prompt optimization, w=weight optimization.
//! Python equivalent: dspy/teleprompt/bettertogether.py

use dspy_core::{Example, Metric, Module};

use crate::bootstrap_finetune::{BootstrapFinetune, BootstrapFinetuneConfig};
use crate::random_search::{BootstrapFewShotWithRandomSearch, RandomSearchConfig};

const STRAT_SEP: &str = " -> ";

/// Configuration for BetterTogether.
pub struct BetterTogetherConfig {
    pub metric: Metric,
    pub seed: u32,
    pub valset_ratio: f64,
    pub strategy: String,
}

impl BetterTogetherConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            seed: 0,
            valset_ratio: 0.1,
            strategy: "p -> w -> p".to_string(),
        }
    }
}

/// BetterTogether optimizer.
///
/// Composes prompt optimization (BootstrapFewShotWithRandomSearch) and
/// weight optimization (BootstrapFinetune) according to a strategy string.
pub struct BetterTogether {
    config: BetterTogetherConfig,
}

impl BetterTogether {
    pub fn new(config: BetterTogetherConfig) -> Self {
        Self { config }
    }

    /// Parse and validate the strategy string.
    fn parse_strategy(strategy: &str) -> dspy_core::Result<Vec<String>> {
        let steps: Vec<String> = strategy
            .to_lowercase()
            .split(STRAT_SEP)
            .map(|s| s.trim().to_string())
            .collect();

        for step in &steps {
            if step != "p" && step != "w" {
                return Err(dspy_core::DspyError::OptimizationError(format!(
                    "Strategy should be a sequence of 'p' and 'w' separated by '{}', but found: {}",
                    STRAT_SEP, strategy
                )));
            }
        }

        Ok(steps)
    }

    /// Compile: execute the strategy sequence on the student program.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
    ) -> dspy_core::Result<Box<dyn Module>> {
        let steps = Self::parse_strategy(&self.config.strategy)?;

        let mut trainset_copy = trainset.to_vec();
        let mut current = student.deep_copy();

        // Seeded shuffle
        let mut rng_state: u32 = self.config.seed;

        for (_ind, step_code) in steps.iter().enumerate() {
            // Shuffle trainset with LCG
            for i in (1..trainset_copy.len()).rev() {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                let j = (rng_state as usize) % (i + 1);
                trainset_copy.swap(i, j);
            }

            // Deep copy student (reset compiled flag)
            current = current.deep_copy();

            if step_code == "p" {
                current = self
                    .compile_prompt_optimizer(current.as_ref(), &trainset_copy)
                    .await?;
            } else if step_code == "w" {
                current = self
                    .compile_weight_optimizer(current.as_ref(), &trainset_copy)
                    .await?;
            }
        }

        Ok(current)
    }

    /// Run prompt optimization (BootstrapFewShotWithRandomSearch).
    async fn compile_prompt_optimizer(
        &self,
        student: &dyn Module,
        trainset: &[Example],
    ) -> dspy_core::Result<Box<dyn Module>> {
        // Strip "hint" from input keys for prompt optimization (Python: set(x.inputs().keys()) - {"hint"})
        let cleaned_trainset: Vec<Example> = trainset
            .iter()
            .map(|ex| {
                let input_keys: Vec<String> = ex
                    .inputs()
                    .keys()
                    .filter(|k| k.as_str() != "hint")
                    .cloned()
                    .collect();
                let key_refs: Vec<&str> = input_keys.iter().map(|s| s.as_str()).collect();
                ex.clone().with_inputs(&key_refs)
            })
            .collect();

        let num_val = (self.config.valset_ratio * cleaned_trainset.len() as f64).floor() as usize;
        let valset = &cleaned_trainset[..num_val];
        let prompt_trainset = &cleaned_trainset[num_val..];

        let prompt_optimizer = BootstrapFewShotWithRandomSearch::new(RandomSearchConfig::new(
            self.config.metric.clone(),
        ));

        // Save predictor LMs before prompt optimization (BFRS may reset them)
        let pred_lms: Vec<_> = student
            .named_predictors()
            .iter()
            .map(|(_, pred)| pred.lm())
            .collect();

        let mut result = prompt_optimizer
            .compile(student, prompt_trainset, None, Some(valset))
            .await?;

        // Restore LMs (Python: for pred, lm in zip(student.predictors(), pred_lms): pred.lm = lm)
        let result_predictors = result.named_predictors_mut();
        for (i, (_, pred)) in result_predictors.into_iter().enumerate() {
            if let Some(Some(lm)) = pred_lms.get(i) {
                pred.set_lm(lm.clone());
            }
        }

        Ok(result)
    }

    /// Run weight optimization (BootstrapFinetune).
    async fn compile_weight_optimizer(
        &self,
        student: &dyn Module,
        trainset: &[Example],
    ) -> dspy_core::Result<Box<dyn Module>> {
        let weight_optimizer = BootstrapFinetune::new(BootstrapFinetuneConfig {
            metric: Some(self.config.metric.clone()),
            ..Default::default()
        });

        weight_optimizer.compile(student, trainset, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{
        Example, LM, LMConfig, LMResponse, Message, Predict, Prediction, Signature,
    };
    use async_trait::async_trait;
    use std::sync::Arc;

    struct MockLM {
        config: LMConfig,
    }

    impl MockLM {
        fn new() -> Self {
            Self {
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse {
                text: "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]".to_string(),
                usage: None,
            }])
        }
        fn model(&self) -> &str {
            "mock"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({"model": "mock"})
        }
    }

    struct SimpleQA {
        predict: Predict,
    }

    impl SimpleQA {
        fn new() -> Self {
            let mut predict =
                Predict::new(Signature::from_string("question -> answer").unwrap());
            predict.set_lm(Arc::new(MockLM::new()));
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for SimpleQA {
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
            Box::new(SimpleQA::new())
        }
    }

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .field("answer", "42")
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[test]
    fn test_parse_strategy_valid() {
        let steps = BetterTogether::parse_strategy("p -> w -> p").unwrap();
        assert_eq!(steps, vec!["p", "w", "p"]);
    }

    #[test]
    fn test_parse_strategy_single() {
        let steps = BetterTogether::parse_strategy("p").unwrap();
        assert_eq!(steps, vec!["p"]);
    }

    #[test]
    fn test_parse_strategy_invalid() {
        let result = BetterTogether::parse_strategy("p -> x -> w");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_better_together_compile() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(10);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let bt = BetterTogether::new(BetterTogetherConfig {
            metric,
            strategy: "p -> w".to_string(),
            valset_ratio: 0.2,
            seed: 42,
        });

        let result = bt.compile(&student, &trainset).await;
        assert!(result.is_ok());
    }
}
