//! BootstrapFewShotWithRandomSearch — multiple bootstrap rounds with varied configs,
//! evaluates each candidate, returns best.
//! Python equivalent: dspy/teleprompt/random_search.py

use dspy_core::{Evaluate, EvaluateConfig, Example, Metric, Module};

use crate::bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};
use crate::labeled_few_shot::LabeledFewShot;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;

pub struct RandomSearchConfig {
    pub metric: Metric,
    pub num_candidate_programs: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub max_errors: usize,
    pub stop_at_score: Option<f64>,
    pub metric_threshold: Option<f64>,
}

impl RandomSearchConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            num_candidate_programs: 16,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            max_errors: 5,
            stop_at_score: None,
            metric_threshold: None,
        }
    }
}

pub struct BootstrapFewShotWithRandomSearch {
    config: RandomSearchConfig,
}

impl BootstrapFewShotWithRandomSearch {
    pub fn new(config: RandomSearchConfig) -> Self {
        Self { config }
    }

    /// Compile: generate multiple candidate programs via bootstrap with different
    /// random seeds/configs, evaluate each, return the best.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
        valset: Option<&[Example]>,
    ) -> dspy_core::Result<Box<dyn Module>> {
        let eval_set = valset.unwrap_or(trainset);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_program: Option<Box<dyn Module>> = None;
        let mut scores: Vec<f64> = Vec::new();

        // Iterate through candidate seeds:
        // seed -3: zero-shot (no demos)
        // seed -2: labeled-only
        // seed -1: unshuffled bootstrap
        // seed 0..N: random-shuffled bootstrap with varying sizes
        for seed in -3..self.config.num_candidate_programs as i64 {
            let program: Box<dyn Module> = if seed == -3 {
                // Zero-shot baseline
                student.deep_copy()
            } else if seed == -2 {
                // Labels only
                let labeled = LabeledFewShot::new(self.config.max_labeled_demos);
                labeled.compile(student, trainset, true)
            } else if seed == -1 {
                // Unshuffled bootstrap
                let bootstrap = BootstrapFewShot::new(BootstrapFewShotConfig {
                    metric: self.config.metric.clone(),
                    metric_threshold: self.config.metric_threshold,
                    max_bootstrapped_demos: self.config.max_bootstrapped_demos,
                    max_labeled_demos: self.config.max_labeled_demos,
                    max_rounds: self.config.max_rounds,
                    max_errors: self.config.max_errors,
                });
                bootstrap.compile(student, trainset, teacher).await?
            } else {
                // Two separate RNG instances, both seeded with the same value.
                // RNG 1: for shuffling the trainset
                // RNG 2: for selecting bootstrap size (first draw from fresh RNG)
                // This matches Python DSPy where random.Random(seed).shuffle() and
                // random.Random(seed).randint() are independent operations.
                let seed = seed as u64;

                let mut shuffle_rng = StdRng::seed_from_u64(seed);
                let mut shuffled = trainset.to_vec();
                shuffled.shuffle(&mut shuffle_rng);

                let mut size_rng = StdRng::seed_from_u64(seed);
                let size = size_rng.gen_range(1..=self.config.max_bootstrapped_demos);

                let bootstrap = BootstrapFewShot::new(BootstrapFewShotConfig {
                    metric: self.config.metric.clone(),
                    metric_threshold: self.config.metric_threshold,
                    max_bootstrapped_demos: size,
                    max_labeled_demos: self.config.max_labeled_demos,
                    max_rounds: self.config.max_rounds,
                    max_errors: self.config.max_errors,
                });
                bootstrap.compile(student, &shuffled, teacher).await?
            };

            // Evaluate candidate
            let evaluator = Evaluate::new(
                eval_set.to_vec(),
                self.config.metric.clone(),
                EvaluateConfig {
                    max_errors: self.config.max_errors,
                    ..Default::default()
                },
            );

            let result = evaluator.run(program.as_ref()).await?;
            let score = result.score;
            scores.push(score);

            if score > best_score {
                best_score = score;
                best_program = Some(program);
            }

            // Early stopping
            if let Some(stop_score) = self.config.stop_at_score {
                if score >= stop_score {
                    break;
                }
            }
        }

        best_program.ok_or_else(|| {
            dspy_core::DspyError::OptimizationError("No candidate programs generated".into())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{Example, LMConfig, LMResponse, Message, Predict, Prediction, Signature, LM};
    use std::sync::Arc;

    struct EchoLM {
        config: LMConfig,
    }

    impl EchoLM {
        fn new() -> Self {
            Self {
                config: LMConfig::new("echo"),
            }
        }
    }

    #[async_trait]
    impl LM for EchoLM {
        async fn call(
            &self,
            messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            // Echo back last user message content as the answer
            let text = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(vec![LMResponse::new(format!("[[ ## answer ## ]]\n{text}"), None)])
        }
        fn model(&self) -> &str {
            "echo"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    struct TestModule {
        predict: Predict,
    }

    impl TestModule {
        fn new(lm: Arc<dyn LM>) -> Self {
            let mut predict = Predict::new(Signature::from_string("question -> answer").unwrap());
            predict.set_lm(lm);
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for TestModule {
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
            Box::new(TestModule {
                predict: self.predict.clone(),
            })
        }
    }

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .field("answer", format!("A{i}"))
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[tokio::test]
    async fn test_random_search_returns_best() {
        dspy_core::reset_settings();
        let lm = Arc::new(EchoLM::new());
        let student = TestModule::new(lm);

        // Metric: always returns 0.5 (all candidates score the same)
        let metric: Metric = Arc::new(|_, _| 0.5);

        let trainset = make_trainset(5);
        let optimizer = BootstrapFewShotWithRandomSearch::new(RandomSearchConfig {
            metric,
            num_candidate_programs: 2,
            max_bootstrapped_demos: 2,
            max_labeled_demos: 2,
            max_rounds: 1,
            max_errors: 10,
            stop_at_score: None,
            metric_threshold: None,
        });

        let compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();
        // Just verify it returns a program
        assert!(!compiled.named_predictors().is_empty());
    }

    #[tokio::test]
    async fn test_random_search_stop_at_score() {
        dspy_core::reset_settings();
        let lm = Arc::new(EchoLM::new());
        let student = TestModule::new(lm);

        let metric: Metric = Arc::new(|_, _| 1.0);

        let trainset = make_trainset(3);
        let optimizer = BootstrapFewShotWithRandomSearch::new(RandomSearchConfig {
            metric,
            num_candidate_programs: 100, // Would take forever without early stop
            max_bootstrapped_demos: 1,
            max_labeled_demos: 1,
            max_rounds: 1,
            max_errors: 10,
            stop_at_score: Some(0.5), // Should stop after first candidate
            metric_threshold: None,
        });

        let compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }

    #[tokio::test]
    async fn test_random_search_with_valset() {
        dspy_core::reset_settings();
        let lm = Arc::new(EchoLM::new());
        let student = TestModule::new(lm);

        let metric: Metric = Arc::new(|_, _| 0.8);

        let trainset = make_trainset(5);
        let valset = make_trainset(3);
        let optimizer = BootstrapFewShotWithRandomSearch::new(RandomSearchConfig {
            metric,
            num_candidate_programs: 2,
            max_bootstrapped_demos: 1,
            max_labeled_demos: 1,
            max_rounds: 1,
            max_errors: 10,
            stop_at_score: None,
            metric_threshold: None,
        });

        let compiled = optimizer
            .compile(&student, &trainset, None, Some(&valset))
            .await
            .unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }
}
