//! BootstrapFewShotWithOptuna: demo selection using TPE Bayesian optimization.
//! Uses the TPE sampler for categorical parameter optimization over demo indices.
//! Python equivalent: dspy/teleprompt/teleprompt_optuna.py

use dspy_core::{
    Evaluate, EvaluateConfig, Example, Metric, Module, Predict,
};
use dspy_tpe::{Direction, Study, TPESampler};
use std::collections::HashMap;

use crate::bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};

/// Configuration for BootstrapFewShotWithOptuna.
pub struct BootstrapFewShotWithOptunaConfig {
    pub metric: Metric,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub num_candidate_programs: usize,
    pub num_threads: usize,
}

impl BootstrapFewShotWithOptunaConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            num_candidate_programs: 16,
            num_threads: 2,
        }
    }
}

/// BootstrapFewShotWithOptuna optimizer.
///
/// 1. Bootstraps demos using BootstrapFewShot
/// 2. Uses TPE to find optimal demo selection for each predictor
/// 3. Evaluates candidates and returns the best
pub struct BootstrapFewShotWithOptuna {
    config: BootstrapFewShotWithOptunaConfig,
}

impl BootstrapFewShotWithOptuna {
    pub fn new(config: BootstrapFewShotWithOptunaConfig) -> Self {
        Self { config }
    }

    /// Compile: use TPE to optimize demo selection.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
        valset: Option<&[Example]>,
        max_demos: usize,
    ) -> dspy_core::Result<Box<dyn Module>> {
        let eval_set = valset.unwrap_or(trainset);

        // Bootstrap demos using BootstrapFewShot
        let bootstrap = BootstrapFewShot::new(BootstrapFewShotConfig {
            metric: self.config.metric.clone(),
            max_bootstrapped_demos: max_demos,
            max_labeled_demos: self.config.max_labeled_demos,
            max_rounds: self.config.max_rounds,
            ..BootstrapFewShotConfig::new(self.config.metric.clone())
        });

        let compiled = bootstrap.compile(student, trainset, teacher).await?;

        // Collect demos per predictor from compiled program
        let compiled_predictors: Vec<(&str, &Predict)> = compiled.named_predictors();
        let demos_per_pred: Vec<(&str, Vec<Example>)> = compiled_predictors
            .iter()
            .map(|(name, pred)| (*name, pred.demos.clone()))
            .collect();

        // Use TPE to optimize demo selection
        let sampler = TPESampler::new(42).with_n_startup_trials(5);
        let mut study = Study::new(Direction::Maximize, sampler);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_program: Box<dyn Module> = student.deep_copy();

        for _trial_idx in 0..self.config.num_candidate_programs {
            let mut candidate = student.deep_copy();
            let candidate_predictors = candidate.named_predictors_mut();

            // Suggest demo indices from TPE for each predictor
            let mut trial_params = HashMap::new();

            for (i, (name, pred)) in candidate_predictors.into_iter().enumerate() {
                let all_demos = &demos_per_pred[i].1;
                if all_demos.is_empty() {
                    continue;
                }

                let param_name = format!("demo_index_for_{}", name);
                let demo_index = study.suggest_categorical(&param_name, all_demos.len());
                trial_params.insert(param_name, demo_index);

                // Assign selected demo
                pred.demos = vec![all_demos[demo_index].clone()];
            }

            // Evaluate candidate
            let eval = Evaluate::new(
                eval_set.to_vec(),
                self.config.metric.clone(),
                EvaluateConfig {
                    num_threads: self.config.num_threads,
                    display_progress: false,
                    ..Default::default()
                },
            );

            let eval_result = eval.run(candidate.as_ref()).await?;
            let score = eval_result.score;

            // Record trial in TPE study
            study.record_trial(trial_params, score);

            if score > best_score {
                best_score = score;
                best_program = candidate;
            }
        }

        Ok(best_program)
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
    fn test_optuna_config_defaults() {
        let metric: Metric = Arc::new(|_, _| 1.0);
        let config = BootstrapFewShotWithOptunaConfig::new(metric);
        assert_eq!(config.max_bootstrapped_demos, 4);
        assert_eq!(config.max_labeled_demos, 16);
        assert_eq!(config.max_rounds, 1);
        assert_eq!(config.num_candidate_programs, 16);
        assert_eq!(config.num_threads, 2);
    }

    #[tokio::test]
    async fn test_optuna_compile_basic() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(5);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let optuna = BootstrapFewShotWithOptuna::new(BootstrapFewShotWithOptunaConfig {
            metric: metric.clone(),
            num_candidate_programs: 3,
            num_threads: 1,
            ..BootstrapFewShotWithOptunaConfig::new(metric)
        });

        let result = optuna
            .compile(&student, &trainset, None, None, 4)
            .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_optuna_compile_with_valset() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(8);
        let valset = make_trainset(3);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let optuna = BootstrapFewShotWithOptuna::new(BootstrapFewShotWithOptunaConfig {
            metric: metric.clone(),
            num_candidate_programs: 2,
            num_threads: 1,
            ..BootstrapFewShotWithOptunaConfig::new(metric)
        });

        let result = optuna
            .compile(&student, &trainset, None, Some(&valset), 4)
            .await;
        assert!(result.is_ok());
    }
}
