//! Evaluate — parallel evaluation of Module on dataset with metric.
//! Python equivalent: dspy/evaluate/evaluate.py

use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::module_trait::Module;
use crate::prediction::Prediction;
use std::sync::Arc;

pub type Metric = Arc<dyn Fn(&Example, &Prediction) -> f64 + Send + Sync>;

pub struct EvaluateConfig {
    pub num_threads: usize,
    pub display_progress: bool,
    pub failure_score: f64,
    pub max_errors: usize,
}

impl Default for EvaluateConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            display_progress: false,
            failure_score: 0.0,
            max_errors: 5,
        }
    }
}

pub struct EvaluationResult {
    pub score: f64,
    pub results: Vec<(Example, Prediction, f64)>,
    pub errors: usize,
}

pub struct Evaluate {
    devset: Vec<Example>,
    metric: Metric,
    config: EvaluateConfig,
}

impl Evaluate {
    pub fn new(devset: Vec<Example>, metric: Metric, config: EvaluateConfig) -> Self {
        Self {
            devset,
            metric,
            config,
        }
    }

    /// Run evaluation on the given module, returning average score and per-example results
    pub async fn run(&self, program: &dyn Module) -> Result<EvaluationResult> {
        let mut results: Vec<(Example, Prediction, f64)> = Vec::new();
        let mut errors = 0usize;
        let mut total_score = 0.0;

        // Evaluate sequentially for simplicity (concurrent version would use tokio::spawn)
        // In the future, we can use semaphore-based concurrency like the TS version
        for example in &self.devset {
            let inputs = example.inputs();

            match program.forward(&inputs).await {
                Ok(prediction) => {
                    let score = (self.metric)(example, &prediction);
                    total_score += score;
                    results.push((example.clone(), prediction, score));
                }
                Err(_e) => {
                    errors += 1;
                    if errors >= self.config.max_errors {
                        return Err(DspyError::EvaluationError(format!(
                            "Too many errors during evaluation: {errors}"
                        )));
                    }
                    // Create a failed result with failure score
                    let empty_pred = Prediction::new(std::collections::HashMap::new());
                    results.push((example.clone(), empty_pred, self.config.failure_score));
                    total_score += self.config.failure_score;
                }
            }
        }

        let count = self.devset.len() as f64;
        // Score as percentage (0-100) matching Python DSPy and TS port
        let avg_score = if count > 0.0 {
            (total_score / count) * 100.0
        } else {
            0.0
        };

        Ok(EvaluationResult {
            score: avg_score,
            results,
            errors,
        })
    }

    pub fn devset(&self) -> &[Example] {
        &self.devset
    }

    pub fn metric(&self) -> &Metric {
        &self.metric
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message, LM};
    use crate::predict::Predict;
    use crate::settings;
    use crate::signature::Signature;
    use async_trait::async_trait;

    struct FixedLM {
        answer: String,
        config: LMConfig,
    }

    impl FixedLM {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                config: LMConfig::new("fixed"),
            }
        }
    }

    #[async_trait]
    impl LM for FixedLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> Result<Vec<LMResponse>> {
            Ok(vec![LMResponse {
                text: format!("[[ ## answer ## ]]\n{}", self.answer),
                usage: None,
            }])
        }
        fn model(&self) -> &str {
            "fixed"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    #[tokio::test]
    async fn test_evaluate_basic() {
        settings::reset_settings();
        let sig = Signature::from_string("question -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(Arc::new(FixedLM::new("42")));

        let devset = vec![
            Example::new()
                .field("question", "What?")
                .field("answer", "42")
                .with_inputs(&["question"]),
            Example::new()
                .field("question", "Why?")
                .field("answer", "42")
                .with_inputs(&["question"]),
        ];

        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let eval = Evaluate::new(devset, metric, EvaluateConfig::default());
        let result = eval.run(&predict).await.unwrap();
        assert_eq!(result.score, 100.0);
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.errors, 0);
    }

    #[tokio::test]
    async fn test_evaluate_partial_match() {
        settings::reset_settings();
        let sig = Signature::from_string("q -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(Arc::new(FixedLM::new("yes")));

        let devset = vec![
            Example::new()
                .field("q", "Q1")
                .field("answer", "yes")
                .with_inputs(&["q"]),
            Example::new()
                .field("q", "Q2")
                .field("answer", "no")
                .with_inputs(&["q"]),
        ];

        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let eval = Evaluate::new(devset, metric, EvaluateConfig::default());
        let result = eval.run(&predict).await.unwrap();
        assert_eq!(result.score, 50.0); // 1 correct out of 2
    }

    #[tokio::test]
    async fn test_evaluate_empty_devset() {
        settings::reset_settings();
        let sig = Signature::from_string("q -> a").unwrap();
        let predict = Predict::new(sig);
        let metric: Metric = Arc::new(|_, _| 1.0);
        let eval = Evaluate::new(vec![], metric, EvaluateConfig::default());
        let result = eval.run(&predict).await.unwrap();
        assert_eq!(result.score, 0.0);
        assert!(result.results.is_empty());
    }

    #[tokio::test]
    async fn test_evaluate_with_errors() {
        settings::reset_settings();
        // No LM configured -> forward will fail
        let sig = Signature::from_string("q -> a").unwrap();
        let predict = Predict::new(sig);

        let devset = vec![
            Example::new().field("q", "Q1").with_inputs(&["q"]),
        ];

        let metric: Metric = Arc::new(|_, _| 1.0);
        let eval = Evaluate::new(
            devset,
            metric,
            EvaluateConfig {
                max_errors: 10,
                ..Default::default()
            },
        );
        let result = eval.run(&predict).await.unwrap();
        assert_eq!(result.errors, 1);
        assert_eq!(result.score, 0.0); // failure_score
    }
}
