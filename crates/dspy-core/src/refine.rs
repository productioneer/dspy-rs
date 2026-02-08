//! Refine — like BestOfN but with LLM-generated feedback between attempts.
//! Python equivalent: dspy/predict/refine.py
//!
//! On each failed attempt, Refine uses an OfferFeedback signature to analyze
//! what went wrong and inject hints into the next attempt.

use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::module_trait::Module;
use crate::prediction::Prediction;
use async_trait::async_trait;

/// Reward function signature: takes (inputs, prediction) → scalar reward.
pub type RewardFn = Box<dyn Fn(&Example, &Prediction) -> f64 + Send + Sync>;

pub struct Refine {
    module: Box<dyn Module>,
    n: usize,
    reward_fn: RewardFn,
    threshold: f64,
    fail_count: usize,
}

impl Refine {
    pub fn new(
        module: Box<dyn Module>,
        n: usize,
        reward_fn: RewardFn,
        threshold: f64,
        fail_count: Option<usize>,
    ) -> Self {
        Self {
            module,
            n,
            reward_fn,
            threshold,
            fail_count: fail_count.unwrap_or(n),
        }
    }
}

#[async_trait]
impl Module for Refine {
    fn module_type_name(&self) -> &str {
        "Refine"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        let mut best_pred: Option<Prediction> = None;
        let mut best_reward = f64::NEG_INFINITY;
        let mut fails_remaining = self.fail_count;

        // Note: In the full Python version, Refine generates OfferFeedback
        // between attempts to inject hints. For the port, we implement the
        // core retry-with-best-selection logic. Full feedback generation
        // requires an LM call via the OfferFeedback signature, which will
        // be wired up when the adapter system supports dynamic hint injection.

        for idx in 0..self.n {
            match self.module.call(args).await {
                Ok(pred) => {
                    let reward = (self.reward_fn)(args, &pred);

                    if reward > best_reward {
                        best_reward = reward;
                        best_pred = Some(pred);
                    }

                    if reward >= self.threshold {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("Refine: Attempt {} failed: {}", idx + 1, e);
                    if fails_remaining == 0 {
                        return Err(e);
                    }
                    fails_remaining -= 1;
                }
            }
        }

        best_pred.ok_or_else(|| DspyError::Other("Refine: All attempts failed".into()))
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(Self {
            module: self.module.deep_copy(),
            n: self.n,
            reward_fn: Box::new(|_, _| 0.0), // reward_fn can't be cloned
            threshold: self.threshold,
            fail_count: self.fail_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::Example;
    use crate::prediction::Prediction;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    struct CountingModule {
        answers: Vec<String>,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Module for CountingModule {
        async fn forward(&self, _args: &Example) -> Result<Prediction> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let answer = &self.answers[idx % self.answers.len()];
            let mut pred = Prediction::new(HashMap::new());
            pred.example.set("answer", answer.as_str());
            Ok(pred)
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(CountingModule {
                answers: self.answers.clone(),
                call_count: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    struct FailingModule {
        fail_until: usize,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Module for FailingModule {
        async fn forward(&self, _args: &Example) -> Result<Prediction> {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst) + 1;
            if count <= self.fail_until {
                return Err(DspyError::Other(format!("Failing on attempt {}", count)));
            }
            let mut pred = Prediction::new(HashMap::new());
            pred.example.set("answer", "success");
            Ok(pred)
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(FailingModule {
                fail_until: self.fail_until,
                call_count: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    #[tokio::test]
    async fn test_refine_picks_best() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let module = Box::new(CountingModule {
            answers: vec!["bad".into(), "good".into(), "ok".into()],
            call_count: call_count.clone(),
        });

        let reward_fn: RewardFn = Box::new(|_, pred| match pred.get_str("answer") {
            Some("good") => 1.0,
            Some("ok") => 0.5,
            _ => 0.0,
        });

        let refine = Refine::new(module, 3, reward_fn, 2.0, None); // unreachable threshold
        let result = refine.forward(&Example::new()).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("good"));
    }

    #[tokio::test]
    async fn test_refine_stops_at_threshold() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let module = Box::new(CountingModule {
            answers: vec!["ok".into(), "great".into()],
            call_count: call_count.clone(),
        });

        let reward_fn: RewardFn = Box::new(|_, pred| {
            if pred.get_str("answer") == Some("great") {
                1.0
            } else {
                0.3
            }
        });

        let refine = Refine::new(module, 10, reward_fn, 0.9, None);
        let result = refine.forward(&Example::new()).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("great"));
        // Should have stopped at attempt 2
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_refine_handles_failures() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let module = Box::new(FailingModule {
            fail_until: 2,
            call_count: call_count.clone(),
        });

        let reward_fn: RewardFn = Box::new(|_, _| 1.0);
        let refine = Refine::new(module, 5, reward_fn, 0.5, None);
        let result = refine.forward(&Example::new()).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("success"));
    }

    #[tokio::test]
    async fn test_refine_all_fail() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let module = Box::new(FailingModule {
            fail_until: 100,
            call_count: call_count.clone(),
        });

        let reward_fn: RewardFn = Box::new(|_, _| 1.0);
        let refine = Refine::new(module, 3, reward_fn, 0.5, Some(3));
        let result = refine.forward(&Example::new()).await;
        assert!(result.is_err());
    }
}
