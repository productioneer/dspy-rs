//! BestOfN — runs a module up to N times, returns the best prediction.
//! Python equivalent: dspy/predict/best_of_n.py

use crate::error::Result;
use crate::example::Example;
use crate::module_trait::Module;
use crate::prediction::Prediction;
use async_trait::async_trait;

/// Reward function signature: takes (inputs, prediction) → scalar reward.
pub type RewardFn = Box<dyn Fn(&Example, &Prediction) -> f64 + Send + Sync>;

pub struct BestOfN {
    module: Box<dyn Module>,
    n: usize,
    reward_fn: RewardFn,
    threshold: f64,
    fail_count: usize,
}

impl BestOfN {
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
impl Module for BestOfN {
    fn module_type_name(&self) -> &str {
        "BestOfN"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        let mut best_pred: Option<Prediction> = None;
        let mut best_reward = f64::NEG_INFINITY;
        let mut fails_remaining = self.fail_count;

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
                    eprintln!("BestOfN: Attempt {} failed: {}", idx + 1, e);
                    if fails_remaining == 0 {
                        return Err(e);
                    }
                    fails_remaining -= 1;
                }
            }
        }

        best_pred
            .ok_or_else(|| crate::error::DspyError::Other("BestOfN: All attempts failed".into()))
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(Self {
            module: self.module.deep_copy(),
            n: self.n,
            reward_fn: Box::new(|_, _| 0.0), // reward_fn can't be cloned; placeholder
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

    struct ConstModule {
        answer: String,
    }

    #[async_trait]
    impl Module for ConstModule {
        async fn forward(&self, _args: &Example) -> Result<Prediction> {
            let mut pred = Prediction::new(HashMap::new());
            pred.example.set("answer", self.answer.as_str());
            Ok(pred)
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(ConstModule {
                answer: self.answer.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_best_of_n_returns_best() {
        let module = Box::new(ConstModule {
            answer: "42".into(),
        });
        let reward_fn: RewardFn = Box::new(|_, pred| {
            if pred.get_str("answer") == Some("42") {
                1.0
            } else {
                0.0
            }
        });

        let best = BestOfN::new(module, 3, reward_fn, 1.0, None);
        let result = best.forward(&Example::new()).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("42"));
    }

    #[tokio::test]
    async fn test_best_of_n_stops_at_threshold() {
        let module = Box::new(ConstModule {
            answer: "yes".into(),
        });
        let reward_fn: RewardFn = Box::new(|_, _| 0.8);
        let best = BestOfN::new(module, 10, reward_fn, 0.5, None);
        let result = best.forward(&Example::new()).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("yes"));
    }
}
