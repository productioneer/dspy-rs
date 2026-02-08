//! LabeledFewShot — simplest optimizer: assigns k random labeled examples as demos.
//! Uses a single seeded RNG (seed=0) shared across predictors so each predictor
//! gets a different random sample (matching Python DSPy behavior).
//! Python equivalent: dspy/teleprompt/vanilla.py

use dspy_core::{Example, Module};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct LabeledFewShot {
    k: usize,
}

impl LabeledFewShot {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compile: deep copies student, randomly samples k demos from trainset,
    /// assigns different samples to each predictor using a shared RNG.
    pub fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        sample: bool,
    ) -> Box<dyn Module> {
        let mut compiled = student.deep_copy();

        if trainset.is_empty() {
            return compiled;
        }

        let count = self.k.min(trainset.len());

        if sample {
            // Single RNG shared across all predictors (Python: random.Random(0))
            // Each predictor consumes from the same RNG, so they get different samples
            let mut rng = StdRng::seed_from_u64(0);

            for (_, pred) in compiled.named_predictors_mut() {
                let mut indices: Vec<usize> = (0..trainset.len()).collect();
                // Partial shuffle using gen_range for correct uniform distribution
                for i in 0..count {
                    let j = rng.gen_range(i..indices.len());
                    indices.swap(i, j);
                }
                pred.demos = indices[..count]
                    .iter()
                    .map(|&i| trainset[i].clone())
                    .collect();
            }
        } else {
            // No sampling: first k examples in order
            let demos: Vec<Example> = trainset[..count].to_vec();
            for (_, pred) in compiled.named_predictors_mut() {
                pred.demos = demos.clone();
            }
        }

        compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{Example, Predict, Signature};

    // Multi-predictor module for testing per-predictor sampling
    struct TwoPredsModule {
        pred1: Predict,
        pred2: Predict,
    }

    impl TwoPredsModule {
        fn new(sig: &str) -> Self {
            Self {
                pred1: Predict::new(Signature::from_string(sig).unwrap()),
                pred2: Predict::new(Signature::from_string(sig).unwrap()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Module for TwoPredsModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<dspy_core::Prediction> {
            self.pred1.forward(args).await
        }

        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("pred1", &self.pred1), ("pred2", &self.pred2)]
        }

        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("pred1", &mut self.pred1), ("pred2", &mut self.pred2)]
        }

        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(Self {
                pred1: self.pred1.clone(),
                pred2: self.pred2.clone(),
            })
        }
    }

    // Simple Module wrapper for testing
    struct SimpleModule {
        predict: Predict,
    }

    impl SimpleModule {
        fn new(sig: &str) -> Self {
            Self {
                predict: Predict::new(Signature::from_string(sig).unwrap()),
            }
        }
    }

    #[async_trait::async_trait]
    impl Module for SimpleModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<dspy_core::Prediction> {
            self.predict.forward(args).await
        }

        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }

        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }

        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(Self {
                predict: self.predict.clone(),
            })
        }
    }

    #[test]
    fn test_labeled_few_shot_basic() {
        let student = SimpleModule::new("question -> answer");
        let trainset: Vec<Example> = (0..10)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .field("answer", format!("A{i}"))
                    .with_inputs(&["question"])
            })
            .collect();

        let optimizer = LabeledFewShot::new(3);
        let compiled = optimizer.compile(&student, &trainset, true);

        let preds = compiled.named_predictors();
        assert_eq!(preds.len(), 1);
        // The predict module should have exactly 3 demos
        assert_eq!(preds[0].1.demos.len(), 3);
    }

    #[test]
    fn test_labeled_few_shot_no_sample() {
        let student = SimpleModule::new("q -> a");
        let trainset: Vec<Example> = (0..5)
            .map(|i| {
                Example::new()
                    .field("q", format!("Q{i}"))
                    .field("a", format!("A{i}"))
                    .with_inputs(&["q"])
            })
            .collect();

        let optimizer = LabeledFewShot::new(3);
        let compiled = optimizer.compile(&student, &trainset, false);

        let preds = compiled.named_predictors();
        // Without sampling, should be first 3
        assert_eq!(preds[0].1.demos.len(), 3);
        assert_eq!(preds[0].1.demos[0].get_str("q"), Some("Q0"));
        assert_eq!(preds[0].1.demos[1].get_str("q"), Some("Q1"));
        assert_eq!(preds[0].1.demos[2].get_str("q"), Some("Q2"));
    }

    #[test]
    fn test_labeled_few_shot_k_larger_than_trainset() {
        let student = SimpleModule::new("q -> a");
        let trainset = vec![
            Example::new()
                .field("q", "Q1")
                .field("a", "A1")
                .with_inputs(&["q"]),
            Example::new()
                .field("q", "Q2")
                .field("a", "A2")
                .with_inputs(&["q"]),
        ];

        let optimizer = LabeledFewShot::new(10);
        let compiled = optimizer.compile(&student, &trainset, true);

        let preds = compiled.named_predictors();
        // Should be capped at trainset length
        assert_eq!(preds[0].1.demos.len(), 2);
    }

    #[test]
    fn test_labeled_few_shot_empty_trainset() {
        let student = SimpleModule::new("q -> a");
        let optimizer = LabeledFewShot::new(5);
        let compiled = optimizer.compile(&student, &[], true);

        let preds = compiled.named_predictors();
        assert_eq!(preds[0].1.demos.len(), 0);
    }

    #[test]
    fn test_labeled_few_shot_deterministic() {
        let student = SimpleModule::new("q -> a");
        let trainset: Vec<Example> = (0..20)
            .map(|i| {
                Example::new()
                    .field("q", format!("Q{i}"))
                    .with_inputs(&["q"])
            })
            .collect();

        let optimizer = LabeledFewShot::new(5);
        let compiled1 = optimizer.compile(&student, &trainset, true);
        let compiled2 = optimizer.compile(&student, &trainset, true);

        let demos1 = &compiled1.named_predictors()[0].1.demos;
        let demos2 = &compiled2.named_predictors()[0].1.demos;

        // Same seed → same sample
        for (d1, d2) in demos1.iter().zip(demos2.iter()) {
            assert_eq!(d1.get_str("q"), d2.get_str("q"));
        }
    }

    #[test]
    fn test_labeled_few_shot_does_not_modify_original() {
        let student = SimpleModule::new("q -> a");
        let trainset = vec![Example::new().field("q", "Q1").with_inputs(&["q"])];

        let optimizer = LabeledFewShot::new(1);
        let _compiled = optimizer.compile(&student, &trainset, true);

        // Original student should be unmodified
        assert_eq!(student.named_predictors()[0].1.demos.len(), 0);
    }

    #[test]
    fn test_labeled_few_shot_per_predictor_sampling() {
        // With 2 predictors and a large enough trainset, each predictor should
        // get different demos because the shared RNG is consumed sequentially.
        let student = TwoPredsModule::new("q -> a");
        let trainset: Vec<Example> = (0..20)
            .map(|i| {
                Example::new()
                    .field("q", format!("Q{i}"))
                    .field("a", format!("A{i}"))
                    .with_inputs(&["q"])
            })
            .collect();

        let optimizer = LabeledFewShot::new(3);
        let compiled = optimizer.compile(&student, &trainset, true);

        let preds = compiled.named_predictors();
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].1.demos.len(), 3);
        assert_eq!(preds[1].1.demos.len(), 3);

        // Different predictors should get different demo sets
        let demos1: Vec<String> = preds[0]
            .1
            .demos
            .iter()
            .filter_map(|d| d.get_str("q").map(String::from))
            .collect();
        let demos2: Vec<String> = preds[1]
            .1
            .demos
            .iter()
            .filter_map(|d| d.get_str("q").map(String::from))
            .collect();
        assert_ne!(
            demos1, demos2,
            "Different predictors should get different demos"
        );
    }
}
