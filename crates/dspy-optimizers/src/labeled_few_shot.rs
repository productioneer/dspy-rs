//! LabeledFewShot — simplest optimizer: assigns k random labeled examples as demos.
//! Python equivalent: dspy/teleprompt/vanilla.py

use dspy_core::{Example, Module};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub struct LabeledFewShot {
    k: usize,
}

impl LabeledFewShot {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    /// Compile: deep copies student, randomly samples k demos from trainset,
    /// assigns them to every predictor.
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
        let demos: Vec<Example> = if sample {
            let mut rng = StdRng::seed_from_u64(0);
            let mut indices: Vec<usize> = (0..trainset.len()).collect();
            indices.shuffle(&mut rng);
            indices[..count].iter().map(|&i| trainset[i].clone()).collect()
        } else {
            trainset[..count].to_vec()
        };

        for (_, pred) in compiled.named_predictors_mut() {
            pred.demos = demos.clone();
        }

        compiled
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{Example, Predict, Signature};

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
            Example::new().field("q", "Q1").field("a", "A1").with_inputs(&["q"]),
            Example::new().field("q", "Q2").field("a", "A2").with_inputs(&["q"]),
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
        let trainset = vec![
            Example::new().field("q", "Q1").with_inputs(&["q"]),
        ];

        let optimizer = LabeledFewShot::new(1);
        let _compiled = optimizer.compile(&student, &trainset, true);

        // Original student should be unmodified
        assert_eq!(student.named_predictors()[0].1.demos.len(), 0);
    }
}
