//! BootstrapFewShot — generates synthetic demos by running teacher on trainset.
//! Successful traces become few-shot examples for the student.
//! Python equivalent: dspy/teleprompt/bootstrap.py

use dspy_core::{Example, Metric, Module};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::labeled_few_shot::LabeledFewShot;

pub struct BootstrapFewShotConfig {
    pub metric: Metric,
    pub metric_threshold: Option<f64>,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub max_errors: usize,
}

impl BootstrapFewShotConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            metric_threshold: None,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            max_errors: 5,
        }
    }
}

pub struct BootstrapFewShot {
    config: BootstrapFewShotConfig,
}

impl BootstrapFewShot {
    pub fn new(config: BootstrapFewShotConfig) -> Self {
        Self { config }
    }

    /// Compile: run teacher on trainset, collect successful traces, assign as demos.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
    ) -> dspy_core::Result<Box<dyn Module>> {
        let mut student_copy = student.deep_copy();

        // Prepare teacher: deep copy of provided teacher or student
        let mut teacher_prog = match teacher {
            Some(t) => t.deep_copy(),
            None => student.deep_copy(),
        };

        // If teacher is uncompiled and we have labeled demos, add them
        if self.config.max_labeled_demos > 0 {
            let labeled = LabeledFewShot::new(self.config.max_labeled_demos);
            teacher_prog = labeled.compile(teacher_prog.as_ref(), trainset, true);
        }

        // Enable train mode on teacher's predictors
        for (_, pred) in teacher_prog.named_predictors_mut() {
            pred.set_train(true);
        }

        // Bootstrap: collect traces per predictor name
        let mut name2traces: HashMap<String, Vec<Example>> = HashMap::new();
        for (name, _) in student_copy.named_predictors() {
            name2traces.insert(name.to_string(), Vec::new());
        }

        let mut bootstrapped_count = 0usize;
        let mut error_count = 0usize;

        for example in trainset {
            if bootstrapped_count >= self.config.max_bootstrapped_demos {
                break;
            }

            for _round in 0..self.config.max_rounds {
                match self.bootstrap_one_example(
                    teacher_prog.as_ref(),
                    example,
                    &self.config.metric,
                ) .await {
                    Ok(true) => {
                        // Collect traces from teacher predictors
                        let mut got_traces = false;
                        for (name, pred) in teacher_prog.named_predictors() {
                            let traces = pred.get_traces();
                            if !traces.is_empty() {
                                if let Some(trace_list) = name2traces.get_mut(name) {
                                    for trace in &traces {
                                        // Merge inputs + outputs into a demo Example
                                        let demo = self.trace_to_demo(trace);
                                        trace_list.push(demo);
                                        got_traces = true;
                                    }
                                }
                            }
                        }
                        if got_traces {
                            bootstrapped_count += 1;
                        }
                        // Clear traces for next example
                        for (_, pred) in teacher_prog.named_predictors() {
                            pred.clear_traces();
                        }
                        break; // Success on this example, move to next
                    }
                    Ok(false) => {
                        // Metric didn't pass; clear traces and try next round
                        for (_, pred) in teacher_prog.named_predictors() {
                            pred.clear_traces();
                        }
                    }
                    Err(_e) => {
                        error_count += 1;
                        // Clear traces after error
                        for (_, pred) in teacher_prog.named_predictors() {
                            pred.clear_traces();
                        }
                        if error_count >= self.config.max_errors {
                            return Err(dspy_core::DspyError::OptimizationError(
                                format!("Too many errors during bootstrap: {error_count}"),
                            ));
                        }
                    }
                }
            }
        }

        // Assign demos to student predictors:
        // Bootstrapped traces + remaining labeled demos
        let mut rng = StdRng::seed_from_u64(0);
        let unused_examples: Vec<&Example> = trainset.iter().collect();
        let mut raw_demos: Vec<Example> = unused_examples.iter().map(|e| (*e).clone()).collect();
        raw_demos.shuffle(&mut rng);

        for (name, pred) in student_copy.named_predictors_mut() {
            let bootstrapped = name2traces
                .get(name)
                .map(|t| t[..self.config.max_bootstrapped_demos.min(t.len())].to_vec())
                .unwrap_or_default();

            let remaining_budget = self.config.max_labeled_demos.saturating_sub(bootstrapped.len());
            let labeled_count = remaining_budget.min(raw_demos.len());
            let labeled: Vec<Example> = raw_demos[..labeled_count].to_vec();

            pred.demos = [bootstrapped, labeled].concat();
        }

        Ok(student_copy)
    }

    /// Run teacher on one example, return whether the metric passed.
    async fn bootstrap_one_example(
        &self,
        teacher: &dyn Module,
        example: &Example,
        metric: &Metric,
    ) -> dspy_core::Result<bool> {
        let inputs = example.inputs();
        let prediction = teacher.call(&inputs).await?;

        let score = metric(example, &prediction);
        let success = match self.config.metric_threshold {
            Some(threshold) => score >= threshold,
            None => score > 0.0,
        };

        Ok(success)
    }

    /// Convert a Trace into a demo Example by merging inputs and outputs.
    fn trace_to_demo(&self, trace: &dspy_core::Trace) -> Example {
        let mut demo = trace.inputs.clone();
        // Merge output fields from the prediction
        for key in trace.outputs.keys() {
            if let Some(val) = trace.outputs.get(key) {
                demo.set(key.clone(), val.clone());
            }
        }
        demo
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

    // Mock LM that returns predictable answers
    struct AnswerLM {
        answer: String,
        config: LMConfig,
    }

    impl AnswerLM {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for AnswerLM {
        async fn call(&self, _messages: &[Message], _config: &LMConfig) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse {
                text: format!("[[ ## answer ## ]]\n{}", self.answer),
                usage: None,
            }])
        }
        fn model(&self) -> &str { "mock" }
        fn config(&self) -> &LMConfig { &self.config }
        fn dump_state(&self) -> serde_json::Value { serde_json::json!({}) }
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
            Box::new(TestModule { predict: self.predict.clone() })
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
    async fn test_bootstrap_basic() {
        dspy_core::reset_settings();
        let lm = Arc::new(AnswerLM::new("A0"));
        let student = TestModule::new(lm);

        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got { 1.0 } else { 0.0 }
        });

        let trainset = make_trainset(5);
        let optimizer = BootstrapFewShot::new(BootstrapFewShotConfig {
            metric,
            max_bootstrapped_demos: 2,
            max_labeled_demos: 3,
            max_rounds: 1,
            max_errors: 5,
            metric_threshold: None,
        });

        let compiled = optimizer.compile(&student, &trainset, None).await.unwrap();
        let preds = compiled.named_predictors();
        // Should have some demos (bootstrapped + labeled)
        assert!(!preds[0].1.demos.is_empty());
    }

    #[tokio::test]
    async fn test_bootstrap_with_all_passing_metric() {
        dspy_core::reset_settings();
        let lm = Arc::new(AnswerLM::new("always_right"));
        let student = TestModule::new(lm);

        // Metric that always passes
        let metric: Metric = Arc::new(|_, _| 1.0);

        let trainset = make_trainset(10);
        let optimizer = BootstrapFewShot::new(BootstrapFewShotConfig {
            metric,
            max_bootstrapped_demos: 3,
            max_labeled_demos: 2,
            max_rounds: 1,
            max_errors: 5,
            metric_threshold: None,
        });

        let compiled = optimizer.compile(&student, &trainset, None).await.unwrap();
        let preds = compiled.named_predictors();
        // Should have bootstrapped demos + labeled demos
        assert!(preds[0].1.demos.len() <= 5); // max 3 bootstrapped + 2 labeled
    }

    #[tokio::test]
    async fn test_bootstrap_with_all_failing_metric() {
        dspy_core::reset_settings();
        let lm = Arc::new(AnswerLM::new("wrong"));
        let student = TestModule::new(lm);

        // Metric that always fails
        let metric: Metric = Arc::new(|_, _| 0.0);

        let trainset = make_trainset(3);
        let optimizer = BootstrapFewShot::new(BootstrapFewShotConfig {
            metric,
            max_bootstrapped_demos: 2,
            max_labeled_demos: 2,
            max_rounds: 1,
            max_errors: 5,
            metric_threshold: None,
        });

        let compiled = optimizer.compile(&student, &trainset, None).await.unwrap();
        let preds = compiled.named_predictors();
        // No bootstrapped demos, only labeled
        assert!(preds[0].1.demos.len() <= 2);
    }

    #[tokio::test]
    async fn test_bootstrap_empty_trainset() {
        dspy_core::reset_settings();
        let lm = Arc::new(AnswerLM::new("test"));
        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|_, _| 1.0);

        let optimizer = BootstrapFewShot::new(BootstrapFewShotConfig::new(metric));
        let compiled = optimizer.compile(&student, &[], None).await.unwrap();
        let preds = compiled.named_predictors();
        assert!(preds[0].1.demos.is_empty());
    }

    #[tokio::test]
    async fn test_bootstrap_does_not_modify_original() {
        dspy_core::reset_settings();
        let lm = Arc::new(AnswerLM::new("test"));
        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|_, _| 1.0);

        let trainset = make_trainset(3);
        let optimizer = BootstrapFewShot::new(BootstrapFewShotConfig::new(metric));
        let _compiled = optimizer.compile(&student, &trainset, None).await.unwrap();

        // Original should be unmodified
        assert!(student.named_predictors()[0].1.demos.is_empty());
    }
}
