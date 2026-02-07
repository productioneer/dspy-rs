//! GRPO: Group Relative Policy Optimization optimizer.
//! Online RL training for LMs using bootstrapped rollouts.
//! Python equivalent: dspy/teleprompt/grpo.py
//!
//! NOTE: This is a structural port — trace collection, grouping, and reward computation work,
//! but actual ReinforceJob management, LM updates, and batch ID coordination are stubbed.
//! Full implementation requires a Provider/ReinforceJob backend that supports online RL training.

use dspy_core::{
    bootstrap_trace_data, BootstrapTraceOptions, Example,
    GRPOChatData, GRPOGroup, Metric, Module,
    TraceData, TrainingMessage,
};

/// Configuration for GRPO.
pub struct GRPOConfig {
    pub metric: Option<Metric>,
    pub multitask: bool,
    pub exclude_demos: bool,
    pub num_threads: usize,
    pub num_train_steps: usize,
    pub num_dspy_examples_per_step: usize,
    pub num_rollouts_per_step: usize,
    pub failure_score: f64,
    pub format_failure_score: f64,
    pub report_train_scores: bool,
    pub use_train_as_val: bool,
}

impl Default for GRPOConfig {
    fn default() -> Self {
        Self {
            metric: None,
            multitask: true,
            exclude_demos: true,
            num_threads: 6,
            num_train_steps: 100,
            num_dspy_examples_per_step: 16,
            num_rollouts_per_step: 8,
            failure_score: 0.0,
            format_failure_score: -1.0,
            report_train_scores: false,
            use_train_as_val: false,
        }
    }
}

impl GRPOConfig {
    pub fn validate(&self) -> dspy_core::Result<()> {
        if self.failure_score <= self.format_failure_score {
            return Err(dspy_core::DspyError::OptimizationError(
                "failureScore must be greater than formatFailureScore".to_string(),
            ));
        }
        if !self.exclude_demos {
            return Err(dspy_core::DspyError::OptimizationError(
                "excludeDemos==false is not supported yet. Please set it to true.".to_string(),
            ));
        }
        if !self.multitask {
            return Err(dspy_core::DspyError::OptimizationError(
                "Independent GRPO training jobs for each predictor is not supported yet.".to_string(),
            ));
        }
        if self.use_train_as_val && !self.report_train_scores {
            return Err(dspy_core::DspyError::OptimizationError(
                "If useTrainAsVal is true, reportTrainScores must be true.".to_string(),
            ));
        }
        Ok(())
    }
}

/// GRPO optimizer.
pub struct GRPO {
    config: GRPOConfig,
}

impl GRPO {
    pub fn new(config: GRPOConfig) -> dspy_core::Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Compile: run GRPO online training loop.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        _valset: Option<&[Example]>,
    ) -> dspy_core::Result<Box<dyn Module>> {
        if trainset.is_empty() {
            return Err(dspy_core::DspyError::OptimizationError(
                "Trainset is empty. Cannot run GRPO.".to_string(),
            ));
        }

        // Verify single LM (GRPO requires all predictors share one LM model)
        let predictors = student.named_predictors();
        let mut models = std::collections::HashSet::new();
        for (_, pred) in &predictors {
            let state = pred.dump_state();
            if let Some(sig) = state.get("signature") {
                let model = sig
                    .get("instructions")
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");
                models.insert(model.to_string());
            }
        }
        // Note: In a real impl, we'd compare LM instances/models more precisely.
        // For now, this is structural validation.

        let student_copy = student.deep_copy();

        // Main training loop
        for step in 0..self.config.num_train_steps {
            // Sample examples for this step
            let step_examples: Vec<Example> = trainset
                .iter()
                .cycle()
                .skip(step * self.config.num_dspy_examples_per_step % trainset.len())
                .take(self.config.num_dspy_examples_per_step.min(trainset.len()))
                .cloned()
                .collect();

            // Bootstrap trace data (rollouts)
            let traces = bootstrap_trace_data(
                student_copy.as_ref(),
                &step_examples,
                &BootstrapTraceOptions {
                    metric: self.config.metric.clone(),
                    num_threads: self.config.num_threads,
                    failure_score: self.config.failure_score,
                    ..Default::default()
                },
            )
            .await?;

            // Format training data as GRPO groups
            let _grpo_groups = self.format_grpo_data(&traces);

            // In a real implementation, we'd submit the GRPO groups to a reinforce job.
            // The mock implementation just collects data.
        }

        Ok(student_copy)
    }

    /// Format trace data into GRPO training groups.
    fn format_grpo_data(&self, traces: &[TraceData]) -> Vec<GRPOGroup> {
        let mut groups = Vec::new();

        for td in traces {
            let mut group_data = Vec::new();
            for entry in &td.trace {
                // Build messages from trace entry
                let mut messages = Vec::new();
                messages.push(TrainingMessage::system("You are a helpful assistant."));

                let mut input_parts = Vec::new();
                for key in entry.inputs.keys() {
                    if let Some(val) = entry.inputs.get(key) {
                        input_parts.push(format!("[[ ## {key} ## ]]\n{val}"));
                    }
                }
                messages.push(TrainingMessage::user(&input_parts.join("\n\n")));

                // Build completion from outputs
                let mut output_parts = Vec::new();
                for key in entry.outputs.keys() {
                    if let Some(val) = entry.outputs.get(key) {
                        output_parts.push(format!("[[ ## {key} ## ]]\n{val}"));
                    }
                }
                output_parts.push("[[ ## completed ## ]]".to_string());

                let completion =
                    TrainingMessage::assistant(&output_parts.join("\n\n"));

                let reward = td.score.unwrap_or(0.0);
                group_data.push(GRPOChatData {
                    messages,
                    completion,
                    reward,
                });
            }

            if !group_data.is_empty() {
                groups.push(GRPOGroup {
                    batch_id: None,
                    group: group_data,
                });
            }
        }

        groups
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
    fn test_grpo_config_defaults() {
        let config = GRPOConfig::default();
        assert!(config.metric.is_none());
        assert!(config.multitask);
        assert!(config.exclude_demos);
        assert_eq!(config.num_threads, 6);
        assert_eq!(config.num_train_steps, 100);
        assert_eq!(config.failure_score, 0.0);
        assert_eq!(config.format_failure_score, -1.0);
    }

    #[test]
    fn test_grpo_validates_failure_scores() {
        let config = GRPOConfig {
            failure_score: -1.0,
            format_failure_score: 0.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_grpo_validates_exclude_demos() {
        let config = GRPOConfig {
            exclude_demos: false,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_grpo_validates_multitask() {
        let config = GRPOConfig {
            multitask: false,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_grpo_validates_use_train_as_val() {
        let config = GRPOConfig {
            use_train_as_val: true,
            report_train_scores: false,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[tokio::test]
    async fn test_grpo_compile_empty_trainset() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let grpo = GRPO::new(GRPOConfig {
            num_train_steps: 1,
            ..Default::default()
        })
        .unwrap();

        let result = grpo.compile(&student, &[], None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_grpo_compile_basic() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(3);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let grpo = GRPO::new(GRPOConfig {
            metric: Some(metric),
            num_train_steps: 2,
            num_threads: 1,
            num_dspy_examples_per_step: 1,
            num_rollouts_per_step: 1,
            ..Default::default()
        })
        .unwrap();

        let result = grpo.compile(&student, &trainset, None).await;
        assert!(result.is_ok());
    }
}
