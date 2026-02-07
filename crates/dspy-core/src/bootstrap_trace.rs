//! Bootstrap trace data: run a program on a dataset, capture execution traces.
//! Used by BootstrapFinetune and GRPO to collect training data.
//! Python equivalent: dspy/teleprompt/bootstrap_trace.py
//!
//! NOTE: Trace capture uses a simplified approach — each example produces trace entries
//! via sequential predictor invocation. Python's approach patches forward() with a trace
//! context and handles AdapterParseError for FailedPrediction. The Rust version captures
//! actual predictor traces but uses a simpler dummy-Predict mapping approach.

use crate::error::Result;
use crate::evaluate::{Evaluate, EvaluateConfig, Metric};
use crate::example::Example;
use crate::module_trait::Module;
use crate::predict::{Predict, Trace};
use crate::prediction::Prediction;
use std::sync::Arc;

/// Represents a prediction that failed to parse but captured the raw LM output.
/// Used by GRPO to assign format failure rewards.
#[derive(Debug, Clone)]
pub struct FailedPrediction {
    pub completion_text: String,
    pub format_reward: Option<f64>,
}

impl FailedPrediction {
    pub fn new(completion_text: &str, format_reward: Option<f64>) -> Self {
        Self {
            completion_text: completion_text.to_string(),
            format_reward,
        }
    }
}

/// A single trace entry from a predictor invocation.
#[derive(Clone)]
pub struct TraceEntry {
    pub predictor: Predict,
    pub inputs: Example,
    pub outputs: Prediction,
}

/// Data returned for each successfully traced example.
#[derive(Clone)]
pub struct TraceData {
    pub example_ind: usize,
    pub example: Example,
    pub prediction: Prediction,
    pub trace: Vec<TraceEntry>,
    pub score: Option<f64>,
}

/// Options for bootstrap trace data collection.
pub struct BootstrapTraceOptions {
    pub metric: Option<Metric>,
    pub num_threads: usize,
    pub failure_score: f64,
    pub format_failure_score: f64,
}

impl Default for BootstrapTraceOptions {
    fn default() -> Self {
        Self {
            metric: None,
            num_threads: 2,
            failure_score: 0.0,
            format_failure_score: -1.0,
        }
    }
}

/// Run a program on a dataset, capture traces from each predictor invocation.
/// Returns a list of TraceData items (one per example).
///
/// In Rust, we use the Predict's built-in trace mechanism (train mode)
/// rather than monkey-patching forward methods.
pub async fn bootstrap_trace_data(
    program: &dyn Module,
    dataset: &[Example],
    options: &BootstrapTraceOptions,
) -> Result<Vec<TraceData>> {
    if dataset.is_empty() {
        return Ok(Vec::new());
    }

    // Create a deep copy to enable train mode
    let mut program_copy = program.deep_copy();

    // Enable train mode on all predictors
    for (_, pred) in program_copy.named_predictors_mut() {
        pred.set_train(true);
    }

    // Define metric for evaluation
    let metric: Metric = if let Some(ref m) = options.metric {
        m.clone()
    } else {
        Arc::new(|_, _| 1.0)
    };

    // Run evaluation to get predictions
    let eval = Evaluate::new(
        dataset.to_vec(),
        metric,
        EvaluateConfig {
            num_threads: options.num_threads,
            display_progress: false,
            failure_score: options.failure_score,
            max_errors: dataset.len() * 10,
        },
    );

    let eval_result = eval.run(program_copy.as_ref()).await?;

    // Collect traces from the predictors
    let predictor_traces: Vec<Trace> = program_copy
        .named_predictors()
        .iter()
        .flat_map(|(_, pred)| pred.get_traces())
        .collect();

    // Build TraceData from results
    let mut data = Vec::new();
    let mut trace_cursor = 0;

    for (i, (example, prediction, score)) in eval_result.results.into_iter().enumerate() {
        // Collect trace entries for this example
        // Each forward() call generates traces equal to the number of predictors
        let num_predictors = program_copy.named_predictors().len();
        let mut trace_entries = Vec::new();

        for _ in 0..num_predictors {
            if trace_cursor < predictor_traces.len() {
                let t = &predictor_traces[trace_cursor];
                trace_entries.push(TraceEntry {
                    predictor: Predict::new(
                        crate::signature::Signature::from_string("q -> a")
                            .unwrap_or_else(|_| crate::signature::Signature::from_string("q -> a").unwrap()),
                    ),
                    inputs: t.inputs.clone(),
                    outputs: t.outputs.clone(),
                });
                trace_cursor += 1;
            }
        }

        let mut trace_data = TraceData {
            example_ind: i,
            example,
            prediction,
            trace: trace_entries,
            score: None,
        };

        if options.metric.is_some() {
            trace_data.score = Some(score);
        }

        data.push(trace_data);
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message, LM};
    use crate::module_trait::Module;
    use crate::predict::Predict;
    use crate::signature::Signature;
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
        ) -> crate::Result<Vec<LMResponse>> {
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
            serde_json::json!({})
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
        async fn forward(&self, args: &Example) -> crate::Result<Prediction> {
            self.predict.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            let mut copy = SimpleQA::new();
            copy.predict.demos = self.predict.demos.clone();
            Box::new(copy)
        }
    }

    fn make_dataset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[test]
    fn test_failed_prediction() {
        let fp = FailedPrediction::new("raw output", None);
        assert_eq!(fp.completion_text, "raw output");
        assert!(fp.format_reward.is_none());

        let fp2 = FailedPrediction::new("raw", Some(-1.0));
        assert_eq!(fp2.format_reward, Some(-1.0));
    }

    #[tokio::test]
    async fn test_bootstrap_trace_basic() {
        crate::settings::reset_settings();
        let program = SimpleQA::new();
        let dataset = make_dataset(2);

        let traces = bootstrap_trace_data(
            &program,
            &dataset,
            &BootstrapTraceOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(traces.len(), 2);
        for td in &traces {
            assert!(!td.trace.is_empty());
        }
    }

    #[tokio::test]
    async fn test_bootstrap_trace_with_metric() {
        crate::settings::reset_settings();
        let program = SimpleQA::new();
        let dataset = vec![
            Example::new()
                .field("question", "Q1")
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

        let traces = bootstrap_trace_data(
            &program,
            &dataset,
            &BootstrapTraceOptions {
                metric: Some(metric),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].score, Some(1.0));
    }

    #[tokio::test]
    async fn test_bootstrap_trace_empty_dataset() {
        crate::settings::reset_settings();
        let program = SimpleQA::new();
        let traces = bootstrap_trace_data(
            &program,
            &[],
            &BootstrapTraceOptions::default(),
        )
        .await
        .unwrap();

        assert!(traces.is_empty());
    }

    #[tokio::test]
    async fn test_bootstrap_trace_example_ind() {
        crate::settings::reset_settings();
        let program = SimpleQA::new();
        let dataset = make_dataset(3);

        let traces = bootstrap_trace_data(
            &program,
            &dataset,
            &BootstrapTraceOptions::default(),
        )
        .await
        .unwrap();

        assert_eq!(traces.len(), 3);
        for (i, td) in traces.iter().enumerate() {
            assert_eq!(td.example_ind, i);
        }
    }
}
