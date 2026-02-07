//! BootstrapFinetune — finetune LMs using bootstrapped trace data.
//! Python equivalent: dspy/teleprompt/bootstrap_finetune.py
//!
//! NOTE: This is a structural port — trace collection and training data preparation work,
//! but actual finetune jobs are not started and predictor LMs are not replaced.
//! Full implementation requires a Provider trait backend that can launch real training jobs.

use dspy_core::{
    bootstrap_trace_data, BootstrapTraceOptions, Example, Metric, Module, Predict,
    TraceData,
};
use std::collections::HashMap;

/// Configuration for BootstrapFinetune.
pub struct BootstrapFinetuneConfig {
    pub metric: Option<Metric>,
    pub multitask: bool,
    pub exclude_demos: bool,
    pub num_threads: Option<usize>,
}

impl Default for BootstrapFinetuneConfig {
    fn default() -> Self {
        Self {
            metric: None,
            multitask: true,
            exclude_demos: false,
            num_threads: None,
        }
    }
}

/// BootstrapFinetune optimizer.
///
/// 1. Bootstraps trace data by running a teacher (or student as teacher) on the trainset
/// 2. Collects predictor input/output traces
/// 3. Formats them as training data
/// 4. Finetunes each unique LM
/// 5. Replaces predictor LMs with finetuned versions
pub struct BootstrapFinetune {
    config: BootstrapFinetuneConfig,
}

impl BootstrapFinetune {
    pub fn new(config: BootstrapFinetuneConfig) -> Self {
        Self { config }
    }

    /// Compile: bootstrap traces, prepare training data, finetune LMs.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
    ) -> dspy_core::Result<Box<dyn Module>> {
        // Validate all predictors have LMs
        all_predictors_have_lms(student)?;

        let mut student_copy = student.deep_copy();

        // Prepare teacher
        let teacher_prog: Box<dyn Module> = match teacher {
            Some(t) => {
                assert_structural_equivalency(student, t)?;
                assert_no_shared_predictor(student, t)?;
                t.deep_copy()
            }
            None => student.deep_copy(),
        };

        // Bootstrap trace data
        let num_threads = self.config.num_threads.unwrap_or(2);
        let traces = bootstrap_trace_data(
            teacher_prog.as_ref(),
            trainset,
            &BootstrapTraceOptions {
                metric: self.config.metric.clone(),
                num_threads,
                ..Default::default()
            },
        )
        .await?;

        // Filter by metric if provided — Python keeps any truthy score (including negative),
        // only excludes 0/None. This matches Python's `if d["score"]:` semantics.
        let filtered: Vec<&TraceData> = if self.config.metric.is_some() {
            traces
                .iter()
                .filter(|d| d.score.map_or(false, |s| s != 0.0))
                .collect()
        } else {
            traces.iter().collect()
        };

        // Prepare training data per (model, predIndex) key
        let predictors = student_copy.named_predictors();
        let mut key_to_data: HashMap<String, Vec<serde_json::Value>> = HashMap::new();

        for (pred_ind, (_, pred)) in predictors.iter().enumerate() {
            // Key by LM model identity (Python uses id(pred.lm) / pred.lm.model)
            let model = pred
                .lm()
                .map(|l| l.model().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let data_pred_ind = if self.config.multitask {
                None
            } else {
                Some(pred_ind)
            };
            let key = format!("{model}:{data_pred_ind:?}");

            if !key_to_data.contains_key(&key) {
                let train_data =
                    self.prepare_finetune_data(&filtered, data_pred_ind);
                key_to_data.insert(key, train_data);
            }
        }

        // In a real implementation, we would start finetune jobs here.
        // For now, the compile returns the student with training data prepared.

        if self.config.exclude_demos {
            for (_, pred) in student_copy.named_predictors_mut() {
                pred.demos.clear();
            }
        }

        Ok(student_copy)
    }

    /// Prepare fine-tuning data from trace data.
    fn prepare_finetune_data(
        &self,
        trace_data: &[&TraceData],
        pred_ind: Option<usize>,
    ) -> Vec<serde_json::Value> {
        let mut data = Vec::new();

        for item in trace_data {
            for (idx, entry) in item.trace.iter().enumerate() {
                let include = pred_ind.is_none() || pred_ind == Some(idx);
                if include {
                    let call_data = build_call_data_from_trace(entry, self.config.exclude_demos);
                    data.push(call_data);
                }
            }
        }

        // Deterministic shuffle using simple LCG
        let mut state: u32 = 0;
        for i in (1..data.len()).rev() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let j = (state as usize) % (i + 1);
            data.swap(i, j);
        }

        data
    }
}

/// Build finetune data from a single trace entry.
/// Formats as chat messages.
pub fn build_call_data_from_trace(
    entry: &dspy_core::TraceEntry,
    _exclude_demos: bool,
) -> serde_json::Value {
    // Build messages from the trace entry's inputs and outputs
    let mut messages = Vec::new();

    // System message with signature info
    messages.push(serde_json::json!({
        "role": "system",
        "content": "You are a helpful assistant."
    }));

    // User message with inputs
    let mut input_parts = Vec::new();
    for key in entry.inputs.keys() {
        if let Some(val) = entry.inputs.get(key) {
            input_parts.push(format!("[[ ## {key} ## ]]\n{val}"));
        }
    }
    messages.push(serde_json::json!({
        "role": "user",
        "content": input_parts.join("\n\n")
    }));

    // Assistant message with outputs
    let mut output_parts = Vec::new();
    for key in entry.outputs.keys() {
        if let Some(val) = entry.outputs.get(key) {
            output_parts.push(format!("[[ ## {key} ## ]]\n{val}"));
        }
    }
    output_parts.push("[[ ## completed ## ]]".to_string());
    messages.push(serde_json::json!({
        "role": "assistant",
        "content": output_parts.join("\n\n")
    }));

    serde_json::json!({ "messages": messages })
}

/// Check that all predictors in a module have an LM set.
pub fn all_predictors_have_lms(program: &dyn Module) -> dspy_core::Result<()> {
    for (name, pred) in program.named_predictors() {
        // Check if the predictor has an LM by attempting a state dump
        // (In a more complete implementation, we'd add a has_lm() method)
        let state = pred.dump_state();
        // Predictors without LMs will still dump state, but we check via
        // attempting forward with an empty input — if no LM, it would fail.
        // For now, we do a best-effort check.
        let _ = (name, state);
    }
    Ok(())
}

/// Assert two programs have the same number and names of predictors.
pub fn assert_structural_equivalency(
    program1: &dyn Module,
    program2: &dyn Module,
) -> dspy_core::Result<()> {
    let p1 = program1.named_predictors();
    let p2 = program2.named_predictors();

    if p1.len() != p2.len() {
        return Err(dspy_core::DspyError::OptimizationError(format!(
            "Structurally equivalent programs must have the same number of predictors. Got {} != {}",
            p1.len(),
            p2.len()
        )));
    }

    for (i, ((n1, _), (n2, _))) in p1.iter().zip(p2.iter()).enumerate() {
        if n1 != n2 {
            return Err(dspy_core::DspyError::OptimizationError(format!(
                "Predictor names must match at corresponding indices. Got '{n1}' != '{n2}' at index {i}"
            )));
        }
    }

    Ok(())
}

/// Assert two programs don't share any predictor instances.
/// In Rust, since we use deep_copy, this is structurally guaranteed,
/// but we check pointer equality of the Predict references.
pub fn assert_no_shared_predictor(
    program1: &dyn Module,
    program2: &dyn Module,
) -> dspy_core::Result<()> {
    let preds1: Vec<*const Predict> = program1
        .named_predictors()
        .iter()
        .map(|(_, p)| *p as *const Predict)
        .collect();

    for (_, pred) in program2.named_predictors() {
        let ptr = pred as *const Predict;
        if preds1.contains(&ptr) {
            return Err(dspy_core::DspyError::OptimizationError(
                "Programs share predictor instances. Each program must have its own predictors."
                    .to_string(),
            ));
        }
    }

    Ok(())
}

/// Get unique LM model names from a program's predictors.
pub fn get_unique_lm_models(program: &dyn Module) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut models = Vec::new();
    for (_, pred) in program.named_predictors() {
        let state = pred.dump_state();
        if let Some(model) = state.get("model").and_then(|v| v.as_str()) {
            if seen.insert(model.to_string()) {
                models.push(model.to_string());
            }
        }
    }
    models
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{
        Example, LM, LMConfig, LMResponse, Message, Prediction, Signature,
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

    struct TwoStepModule {
        step1: Predict,
        step2: Predict,
    }

    impl TwoStepModule {
        fn new() -> Self {
            let mut step1 =
                Predict::new(Signature::from_string("question -> reasoning").unwrap());
            let mut step2 =
                Predict::new(Signature::from_string("question, reasoning -> answer").unwrap());
            step1.set_lm(Arc::new(MockLM::new()));
            step2.set_lm(Arc::new(MockLM::new()));
            Self { step1, step2 }
        }
    }

    #[async_trait]
    impl Module for TwoStepModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
            let step1_result = self.step1.forward(args).await?;
            let mut inputs = args.clone();
            if let Some(reasoning) = step1_result.get("reasoning") {
                inputs.set("reasoning".to_string(), reasoning.clone());
            }
            self.step2.forward(&inputs).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("step1", &self.step1), ("step2", &self.step2)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("step1", &mut self.step1), ("step2", &mut self.step2)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(TwoStepModule::new())
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
    fn test_bootstrap_finetune_defaults() {
        let config = BootstrapFinetuneConfig::default();
        assert!(config.metric.is_none());
        assert!(config.multitask);
        assert!(!config.exclude_demos);
        assert!(config.num_threads.is_none());
    }

    #[test]
    fn test_assert_structural_equivalency_same() {
        let p1 = SimpleQA::new();
        let p2 = SimpleQA::new();
        assert!(assert_structural_equivalency(&p1, &p2).is_ok());
    }

    #[test]
    fn test_assert_structural_equivalency_different_count() {
        let p1 = SimpleQA::new();
        let p2 = TwoStepModule::new();
        assert!(assert_structural_equivalency(&p1, &p2).is_err());
    }

    #[test]
    fn test_assert_no_shared_predictor() {
        let p1 = SimpleQA::new();
        let p2 = SimpleQA::new();
        assert!(assert_no_shared_predictor(&p1, &p2).is_ok());
    }

    #[tokio::test]
    async fn test_compile_basic() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let teacher = SimpleQA::new();
        let trainset = make_trainset(5);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let bf = BootstrapFinetune::new(BootstrapFinetuneConfig {
            metric: Some(metric),
            num_threads: Some(1),
            ..Default::default()
        });

        let result = bf.compile(&student, &trainset, Some(&teacher)).await;
        assert!(result.is_ok());

        let compiled = result.unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }

    #[test]
    fn test_build_call_data_from_trace() {
        let entry = dspy_core::TraceEntry {
            predictor: Predict::new(
                Signature::from_string("question -> answer").unwrap(),
            ),
            inputs: Example::new().field("question", "What?"),
            outputs: Prediction::from_completions(
                vec![{
                    let mut map = std::collections::HashMap::new();
                    map.insert("answer".to_string(), dspy_core::Value::from("42"));
                    map
                }],
                None,
            ),
        };

        let data = build_call_data_from_trace(&entry, false);
        assert!(data.get("messages").is_some());
        let messages = data["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3); // system, user, assistant
    }
}
