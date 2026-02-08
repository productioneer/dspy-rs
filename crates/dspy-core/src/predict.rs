//! Predict — core building block module.
//! Python equivalent: dspy/predict/predict.py

use crate::adapter::{Adapter, ChatAdapter};
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::LM;
use crate::prediction::Prediction;
use crate::settings::get_settings;
use crate::signature::Signature;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A trace entry recording one predict call
#[derive(Debug, Clone)]
pub struct Trace {
    pub predictor_name: String,
    pub inputs: Example,
    pub outputs: Prediction,
}

#[derive(Clone)]
pub struct Predict {
    pub signature: Signature,
    pub demos: Vec<Example>,
    lm: Option<Arc<dyn LM>>,
    adapter: Option<Arc<dyn Adapter>>,
    traces: Arc<Mutex<Vec<Trace>>>,
    train: bool,
}

impl Predict {
    pub fn new(signature: Signature) -> Self {
        Self {
            signature,
            demos: Vec::new(),
            lm: None,
            adapter: None,
            traces: Arc::new(Mutex::new(Vec::new())),
            train: false,
        }
    }

    pub fn lm(&self) -> Option<Arc<dyn LM>> {
        self.lm.clone()
    }

    pub fn set_lm(&mut self, lm: Arc<dyn LM>) {
        self.lm = Some(lm);
    }

    pub fn set_adapter(&mut self, adapter: Arc<dyn Adapter>) {
        self.adapter = Some(adapter);
    }

    pub fn set_train(&mut self, train: bool) {
        self.train = train;
    }

    pub fn is_train(&self) -> bool {
        self.train
    }

    pub fn get_traces(&self) -> Vec<Trace> {
        self.traces.lock().unwrap().clone()
    }

    pub fn clear_traces(&self) {
        self.traces.lock().unwrap().clear();
    }

    pub fn reset(&mut self) {
        self.demos.clear();
        self.traces.lock().unwrap().clear();
    }

    /// Get effective LM (own > settings > error)
    fn effective_lm(&self) -> Result<Arc<dyn LM>> {
        if let Some(ref lm) = self.lm {
            return Ok(lm.clone());
        }
        let settings = get_settings();
        settings.lm.ok_or_else(|| {
            DspyError::LMError("No LM configured. Use set_lm() or configure().".into())
        })
    }

    /// Get effective adapter (own > settings > ChatAdapter default)
    fn effective_adapter(&self) -> Arc<dyn Adapter> {
        if let Some(ref adapter) = self.adapter {
            return adapter.clone();
        }
        let settings = get_settings();
        settings
            .adapter
            .unwrap_or_else(|| Arc::new(ChatAdapter::new()))
    }

    pub fn dump_state(&self) -> serde_json::Value {
        let demos_state: Vec<serde_json::Value> = self
            .demos
            .iter()
            .map(|d| serde_json::to_value(d).unwrap_or(serde_json::json!(null)))
            .collect();

        serde_json::json!({
            "signature": self.signature.dump_state(),
            "demos": demos_state,
        })
    }

    pub fn load_state(&mut self, state: &serde_json::Value) -> Result<()> {
        if let Some(sig_state) = state.get("signature") {
            self.signature = Signature::load_state(sig_state)?;
        }
        if let Some(demos_arr) = state.get("demos").and_then(|v| v.as_array()) {
            self.demos = demos_arr
                .iter()
                .filter_map(|d| serde_json::from_value(d.clone()).ok())
                .collect();
        }
        Ok(())
    }
}

#[async_trait]
impl crate::module_trait::Module for Predict {
    fn module_type_name(&self) -> &str {
        "Predict"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        let lm = self.effective_lm()?;
        let adapter = self.effective_adapter();
        let mut config = lm.config().clone();

        // Temperature auto-adjustment matching Python DSPy:
        // If temperature is unset or <= 0.15, and n > 1, set to 0.7 to keep randomness
        let n = config.n.unwrap_or(1);
        if n > 1 {
            let temp = config.temperature.unwrap_or(0.0);
            if temp <= 0.15 {
                config.temperature = Some(0.7);
            }
        }

        // Call adapter to format + call LM + parse
        let completions = adapter
            .call(lm.as_ref(), &self.signature, &self.demos, args, &config)
            .await?;

        let prediction = Prediction::from_completions(completions, Some(&self.signature));

        // Record trace if in training mode
        if self.train {
            let trace = Trace {
                predictor_name: String::new(),
                inputs: args.clone(),
                outputs: prediction.clone(),
            };

            // Try to add to settings trace collector
            let settings = get_settings();
            if let Some(ref trace_vec) = settings.trace {
                if let Ok(mut tv) = trace_vec.lock() {
                    tv.push(trace.clone());
                }
            }

            // Always add to own traces
            if let Ok(mut traces) = self.traces.lock() {
                traces.push(trace);
            }
        }

        Ok(prediction)
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![("self", self)]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![("self", self)]
    }

    fn deep_copy(&self) -> Box<dyn crate::module_trait::Module> {
        Box::new(self.clone())
    }

    fn dump_state(&self) -> serde_json::Value {
        Predict::dump_state(self)
    }

    fn load_state(&mut self, state: &serde_json::Value) -> Result<()> {
        Predict::load_state(self, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message};
    use crate::module_trait::Module;
    use crate::settings;
    /// Mock LM for testing
    struct MockLM {
        responses: Vec<String>,
        config: LMConfig,
    }

    impl MockLM {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.into_iter().map(|s| s.to_string()).collect(),
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(&self, _messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            Ok(self
                .responses
                .iter()
                .map(|text| LMResponse {
                    text: text.clone(),
                    usage: None,
                })
                .collect())
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

    #[tokio::test]
    async fn test_predict_forward() {
        settings::reset_settings();
        let sig = Signature::from_string("question -> answer").unwrap();
        let mut predict = Predict::new(sig);
        let lm = Arc::new(MockLM::new(vec!["[[ ## answer ## ]]\n42"]));
        predict.set_lm(lm);

        let inputs = Example::new().field("question", "What is the meaning of life?");
        let result = predict.forward(&inputs).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("42"));
    }

    #[tokio::test]
    async fn test_predict_forward_no_markers() {
        settings::reset_settings();
        let sig = Signature::from_string("question -> answer").unwrap();
        let mut predict = Predict::new(sig);
        let lm = Arc::new(MockLM::new(vec!["The answer is 42"]));
        predict.set_lm(lm);

        let inputs = Example::new().field("question", "What?");
        let result = predict.forward(&inputs).await.unwrap();
        // Without markers, entire text goes to first output field
        assert!(result.get_str("answer").unwrap().contains("42"));
    }

    #[tokio::test]
    async fn test_predict_no_lm_error() {
        settings::reset_settings();
        let sig = Signature::from_string("q -> a").unwrap();
        let predict = Predict::new(sig);
        let inputs = Example::new().field("q", "test");
        let result = predict.forward(&inputs).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_predict_with_demos() {
        settings::reset_settings();
        let sig = Signature::from_string("question -> answer").unwrap();
        let mut predict = Predict::new(sig);
        let lm = Arc::new(MockLM::new(vec!["[[ ## answer ## ]]\nParis"]));
        predict.set_lm(lm);
        predict.demos.push(
            Example::new()
                .field("question", "Capital of Germany?")
                .field("answer", "Berlin"),
        );

        let inputs = Example::new().field("question", "Capital of France?");
        let result = predict.forward(&inputs).await.unwrap();
        assert_eq!(result.get_str("answer"), Some("Paris"));
    }

    #[tokio::test]
    async fn test_predict_train_mode_traces() {
        settings::reset_settings();
        let sig = Signature::from_string("q -> a").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_train(true);
        let lm = Arc::new(MockLM::new(vec!["[[ ## a ## ]]\ntest"]));
        predict.set_lm(lm);

        let inputs = Example::new().field("q", "hello");
        let _ = predict.forward(&inputs).await.unwrap();

        let traces = predict.get_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].inputs.get_str("q"), Some("hello"));
    }

    #[test]
    fn test_dump_load_state() {
        let sig = Signature::from_string("q -> a").unwrap();
        let mut predict = Predict::new(sig);
        predict
            .demos
            .push(Example::new().field("q", "demo").field("a", "ans"));

        let state = predict.dump_state();
        let mut predict2 = Predict::new(Signature::from_string("x -> y").unwrap());
        predict2.load_state(&state).unwrap();

        assert_eq!(predict2.demos.len(), 1);
        assert_eq!(predict2.demos[0].get_str("q"), Some("demo"));
    }

    #[test]
    fn test_named_predictors() {
        let sig = Signature::from_string("q -> a").unwrap();
        let predict = Predict::new(sig);
        let preds = predict.named_predictors();
        assert_eq!(preds.len(), 1);
        assert_eq!(preds[0].0, "self");
    }

    #[test]
    fn test_reset() {
        let sig = Signature::from_string("q -> a").unwrap();
        let mut predict = Predict::new(sig);
        predict.demos.push(Example::new().field("q", "demo"));
        predict.reset();
        assert!(predict.demos.is_empty());
    }

    #[tokio::test]
    async fn test_predict_via_settings_lm() {
        settings::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new(vec!["[[ ## a ## ]]\nfrom_settings"]));
        settings::configure(Settings::new().with_lm(lm));

        let sig = Signature::from_string("q -> a").unwrap();
        let predict = Predict::new(sig);
        let inputs = Example::new().field("q", "test");
        let result = predict.forward(&inputs).await.unwrap();
        assert_eq!(result.get_str("a"), Some("from_settings"));

        settings::reset_settings();
    }

    use crate::settings::Settings;

    /// Mock LM that captures the config it receives
    struct ConfigCaptureLM {
        received_config: Mutex<Option<LMConfig>>,
        config: LMConfig,
    }

    impl ConfigCaptureLM {
        fn new(config: LMConfig) -> Self {
            Self {
                received_config: Mutex::new(None),
                config,
            }
        }

        fn received_temperature(&self) -> Option<f64> {
            self.received_config
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|c| c.temperature)
        }
    }

    #[async_trait]
    impl LM for ConfigCaptureLM {
        async fn call(&self, _messages: &[Message], config: &LMConfig) -> Result<Vec<LMResponse>> {
            *self.received_config.lock().unwrap() = Some(config.clone());
            Ok(vec![LMResponse {
                text: "[[ ## answer ## ]]\ntest".to_string(),
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

    #[tokio::test]
    async fn test_temperature_auto_adjust_n_gt_1_low_temp() {
        // When n > 1 and temperature <= 0.15, should auto-adjust to 0.7
        settings::reset_settings();
        let mut config = LMConfig::new("mock");
        config.n = Some(3);
        config.temperature = Some(0.1);
        let lm = Arc::new(ConfigCaptureLM::new(config));

        let sig = Signature::from_string("q -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(lm.clone());

        let inputs = Example::new().field("q", "test");
        let _ = predict.forward(&inputs).await.unwrap();

        assert_eq!(lm.received_temperature(), Some(0.7));
    }

    #[tokio::test]
    async fn test_temperature_no_adjust_n_eq_1() {
        // When n = 1, temperature should not be adjusted
        settings::reset_settings();
        let mut config = LMConfig::new("mock");
        config.n = Some(1);
        config.temperature = Some(0.1);
        let lm = Arc::new(ConfigCaptureLM::new(config));

        let sig = Signature::from_string("q -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(lm.clone());

        let inputs = Example::new().field("q", "test");
        let _ = predict.forward(&inputs).await.unwrap();

        // Should remain at 0.1 — no adjustment when n=1
        assert_eq!(lm.received_temperature(), Some(0.1));
    }

    #[tokio::test]
    async fn test_temperature_no_adjust_high_temp() {
        // When n > 1 but temperature > 0.15, should NOT adjust
        settings::reset_settings();
        let mut config = LMConfig::new("mock");
        config.n = Some(5);
        config.temperature = Some(0.5);
        let lm = Arc::new(ConfigCaptureLM::new(config));

        let sig = Signature::from_string("q -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(lm.clone());

        let inputs = Example::new().field("q", "test");
        let _ = predict.forward(&inputs).await.unwrap();

        // Should remain at 0.5 — already above threshold
        assert_eq!(lm.received_temperature(), Some(0.5));
    }

    #[tokio::test]
    async fn test_temperature_adjust_none_temp() {
        // When n > 1 and temperature is None (defaults to 0.0), should adjust to 0.7
        settings::reset_settings();
        let mut config = LMConfig::new("mock");
        config.n = Some(2);
        // temperature is None by default
        let lm = Arc::new(ConfigCaptureLM::new(config));

        let sig = Signature::from_string("q -> answer").unwrap();
        let mut predict = Predict::new(sig);
        predict.set_lm(lm.clone());

        let inputs = Example::new().field("q", "test");
        let _ = predict.forward(&inputs).await.unwrap();

        assert_eq!(lm.received_temperature(), Some(0.7));
    }
}
