//! Provider — abstractions for model training (finetune, RL).
//! Python equivalent: dspy/clients/provider.py

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::oneshot;

use crate::finetune_types::TrainDataFormat;
use crate::lm::LM;

/// TrainingJob — represents an asynchronous fine-tuning job.
/// Uses a oneshot channel for async result delivery.
pub struct TrainingJob {
    pub model: Option<String>,
    pub train_data: Option<Vec<serde_json::Value>>,
    pub train_data_format: Option<TrainDataFormat>,
    pub train_kwargs: HashMap<String, serde_json::Value>,
    done: bool,
    cancelled: bool,
    sender: Option<oneshot::Sender<Result<Arc<dyn LM>, String>>>,
    receiver: Option<oneshot::Receiver<Result<Arc<dyn LM>, String>>>,
}

impl TrainingJob {
    pub fn new() -> Self {
        let (tx, rx) = oneshot::channel();
        Self {
            model: None,
            train_data: None,
            train_data_format: None,
            train_kwargs: HashMap::new(),
            done: false,
            cancelled: false,
            sender: Some(tx),
            receiver: Some(rx),
        }
    }

    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    pub fn with_train_data(mut self, data: Vec<serde_json::Value>) -> Self {
        self.train_data = Some(data);
        self
    }

    pub fn with_train_kwargs(mut self, kwargs: HashMap<String, serde_json::Value>) -> Self {
        self.train_kwargs = kwargs;
        self
    }

    /// Complete the job with a result (finetuned LM or error message).
    pub fn complete(&mut self, result: Result<Arc<dyn LM>, String>) {
        self.done = true;
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(result);
        }
    }

    /// Get the result (blocks until complete).
    pub async fn result(&mut self) -> Result<Arc<dyn LM>, String> {
        if let Some(rx) = self.receiver.take() {
            rx.await.unwrap_or(Err("Job channel closed".to_string()))
        } else {
            Err("Result already consumed".to_string())
        }
    }

    /// Cancel the job.
    pub fn cancel(&mut self) {
        self.cancelled = true;
        self.done = true;
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(Err("Job cancelled".to_string()));
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn status(&self) -> &str {
        if self.cancelled {
            "cancelled"
        } else if self.done {
            "done"
        } else {
            "running"
        }
    }
}

impl Default for TrainingJob {
    fn default() -> Self {
        Self::new()
    }
}

/// ReinforceJob — for GRPO-style online training.
/// Abstract base; providers must implement the methods.
#[async_trait]
pub trait ReinforceJob: Send + Sync {
    fn lm(&self) -> &dyn LM;
    fn train_kwargs(&self) -> &HashMap<String, serde_json::Value>;
    fn checkpoints(&self) -> &HashMap<String, String>;
    fn last_checkpoint(&self) -> Option<&str>;

    async fn initialize(&mut self) -> crate::Result<()>;
    async fn step(
        &mut self,
        train_data: &[serde_json::Value],
        train_data_format: Option<TrainDataFormat>,
    ) -> crate::Result<()>;
    async fn terminate(&mut self) -> crate::Result<()>;
    async fn save_checkpoint(&mut self, name: &str) -> crate::Result<()>;
    fn get_status(&self) -> serde_json::Value;

    fn cancel(&mut self) -> crate::Result<()> {
        Err(crate::DspyError::Other("Not implemented".to_string()))
    }
}

/// Provider — abstract base for model providers.
/// Concrete providers (OpenAI, local, etc.) implement these methods.
#[async_trait]
pub trait Provider: Send + Sync {
    fn is_finetunable(&self) -> bool {
        false
    }

    fn is_reinforceable(&self) -> bool {
        false
    }

    fn is_provider_model(&self, _model: &str) -> bool {
        false
    }

    async fn launch(&self, _lm: &dyn LM) -> crate::Result<()> {
        Ok(())
    }

    async fn kill(&self, _lm: &dyn LM) -> crate::Result<()> {
        Ok(())
    }

    async fn finetune(
        &self,
        _job: &mut TrainingJob,
        _model: &str,
        _train_data: &[serde_json::Value],
        _train_data_format: Option<TrainDataFormat>,
    ) -> crate::Result<String> {
        Err(crate::DspyError::Other("Not implemented".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message};

    struct MockLM {
        model: String,
        config: LMConfig,
    }

    impl MockLM {
        fn new(model: &str) -> Self {
            Self {
                model: model.to_string(),
                config: LMConfig::new(model),
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
            Ok(vec![LMResponse::new("mock", None)])
        }
        fn model(&self) -> &str {
            &self.model
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({"model": self.model})
        }
    }

    #[test]
    fn test_training_job_defaults() {
        let job = TrainingJob::new();
        assert!(job.model.is_none());
        assert!(job.train_data.is_none());
        assert!(job.train_data_format.is_none());
        assert!(job.train_kwargs.is_empty());
        assert!(!job.is_done());
        assert!(!job.is_cancelled());
        assert_eq!(job.status(), "running");
    }

    #[test]
    fn test_training_job_with_options() {
        let job = TrainingJob::new()
            .with_model("gpt-4")
            .with_train_data(vec![serde_json::json!({"text": "hello"})]);
        assert_eq!(job.model.as_deref(), Some("gpt-4"));
        assert!(job.train_data.is_some());
    }

    #[tokio::test]
    async fn test_training_job_complete() {
        let mut job = TrainingJob::new();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("finetuned"));
        job.complete(Ok(lm));
        assert!(job.is_done());
        assert_eq!(job.status(), "done");

        let result = job.result().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_training_job_cancel() {
        let mut job = TrainingJob::new();
        job.cancel();
        assert!(job.is_cancelled());
        assert!(job.is_done());
        assert_eq!(job.status(), "cancelled");

        let result = job.result().await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.contains("cancelled"));
    }

    #[tokio::test]
    async fn test_training_job_complete_with_error() {
        let mut job = TrainingJob::new();
        job.complete(Err("Training failed".to_string()));
        assert!(job.is_done());

        let result = job.result().await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert_eq!(err, "Training failed");
    }
}
