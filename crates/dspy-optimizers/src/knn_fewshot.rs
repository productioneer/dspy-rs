//! KNNFewShot — k-nearest neighbor few-shot optimizer.
//!
//! Uses an in-memory KNN retriever to find the k nearest neighbors in a trainset
//! at test time. For each input, it finds similar examples from the trainset
//! and bootstraps few-shot demos from those neighbors.
//!
//! Python equivalent: dspy/teleprompt/knn_fewshot.py

use crate::bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};
use dspy_core::{Embedder, Example, Metric, Module, Predict, Prediction, KNN};
use std::sync::Arc;

/// Configuration for KNNFewShot.
pub struct KNNFewShotConfig {
    /// Number of nearest neighbors to retrieve.
    pub k: usize,
    /// Max bootstrapped demos for inner BootstrapFewShot.
    pub max_bootstrapped_demos: usize,
    /// Max labeled demos for inner BootstrapFewShot.
    pub max_labeled_demos: usize,
    /// Max rounds for inner BootstrapFewShot.
    pub max_rounds: usize,
}

impl KNNFewShotConfig {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
        }
    }
}

/// KNNFewShot optimizer.
///
/// At compile time, pre-embeds the trainset. At inference time, the compiled
/// program finds k-nearest neighbors for each query, bootstraps few-shot demos
/// from them, and uses those demos for the forward pass.
pub struct KNNFewShot {
    config: KNNFewShotConfig,
}

impl KNNFewShot {
    pub fn new(config: KNNFewShotConfig) -> Self {
        Self { config }
    }

    /// Compile a student program with KNN-based dynamic few-shot selection.
    ///
    /// Returns a `KNNCompiledProgram` that wraps the student and dynamically
    /// selects demos based on input similarity at inference time.
    pub fn compile(
        &self,
        student: &dyn Module,
        trainset: Vec<Example>,
        embedder: Embedder,
        metric: Metric,
    ) -> KNNCompiledProgram {
        let knn = KNN::new(self.config.k, trainset, embedder);

        KNNCompiledProgram {
            student: student.deep_copy(),
            knn: Arc::new(knn),
            metric,
            max_bootstrapped_demos: self.config.max_bootstrapped_demos,
            max_labeled_demos: self.config.max_labeled_demos,
            max_rounds: self.config.max_rounds,
        }
    }
}

/// A compiled program that dynamically selects demos via KNN at inference time.
pub struct KNNCompiledProgram {
    student: Box<dyn Module>,
    knn: Arc<KNN>,
    metric: Metric,
    max_bootstrapped_demos: usize,
    max_labeled_demos: usize,
    max_rounds: usize,
}

impl KNNCompiledProgram {
    /// Run the program with KNN-based demo selection.
    ///
    /// Steps:
    /// 1. Extract input fields from the example
    /// 2. Query KNN to find k nearest neighbors
    /// 3. Run BootstrapFewShot with those neighbors as trainset
    /// 4. Forward the input through the compiled program
    pub async fn call(&self, input: &Example) -> Result<Prediction, dspy_core::DspyError> {
        // Extract input fields as key-value pairs for KNN query
        let input_example = input.inputs();
        let fields: Vec<(String, String)> = input_example
            .keys()
            .filter_map(|k| input_example.get_str(k).map(|v| (k.clone(), v.to_string())))
            .collect();

        let field_refs: Vec<(&str, &str)> = fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        // Find k nearest neighbors
        let knn_trainset = self.knn.query(&field_refs);

        // Bootstrap with neighbors as trainset
        let bootstrap_config = BootstrapFewShotConfig {
            metric: self.metric.clone(),
            metric_threshold: None,
            max_bootstrapped_demos: self.max_bootstrapped_demos,
            max_labeled_demos: self.max_labeled_demos,
            max_rounds: self.max_rounds,
            max_errors: 5,
        };
        let bootstrap = BootstrapFewShot::new(bootstrap_config);
        let compiled = bootstrap
            .compile(self.student.as_ref(), &knn_trainset, None)
            .await?;

        // Forward through compiled program
        compiled.call(input).await
    }
}

#[async_trait::async_trait]
impl Module for KNNCompiledProgram {
    fn module_type_name(&self) -> &str {
        "KNNCompiledProgram"
    }

    async fn forward(&self, input: &Example) -> Result<Prediction, dspy_core::DspyError> {
        self.call(input).await
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        self.student.named_predictors()
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        self.student.named_predictors_mut()
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(KNNCompiledProgram {
            student: self.student.deep_copy(),
            knn: Arc::clone(&self.knn),
            metric: self.metric.clone(),
            max_bootstrapped_demos: self.max_bootstrapped_demos,
            max_labeled_demos: self.max_labeled_demos,
            max_rounds: self.max_rounds,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::*;
    use std::sync::Arc;

    struct MockKNNLM {
        config: LMConfig,
    }

    impl MockKNNLM {
        fn new() -> Self {
            Self {
                config: LMConfig::new("mock-knn"),
            }
        }
    }

    #[async_trait]
    impl LM for MockKNNLM {
        async fn call(&self, messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            let last = messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            let answer = if last.contains("Paris") {
                "Paris is the capital of France"
            } else if last.contains("Berlin") {
                "Berlin is the capital of Germany"
            } else {
                "I don't know"
            };
            Ok(vec![LMResponse {
                text: format!("[[ ## answer ## ]]\n{}", answer),
                usage: None,
            }])
        }

        fn model(&self) -> &str {
            "mock-knn"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /// Simple mock embedder: sum of char values as first dim, len as second dim.
    fn mock_embedder() -> Embedder {
        Arc::new(|texts: &[String]| {
            texts
                .iter()
                .map(|t| {
                    vec![
                        t.chars().map(|c| c as u32 as f32).sum::<f32>(),
                        t.len() as f32,
                    ]
                })
                .collect()
        })
    }

    struct SimpleQA {
        predict: Predict,
    }

    impl SimpleQA {
        fn new(lm: Arc<dyn LM>) -> Self {
            let sig = Signature::from_string("question -> answer").unwrap();
            let mut predict = Predict::new(sig);
            predict.set_lm(lm);
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for SimpleQA {
        async fn forward(&self, input: &Example) -> Result<Prediction> {
            self.predict.forward(input).await
        }

        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }

        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }

        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(SimpleQA {
                predict: self.predict.clone(),
            })
        }
    }

    #[tokio::test]
    async fn test_knn_fewshot_basic() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockKNNLM::new());
        let student = SimpleQA::new(lm);

        let trainset = vec![
            Example::new()
                .field("question", "What is Paris?")
                .field("answer", "Capital of France")
                .with_inputs(&["question"]),
            Example::new()
                .field("question", "What is Berlin?")
                .field("answer", "Capital of Germany")
                .with_inputs(&["question"]),
            Example::new()
                .field("question", "What is London?")
                .field("answer", "Capital of UK")
                .with_inputs(&["question"]),
        ];

        let metric: Metric = Arc::new(|_example, _prediction| 1.0);

        let config = KNNFewShotConfig::new(2);
        let optimizer = KNNFewShot::new(config);
        let compiled = optimizer.compile(&student, trainset, mock_embedder(), metric);

        let input = Example::new()
            .field("question", "What is Paris?")
            .with_inputs(&["question"]);

        let result = compiled.forward(&input).await.unwrap();
        assert!(result.get_str("answer").is_some());
    }

    #[tokio::test]
    async fn test_knn_fewshot_deep_copy() {
        let lm: Arc<dyn LM> = Arc::new(MockKNNLM::new());
        let student = SimpleQA::new(lm);

        let trainset = vec![Example::new()
            .field("q", "hello")
            .field("a", "world")
            .with_inputs(&["q"])];

        let metric: Metric = Arc::new(|_, _| 1.0);
        let config = KNNFewShotConfig::new(1);
        let optimizer = KNNFewShot::new(config);
        let compiled = optimizer.compile(&student, trainset, mock_embedder(), metric);

        let copy = compiled.deep_copy();
        assert_eq!(copy.named_predictors().len(), 1);
    }

    #[tokio::test]
    async fn test_knn_fewshot_empty_trainset() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockKNNLM::new());
        let student = SimpleQA::new(lm);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let config = KNNFewShotConfig::new(3);
        let optimizer = KNNFewShot::new(config);
        let compiled = optimizer.compile(&student, vec![], mock_embedder(), metric);

        let input = Example::new()
            .field("question", "anything")
            .with_inputs(&["question"]);

        let result = compiled.forward(&input).await.unwrap();
        assert!(result.get_str("answer").is_some());
    }
}
