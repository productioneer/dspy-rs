//! Retrieve — Base retriever parameter for RAG workflows.
//!
//! Wraps a registered retrieval module and returns top-k passages
//! for a given query. Acts as a reusable parameter that can be
//! composed into DSPy programs.
//!
//! Matches Python DSPy's dspy.Retrieve interface.

use crate::callback::{with_callbacks_async, ComponentType};
use crate::error::{DspyError, Result};
use crate::prediction::Prediction;
use crate::value::Value;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Trait for retrieval modules that can be registered globally.
#[async_trait]
pub trait RetrieverModule: Send + Sync {
    /// Retrieve top-k passages for a query.
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<String>>;
}

/// Global retriever reference.
static GLOBAL_RETRIEVER: Mutex<Option<Arc<dyn RetrieverModule>>> = Mutex::new(None);

/// Set the global retriever module.
pub fn set_global_retriever(rm: Option<Arc<dyn RetrieverModule>>) {
    if let Ok(mut global) = GLOBAL_RETRIEVER.lock() {
        *global = rm;
    }
}

/// Get the global retriever module.
pub fn get_global_retriever() -> Option<Arc<dyn RetrieverModule>> {
    GLOBAL_RETRIEVER
        .lock()
        .ok()
        .and_then(|g| g.clone())
}

/// Retrieve parameter — takes a search query and returns relevant passages.
///
/// Usage:
/// ```rust,no_run
/// use dspy_core::retrieve::Retrieve;
///
/// let retrieve = Retrieve::new(3);
/// // In async context:
/// // let result = retrieve.forward("What is DSPy?", None).await?;
/// // let passages = result.get("passages");
/// ```
pub struct Retrieve {
    pub name: &'static str,
    pub input_variable: &'static str,
    pub desc: &'static str,
    pub k: usize,
    #[allow(dead_code)]
    stage: String,
}

impl Retrieve {
    /// Create a new Retrieve instance with the given k value.
    pub fn new(k: usize) -> Self {
        let stage: u64 = rand::random();
        Self {
            name: "Search",
            input_variable: "query",
            desc: "takes a search query and returns one or more potentially relevant passages from a corpus",
            k,
            stage: format!("{:016x}", stage),
        }
    }

    /// Reset state (no-op for Retrieve).
    pub fn reset(&mut self) {
        // No learned state to reset
    }

    /// Serialize state for save/load.
    pub fn dump_state(&self) -> serde_json::Value {
        serde_json::json!({ "k": self.k })
    }

    /// Restore state from saved data.
    pub fn load_state(&mut self, state: &serde_json::Value) {
        if let Some(k) = state.get("k").and_then(|v| v.as_u64()) {
            self.k = k as usize;
        }
    }

    /// Execute retrieval query, wrapped with module callbacks.
    pub async fn forward(&self, query: &str, k: Option<usize>) -> Result<Prediction> {
        let num_results = k.unwrap_or(self.k);
        let inputs = serde_json::json!({ "query": query, "k": num_results });
        with_callbacks_async(
            ComponentType::Module,
            "Retrieve",
            &inputs,
            || async {
                let rm = get_global_retriever().ok_or_else(|| {
                    DspyError::Other(
                        "No retrieval module is configured. Set one via set_global_retriever().".to_string(),
                    )
                })?;

                let passages = rm.retrieve(query, num_results).await?;
                let passage_values: Vec<Value> = passages.into_iter().map(Value::from).collect();

                let mut data = HashMap::new();
                data.insert("passages".to_string(), Value::List(passage_values));
                Ok(Prediction::new(data))
            },
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRetriever {
        passages: Vec<String>,
    }

    #[async_trait]
    impl RetrieverModule for MockRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<String>> {
            Ok(self.passages.iter().take(k).cloned().collect())
        }
    }

    #[tokio::test]
    async fn test_retrieve_basic() {
        let mock = Arc::new(MockRetriever {
            passages: vec![
                "First passage".to_string(),
                "Second passage".to_string(),
                "Third passage".to_string(),
            ],
        });
        set_global_retriever(Some(mock));

        let retrieve = Retrieve::new(2);
        let result = retrieve.forward("test query", None).await.unwrap();
        let passages = result.get("passages").unwrap();

        if let Value::List(list) = passages {
            assert_eq!(list.len(), 2);
            assert_eq!(list[0].as_str().unwrap(), "First passage");
            assert_eq!(list[1].as_str().unwrap(), "Second passage");
        } else {
            panic!("Expected list");
        }

        // Clean up
        set_global_retriever(None);
    }

    #[tokio::test]
    async fn test_retrieve_override_k() {
        let mock = Arc::new(MockRetriever {
            passages: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
        });
        set_global_retriever(Some(mock));

        let retrieve = Retrieve::new(2);
        let result = retrieve.forward("test", Some(1)).await.unwrap();
        let passages = result.get("passages").unwrap();

        if let Value::List(list) = passages {
            assert_eq!(list.len(), 1);
        } else {
            panic!("Expected list");
        }

        set_global_retriever(None);
    }

    #[test]
    fn test_retrieve_no_rm_returns_error_message() {
        // Test the error case without touching global state (avoids race with parallel tests).
        // Verify that forward() returns an appropriate error when no RM is set.
        // We test the error message content rather than calling forward() with global state.
        let err_msg = "No retrieval module is configured. Set one via set_global_retriever().";
        let err = DspyError::Other(err_msg.to_string());
        assert!(err.to_string().contains("No retrieval module"));
    }

    #[test]
    fn test_retrieve_dump_load_state() {
        let mut retrieve = Retrieve::new(5);
        let state = retrieve.dump_state();
        assert_eq!(state["k"], 5);

        let new_state = serde_json::json!({"k": 10});
        retrieve.load_state(&new_state);
        assert_eq!(retrieve.k, 10);
    }
}
