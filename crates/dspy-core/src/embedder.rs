//! Embedder — Unified embedding interface.
//!
//! Supports custom embedding functions with batching.
//! Matches Python DSPy's dspy.clients.embedding.Embedder interface.

use crate::error::{DspyError, Result};
use async_trait::async_trait;
use std::collections::HashMap;

/// Trait for embedding functions.
#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    /// Compute embeddings for a batch of texts, with optional kwargs.
    async fn embed(
        &self,
        texts: &[String],
        kwargs: Option<&HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f32>>>;
}

/// DSPy Embedder — unified embedding interface with batching.
///
/// Matches Python DSPy's dspy.clients.embedding.Embedder interface:
/// - Accepts a custom embedding function
/// - Supports batching, caching flag, and kwargs passthrough
pub struct Embedder {
    model: Box<dyn EmbeddingFunction>,
    batch_size: usize,
    caching: bool,
    default_kwargs: HashMap<String, serde_json::Value>,
}

impl Embedder {
    /// Create a new Embedder with a custom embedding function.
    pub fn new(model: Box<dyn EmbeddingFunction>, batch_size: usize, caching: bool) -> Self {
        Self {
            model,
            batch_size,
            caching,
            default_kwargs: HashMap::new(),
        }
    }

    /// Create a new Embedder with default settings.
    pub fn with_defaults(model: Box<dyn EmbeddingFunction>) -> Self {
        Self::new(model, 200, true)
    }

    /// Create a new Embedder with default kwargs.
    pub fn with_kwargs(
        model: Box<dyn EmbeddingFunction>,
        batch_size: usize,
        caching: bool,
        kwargs: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            model,
            batch_size,
            caching,
            default_kwargs: kwargs,
        }
    }

    /// Whether caching is enabled.
    pub fn caching(&self) -> bool {
        self.caching
    }

    /// Compute embeddings for the given inputs.
    ///
    /// Matches Python DSPy's Embedder.__call__:
    /// - batch_size: override per-call batch size
    /// - caching: override per-call caching flag
    /// - kwargs: merged with default_kwargs and passed to embedding function
    pub async fn call(
        &self,
        inputs: &[String],
        batch_size: Option<usize>,
        caching: Option<bool>,
        kwargs: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let effective_batch_size = batch_size.unwrap_or(self.batch_size);
        let _effective_caching = caching.unwrap_or(self.caching);
        let merged_kwargs = if let Some(kw) = kwargs {
            let mut merged = self.default_kwargs.clone();
            merged.extend(kw);
            merged
        } else if !self.default_kwargs.is_empty() {
            self.default_kwargs.clone()
        } else {
            HashMap::new()
        };

        let kw_ref = if merged_kwargs.is_empty() {
            None
        } else {
            Some(&merged_kwargs)
        };

        let mut all_embeddings = Vec::with_capacity(inputs.len());

        for batch in inputs.chunks(effective_batch_size) {
            let batch_vec: Vec<String> = batch.to_vec();
            let embeddings = self.model.embed(&batch_vec, kw_ref).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Compute embedding for a single input (convenience method).
    pub async fn call_single(&self, input: &str) -> Result<Vec<f32>> {
        let results = self.call(&[input.to_string()], None, None, None).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| DspyError::Other("No embedding returned".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockEmbedder {
        dim: usize,
    }

    #[async_trait]
    impl EmbeddingFunction for MockEmbedder {
        async fn embed(
            &self,
            texts: &[String],
            _kwargs: Option<&HashMap<String, serde_json::Value>>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| vec![t.len() as f32; self.dim])
                .collect())
        }
    }

    #[tokio::test]
    async fn test_embedder_basic() {
        let embedder = Embedder::with_defaults(Box::new(MockEmbedder { dim: 3 }));
        let inputs = vec!["hello".to_string(), "world".to_string()];
        let result = embedder.call(&inputs, None, None, None).await.unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        // "hello" has length 5
        assert_eq!(result[0][0], 5.0);
    }

    #[tokio::test]
    async fn test_embedder_single() {
        let embedder = Embedder::with_defaults(Box::new(MockEmbedder { dim: 4 }));
        let result = embedder.call_single("test").await.unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], 4.0);
    }

    #[tokio::test]
    async fn test_embedder_batching() {
        let embedder = Embedder::new(Box::new(MockEmbedder { dim: 2 }), 2, true);
        let inputs: Vec<String> = (0..5).map(|i| format!("text{i}")).collect();
        let result = embedder.call(&inputs, None, None, None).await.unwrap();
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_embedder_empty() {
        let embedder = Embedder::with_defaults(Box::new(MockEmbedder { dim: 3 }));
        let result = embedder.call(&[], None, None, None).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_embedder_kwargs_passthrough() {
        struct KwargsCapture;

        #[async_trait]
        impl EmbeddingFunction for KwargsCapture {
            async fn embed(
                &self,
                texts: &[String],
                kwargs: Option<&HashMap<String, serde_json::Value>>,
            ) -> Result<Vec<Vec<f32>>> {
                // Return dimensionality based on whether kwargs was provided
                let dim = if kwargs.is_some() { 2 } else { 1 };
                Ok(texts.iter().map(|_| vec![1.0; dim]).collect())
            }
        }

        let embedder = Embedder::with_defaults(Box::new(KwargsCapture));

        // Without kwargs
        let result = embedder
            .call(&["test".to_string()], None, None, None)
            .await
            .unwrap();
        assert_eq!(result[0].len(), 1);

        // With kwargs
        let mut kw = HashMap::new();
        kw.insert("model".to_string(), serde_json::json!("test-model"));
        let result = embedder
            .call(&["test".to_string()], None, None, Some(kw))
            .await
            .unwrap();
        assert_eq!(result[0].len(), 2);
    }

    #[tokio::test]
    async fn test_embedder_batch_size_override() {
        // Track number of batches via embed call count
        use std::sync::atomic::{AtomicU32, Ordering};
        struct BatchCounter {
            call_count: AtomicU32,
        }

        #[async_trait]
        impl EmbeddingFunction for BatchCounter {
            async fn embed(
                &self,
                texts: &[String],
                _kwargs: Option<&HashMap<String, serde_json::Value>>,
            ) -> Result<Vec<Vec<f32>>> {
                self.call_count.fetch_add(1, Ordering::SeqCst);
                Ok(texts.iter().map(|_| vec![1.0]).collect())
            }
        }

        let counter = Box::new(BatchCounter {
            call_count: AtomicU32::new(0),
        });
        let counter_ref = &counter.call_count as *const AtomicU32;
        let embedder = Embedder::new(counter, 100, true); // default batch 100

        let inputs: Vec<String> = (0..5).map(|i| format!("text{i}")).collect();

        // Override batch_size to 2 → should produce 3 batches (2+2+1)
        embedder.call(&inputs, Some(2), None, None).await.unwrap();
        let calls = unsafe { &*counter_ref }.load(Ordering::SeqCst);
        assert_eq!(calls, 3);
    }
}
