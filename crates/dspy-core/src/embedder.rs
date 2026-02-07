//! Embedder — Unified embedding interface.
//!
//! Supports custom embedding functions with batching.
//! Matches Python DSPy's dspy.clients.embedding.Embedder interface.

use crate::error::{DspyError, Result};
use async_trait::async_trait;

/// Trait for embedding functions.
#[async_trait]
pub trait EmbeddingFunction: Send + Sync {
    /// Compute embeddings for a batch of texts.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// DSPy Embedder — unified embedding interface with batching.
pub struct Embedder {
    model: Box<dyn EmbeddingFunction>,
    batch_size: usize,
    caching: bool,
}

impl Embedder {
    /// Create a new Embedder with a custom embedding function.
    pub fn new(model: Box<dyn EmbeddingFunction>, batch_size: usize, caching: bool) -> Self {
        Self {
            model,
            batch_size,
            caching,
        }
    }

    /// Create a new Embedder with default settings.
    pub fn with_defaults(model: Box<dyn EmbeddingFunction>) -> Self {
        Self::new(model, 200, true)
    }

    /// Whether caching is enabled.
    pub fn caching(&self) -> bool {
        self.caching
    }

    /// Compute embeddings for the given inputs.
    pub async fn call(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        let mut all_embeddings = Vec::with_capacity(inputs.len());

        for batch in inputs.chunks(self.batch_size) {
            let batch_vec: Vec<String> = batch.to_vec();
            let embeddings = self.model.embed(&batch_vec).await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    /// Compute embedding for a single input.
    pub async fn call_single(&self, input: &str) -> Result<Vec<f32>> {
        let results = self.call(&[input.to_string()]).await?;
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
        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
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
        let result = embedder.call(&inputs).await.unwrap();

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
        let result = embedder.call(&inputs).await.unwrap();
        assert_eq!(result.len(), 5);
    }

    #[tokio::test]
    async fn test_embedder_empty() {
        let embedder = Embedder::with_defaults(Box::new(MockEmbedder { dim: 3 }));
        let result = embedder.call(&[]).await.unwrap();
        assert!(result.is_empty());
    }
}
