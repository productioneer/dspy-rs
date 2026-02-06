//! KNN — k-nearest neighbor retriever over embedded examples.
//!
//! Pre-computes embeddings of trainset examples, then at query time finds the
//! k most similar examples by cosine similarity (dot product of normalized vectors).
//!
//! Python equivalent: dspy/predict/knn.py

use crate::Example;
use std::sync::Arc;

/// An embedding function that converts strings to vectors.
/// Takes a batch of strings and returns a vector of embeddings.
pub type Embedder = Arc<dyn Fn(&[String]) -> Vec<Vec<f32>> + Send + Sync>;

/// K-nearest neighbor retriever over a training set.
///
/// Pre-computes embedding vectors for all training examples (using their input
/// fields as text), then retrieves the k most similar examples for a given query.
pub struct KNN {
    k: usize,
    trainset: Vec<Example>,
    trainset_vectors: Vec<Vec<f32>>,
    embedder: Embedder,
}

impl KNN {
    /// Create a new KNN retriever.
    ///
    /// Immediately embeds the entire trainset. Each example is serialized as
    /// `"key1: value1 | key2: value2"` using only input keys (or all keys if
    /// no input keys are set).
    pub fn new(k: usize, trainset: Vec<Example>, embedder: Embedder) -> Self {
        let texts: Vec<String> = trainset
            .iter()
            .map(|ex| Self::example_to_text(ex))
            .collect();

        let trainset_vectors = if texts.is_empty() {
            vec![]
        } else {
            let raw = embedder(&texts);
            raw.into_iter().map(|v| normalize(&v)).collect()
        };

        Self {
            k,
            trainset,
            trainset_vectors,
            embedder,
        }
    }

    /// Find the k nearest neighbors for the given input fields.
    pub fn query(&self, fields: &[(&str, &str)]) -> Vec<Example> {
        if self.trainset.is_empty() || self.k == 0 {
            return vec![];
        }

        // Build query text
        let query_text = fields
            .iter()
            .map(|(k, v)| format!("{}: {}", k, v))
            .collect::<Vec<_>>()
            .join(" | ");

        let query_vecs = (self.embedder)(&[query_text]);
        if query_vecs.is_empty() {
            return vec![];
        }
        let query_vec = normalize(&query_vecs[0]);

        // Compute dot-product scores
        let mut scored: Vec<(usize, f32)> = self
            .trainset_vectors
            .iter()
            .enumerate()
            .map(|(i, tv)| (i, dot(&query_vec, tv)))
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top-k
        scored
            .iter()
            .take(self.k)
            .map(|(idx, _)| self.trainset[*idx].clone())
            .collect()
    }

    /// Serialize an example's input fields to a text string.
    fn example_to_text(ex: &Example) -> String {
        let inputs = ex.inputs();
        let keys_and_vals: Vec<String> = if inputs.keys().count() > 0 {
            inputs
                .keys()
                .map(|k| format!("{}: {}", k, inputs.get_str(k).unwrap_or_default()))
                .collect()
        } else {
            ex.keys()
                .map(|k| format!("{}: {}", k, ex.get_str(k).unwrap_or_default()))
                .collect()
        };
        keys_and_vals.join(" | ")
    }
}

/// L2-normalize a vector. Returns zero vector if input is all zeros.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-9 {
        vec![0.0; v.len()]
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Dot product of two vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Mock embedder that assigns distinct fixed vectors based on content.
    fn mock_embedder() -> Embedder {
        Arc::new(|texts: &[String]| {
            texts
                .iter()
                .map(|t| {
                    // Assign clearly distinct vectors for known strings
                    if t.contains("Paris") {
                        vec![1.0, 0.0, 0.0]
                    } else if t.contains("Berlin") {
                        vec![0.0, 1.0, 0.0]
                    } else if t.contains("London") {
                        vec![0.0, 0.0, 1.0]
                    } else if t.contains("hello") || t.contains("art") {
                        vec![0.5, 0.5, 0.0]
                    } else if t.contains("world") || t.contains("zoo") {
                        vec![0.0, 0.5, 0.5]
                    } else {
                        vec![0.33, 0.33, 0.34]
                    }
                })
                .collect()
        })
    }

    #[test]
    fn test_knn_basic() {
        let trainset = vec![
            Example::new().field("question", "What is Paris?").with_inputs(&["question"]),
            Example::new().field("question", "What is Berlin?").with_inputs(&["question"]),
            Example::new().field("question", "What is London?").with_inputs(&["question"]),
        ];

        let knn = KNN::new(2, trainset, mock_embedder());
        let results = knn.query(&[("question", "What is Paris?")]);

        assert_eq!(results.len(), 2);
        // First result should be exact match (same embedding vector)
        assert_eq!(results[0].get_str("question").unwrap(), "What is Paris?");
    }

    #[test]
    fn test_knn_empty_trainset() {
        let knn = KNN::new(3, vec![], mock_embedder());
        let results = knn.query(&[("question", "anything")]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_knn_k_larger_than_trainset() {
        let trainset = vec![
            Example::new().field("q", "hello").with_inputs(&["q"]),
            Example::new().field("q", "world").with_inputs(&["q"]),
        ];

        let knn = KNN::new(10, trainset, mock_embedder());
        let results = knn.query(&[("q", "hello")]);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_knn_zero_k() {
        let trainset = vec![
            Example::new().field("q", "hello").with_inputs(&["q"]),
        ];

        let knn = KNN::new(0, trainset, mock_embedder());
        let results = knn.query(&[("q", "hello")]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_knn_similarity_ordering() {
        // Use an embedder where the vector is just [first_char_code, word_len]
        let embedder: Embedder = Arc::new(|texts: &[String]| {
            texts
                .iter()
                .map(|t| {
                    let first = t.chars().next().unwrap_or('a') as u32 as f32;
                    let len = t.len() as f32;
                    vec![first, len]
                })
                .collect()
        });

        let trainset = vec![
            Example::new().field("word", "apple").with_inputs(&["word"]),  // 'a'=97, len=5
            Example::new().field("word", "ant").with_inputs(&["word"]),    // 'a'=97, len=3
            Example::new().field("word", "zoo").with_inputs(&["word"]),    // 'z'=122, len=3
        ];

        let knn = KNN::new(3, trainset, embedder);
        let results = knn.query(&[("word", "art")]); // 'a'=97, len=3
        // "ant" has same first char 'a' and len 3 — most similar
        // "apple" has same first char 'a' but different len
        // "zoo" different first char
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0, 4.0];
        let n = normalize(&v);
        assert!((n[0] - 0.6).abs() < 1e-6);
        assert!((n[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let v = vec![0.0, 0.0, 0.0];
        let n = normalize(&v);
        assert!(n.iter().all(|x| *x == 0.0));
    }
}
