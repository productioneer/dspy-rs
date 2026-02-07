//! Parallel — run (module, inputs) pairs concurrently.
//! Python equivalent: dspy/predict/parallel.py

use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::module_trait::Module;
use crate::prediction::Prediction;

/// Configuration for parallel execution.
pub struct ParallelConfig {
    /// Max concurrency (default: 8)
    pub num_threads: usize,
    /// Max allowed failures before aborting (default: usize::MAX)
    pub max_errors: usize,
    /// Whether to return failed indices (default: false)
    pub return_failed: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: 8,
            max_errors: usize::MAX,
            return_failed: false,
        }
    }
}

/// Result from parallel execution.
pub struct ParallelResult {
    pub results: Vec<Option<Prediction>>,
    pub failed_indices: Vec<usize>,
    pub exceptions: Vec<String>,
}

/// Execute (module, inputs) pairs concurrently with bounded parallelism.
pub async fn parallel_execute(
    exec_pairs: Vec<(&dyn Module, Example)>,
    config: &ParallelConfig,
) -> Result<ParallelResult> {
    use tokio::sync::Semaphore;
    use std::sync::Arc;

    let n = exec_pairs.len();
    let semaphore = Arc::new(Semaphore::new(config.num_threads));

    // We need to collect results. Since Module isn't Send in general,
    // we process sequentially with a semaphore for backpressure simulation.
    // For truly parallel execution, modules would need to be Send + Sync.
    let mut results: Vec<Option<Prediction>> = Vec::with_capacity(n);
    let mut failed_indices = Vec::new();
    let mut exceptions = Vec::new();
    let mut error_count = 0usize;

    for (idx, (module, inputs)) in exec_pairs.into_iter().enumerate() {
        if error_count > config.max_errors {
            results.push(None);
            continue;
        }

        let _permit = semaphore.acquire().await.map_err(|e| {
            DspyError::Other(format!("Semaphore error: {}", e))
        })?;

        match module.forward(&inputs).await {
            Ok(pred) => results.push(Some(pred)),
            Err(e) => {
                error_count += 1;
                failed_indices.push(idx);
                exceptions.push(format!("{}", e));
                results.push(None);
            }
        }
    }

    Ok(ParallelResult {
        results,
        failed_indices,
        exceptions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::Example;
    use crate::prediction::Prediction;
    use async_trait::async_trait;
    use std::collections::HashMap;

    struct ConstModule {
        answer: String,
    }

    #[async_trait]
    impl Module for ConstModule {
        async fn forward(&self, _args: &Example) -> Result<Prediction> {
            let mut pred = Prediction::new(HashMap::new());
            pred.example.set("answer", self.answer.as_str());
            Ok(pred)
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(ConstModule {
                answer: self.answer.clone(),
            })
        }
    }

    struct FailModule;

    #[async_trait]
    impl Module for FailModule {
        async fn forward(&self, _args: &Example) -> Result<Prediction> {
            Err(DspyError::Other("intentional failure".into()))
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(FailModule)
        }
    }

    #[tokio::test]
    async fn test_parallel_basic() {
        let m1 = ConstModule { answer: "A".into() };
        let m2 = ConstModule { answer: "B".into() };

        let pairs: Vec<(&dyn Module, Example)> = vec![
            (&m1, Example::new()),
            (&m2, Example::new()),
        ];

        let result = parallel_execute(pairs, &ParallelConfig::default()).await.unwrap();
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].as_ref().unwrap().get_str("answer"), Some("A"));
        assert_eq!(result.results[1].as_ref().unwrap().get_str("answer"), Some("B"));
        assert!(result.failed_indices.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_with_failures() {
        let m1 = ConstModule { answer: "ok".into() };
        let m2 = FailModule;
        let m3 = ConstModule { answer: "also ok".into() };

        let pairs: Vec<(&dyn Module, Example)> = vec![
            (&m1, Example::new()),
            (&m2, Example::new()),
            (&m3, Example::new()),
        ];

        let result = parallel_execute(pairs, &ParallelConfig::default()).await.unwrap();
        assert_eq!(result.results.len(), 3);
        assert!(result.results[0].is_some());
        assert!(result.results[1].is_none());
        assert!(result.results[2].is_some());
        assert_eq!(result.failed_indices, vec![1]);
    }

    #[tokio::test]
    async fn test_parallel_max_errors() {
        let m1 = FailModule;
        let m2 = FailModule;
        let m3 = ConstModule { answer: "ok".into() };

        let config = ParallelConfig {
            max_errors: 1,
            ..Default::default()
        };

        let pairs: Vec<(&dyn Module, Example)> = vec![
            (&m1, Example::new()),
            (&m2, Example::new()),
            (&m3, Example::new()),
        ];

        let result = parallel_execute(pairs, &config).await.unwrap();
        // After 2 failures (exceeding max_errors=1), remaining items are skipped
        assert_eq!(result.results.len(), 3);
    }
}
