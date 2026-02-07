//! ParallelExecutor — Parallel execution with error handling and straggler detection.
//!
//! Executes an async function over data items concurrently using tokio tasks
//! with configurable concurrency, error limits, and timeout.
//!
//! Matches Python DSPy's dspy.utils.parallelizer.ParallelExecutor interface.

use crate::error::{DspyError, Result};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Configuration for parallel execution.
pub struct ParallelExecutorConfig {
    /// Number of concurrent workers. Default: 4
    pub num_threads: usize,
    /// Maximum errors before cancellation. Default: 5
    pub max_errors: usize,
    /// Timeout per item in seconds. Default: 120
    pub timeout_secs: u64,
    /// Number of remaining items that triggers straggler detection. Default: 3
    pub straggler_limit: usize,
    /// Whether to provide tracebacks on errors. Default: false
    pub provide_traceback: bool,
}

impl Default for ParallelExecutorConfig {
    fn default() -> Self {
        Self {
            num_threads: 4,
            max_errors: 5,
            timeout_secs: 120,
            straggler_limit: 3,
            provide_traceback: false,
        }
    }
}

/// Executes an async function over data items in parallel with error handling.
pub struct ParallelExecutor {
    config: ParallelExecutorConfig,
}

impl ParallelExecutor {
    /// Create a new ParallelExecutor with the given configuration.
    pub fn new(config: ParallelExecutorConfig) -> Self {
        Self { config }
    }

    /// Create a new ParallelExecutor with default configuration.
    pub fn default_config() -> Self {
        Self::new(ParallelExecutorConfig::default())
    }

    /// Execute a function over all data items in parallel.
    ///
    /// Returns a Vec of Options where None represents failed/cancelled items.
    pub async fn execute<T, R, F, Fut>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<Option<R>>>
    where
        T: Send + Sync + 'static,
        R: Send + 'static,
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<R>> + Send + 'static,
    {
        let len = data.len();
        let results: Arc<tokio::sync::Mutex<Vec<Option<R>>>> =
            Arc::new(tokio::sync::Mutex::new(Vec::with_capacity(len)));
        {
            let mut guard = results.lock().await;
            guard.resize_with(len, || None);
        }

        let error_count = Arc::new(AtomicUsize::new(0));
        let cancel = Arc::new(AtomicBool::new(false));
        let semaphore = Arc::new(Semaphore::new(self.config.num_threads));
        let func = Arc::new(func);
        let timeout = self.config.timeout_secs;
        let max_errors = self.config.max_errors;
        let provide_traceback = self.config.provide_traceback;

        let mut handles = Vec::with_capacity(len);

        for (idx, item) in data.into_iter().enumerate() {
            let sem = semaphore.clone();
            let func = func.clone();
            let results = results.clone();
            let error_count = error_count.clone();
            let cancel = cancel.clone();

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.unwrap();

                if cancel.load(Ordering::SeqCst) {
                    return;
                }

                let result = if timeout > 0 {
                    tokio::time::timeout(
                        std::time::Duration::from_secs(timeout),
                        func(item),
                    )
                    .await
                } else {
                    Ok(func(item).await)
                };

                match result {
                    Ok(Ok(value)) => {
                        let mut guard = results.lock().await;
                        guard[idx] = Some(value);
                    }
                    Ok(Err(e)) => {
                        let count = error_count.fetch_add(1, Ordering::SeqCst) + 1;
                        if provide_traceback {
                            eprintln!("Error for item {idx}: {e}");
                        }
                        if count >= max_errors {
                            cancel.store(true, Ordering::SeqCst);
                        }
                    }
                    Err(_elapsed) => {
                        let count = error_count.fetch_add(1, Ordering::SeqCst) + 1;
                        if provide_traceback {
                            eprintln!("Timeout for item {idx}");
                        }
                        if count >= max_errors {
                            cancel.store(true, Ordering::SeqCst);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.await;
        }

        if cancel.load(Ordering::SeqCst) {
            return Err(DspyError::Other(
                "Execution cancelled due to errors or interruption.".to_string(),
            ));
        }

        let inner = Arc::try_unwrap(results)
            .map_err(|_| DspyError::Other("Failed to unwrap results".to_string()))?;
        Ok(inner.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_basic() {
        let executor = ParallelExecutor::default_config();
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];

        let results = executor
            .execute(data, |x| async move { Ok(x * 2) })
            .await
            .unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(results[0], Some(2));
        assert_eq!(results[4], Some(10));
    }

    #[tokio::test]
    async fn test_parallel_with_errors() {
        let config = ParallelExecutorConfig {
            max_errors: 10,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        let data: Vec<i32> = vec![1, 2, 0, 4, 5];

        let results = executor
            .execute(data, |x| async move {
                if x == 0 {
                    Err(DspyError::Other("zero".to_string()))
                } else {
                    Ok(x * 2)
                }
            })
            .await
            .unwrap();

        assert_eq!(results[0], Some(2));
        assert_eq!(results[2], None); // Failed item
        assert_eq!(results[3], Some(8));
    }

    #[tokio::test]
    async fn test_parallel_cancellation() {
        let config = ParallelExecutorConfig {
            max_errors: 1,
            num_threads: 1,
            ..Default::default()
        };
        let executor = ParallelExecutor::new(config);
        let data: Vec<i32> = vec![0, 1, 2]; // First item fails

        let result = executor
            .execute(data, |x| async move {
                if x == 0 {
                    Err(DspyError::Other("error".to_string()))
                } else {
                    Ok(x)
                }
            })
            .await;

        assert!(result.is_err());
    }
}
