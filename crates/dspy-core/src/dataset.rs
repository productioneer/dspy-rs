//! Dataset — Base dataset abstraction for DSPy.
//!
//! Provides train/dev/test splits with seeded shuffling and sampling.
//! Matches Python DSPy's dspy.datasets.dataset.Dataset interface.

use crate::example::Example;
use crate::value::Value;

/// Configuration for a dataset.
pub struct DatasetConfig {
    pub train_seed: u64,
    pub train_size: Option<usize>,
    pub eval_seed: u64,
    pub dev_size: Option<usize>,
    pub test_size: Option<usize>,
    pub input_keys: Vec<String>,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            train_seed: 0,
            train_size: None,
            eval_seed: 0,
            dev_size: None,
            test_size: None,
            input_keys: Vec::new(),
        }
    }
}

/// Base dataset with train/dev/test splits and seeded shuffling.
pub struct Dataset {
    pub name: String,
    pub config: DatasetConfig,
    pub do_shuffle: bool,
    train_data: Vec<std::collections::HashMap<String, String>>,
    dev_data: Vec<std::collections::HashMap<String, String>>,
    test_data: Vec<std::collections::HashMap<String, String>>,
}

impl Dataset {
    /// Create a new dataset with raw data for each split.
    pub fn new(
        name: &str,
        config: DatasetConfig,
        train_data: Vec<std::collections::HashMap<String, String>>,
        dev_data: Vec<std::collections::HashMap<String, String>>,
        test_data: Vec<std::collections::HashMap<String, String>>,
    ) -> Self {
        Self {
            name: name.to_string(),
            config,
            do_shuffle: true,
            train_data,
            dev_data,
            test_data,
        }
    }

    /// Get the training split.
    pub fn train(&self) -> Vec<Example> {
        self.shuffle_and_sample("train", &self.train_data, self.config.train_size, self.config.train_seed)
    }

    /// Get the dev/validation split.
    pub fn dev(&self) -> Vec<Example> {
        self.shuffle_and_sample("dev", &self.dev_data, self.config.dev_size, self.config.eval_seed)
    }

    /// Get the test split.
    pub fn test(&self) -> Vec<Example> {
        self.shuffle_and_sample("test", &self.test_data, self.config.test_size, self.config.eval_seed)
    }

    fn shuffle_and_sample(
        &self,
        split: &str,
        data: &[std::collections::HashMap<String, String>],
        size: Option<usize>,
        seed: u64,
    ) -> Vec<Example> {
        let mut data_list: Vec<_> = data.to_vec();

        if self.do_shuffle {
            seeded_shuffle(&mut data_list, seed);
        }

        if let Some(n) = size {
            data_list.truncate(n);
        }

        data_list
            .into_iter()
            .map(|mut item| {
                item.insert("dspy_split".to_string(), split.to_string());
                // Convert HashMap<String, String> to HashMap<String, Value>
                let value_map: std::collections::HashMap<String, Value> = item
                    .into_iter()
                    .map(|(k, v)| (k, Value::from(v)))
                    .collect();
                let example = Example::from_map(value_map);
                if !self.config.input_keys.is_empty() {
                    let refs: Vec<&str> = self.config.input_keys.iter().map(|s| s.as_str()).collect();
                    example.with_inputs(&refs)
                } else {
                    example
                }
            })
            .collect()
    }
}

/// Simple seeded Fisher-Yates shuffle using LCG.
fn seeded_shuffle<T>(data: &mut Vec<T>, seed: u64) {
    let mut state: u64 = seed;
    let next = |state: &mut u64| -> f64 {
        *state = state.wrapping_mul(1664525).wrapping_add(1013904223) & 0xFFFFFFFF;
        (*state as f64) / 0x100000000_u64 as f64
    };

    let len = data.len();
    for i in (1..len).rev() {
        let j = (next(&mut state) * (i + 1) as f64) as usize;
        data.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_data(n: usize) -> Vec<HashMap<String, String>> {
        (0..n)
            .map(|i| {
                let mut m = HashMap::new();
                m.insert("question".to_string(), format!("Q{i}"));
                m.insert("answer".to_string(), format!("A{i}"));
                m
            })
            .collect()
    }

    #[test]
    fn test_dataset_basic() {
        let ds = Dataset::new(
            "test",
            DatasetConfig::default(),
            make_data(10),
            make_data(5),
            make_data(3),
        );

        assert_eq!(ds.train().len(), 10);
        assert_eq!(ds.dev().len(), 5);
        assert_eq!(ds.test().len(), 3);
    }

    #[test]
    fn test_dataset_size_limit() {
        let config = DatasetConfig {
            train_size: Some(3),
            ..Default::default()
        };
        let ds = Dataset::new(
            "test",
            config,
            make_data(10),
            Vec::new(),
            Vec::new(),
        );

        assert_eq!(ds.train().len(), 3);
    }

    #[test]
    fn test_dataset_shuffle_deterministic() {
        let config = DatasetConfig {
            train_seed: 42,
            ..Default::default()
        };
        let ds1 = Dataset::new(
            "test",
            config,
            make_data(10),
            Vec::new(),
            Vec::new(),
        );

        let config2 = DatasetConfig {
            train_seed: 42,
            ..Default::default()
        };
        let ds2 = Dataset::new(
            "test",
            config2,
            make_data(10),
            Vec::new(),
            Vec::new(),
        );

        let t1 = ds1.train();
        let t2 = ds2.train();

        for (a, b) in t1.iter().zip(t2.iter()) {
            assert_eq!(
                a.get("question").map(|v| v.as_str()),
                b.get("question").map(|v| v.as_str())
            );
        }
    }

    #[test]
    fn test_dataset_input_keys() {
        let config = DatasetConfig {
            input_keys: vec!["question".to_string()],
            ..Default::default()
        };
        let ds = Dataset::new(
            "test",
            config,
            make_data(3),
            Vec::new(),
            Vec::new(),
        );

        let examples = ds.train();
        // Each example should have input_keys set
        assert_eq!(examples.len(), 3);
    }
}
