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

    /// Prepare multiple train/eval sets indexed by seed for reproducible cross-validation.
    /// Mirrors Python DSPy's Dataset.prepare_by_seed.
    pub fn prepare_by_seed(
        &mut self,
        train_seeds: Option<&[u64]>,
        train_size: usize,
        dev_size: usize,
        divide_eval_per_seed: bool,
        eval_seed: u64,
    ) -> (Vec<Vec<Example>>, Vec<Vec<Example>>) {
        let seeds = train_seeds.unwrap_or(&[1, 2, 3, 4, 5]);

        // Set up initial config for eval
        self.config.train_size = Some(train_size);
        self.config.eval_seed = eval_seed;
        self.config.dev_size = Some(dev_size);
        self.config.test_size = Some(0);

        let eval_set = self.dev();
        let mut eval_sets: Vec<Vec<Example>> = Vec::new();
        let mut train_sets: Vec<Vec<Example>> = Vec::new();

        let examples_per_seed = if divide_eval_per_seed {
            dev_size / seeds.len()
        } else {
            dev_size
        };
        let mut eval_offset = 0;

        for &train_seed in seeds {
            self.config.train_seed = train_seed;
            self.config.train_size = Some(train_size);
            self.config.eval_seed = eval_seed;
            self.config.dev_size = Some(dev_size);
            self.config.test_size = Some(0);

            let end = (eval_offset + examples_per_seed).min(eval_set.len());
            eval_sets.push(eval_set[eval_offset..end].to_vec());
            train_sets.push(self.train());

            if divide_eval_per_seed {
                eval_offset += examples_per_seed;
            }
        }

        (train_sets, eval_sets)
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
    fn test_prepare_by_seed() {
        let mut ds = Dataset::new(
            "test",
            DatasetConfig {
                train_size: Some(5),
                dev_size: Some(90),
                eval_seed: 42,
                ..Default::default()
            },
            make_data(50),
            make_data(100),
            Vec::new(),
        );

        let (train_sets, eval_sets) = ds.prepare_by_seed(
            Some(&[1, 2, 3]),
            5,
            90,
            true,
            42,
        );

        assert_eq!(train_sets.len(), 3);
        assert_eq!(eval_sets.len(), 3);

        // Each train set has 5 examples
        for ts in &train_sets {
            assert_eq!(ts.len(), 5);
        }

        // Each eval set has 30 (90/3) examples
        for es in &eval_sets {
            assert_eq!(es.len(), 30);
        }

        // Different seeds produce different train orders
        let t1: Vec<_> = train_sets[0].iter().map(|e| e.get_str("question").unwrap_or("").to_string()).collect();
        let t2: Vec<_> = train_sets[1].iter().map(|e| e.get_str("question").unwrap_or("").to_string()).collect();
        assert_ne!(t1, t2, "Different seeds should produce different train sets");
    }

    #[test]
    fn test_prepare_by_seed_no_divide() {
        let mut ds = Dataset::new(
            "test",
            DatasetConfig::default(),
            make_data(50),
            make_data(100),
            Vec::new(),
        );

        let (_, eval_sets) = ds.prepare_by_seed(
            Some(&[1, 2]),
            5,
            50,
            false,
            42,
        );

        // Each eval set gets the full devSize
        for es in &eval_sets {
            assert_eq!(es.len(), 50);
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
