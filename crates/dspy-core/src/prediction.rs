//! Prediction — extends Example with LM completion data and score.
//! Python equivalent: dspy/primitives/prediction.py

use crate::example::Example;
use crate::signature::Signature;
use crate::value::Value;
use serde::Serialize;
use std::collections::HashMap;
use std::ops::Deref;

#[derive(Debug, Clone, Serialize)]
pub struct Prediction {
    pub example: Example,
    completions: HashMap<String, Vec<Value>>,
    lm_usage: HashMap<String, u64>,
    score: Option<f64>,
}

impl Prediction {
    pub fn new(data: HashMap<String, Value>) -> Self {
        Self {
            example: Example::from_map(data),
            completions: HashMap::new(),
            lm_usage: HashMap::new(),
            score: None,
        }
    }

    pub fn from_example(example: Example) -> Self {
        Self {
            example,
            completions: HashMap::new(),
            lm_usage: HashMap::new(),
            score: None,
        }
    }

    /// Build a Prediction from a list of completions (one map per completion).
    /// The first completion's values populate the example, all values stored in completions.
    pub fn from_completions(
        completions_list: Vec<HashMap<String, Value>>,
        signature: Option<&Signature>,
    ) -> Self {
        if completions_list.is_empty() {
            return Self::new(HashMap::new());
        }

        // First completion populates the main example
        let primary = completions_list[0].clone();

        // Build completions map: for each output field, collect all values across completions
        let mut completions_map: HashMap<String, Vec<Value>> = HashMap::new();

        let output_keys: Vec<String> = if let Some(sig) = signature {
            sig.output_fields().map(|(k, _)| k.clone()).collect()
        } else {
            primary.keys().cloned().collect()
        };

        for key in &output_keys {
            let vals: Vec<Value> = completions_list
                .iter()
                .filter_map(|c| c.get(key).cloned())
                .collect();
            completions_map.insert(key.clone(), vals);
        }

        Self {
            example: Example::from_map(primary),
            completions: completions_map,
            lm_usage: HashMap::new(),
            score: None,
        }
    }

    // Score
    pub fn score(&self) -> Option<f64> {
        self.score
    }

    pub fn set_score(&mut self, score: f64) {
        self.score = Some(score);
    }

    // Completions
    pub fn completions(&self) -> &HashMap<String, Vec<Value>> {
        &self.completions
    }

    // LM usage
    pub fn lm_usage(&self) -> &HashMap<String, u64> {
        &self.lm_usage
    }

    pub fn set_lm_usage(&mut self, usage: HashMap<String, u64>) {
        self.lm_usage = usage;
    }
}

/// Delegate field access to the underlying Example
impl Deref for Prediction {
    type Target = Example;
    fn deref(&self) -> &Example {
        &self.example
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_prediction() {
        let mut data = HashMap::new();
        data.insert("answer".to_string(), Value::from("42"));
        let pred = Prediction::new(data);
        assert_eq!(pred.get_str("answer"), Some("42"));
        assert!(pred.score().is_none());
    }

    #[test]
    fn test_score() {
        let pred_data = HashMap::new();
        let mut pred = Prediction::new(pred_data);
        assert!(pred.score().is_none());
        pred.set_score(0.95);
        assert_eq!(pred.score(), Some(0.95));
    }

    #[test]
    fn test_deref_to_example() {
        let mut data = HashMap::new();
        data.insert("question".to_string(), Value::from("What?"));
        data.insert("answer".to_string(), Value::from("42"));
        let pred = Prediction::new(data);
        // Access via Deref
        assert!(pred.has("question"));
        assert_eq!(pred.get_str("answer"), Some("42"));
    }

    #[test]
    fn test_from_completions() {
        let c1: HashMap<String, Value> =
            [("answer".to_string(), Value::from("42"))].into();
        let c2: HashMap<String, Value> =
            [("answer".to_string(), Value::from("43"))].into();
        let pred = Prediction::from_completions(vec![c1, c2], None);
        // Primary value from first completion
        assert_eq!(pred.get_str("answer"), Some("42"));
        // All completions stored
        assert_eq!(pred.completions()["answer"].len(), 2);
    }

    #[test]
    fn test_from_completions_empty() {
        let pred = Prediction::from_completions(vec![], None);
        assert!(pred.example.is_empty());
    }

    #[test]
    fn test_from_example() {
        let ex = Example::new().field("q", "test");
        let pred = Prediction::from_example(ex);
        assert_eq!(pred.get_str("q"), Some("test"));
    }

    #[test]
    fn test_clone() {
        let mut data = HashMap::new();
        data.insert("a".to_string(), Value::from("1"));
        let mut pred = Prediction::new(data);
        pred.set_score(0.5);
        let cloned = pred.clone();
        assert_eq!(cloned.score(), Some(0.5));
        assert_eq!(cloned.get_str("a"), Some("1"));
    }
}
