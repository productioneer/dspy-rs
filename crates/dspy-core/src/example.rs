//! Example — flexible key-value data container for training data, demos, and predictions.
//! Python equivalent: dspy/primitives/example.py

use crate::value::Value;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::Index;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    store: HashMap<String, Value>,
    input_keys: Option<HashSet<String>>,
    #[serde(skip)]
    demos: Vec<Example>,
}

impl Example {
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            input_keys: None,
            demos: Vec::new(),
        }
    }

    pub fn from_map(data: HashMap<String, Value>) -> Self {
        Self {
            store: data,
            input_keys: None,
            demos: Vec::new(),
        }
    }

    // Field access
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.store.get(key)
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.store.get(key).and_then(|v| v.as_str())
    }

    pub fn set(&mut self, key: impl Into<String>, value: impl Into<Value>) -> &mut Self {
        self.store.insert(key.into(), value.into());
        self
    }

    pub fn has(&self, key: &str) -> bool {
        self.store.contains_key(key)
    }

    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.store.remove(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.store.keys()
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    pub fn to_map(&self) -> &HashMap<String, Value> {
        &self.store
    }

    pub fn into_map(self) -> HashMap<String, Value> {
        self.store
    }

    // Input/output separation
    pub fn with_inputs(mut self, keys: &[&str]) -> Self {
        self.input_keys = Some(keys.iter().map(|k| k.to_string()).collect());
        self
    }

    pub fn inputs(&self) -> Example {
        match &self.input_keys {
            Some(keys) => {
                let data: HashMap<String, Value> = self
                    .store
                    .iter()
                    .filter(|(k, _)| keys.contains(k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                let mut ex = Example::from_map(data);
                ex.input_keys = self.input_keys.clone();
                ex
            }
            None => self.clone(),
        }
    }

    pub fn labels(&self) -> Example {
        match &self.input_keys {
            Some(keys) => {
                let data: HashMap<String, Value> = self
                    .store
                    .iter()
                    .filter(|(k, _)| !keys.contains(k.as_str()))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                Example::from_map(data)
            }
            None => Example::new(),
        }
    }

    pub fn input_keys(&self) -> Option<&HashSet<String>> {
        self.input_keys.as_ref()
    }

    // Demos
    pub fn demos(&self) -> &[Example] {
        &self.demos
    }

    pub fn demos_mut(&mut self) -> &mut Vec<Example> {
        &mut self.demos
    }

    pub fn set_demos(&mut self, demos: Vec<Example>) {
        self.demos = demos;
    }

    // Builder pattern
    pub fn field(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.store.insert(key.into(), value.into());
        self
    }
}

impl Default for Example {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<&str> for Example {
    type Output = Value;
    fn index(&self, key: &str) -> &Value {
        self.store
            .get(key)
            .unwrap_or_else(|| panic!("Example has no field '{key}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_and_field_builder() {
        let ex = Example::new()
            .field("question", "What is 2+2?")
            .field("answer", "4");
        assert_eq!(ex.get_str("question"), Some("What is 2+2?"));
        assert_eq!(ex.get_str("answer"), Some("4"));
        assert_eq!(ex.len(), 2);
    }

    #[test]
    fn test_set_and_get() {
        let mut ex = Example::new();
        ex.set("name", "Alice");
        assert!(ex.has("name"));
        assert!(!ex.has("age"));
        assert_eq!(ex["name"].as_str(), Some("Alice"));
    }

    #[test]
    fn test_with_inputs_and_labels() {
        let ex = Example::new()
            .field("question", "What?")
            .field("context", "Some context")
            .field("answer", "42")
            .with_inputs(&["question", "context"]);

        let inputs = ex.inputs();
        assert_eq!(inputs.len(), 2);
        assert!(inputs.has("question"));
        assert!(inputs.has("context"));
        assert!(!inputs.has("answer"));

        let labels = ex.labels();
        assert_eq!(labels.len(), 1);
        assert!(labels.has("answer"));
    }

    #[test]
    fn test_inputs_without_input_keys_returns_clone() {
        let ex = Example::new().field("a", "1").field("b", "2");
        let inputs = ex.inputs();
        assert_eq!(inputs.len(), 2);
    }

    #[test]
    fn test_labels_without_input_keys_returns_empty() {
        let ex = Example::new().field("a", "1");
        let labels = ex.labels();
        assert_eq!(labels.len(), 0);
    }

    #[test]
    fn test_remove() {
        let mut ex = Example::new().field("a", "1").field("b", "2");
        let removed = ex.remove("a");
        assert!(removed.is_some());
        assert!(!ex.has("a"));
        assert_eq!(ex.len(), 1);
    }

    #[test]
    fn test_demos() {
        let mut ex = Example::new().field("q", "test");
        let demo = Example::new().field("q", "demo").field("a", "ans");
        ex.demos_mut().push(demo);
        assert_eq!(ex.demos().len(), 1);
        assert_eq!(ex.demos()[0].get_str("q"), Some("demo"));
    }

    #[test]
    fn test_from_map() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), Value::from("val"));
        let ex = Example::from_map(map);
        assert_eq!(ex.get_str("key"), Some("val"));
    }

    #[test]
    fn test_clone_preserves_input_keys() {
        let ex = Example::new()
            .field("q", "test")
            .field("a", "ans")
            .with_inputs(&["q"]);
        let cloned = ex.clone();
        assert_eq!(cloned.inputs().len(), 1);
        assert!(cloned.inputs().has("q"));
    }

    #[test]
    #[should_panic(expected = "Example has no field 'missing'")]
    fn test_index_panics_on_missing() {
        let ex = Example::new();
        let _ = &ex["missing"];
    }

    #[test]
    fn test_serde_roundtrip() {
        let ex = Example::new()
            .field("question", "What?")
            .field("count", Value::Integer(42));
        let json = serde_json::to_string(&ex).unwrap();
        let ex2: Example = serde_json::from_str(&json).unwrap();
        assert_eq!(ex2.get_str("question"), Some("What?"));
        assert_eq!(ex2.get("count").and_then(|v| v.as_i64()), Some(42));
        // demos are not serialized (serde skip)
        assert!(ex2.demos().is_empty());
    }
}
