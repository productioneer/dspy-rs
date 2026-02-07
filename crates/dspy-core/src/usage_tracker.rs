//! Usage Tracking — Token and cost tracking for LM calls.
//!
//! UsageTracker accumulates usage data (prompt tokens, completion tokens, etc.)
//! per LM model. Matches Python DSPy's dspy.utils.usage_tracker interface.

use std::collections::HashMap;

/// Tracks LM usage data within a context.
pub struct UsageTracker {
    /// Map of LM name to list of usage entries.
    pub usage_data: HashMap<String, Vec<serde_json::Value>>,
}

impl UsageTracker {
    /// Create a new empty usage tracker.
    pub fn new() -> Self {
        Self {
            usage_data: HashMap::new(),
        }
    }

    /// Add a usage entry for a specific LM.
    pub fn add_usage(&mut self, lm: &str, usage_entry: serde_json::Value) {
        if let Some(obj) = usage_entry.as_object() {
            if obj.is_empty() {
                return;
            }
        }
        self.usage_data
            .entry(lm.to_string())
            .or_default()
            .push(usage_entry);
    }

    /// Calculate total tokens from all tracked usage, aggregated by LM.
    pub fn get_total_tokens(&self) -> HashMap<String, serde_json::Value> {
        let mut result = HashMap::new();
        for (lm, entries) in &self.usage_data {
            let mut total = serde_json::Value::Object(serde_json::Map::new());
            for entry in entries {
                total = merge_usage_entries(&total, entry);
            }
            result.insert(lm.clone(), total);
        }
        result
    }
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge two usage entry values (recursively sum numeric fields, recurse into objects).
fn merge_usage_entries(a: &serde_json::Value, b: &serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Object(ma), serde_json::Value::Object(mb)) => {
            let mut result = mb.clone();
            for (k, v) in ma {
                let current = result.get(k);
                match (current, v) {
                    (Some(cv), _)
                        if cv.is_object() || v.is_object() =>
                    {
                        result.insert(k.clone(), merge_usage_entries(cv, v));
                    }
                    (Some(cv), _) => {
                        let sum = cv.as_f64().unwrap_or(0.0) + v.as_f64().unwrap_or(0.0);
                        result.insert(
                            k.clone(),
                            serde_json::Value::Number(
                                serde_json::Number::from_f64(sum)
                                    .unwrap_or(serde_json::Number::from(0)),
                            ),
                        );
                    }
                    (None, _) => {
                        result.insert(k.clone(), v.clone());
                    }
                }
            }
            serde_json::Value::Object(result)
        }
        (serde_json::Value::Object(_), _) => a.clone(),
        (_, serde_json::Value::Object(_)) => b.clone(),
        _ => {
            let sum = a.as_f64().unwrap_or(0.0) + b.as_f64().unwrap_or(0.0);
            serde_json::Value::Number(
                serde_json::Number::from_f64(sum).unwrap_or(serde_json::Number::from(0)),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_add_usage() {
        let mut tracker = UsageTracker::new();
        tracker.add_usage("gpt-4", json!({"prompt_tokens": 100, "completion_tokens": 200}));
        tracker.add_usage("gpt-4", json!({"prompt_tokens": 50, "completion_tokens": 100}));
        assert_eq!(tracker.usage_data["gpt-4"].len(), 2);
    }

    #[test]
    fn test_empty_entry_skipped() {
        let mut tracker = UsageTracker::new();
        tracker.add_usage("gpt-4", json!({}));
        assert!(!tracker.usage_data.contains_key("gpt-4"));
    }

    #[test]
    fn test_get_total_tokens() {
        let mut tracker = UsageTracker::new();
        tracker.add_usage("gpt-4", json!({"prompt_tokens": 100, "completion_tokens": 200}));
        tracker.add_usage("gpt-4", json!({"prompt_tokens": 50, "completion_tokens": 100}));
        tracker.add_usage("gpt-3.5", json!({"prompt_tokens": 10}));

        let totals = tracker.get_total_tokens();
        assert_eq!(totals["gpt-4"]["prompt_tokens"], 150.0);
        assert_eq!(totals["gpt-4"]["completion_tokens"], 300.0);
        assert_eq!(totals["gpt-3.5"]["prompt_tokens"], 10.0);
    }

    #[test]
    fn test_merge_nested() {
        let mut tracker = UsageTracker::new();
        tracker.add_usage("m", json!({"tokens": {"cached": 5, "new": 10}}));
        tracker.add_usage("m", json!({"tokens": {"cached": 3, "new": 7}}));

        let totals = tracker.get_total_tokens();
        assert_eq!(totals["m"]["tokens"]["cached"], 8.0);
        assert_eq!(totals["m"]["tokens"]["new"], 17.0);
    }
}
