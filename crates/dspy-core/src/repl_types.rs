//! REPL data types for RLM and interpreter interactions.
//! Python equivalent: dspy/primitives/repl_types.py
//!
//! Types:
//! - REPLVariable: Metadata about variables available in the REPL
//! - REPLEntry: A single interaction (reasoning, code, output)
//! - REPLHistory: Container for the full interaction history (immutable)

use serde::{Deserialize, Serialize};

/// Metadata about a variable available in the REPL environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REPLVariable {
    pub name: String,
    pub type_name: String,
    pub desc: String,
    pub constraints: String,
    pub total_length: usize,
    pub preview: String,
}

/// Create a REPLVariable from an actual JSON value and optional field metadata.
pub fn create_repl_variable(
    name: &str,
    value: &serde_json::Value,
    desc: Option<&str>,
    constraints: Option<&str>,
    preview_chars: Option<usize>,
) -> REPLVariable {
    let preview_limit = preview_chars.unwrap_or(500);
    let value_str = match value {
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
            serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Null => String::new(),
        other => other.to_string(),
    };

    let is_truncated = value_str.len() > preview_limit;
    let preview = if is_truncated {
        format!("{}...", &value_str[..preview_limit])
    } else {
        value_str.clone()
    };

    REPLVariable {
        name: name.to_string(),
        type_name: get_type_name(value),
        desc: desc.unwrap_or("").to_string(),
        constraints: constraints.unwrap_or("").to_string(),
        total_length: value_str.len(),
        preview,
    }
}

/// Format a REPLVariable for inclusion in prompts.
pub fn format_repl_variable(v: &REPLVariable) -> String {
    let mut lines = vec![format!("Variable: `{}` (access it in your code)", v.name)];
    lines.push(format!("Type: {}", v.type_name));
    if !v.desc.is_empty() {
        lines.push(format!("Description: {}", v.desc));
    }
    if !v.constraints.is_empty() {
        lines.push(format!("Constraints: {}", v.constraints));
    }
    lines.push(format!("Total length: {} characters", v.total_length));
    lines.push(format!("Preview:\n```\n{}\n```", v.preview));
    lines.join("\n")
}

/// A single REPL interaction entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REPLEntry {
    pub reasoning: String,
    pub code: String,
    pub output: String,
}

/// Format a single REPL entry for prompts.
pub fn format_repl_entry(entry: &REPLEntry, index: usize, max_output_chars: usize) -> String {
    let output = if entry.output.len() > max_output_chars {
        format!(
            "{}\n... (truncated to {}/{} chars)",
            &entry.output[..max_output_chars],
            max_output_chars,
            entry.output.len()
        )
    } else {
        entry.output.clone()
    };

    let reasoning_line = if !entry.reasoning.is_empty() {
        format!("Reasoning: {}\n", entry.reasoning)
    } else {
        String::new()
    };

    format!(
        "=== Step {} ===\n{}Code:\n```python\n{}\n```\nOutput ({} chars):\n{}",
        index + 1,
        reasoning_line,
        entry.code,
        entry.output.len(),
        output
    )
}

/// Immutable container for REPL interaction history.
/// append() returns a new instance with the entry added.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REPLHistory {
    entries: Vec<REPLEntry>,
}

impl REPLHistory {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return a new REPLHistory with the entry appended.
    pub fn append(&self, reasoning: &str, code: &str, output: &str) -> REPLHistory {
        let mut new_entries = self.entries.clone();
        new_entries.push(REPLEntry {
            reasoning: reasoning.to_string(),
            code: code.to_string(),
            output: output.to_string(),
        });
        REPLHistory {
            entries: new_entries,
        }
    }

    /// Format the full history for prompt inclusion.
    pub fn format(&self, max_output_chars: Option<usize>) -> String {
        let max_chars = max_output_chars.unwrap_or(5000);
        if self.entries.is_empty() {
            return "You have not interacted with the REPL environment yet.".to_string();
        }
        self.entries
            .iter()
            .enumerate()
            .map(|(i, entry)| format_repl_entry(entry, i, max_chars))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get entries as a slice.
    pub fn entries(&self) -> &[REPLEntry] {
        &self.entries
    }

    /// Get number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if history is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize to JSON value (array of entries).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(&self.entries).unwrap_or(serde_json::json!([]))
    }
}

impl Default for REPLHistory {
    fn default() -> Self {
        Self::new()
    }
}

// Helpers

fn get_type_name(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(_) => "boolean".to_string(),
        serde_json::Value::Number(_) => "number".to_string(),
        serde_json::Value::String(_) => "string".to_string(),
        serde_json::Value::Array(_) => "list".to_string(),
        serde_json::Value::Object(_) => "dict".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repl_history_starts_empty() {
        let history = REPLHistory::new();
        assert_eq!(history.len(), 0);
        assert!(history.is_empty());
        assert_eq!(
            history.format(None),
            "You have not interacted with the REPL environment yet."
        );
    }

    #[test]
    fn test_repl_history_append_returns_new() {
        let h1 = REPLHistory::new();
        let h2 = h1.append("Let me check", "print(1)", "1");

        assert_eq!(h1.len(), 0); // Original unchanged
        assert_eq!(h2.len(), 1);
        assert_eq!(h2.entries()[0].reasoning, "Let me check");
        assert_eq!(h2.entries()[0].code, "print(1)");
        assert_eq!(h2.entries()[0].output, "1");
    }

    #[test]
    fn test_repl_history_format_includes_all() {
        let history = REPLHistory::new()
            .append("Step 1", "x = 1", "no output")
            .append("Step 2", "print(x + 1)", "2");

        let formatted = history.format(None);
        assert!(formatted.contains("=== Step 1 ==="));
        assert!(formatted.contains("=== Step 2 ==="));
        assert!(formatted.contains("x = 1"));
        assert!(formatted.contains("print(x + 1)"));
    }

    #[test]
    fn test_repl_history_format_truncates() {
        let long_output = "x".repeat(10000);
        let history = REPLHistory::new().append("", "print('x' * 10000)", &long_output);

        let formatted = history.format(Some(100));
        assert!(formatted.contains("truncated"));
    }

    #[test]
    fn test_repl_history_to_json() {
        let history = REPLHistory::new().append("r", "c", "o");
        let json = history.to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["reasoning"], "r");
        assert_eq!(arr[0]["code"], "c");
        assert_eq!(arr[0]["output"], "o");
    }

    #[test]
    fn test_create_repl_variable_string() {
        let v = create_repl_variable(
            "msg",
            &serde_json::json!("hello world"),
            None,
            None,
            None,
        );
        assert_eq!(v.name, "msg");
        assert_eq!(v.type_name, "string");
        assert_eq!(v.total_length, 11);
        assert_eq!(v.preview, "hello world");
    }

    #[test]
    fn test_create_repl_variable_object() {
        let v = create_repl_variable(
            "data",
            &serde_json::json!({"key": "value"}),
            None,
            None,
            None,
        );
        assert_eq!(v.type_name, "dict");
        assert!(v.preview.contains("key"));
    }

    #[test]
    fn test_create_repl_variable_array() {
        let v = create_repl_variable("items", &serde_json::json!([1, 2, 3]), None, None, None);
        assert_eq!(v.type_name, "list");
    }

    #[test]
    fn test_create_repl_variable_truncates() {
        let long_str = "x".repeat(1000);
        let v = create_repl_variable(
            "big",
            &serde_json::Value::String(long_str),
            None,
            None,
            Some(100),
        );
        assert!(v.preview.len() < 110);
        assert!(v.preview.contains("..."));
        assert_eq!(v.total_length, 1000);
    }

    #[test]
    fn test_create_repl_variable_with_desc() {
        let v = create_repl_variable(
            "q",
            &serde_json::json!("test"),
            Some("The question"),
            None,
            None,
        );
        assert_eq!(v.desc, "The question");
    }

    #[test]
    fn test_format_repl_variable_includes_metadata() {
        let v = create_repl_variable(
            "ctx",
            &serde_json::json!("some context"),
            Some("The context to analyze"),
            Some("max 1000 chars"),
            None,
        );
        let formatted = format_repl_variable(&v);
        assert!(formatted.contains("`ctx`"));
        assert!(formatted.contains("string"));
        assert!(formatted.contains("The context to analyze"));
        assert!(formatted.contains("max 1000 chars"));
    }
}
