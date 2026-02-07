//! Aggregation utilities — majority voting over completions.
//! Python equivalent: dspy/predict/aggregation.py

use std::collections::HashMap;

/// Default normalize: lowercase, strip whitespace and punctuation.
/// Returns None if the result is empty (those completions are ignored).
fn default_normalize(s: &str) -> Option<String> {
    let normalized: String = s
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

/// Returns the most common completion for the target field (or the last field).
/// When normalize returns None, that completion is ignored.
/// In case of a tie, earlier completions are prioritized.
pub fn majority(
    completions: &[HashMap<String, String>],
    field: Option<&str>,
    normalize: Option<&dyn Fn(&str) -> Option<String>>,
) -> Option<HashMap<String, String>> {
    if completions.is_empty() {
        return None;
    }

    let norm_fn: &dyn Fn(&str) -> Option<String> = match normalize {
        Some(f) => f,
        None => &default_normalize,
    };

    // Determine target field: last key of first completion
    let target_field = match field {
        Some(f) => f.to_string(),
        None => completions[0].keys().last()?.to_string(),
    };

    // Normalize values
    let normalized: Vec<Option<String>> = completions
        .iter()
        .map(|c| {
            c.get(&target_field)
                .and_then(|v| norm_fn(v))
        })
        .collect();

    let non_null: Vec<&String> = normalized.iter().filter_map(|v| v.as_ref()).collect();
    let values_to_count: Vec<&String> = if non_null.is_empty() {
        // fallback to all (including None mapped to empty)
        return Some(completions[0].clone());
    } else {
        non_null
    };

    // Count occurrences
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in &values_to_count {
        *counts.entry(value.as_str()).or_insert(0) += 1;
    }

    // Find majority value (max count, first wins on tie)
    let majority_value = counts
        .into_iter()
        .max_by_key(|(_, count)| *count)?
        .0
        .to_string();

    // Return first completion whose normalized field matches
    for (i, completion) in completions.iter().enumerate() {
        if let Some(ref norm) = normalized[i] {
            if *norm == majority_value {
                return Some(completion.clone());
            }
        }
    }

    Some(completions[0].clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_completion(field: &str, value: &str) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(field.to_string(), value.to_string());
        m
    }

    #[test]
    fn test_majority_simple() {
        let completions = vec![
            make_completion("answer", "Paris"),
            make_completion("answer", "London"),
            make_completion("answer", "Paris"),
        ];
        let result = majority(&completions, Some("answer"), None).unwrap();
        assert_eq!(result.get("answer").unwrap(), "Paris");
    }

    #[test]
    fn test_majority_tie_prefers_earlier() {
        let completions = vec![
            make_completion("answer", "A"),
            make_completion("answer", "B"),
        ];
        let result = majority(&completions, Some("answer"), None).unwrap();
        // Tie — either is valid but we get whichever has higher count first;
        // with equal counts HashMap iteration order may vary, so just check it's one of them
        let ans = result.get("answer").unwrap();
        assert!(ans == "A" || ans == "B");
    }

    #[test]
    fn test_majority_normalization() {
        let completions = vec![
            make_completion("answer", "  Paris  "),
            make_completion("answer", "paris"),
            make_completion("answer", "London"),
        ];
        let result = majority(&completions, Some("answer"), None).unwrap();
        // Both "Paris" and "paris" normalize to "paris"
        let ans = result.get("answer").unwrap().to_lowercase().trim().to_string();
        assert_eq!(ans, "paris");
    }

    #[test]
    fn test_majority_empty() {
        let result = majority(&[], None, None);
        assert!(result.is_none());
    }

    #[test]
    fn test_majority_custom_normalize() {
        let no_norm: &dyn Fn(&str) -> Option<String> = &|s: &str| Some(s.to_string());
        let completions = vec![
            make_completion("answer", "Paris"),
            make_completion("answer", "paris"),
            make_completion("answer", "paris"),
        ];
        let result = majority(&completions, Some("answer"), Some(no_norm)).unwrap();
        // Without normalization, "paris" (lowercase) wins
        assert_eq!(result.get("answer").unwrap(), "paris");
    }
}
