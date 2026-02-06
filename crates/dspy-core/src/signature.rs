//! Signature — task input/output schema definition.
//! Python equivalent: dspy/signatures/signature.py

use crate::error::{DspyError, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FieldType {
    Input,
    Output,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub name: String,
    pub field_type: FieldType,
    pub description: Option<String>,
    pub prefix: Option<String>,
    pub format: Option<String>,
}

impl FieldDef {
    pub fn with_desc(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.prefix = Some(prefix.to_string());
        self
    }

    pub fn with_format(mut self, format: &str) -> Self {
        self.format = Some(format.to_string());
        self
    }
}

pub fn input_field(name: &str) -> FieldDef {
    FieldDef {
        name: name.to_string(),
        field_type: FieldType::Input,
        description: None,
        prefix: None,
        format: None,
    }
}

pub fn output_field(name: &str) -> FieldDef {
    FieldDef {
        name: name.to_string(),
        field_type: FieldType::Output,
        description: None,
        prefix: None,
        format: None,
    }
}

/// Updates to apply to a FieldDef
pub struct FieldUpdate {
    pub description: Option<String>,
    pub prefix: Option<String>,
    pub format: Option<String>,
}

impl Default for FieldUpdate {
    fn default() -> Self {
        Self {
            description: None,
            prefix: None,
            format: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Signature {
    instructions: String,
    fields: IndexMap<String, FieldDef>,
}

impl Signature {
    pub fn new(fields: Vec<FieldDef>, instructions: &str) -> Self {
        let mut map = IndexMap::new();
        for f in fields {
            map.insert(f.name.clone(), f);
        }
        Self {
            instructions: instructions.to_string(),
            fields: map,
        }
    }

    /// Parse shorthand like "question -> answer" or "question, context -> answer: detailed answer"
    pub fn from_string(spec: &str) -> Result<Self> {
        // Split on "->"
        let parts: Vec<&str> = spec.splitn(2, "->").collect();
        if parts.len() != 2 {
            return Err(DspyError::InvalidSignature(format!(
                "Expected 'inputs -> outputs', got: {spec}"
            )));
        }

        let (input_str, output_str) = (parts[0].trim(), parts[1].trim());
        let mut fields = IndexMap::new();

        // Parse input fields
        for field_str in input_str.split(',') {
            let field_str = field_str.trim();
            if field_str.is_empty() {
                continue;
            }
            let (name, desc) = parse_field_with_desc(field_str);
            let mut f = input_field(&name);
            if let Some(d) = desc {
                f.description = Some(d);
            }
            fields.insert(name, f);
        }

        // Parse output fields
        for field_str in output_str.split(',') {
            let field_str = field_str.trim();
            if field_str.is_empty() {
                continue;
            }
            let (name, desc) = parse_field_with_desc(field_str);
            let mut f = output_field(&name);
            if let Some(d) = desc {
                f.description = Some(d);
            }
            fields.insert(name, f);
        }

        if fields.is_empty() {
            return Err(DspyError::InvalidSignature(
                "Signature must have at least one field".into(),
            ));
        }

        Ok(Self {
            instructions: String::new(),
            fields,
        })
    }

    // Accessors
    pub fn instructions(&self) -> &str {
        &self.instructions
    }

    pub fn fields(&self) -> &IndexMap<String, FieldDef> {
        &self.fields
    }

    pub fn input_fields(&self) -> impl Iterator<Item = (&String, &FieldDef)> {
        self.fields
            .iter()
            .filter(|(_, f)| f.field_type == FieldType::Input)
    }

    pub fn output_fields(&self) -> impl Iterator<Item = (&String, &FieldDef)> {
        self.fields
            .iter()
            .filter(|(_, f)| f.field_type == FieldType::Output)
    }

    pub fn get_field(&self, name: &str) -> Option<&FieldDef> {
        self.fields.get(name)
    }

    // Manipulation (returns new Signature — immutable style)
    pub fn with_instructions(&self, instructions: &str) -> Self {
        Self {
            instructions: instructions.to_string(),
            fields: self.fields.clone(),
        }
    }

    pub fn with_updated_field(&self, name: &str, updates: FieldUpdate) -> Self {
        let mut fields = self.fields.clone();
        if let Some(f) = fields.get_mut(name) {
            if let Some(d) = updates.description {
                f.description = Some(d);
            }
            if let Some(p) = updates.prefix {
                f.prefix = Some(p);
            }
            if let Some(fmt) = updates.format {
                f.format = Some(fmt);
            }
        }
        Self {
            instructions: self.instructions.clone(),
            fields,
        }
    }

    /// Prepend a field (insert at beginning of its type group).
    /// Output fields go before existing output fields.
    pub fn prepend(&self, field: FieldDef) -> Self {
        let mut fields = IndexMap::new();
        let name = field.name.clone();

        match field.field_type {
            FieldType::Input => {
                // Insert at beginning
                fields.insert(name, field);
                for (k, v) in &self.fields {
                    fields.insert(k.clone(), v.clone());
                }
            }
            FieldType::Output => {
                // Insert all inputs first, then this field, then existing outputs
                for (k, v) in &self.fields {
                    if v.field_type == FieldType::Input {
                        fields.insert(k.clone(), v.clone());
                    }
                }
                fields.insert(name, field);
                for (k, v) in &self.fields {
                    if v.field_type == FieldType::Output {
                        fields.insert(k.clone(), v.clone());
                    }
                }
            }
        }

        Self {
            instructions: self.instructions.clone(),
            fields,
        }
    }

    pub fn append(&self, field: FieldDef) -> Self {
        let mut fields = self.fields.clone();
        fields.insert(field.name.clone(), field);
        Self {
            instructions: self.instructions.clone(),
            fields,
        }
    }

    pub fn insert(&self, index: usize, field: FieldDef) -> Self {
        let mut fields = IndexMap::new();
        let entries: Vec<_> = self.fields.iter().collect();
        let name = field.name.clone();

        for (i, (k, v)) in entries.iter().enumerate() {
            if i == index {
                fields.insert(name.clone(), field.clone());
            }
            fields.insert((*k).clone(), (*v).clone());
        }
        // If index >= length, append
        if index >= entries.len() {
            fields.insert(name, field);
        }

        Self {
            instructions: self.instructions.clone(),
            fields,
        }
    }

    pub fn delete_field(&self, name: &str) -> Self {
        let mut fields = self.fields.clone();
        fields.shift_remove(name);
        Self {
            instructions: self.instructions.clone(),
            fields,
        }
    }

    // Serialization
    pub fn dump_state(&self) -> serde_json::Value {
        let fields_state: Vec<serde_json::Value> = self
            .fields
            .iter()
            .map(|(_, f)| {
                serde_json::json!({
                    "name": f.name,
                    "field_type": match f.field_type {
                        FieldType::Input => "input",
                        FieldType::Output => "output",
                    },
                    "description": f.description,
                    "prefix": f.prefix,
                    "format": f.format,
                })
            })
            .collect();

        serde_json::json!({
            "instructions": self.instructions,
            "fields": fields_state,
        })
    }

    pub fn load_state(state: &serde_json::Value) -> Result<Self> {
        let instructions = state["instructions"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let fields_arr = state["fields"]
            .as_array()
            .ok_or_else(|| DspyError::ParseError("Expected 'fields' array".into()))?;

        let mut fields = IndexMap::new();
        for f_val in fields_arr {
            let name = f_val["name"]
                .as_str()
                .ok_or_else(|| DspyError::ParseError("Field missing 'name'".into()))?
                .to_string();
            let ft = match f_val["field_type"].as_str() {
                Some("input") => FieldType::Input,
                Some("output") => FieldType::Output,
                _ => {
                    return Err(DspyError::ParseError(
                        "Invalid field_type".into(),
                    ))
                }
            };
            let f = FieldDef {
                name: name.clone(),
                field_type: ft,
                description: f_val["description"].as_str().map(|s| s.to_string()),
                prefix: f_val["prefix"].as_str().map(|s| s.to_string()),
                format: f_val["format"].as_str().map(|s| s.to_string()),
            };
            fields.insert(name, f);
        }

        Ok(Self {
            instructions,
            fields,
        })
    }

    /// Generate the shorthand string representation
    pub fn to_shorthand(&self) -> String {
        let inputs: Vec<&str> = self
            .input_fields()
            .map(|(k, _)| k.as_str())
            .collect();
        let outputs: Vec<&str> = self
            .output_fields()
            .map(|(k, _)| k.as_str())
            .collect();
        format!("{} -> {}", inputs.join(", "), outputs.join(", "))
    }
}

/// Parse "name: description" or just "name"
fn parse_field_with_desc(s: &str) -> (String, Option<String>) {
    if let Some(idx) = s.find(':') {
        let name = s[..idx].trim().to_string();
        let desc = s[idx + 1..].trim().to_string();
        if desc.is_empty() {
            (name, None)
        } else {
            (name, Some(desc))
        }
    } else {
        (s.trim().to_string(), None)
    }
}

/// Builder pattern for Signature
pub struct SignatureBuilder {
    instructions: String,
    fields: Vec<FieldDef>,
}

impl SignatureBuilder {
    pub fn new() -> Self {
        Self {
            instructions: String::new(),
            fields: Vec::new(),
        }
    }

    pub fn instructions(mut self, instructions: &str) -> Self {
        self.instructions = instructions.to_string();
        self
    }

    pub fn input(mut self, name: &str) -> Self {
        self.fields.push(input_field(name));
        self
    }

    pub fn input_with_desc(mut self, name: &str, desc: &str) -> Self {
        self.fields.push(input_field(name).with_desc(desc));
        self
    }

    pub fn output(mut self, name: &str) -> Self {
        self.fields.push(output_field(name));
        self
    }

    pub fn output_with_desc(mut self, name: &str, desc: &str) -> Self {
        self.fields.push(output_field(name).with_desc(desc));
        self
    }

    pub fn build(self) -> Signature {
        Signature::new(self.fields, &self.instructions)
    }
}

impl Default for SignatureBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: Signature::define().input("q").output("a").build()
impl Signature {
    pub fn define() -> SignatureBuilder {
        SignatureBuilder::new()
    }
}

/// Allow Signature::from_string via Into<Signature> for &str
impl TryFrom<&str> for Signature {
    type Error = DspyError;
    fn try_from(s: &str) -> Result<Self> {
        Signature::from_string(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_string_basic() {
        let sig = Signature::from_string("question -> answer").unwrap();
        assert_eq!(sig.input_fields().count(), 1);
        assert_eq!(sig.output_fields().count(), 1);
        assert_eq!(sig.fields().get_index(0).unwrap().0, "question");
        assert_eq!(sig.fields().get_index(1).unwrap().0, "answer");
    }

    #[test]
    fn test_from_string_multiple_fields() {
        let sig = Signature::from_string("question, context -> answer, confidence").unwrap();
        assert_eq!(sig.input_fields().count(), 2);
        assert_eq!(sig.output_fields().count(), 2);
    }

    #[test]
    fn test_from_string_with_descriptions() {
        let sig =
            Signature::from_string("question: the question to answer -> answer: the final answer")
                .unwrap();
        let q = sig.fields().get("question").unwrap();
        assert_eq!(q.description.as_deref(), Some("the question to answer"));
        let a = sig.fields().get("answer").unwrap();
        assert_eq!(a.description.as_deref(), Some("the final answer"));
    }

    #[test]
    fn test_from_string_error() {
        let result = Signature::from_string("no arrow here");
        assert!(result.is_err());
    }

    #[test]
    fn test_builder() {
        let sig = Signature::define()
            .instructions("Answer questions")
            .input("question")
            .output_with_desc("answer", "The answer")
            .build();
        assert_eq!(sig.instructions(), "Answer questions");
        assert_eq!(sig.input_fields().count(), 1);
        assert_eq!(sig.output_fields().count(), 1);
        let a = sig.fields().get("answer").unwrap();
        assert_eq!(a.description.as_deref(), Some("The answer"));
    }

    #[test]
    fn test_with_instructions() {
        let sig = Signature::from_string("q -> a").unwrap();
        assert_eq!(sig.instructions(), "");
        let sig2 = sig.with_instructions("Do something");
        assert_eq!(sig2.instructions(), "Do something");
        // Original unchanged
        assert_eq!(sig.instructions(), "");
    }

    #[test]
    fn test_prepend_output() {
        let sig = Signature::from_string("question -> answer").unwrap();
        let sig2 = sig.prepend(output_field("reasoning").with_desc("Step by step reasoning"));
        let fields: Vec<_> = sig2.fields().keys().collect();
        // Should be: question, reasoning, answer
        assert_eq!(fields, vec!["question", "reasoning", "answer"]);
        assert_eq!(sig2.output_fields().count(), 2);
    }

    #[test]
    fn test_prepend_input() {
        let sig = Signature::from_string("question -> answer").unwrap();
        let sig2 = sig.prepend(input_field("context"));
        let fields: Vec<_> = sig2.fields().keys().collect();
        assert_eq!(fields, vec!["context", "question", "answer"]);
    }

    #[test]
    fn test_append() {
        let sig = Signature::from_string("q -> a").unwrap();
        let sig2 = sig.append(output_field("confidence"));
        assert_eq!(sig2.fields().len(), 3);
        assert_eq!(sig2.fields().get_index(2).unwrap().0, "confidence");
    }

    #[test]
    fn test_delete_field() {
        let sig = Signature::from_string("q, context -> a").unwrap();
        let sig2 = sig.delete_field("context");
        assert_eq!(sig2.fields().len(), 2);
        assert!(!sig2.fields().contains_key("context"));
    }

    #[test]
    fn test_insert_at_index() {
        let sig = Signature::from_string("q -> a").unwrap();
        let sig2 = sig.insert(1, output_field("reasoning"));
        let fields: Vec<_> = sig2.fields().keys().collect();
        assert_eq!(fields, vec!["q", "reasoning", "a"]);
    }

    #[test]
    fn test_with_updated_field() {
        let sig = Signature::from_string("q -> a").unwrap();
        let sig2 = sig.with_updated_field(
            "a",
            FieldUpdate {
                description: Some("The answer".into()),
                ..Default::default()
            },
        );
        assert_eq!(
            sig2.fields().get("a").unwrap().description.as_deref(),
            Some("The answer")
        );
    }

    #[test]
    fn test_dump_load_state() {
        let sig = Signature::define()
            .instructions("Test")
            .input_with_desc("q", "question")
            .output("a")
            .build();
        let state = sig.dump_state();
        let sig2 = Signature::load_state(&state).unwrap();
        assert_eq!(sig2.instructions(), "Test");
        assert_eq!(sig2.fields().len(), 2);
        assert_eq!(
            sig2.fields().get("q").unwrap().description.as_deref(),
            Some("question")
        );
    }

    #[test]
    fn test_to_shorthand() {
        let sig = Signature::from_string("question, context -> answer").unwrap();
        let sh = sig.to_shorthand();
        assert!(sh.contains("question"));
        assert!(sh.contains("context"));
        assert!(sh.contains("answer"));
        assert!(sh.contains("->"));
    }

    #[test]
    fn test_clone_preserves_order() {
        let sig = Signature::from_string("a, b, c -> x, y").unwrap();
        let cloned = sig.clone();
        let orig_keys: Vec<_> = sig.fields().keys().collect();
        let clone_keys: Vec<_> = cloned.fields().keys().collect();
        assert_eq!(orig_keys, clone_keys);
    }
}
