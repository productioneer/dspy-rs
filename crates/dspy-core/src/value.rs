//! Dynamic value type for Example/Prediction stores.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    String(String),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Value>),
    Object(HashMap<String, Value>),
    Null,
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Integer(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Integer(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn into_string(self) -> Option<String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::String(s) => write!(f, "{s}"),
            Value::Integer(n) => write!(f, "{n}"),
            Value::Float(n) => write!(f, "{n}"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::List(v) => write!(f, "{v:?}"),
            Value::Object(m) => write!(f, "{m:?}"),
            Value::Null => write!(f, "null"),
        }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}

impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Integer(n)
    }
}

impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Integer(n as i64)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl<T: Into<Value>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Value::List(v.into_iter().map(Into::into).collect())
    }
}

impl From<serde_json::Value> for Value {
    fn from(v: serde_json::Value) -> Self {
        match v {
            serde_json::Value::Null => Value::Null,
            serde_json::Value::Bool(b) => Value::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Integer(i)
                } else {
                    Value::Float(n.as_f64().unwrap_or(0.0))
                }
            }
            serde_json::Value::String(s) => Value::String(s),
            serde_json::Value::Array(arr) => {
                Value::List(arr.into_iter().map(Value::from).collect())
            }
            serde_json::Value::Object(map) => {
                Value::Object(map.into_iter().map(|(k, v)| (k, Value::from(v))).collect())
            }
        }
    }
}

impl From<Value> for serde_json::Value {
    fn from(v: Value) -> Self {
        match v {
            Value::Null => serde_json::Value::Null,
            Value::Bool(b) => serde_json::Value::Bool(b),
            Value::Integer(i) => serde_json::json!(i),
            Value::Float(f) => serde_json::json!(f),
            Value::String(s) => serde_json::Value::String(s),
            Value::List(arr) => {
                serde_json::Value::Array(arr.into_iter().map(serde_json::Value::from).collect())
            }
            Value::Object(map) => serde_json::Value::Object(
                map.into_iter()
                    .map(|(k, v)| (k, serde_json::Value::from(v)))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_from_str() {
        let v: Value = "hello".into();
        assert_eq!(v.as_str(), Some("hello"));
    }

    #[test]
    fn test_value_display() {
        assert_eq!(Value::String("hello".into()).to_string(), "hello");
        assert_eq!(Value::Integer(42).to_string(), "42");
        assert_eq!(Value::Bool(true).to_string(), "true");
    }

    #[test]
    fn test_value_as_f64_from_int() {
        let v = Value::Integer(42);
        assert_eq!(v.as_f64(), Some(42.0));
    }

    #[test]
    fn test_value_serde_roundtrip() {
        let v = Value::String("test".into());
        let json = serde_json::to_string(&v).unwrap();
        let v2: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }
}
