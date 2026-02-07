//! Streaming — Stream LM responses incrementally.
//!
//! StreamListener captures streaming output for specific fields of a predictor.
//! Matches Python DSPy's dspy.streaming interface.

use regex::Regex;

/// Response chunk from a stream listener.
#[derive(Debug, Clone)]
pub struct StreamResponse {
    /// Name of the predictor that produced this chunk.
    pub predict_name: Option<String>,
    /// Name of the signature field being streamed.
    pub field_name: String,
    /// The text chunk content.
    pub chunk: Option<String>,
    /// Whether this is the final chunk for this field.
    pub is_last_chunk: bool,
}

/// Adapter type for stream processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterType {
    ChatAdapter,
    JsonAdapter,
    XmlAdapter,
}

struct AdapterIdentifiers {
    start_identifier: String,
    end_identifier: Regex,
    start_indicator: String,
    end_pattern_prefixes: Vec<String>,
    end_pattern_contains: String,
}

/// Listens to an LM response stream and captures output for a specific field.
pub struct StreamListener {
    pub signature_field_name: String,
    pub predict_name: Option<String>,
    pub allow_reuse: bool,

    field_start_queue: Vec<String>,
    field_end_queue: Vec<String>,
    pub stream_start: bool,
    pub stream_end: bool,
    pub cache_hit: bool,

    identifiers: std::collections::HashMap<String, AdapterIdentifiers>,
}

impl StreamListener {
    /// Create a new StreamListener for a specific field.
    pub fn new(field_name: &str, predict_name: Option<String>, allow_reuse: bool) -> Self {
        let mut identifiers = std::collections::HashMap::new();

        identifiers.insert(
            "ChatAdapter".to_string(),
            AdapterIdentifiers {
                start_identifier: format!("[[ ## {} ## ]]", field_name),
                end_identifier: Regex::new(r"\[\[ ## (\w+) ## \]\]").unwrap(),
                start_indicator: "[".to_string(),
                end_pattern_prefixes: vec![
                    "[".to_string(), "[[".to_string(), "[[ ".to_string(),
                    "[[ #".to_string(), "[[ ##".to_string(),
                ],
                end_pattern_contains: "[[ ##".to_string(),
            },
        );

        identifiers.insert(
            "JsonAdapter".to_string(),
            AdapterIdentifiers {
                start_identifier: format!("\"{}\":", field_name),
                end_identifier: Regex::new(r#"\w*"(,|\s*\})"#).unwrap(),
                start_indicator: "\"".to_string(),
                end_pattern_prefixes: vec![
                    "\"".to_string(), "\",".to_string(), "\" ".to_string(), "\"}".to_string(),
                ],
                end_pattern_contains: "}".to_string(),
            },
        );

        identifiers.insert(
            "XmlAdapter".to_string(),
            AdapterIdentifiers {
                start_identifier: format!("<{}>", field_name),
                end_identifier: Regex::new(&format!("</{}>", field_name)).unwrap(),
                start_indicator: "<".to_string(),
                end_pattern_prefixes: vec!["<".to_string(), "</".to_string()],
                end_pattern_contains: "</".to_string(),
            },
        );

        Self {
            signature_field_name: field_name.to_string(),
            predict_name,
            allow_reuse,
            field_start_queue: Vec::new(),
            field_end_queue: Vec::new(),
            stream_start: false,
            stream_end: false,
            cache_hit: false,
            identifiers,
        }
    }

    /// Receive a chunk from the LM response stream.
    pub fn receive(&mut self, chunk_message: &str, adapter_type: AdapterType) -> Option<StreamResponse> {
        let adapter_name = match adapter_type {
            AdapterType::ChatAdapter => "ChatAdapter",
            AdapterType::JsonAdapter => "JsonAdapter",
            AdapterType::XmlAdapter => "XmlAdapter",
        };

        // Clone config values upfront to avoid holding immutable borrow across mutable ops
        let (start_identifier, end_identifier, start_indicator) = {
            let config = self.identifiers.get(adapter_name)?;
            (
                config.start_identifier.clone(),
                config.end_identifier.clone(),
                config.start_indicator.clone(),
            )
        };

        if self.stream_end {
            if self.allow_reuse {
                self.reset();
            } else {
                return None;
            }
        }

        // Check for cache hit (full response in single chunk)
        if chunk_message.contains(&start_identifier)
            && adapter_type != AdapterType::JsonAdapter
        {
            let after_start = &chunk_message[chunk_message.find(&start_identifier).unwrap()
                + start_identifier.len()..];
            if end_identifier.is_match(after_start) {
                self.cache_hit = true;
                self.stream_start = true;
                self.stream_end = true;
                return None;
            }
        }

        // Look for start indicator
        if self.field_start_queue.is_empty()
            && !self.stream_start
            && chunk_message.contains(&start_indicator)
        {
            self.field_start_queue.push(chunk_message.to_string());
            return None;
        }

        // Accumulate start tokens
        let mut chunk_msg = chunk_message.to_string();
        if !self.field_start_queue.is_empty() && !self.stream_start {
            self.field_start_queue.push(chunk_msg.clone());
            let concat: String = self.field_start_queue.join("");

            if concat.contains(&start_identifier) {
                self.stream_start = true;
                self.field_start_queue.clear();
                let value_start =
                    concat.find(&start_identifier).unwrap() + start_identifier.len();
                chunk_msg = concat[value_start..].trim_start().to_string();
            } else if self.could_form_start(concat.trim(), &start_identifier) {
                return None;
            } else {
                self.field_start_queue.clear();
                return None;
            }
        }

        // Stream content
        if self.stream_start && !chunk_msg.is_empty() {
            self.field_end_queue.push(chunk_msg);
            let concat: String = self.field_end_queue.join("").trim().to_string();

            let mut token: Option<String> = None;

            if !self.could_form_end(&concat, adapter_name) {
                token = Some(self.flush());
            } else if self.field_end_queue.len() > 10 {
                token = Some(self.field_end_queue.remove(0));
            }

            // Check for end identifier
            if end_identifier.is_match(&concat) {
                self.stream_end = true;
                let last = self.flush();
                let combined = format!("{}{}", token.unwrap_or_default(), last);
                token = Some(combined.trim_end().to_string());
            }

            if token.is_some() || self.stream_end {
                return Some(StreamResponse {
                    predict_name: self.predict_name.clone(),
                    field_name: self.signature_field_name.clone(),
                    chunk: token,
                    is_last_chunk: self.stream_end,
                });
            }
        }

        None
    }

    /// Flush remaining buffered tokens.
    pub fn flush(&mut self) -> String {
        let tokens: String = self.field_end_queue.join("");
        self.field_end_queue.clear();
        tokens
    }

    /// Finalize the stream, flushing any remaining buffered tokens.
    pub fn finalize(&mut self) -> Option<StreamResponse> {
        if self.stream_end || !self.stream_start {
            return None;
        }

        self.stream_end = true;
        if !self.field_end_queue.is_empty() {
            let token = self.flush();
            if !token.is_empty() {
                return Some(StreamResponse {
                    predict_name: self.predict_name.clone(),
                    field_name: self.signature_field_name.clone(),
                    chunk: Some(token),
                    is_last_chunk: true,
                });
            }
        }
        None
    }

    fn reset(&mut self) {
        self.stream_end = false;
        self.cache_hit = false;
        self.field_start_queue.clear();
        self.field_end_queue.clear();
        self.stream_start = false;
    }

    fn could_form_start(&self, concat: &str, start_identifier: &str) -> bool {
        for i in 0..concat.len() {
            if start_identifier.starts_with(&concat[concat.len() - i - 1..]) {
                return true;
            }
        }
        false
    }

    fn could_form_end(&self, concat: &str, adapter_name: &str) -> bool {
        let config = match self.identifiers.get(adapter_name) {
            Some(c) => c,
            None => return false,
        };
        if config.end_pattern_prefixes.iter().any(|p| concat.ends_with(p.as_str())) {
            return true;
        }
        if !config.end_pattern_contains.is_empty() && concat.contains(&config.end_pattern_contains) {
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_listener_basic() {
        let mut listener = StreamListener::new("answer", None, false);

        // Simulate ChatAdapter streaming — header arrives in pieces
        assert!(listener.receive("[[ ## answer", AdapterType::ChatAdapter).is_none());
        assert!(!listener.stream_start); // not yet complete

        assert!(listener.receive(" ## ]]", AdapterType::ChatAdapter).is_none());
        assert!(listener.stream_start); // now the full identifier was found

        let resp = listener.receive("Hello ", AdapterType::ChatAdapter);
        // May or may not yield immediately depending on buffering
        assert!(resp.is_some() || !listener.field_end_queue.is_empty());
    }

    #[test]
    fn test_stream_listener_cache_hit() {
        let mut listener = StreamListener::new("answer", None, false);

        // Full response in one chunk
        let msg = "[[ ## answer ## ]] Hello world [[ ## completed ## ]]";
        listener.receive(msg, AdapterType::ChatAdapter);

        assert!(listener.cache_hit);
        assert!(listener.stream_end);
    }

    #[test]
    fn test_stream_listener_finalize() {
        let mut listener = StreamListener::new("answer", None, false);

        // Start the stream manually
        listener.stream_start = true;
        listener.field_end_queue.push("buffered content".to_string());

        let result = listener.finalize();
        assert!(result.is_some());
        assert_eq!(result.unwrap().chunk.unwrap(), "buffered content");
        assert!(listener.stream_end);
    }

    #[test]
    fn test_stream_listener_reuse() {
        let mut listener = StreamListener::new("answer", None, true);

        // First use
        listener.stream_start = true;
        listener.stream_end = true;

        // Should reset on next receive when allow_reuse is true
        listener.receive("test", AdapterType::ChatAdapter);
        assert!(!listener.stream_end);
    }
}
