//! Streaming — Stream LM responses incrementally.
//!
//! StreamListener captures streaming output for specific fields of a predictor.
//! streamify() wraps a DSPy program to yield incremental results.
//!
//! Matches Python DSPy's dspy.streaming interface.

use crate::prediction::Prediction;
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

/// Status message emitted during streaming.
#[derive(Debug, Clone)]
pub struct StatusMessage {
    pub message: String,
    pub module_name: Option<String>,
}

/// Union type for values that can flow through a stream.
#[derive(Debug, Clone)]
pub enum StreamValue {
    Chunk(StreamResponse),
    Status(StatusMessage),
    Result(Prediction),
}

/// Provider for customizable status messages during streaming.
///
/// Implement this trait to customize status messages emitted at
/// module/LM/tool lifecycle boundaries. Matches Python DSPy's
/// StatusMessageProvider.
pub trait StatusMessageProvider: Send + Sync {
    fn module_start_status_message(&self, _module_name: &str) -> Option<String> {
        None
    }
    fn module_end_status_message(&self, _output: &Prediction) -> Option<String> {
        None
    }
    fn lm_start_status_message(&self, _model: &str) -> Option<String> {
        None
    }
    fn lm_end_status_message(&self) -> Option<String> {
        None
    }
    fn tool_start_status_message(&self, _tool_name: &str) -> Option<String> {
        None
    }
    fn tool_end_status_message(&self) -> Option<String> {
        None
    }
}

/// Options for streamify().
pub struct StreamifyOptions {
    pub stream_listeners: Vec<std::sync::Arc<std::sync::Mutex<StreamListener>>>,
    pub status_message_provider: Option<Box<dyn StatusMessageProvider>>,
    pub include_final_prediction: bool,
}

impl Default for StreamifyOptions {
    fn default() -> Self {
        Self {
            stream_listeners: Vec::new(),
            status_message_provider: None,
            include_final_prediction: true,
        }
    }
}

/// Wrap a DSPy module to stream its outputs incrementally.
///
/// Runs the program with streaming context and collects:
/// - StreamResponse chunks from each listener as fields are populated
/// - StatusMessage for lifecycle events
/// - The final Prediction as the last item
///
/// LM implementations that support streaming should check settings.send_stream
/// and settings.stream_listeners to route response chunks through the listeners
/// during generation. Without LM-level streaming, listeners are finalized after
/// the forward pass completes (post-call finalization).
///
/// Matches Python DSPy's dspy.streamify() interface.
pub async fn streamify<F, Fut>(
    module_name: &str,
    forward_fn: F,
    options: StreamifyOptions,
) -> Vec<StreamValue>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = crate::error::Result<Prediction>>,
{
    use crate::settings::{get_settings, with_settings, SendStreamFn};
    use std::sync::{Arc, Mutex};

    let mut output: Vec<StreamValue> = Vec::new();

    // Emit start status
    let start_msg = options
        .status_message_provider
        .as_ref()
        .and_then(|p| p.module_start_status_message(module_name))
        .unwrap_or_else(|| "Starting program execution".to_string());
    output.push(StreamValue::Status(StatusMessage {
        message: start_msg,
        module_name: Some(module_name.to_string()),
    }));

    // Collect streamed chunks via send_stream callback
    let streamed_chunks: Arc<Mutex<Vec<StreamValue>>> = Arc::new(Mutex::new(Vec::new()));
    let chunks_clone = streamed_chunks.clone();
    let send_stream: SendStreamFn = Arc::new(move |value: StreamValue| {
        if matches!(&value, StreamValue::Result(_)) {
            return; // predictions handled separately
        }
        chunks_clone.lock().unwrap().push(value);
    });

    // Execute with streaming context
    let mut settings = get_settings();
    settings.send_stream = Some(send_stream);
    settings.stream_listeners = Some(options.stream_listeners.clone());

    let result = with_settings(settings, || forward_fn());
    let result = result.await;

    // Yield collected chunks
    {
        let chunks = streamed_chunks.lock().unwrap();
        output.extend(chunks.iter().cloned());
    }

    // Finalize listeners and yield remaining buffered content
    for listener_arc in &options.stream_listeners {
        let mut listener = listener_arc.lock().unwrap();
        if let Some(final_chunk) = listener.finalize() {
            output.push(StreamValue::Chunk(final_chunk));
        }
    }

    match result {
        Ok(prediction) => {
            // Status message for completion
            if let Some(ref provider) = options.status_message_provider {
                if let Some(end_msg) = provider.module_end_status_message(&prediction) {
                    output.push(StreamValue::Status(StatusMessage {
                        message: end_msg,
                        module_name: Some(module_name.to_string()),
                    }));
                }
            }

            // Yield the final prediction
            let should_include = if options.include_final_prediction {
                true
            } else if options.stream_listeners.is_empty() {
                true
            } else {
                let any_cache_hit = options
                    .stream_listeners
                    .iter()
                    .any(|l| l.lock().unwrap().cache_hit);
                let any_started = options
                    .stream_listeners
                    .iter()
                    .any(|l| l.lock().unwrap().stream_start);
                any_cache_hit || !any_started
            };

            if should_include {
                output.push(StreamValue::Result(prediction));
            }
        }
        Err(e) => {
            output.push(StreamValue::Status(StatusMessage {
                message: format!("Error: {}", e),
                module_name: Some(module_name.to_string()),
            }));
        }
    }

    output
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

    active_adapter_name: Option<String>,
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
                    "[".to_string(),
                    "[[".to_string(),
                    "[[ ".to_string(),
                    "[[ #".to_string(),
                    "[[ ##".to_string(),
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
                    "\"".to_string(),
                    "\",".to_string(),
                    "\" ".to_string(),
                    "\"}".to_string(),
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
            active_adapter_name: None,
            identifiers,
        }
    }

    /// Receive a chunk from the LM response stream.
    pub fn receive(
        &mut self,
        chunk_message: &str,
        adapter_type: AdapterType,
    ) -> Option<StreamResponse> {
        let adapter_name = match adapter_type {
            AdapterType::ChatAdapter => "ChatAdapter",
            AdapterType::JsonAdapter => "JsonAdapter",
            AdapterType::XmlAdapter => "XmlAdapter",
        };
        self.active_adapter_name = Some(adapter_name.to_string());

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
        if chunk_message.contains(&start_identifier) && adapter_type != AdapterType::JsonAdapter {
            let after_start = &chunk_message
                [chunk_message.find(&start_identifier).unwrap() + start_identifier.len()..];
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
                let value_start = concat.find(&start_identifier).unwrap() + start_identifier.len();
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

    /// Flush remaining buffered tokens, trimming adapter-specific end delimiters.
    pub fn flush(&mut self) -> String {
        let mut tokens: String = self.field_end_queue.join("");
        self.field_end_queue.clear();

        // Trim adapter-specific end delimiters to prevent delimiter leakage
        if let Some(ref adapter_name) = self.active_adapter_name {
            if let Some(config) = self.identifiers.get(adapter_name.as_str()) {
                if let Some(m) = config.end_identifier.find(&tokens) {
                    tokens = tokens[..m.start()].to_string();
                }
            }
        }

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
        self.active_adapter_name = None;
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
        if config
            .end_pattern_prefixes
            .iter()
            .any(|p| concat.ends_with(p.as_str()))
        {
            return true;
        }
        if !config.end_pattern_contains.is_empty() && concat.contains(&config.end_pattern_contains)
        {
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
        assert!(listener
            .receive("[[ ## answer", AdapterType::ChatAdapter)
            .is_none());
        assert!(!listener.stream_start); // not yet complete

        assert!(listener
            .receive(" ## ]]", AdapterType::ChatAdapter)
            .is_none());
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
        listener
            .field_end_queue
            .push("buffered content".to_string());

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

    #[test]
    fn test_status_message() {
        let msg = StatusMessage {
            message: "Starting".to_string(),
            module_name: Some("ChainOfThought".to_string()),
        };
        assert_eq!(msg.message, "Starting");
        assert_eq!(msg.module_name.unwrap(), "ChainOfThought");

        let msg2 = StatusMessage {
            message: "Done".to_string(),
            module_name: None,
        };
        assert!(msg2.module_name.is_none());
    }

    #[test]
    fn test_stream_value_variants() {
        let chunk = StreamValue::Chunk(StreamResponse {
            predict_name: None,
            field_name: "answer".to_string(),
            chunk: Some("Hello".to_string()),
            is_last_chunk: false,
        });
        assert!(matches!(chunk, StreamValue::Chunk(_)));

        let status = StreamValue::Status(StatusMessage {
            message: "Running".to_string(),
            module_name: None,
        });
        assert!(matches!(status, StreamValue::Status(_)));

        let pred = Prediction::from_completions(vec![std::collections::HashMap::new()], None);
        let result = StreamValue::Result(pred);
        assert!(matches!(result, StreamValue::Result(_)));
    }

    #[test]
    fn test_streamify_options_default() {
        let opts = StreamifyOptions::default();
        assert!(opts.stream_listeners.is_empty());
        assert!(opts.status_message_provider.is_none());
        assert!(opts.include_final_prediction);
    }

    struct TestStatusProvider;
    impl StatusMessageProvider for TestStatusProvider {
        fn module_start_status_message(&self, module_name: &str) -> Option<String> {
            Some(format!("Starting {}", module_name))
        }
        fn module_end_status_message(&self, _output: &Prediction) -> Option<String> {
            Some("Completed".to_string())
        }
    }

    #[tokio::test]
    async fn test_streamify_basic() {
        use crate::settings::reset_settings;
        reset_settings();

        let result = streamify(
            "TestModule",
            || async {
                Ok(Prediction::from_completions(
                    vec![std::collections::HashMap::new()],
                    None,
                ))
            },
            StreamifyOptions::default(),
        )
        .await;

        // Should have at least start status and final prediction
        assert!(result.len() >= 2);
        assert!(
            matches!(&result[0], StreamValue::Status(s) if s.message == "Starting program execution")
        );
        assert!(matches!(&result[result.len() - 1], StreamValue::Result(_)));
    }

    #[tokio::test]
    async fn test_streamify_with_status_provider() {
        use crate::settings::reset_settings;
        reset_settings();

        let result = streamify(
            "MyModule",
            || async {
                Ok(Prediction::from_completions(
                    vec![std::collections::HashMap::new()],
                    None,
                ))
            },
            StreamifyOptions {
                status_message_provider: Some(Box::new(TestStatusProvider)),
                ..Default::default()
            },
        )
        .await;

        // Start message should come from provider
        assert!(matches!(&result[0], StreamValue::Status(s) if s.message == "Starting MyModule"));
        // End message should come from provider
        let has_end = result
            .iter()
            .any(|v| matches!(v, StreamValue::Status(s) if s.message == "Completed"));
        assert!(has_end);
    }

    #[tokio::test]
    async fn test_streamify_error_handling() {
        use crate::settings::reset_settings;
        reset_settings();

        let result: Vec<StreamValue> = streamify(
            "ErrorModule",
            || async { Err(crate::error::DspyError::Other("test error".to_string())) },
            StreamifyOptions::default(),
        )
        .await;

        // Should have start status and error status
        assert!(result.len() >= 2);
        let has_error = result
            .iter()
            .any(|v| matches!(v, StreamValue::Status(s) if s.message.contains("test error")));
        assert!(has_error);
    }

    #[tokio::test]
    async fn test_streamify_exclude_final_prediction() {
        use crate::settings::reset_settings;
        reset_settings();

        // With no listeners, prediction is always included even when include_final_prediction=false
        let result = streamify(
            "TestModule",
            || async {
                Ok(Prediction::from_completions(
                    vec![std::collections::HashMap::new()],
                    None,
                ))
            },
            StreamifyOptions {
                include_final_prediction: false,
                ..Default::default()
            },
        )
        .await;

        // Should still include prediction because no listeners are present
        let has_prediction = result.iter().any(|v| matches!(v, StreamValue::Result(_)));
        assert!(has_prediction);
    }

    #[tokio::test]
    async fn test_streamify_with_listener() {
        use crate::settings::reset_settings;
        use std::sync::{Arc, Mutex};
        reset_settings();

        let listener = Arc::new(Mutex::new(StreamListener::new("answer", None, false)));

        let result = streamify(
            "TestModule",
            || async {
                Ok(Prediction::from_completions(
                    vec![std::collections::HashMap::new()],
                    None,
                ))
            },
            StreamifyOptions {
                stream_listeners: vec![listener],
                ..Default::default()
            },
        )
        .await;

        assert!(!result.is_empty());
        assert!(matches!(&result[0], StreamValue::Status(_)));
    }
}
