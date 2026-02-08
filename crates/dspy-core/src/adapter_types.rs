//! Adapter Types — rich input/output types for DSPy signatures.
//!
//! These types format content for multimodal LLM APIs:
//! - Image: image URL/base64 data → image_url content block
//! - Audio: audio data → input_audio content block
//! - DSPyFile: file data/path → file content block
//! - History: conversation history as message list
//! - Code: code with language → markdown code block
//! - Reasoning: str-like wrapper for reasoning content
//!
//! Python equivalent: dspy/adapters/types/

use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::fmt;

/// Custom type markers for message content splitting.
pub const CUSTOM_TYPE_START: &str = "<<CUSTOM-TYPE-START-IDENTIFIER>>";
pub const CUSTOM_TYPE_END: &str = "<<CUSTOM-TYPE-END-IDENTIFIER>>";

/// Content block as used by OpenAI-style multimodal APIs.
pub type ContentBlock = HashMap<String, JsonValue>;

// ============================================================================
// AdapterType — base trait
// ============================================================================

/// Base trait for all custom DSPy adapter types.
/// Implementations provide `format()` to return content blocks for the LLM API.
pub trait AdapterType: fmt::Display + Send + Sync {
    /// Format this value as LLM API content blocks.
    fn format(&self) -> AdapterTypeOutput;

    /// Description of this type for prompt generation.
    fn description() -> String
    where
        Self: Sized,
    {
        String::new()
    }

    /// Serialize for embedding in messages.
    /// Wraps format() output in custom type markers for later splitting.
    fn serialize(&self) -> String {
        match self.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                let json = serde_json::to_string(&blocks).unwrap_or_default();
                format!("{}{}{}", CUSTOM_TYPE_START, json, CUSTOM_TYPE_END)
            }
            AdapterTypeOutput::Text(s) => s,
        }
    }
}

/// Output of AdapterType::format() — either content blocks or a plain string.
#[derive(Debug, Clone)]
pub enum AdapterTypeOutput {
    Blocks(Vec<ContentBlock>),
    Text(String),
}

// ============================================================================
// Image
// ============================================================================

/// Image type — URL or base64-encoded data URI.
#[derive(Debug, Clone)]
pub struct Image {
    pub url: String,
}

impl Image {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Create an Image from raw bytes with specified MIME type.
    pub fn from_bytes(bytes: &[u8], mime_type: &str) -> Self {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self {
            url: format!("data:{};base64,{}", mime_type, b64),
        }
    }

    /// Create an Image from a file path, encoding as base64 data URI.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let mime = match ext.as_str() {
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            "bmp" => "image/bmp",
            _ => "application/octet-stream",
        };
        Ok(Self::from_bytes(&data, mime))
    }
}

impl AdapterType for Image {
    fn format(&self) -> AdapterTypeOutput {
        let mut block = ContentBlock::new();
        block.insert("type".into(), JsonValue::String("image_url".into()));
        let mut inner = serde_json::Map::new();
        inner.insert("url".into(), JsonValue::String(self.url.clone()));
        block.insert("image_url".into(), JsonValue::Object(inner));
        AdapterTypeOutput::Blocks(vec![block])
    }

    fn description() -> String {
        "An image URL or base64-encoded data URI.".into()
    }
}

impl fmt::Display for Image {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.serialize())
    }
}

// ============================================================================
// Audio
// ============================================================================

/// Audio type — base64-encoded audio data with format.
#[derive(Debug, Clone)]
pub struct Audio {
    pub data: String, // base64-encoded audio data
    pub audio_format: String,
}

impl Audio {
    pub fn new(data: impl Into<String>, audio_format: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            audio_format: audio_format.into(),
        }
    }

    /// Create Audio from raw bytes.
    pub fn from_bytes(bytes: &[u8], audio_format: &str) -> Self {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self {
            data: b64,
            audio_format: audio_format.into(),
        }
    }

    /// Create Audio from a file path.
    pub fn from_file(path: &std::path::Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let fmt = match ext.as_str() {
            "wav" => "wav",
            "mp3" => "mp3",
            "ogg" => "ogg",
            "flac" => "flac",
            "m4a" => "m4a",
            "webm" => "webm",
            _ => &ext,
        };
        Ok(Self::from_bytes(&data, fmt))
    }

    /// Create Audio from a base64 data URI (data:audio/wav;base64,...).
    pub fn from_data_uri(uri: &str) -> Self {
        let parts: Vec<&str> = uri.splitn(2, ',').collect();
        let b64data = if parts.len() == 2 { parts[1] } else { "" };
        let header = parts[0];
        // Parse "data:audio/wav;base64" -> "audio/wav" -> "wav"
        let mime = header
            .split(';')
            .next()
            .unwrap_or("")
            .trim_start_matches("data:");
        let fmt = mime.split('/').nth(1).unwrap_or("").replace("x-", "");
        Self {
            data: b64data.into(),
            audio_format: fmt,
        }
    }
}

impl AdapterType for Audio {
    fn format(&self) -> AdapterTypeOutput {
        let mut block = ContentBlock::new();
        block.insert("type".into(), JsonValue::String("input_audio".into()));
        let mut inner = serde_json::Map::new();
        inner.insert("data".into(), JsonValue::String(self.data.clone()));
        inner.insert(
            "format".into(),
            JsonValue::String(self.audio_format.clone()),
        );
        block.insert("input_audio".into(), JsonValue::Object(inner));
        AdapterTypeOutput::Blocks(vec![block])
    }

    fn description() -> String {
        "Audio data encoded as base64 with an audio format.".into()
    }
}

impl fmt::Display for Audio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.serialize())
    }
}

// ============================================================================
// DSPyFile
// ============================================================================

/// File type — file data URI, file ID, or filename.
#[derive(Debug, Clone)]
pub struct DSPyFile {
    pub file_data: Option<String>,
    pub file_id: Option<String>,
    pub filename: Option<String>,
}

impl DSPyFile {
    pub fn new(
        file_data: Option<String>,
        file_id: Option<String>,
        filename: Option<String>,
    ) -> Self {
        assert!(
            file_data.is_some() || file_id.is_some() || filename.is_some(),
            "DSPyFile must have at least one of: file_data, file_id, filename"
        );
        Self {
            file_data,
            file_id,
            filename,
        }
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8], filename: Option<&str>, mime_type: &str) -> Self {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        Self {
            file_data: Some(format!("data:{};base64,{}", mime_type, b64)),
            file_id: None,
            filename: filename.map(|s| s.into()),
        }
    }

    /// Create from a file path.
    pub fn from_path(
        path: &std::path::Path,
        filename: Option<&str>,
        mime_type: Option<&str>,
    ) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let fname = filename
            .map(|s| s.to_string())
            .or_else(|| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
            });
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let mime = mime_type.unwrap_or_else(|| match ext.as_str() {
            "pdf" => "application/pdf",
            "txt" => "text/plain",
            "csv" => "text/csv",
            "json" => "application/json",
            "xml" => "application/xml",
            "html" => "text/html",
            "md" => "text/markdown",
            "png" => "image/png",
            "jpg" | "jpeg" => "image/jpeg",
            _ => "application/octet-stream",
        });
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&data);
        Ok(Self {
            file_data: Some(format!("data:{};base64,{}", mime, b64)),
            file_id: None,
            filename: fname,
        })
    }

    /// Create from an uploaded file ID.
    pub fn from_file_id(file_id: impl Into<String>, filename: Option<String>) -> Self {
        Self {
            file_data: None,
            file_id: Some(file_id.into()),
            filename,
        }
    }
}

impl AdapterType for DSPyFile {
    fn format(&self) -> AdapterTypeOutput {
        let mut block = ContentBlock::new();
        block.insert("type".into(), JsonValue::String("file".into()));
        let mut file_map = serde_json::Map::new();
        if let Some(ref fd) = self.file_data {
            file_map.insert("file_data".into(), JsonValue::String(fd.clone()));
        }
        if let Some(ref fid) = self.file_id {
            file_map.insert("file_id".into(), JsonValue::String(fid.clone()));
        }
        if let Some(ref fname) = self.filename {
            file_map.insert("filename".into(), JsonValue::String(fname.clone()));
        }
        block.insert("file".into(), JsonValue::Object(file_map));
        AdapterTypeOutput::Blocks(vec![block])
    }

    fn description() -> String {
        "A file input with data URI, file ID, or filename.".into()
    }
}

impl fmt::Display for DSPyFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.serialize())
    }
}

// ============================================================================
// History
// ============================================================================

/// Conversation history — a list of message dicts keyed by signature fields.
#[derive(Debug, Clone)]
pub struct History {
    pub messages: Vec<HashMap<String, JsonValue>>,
}

impl History {
    pub fn new(messages: Vec<HashMap<String, JsonValue>>) -> Self {
        Self { messages }
    }
}

impl fmt::Display for History {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let json = serde_json::to_string(&self.messages).unwrap_or_default();
        write!(f, "{}", json)
    }
}

// ============================================================================
// Code
// ============================================================================

/// Code type with language annotation.
/// Strips markdown code blocks during construction.
#[derive(Debug, Clone)]
pub struct Code {
    pub code: String,
    pub language: String,
}

impl Code {
    pub fn new(code: impl Into<String>, language: impl Into<String>) -> Self {
        let code_str = code.into();
        Self {
            code: filter_code(&code_str),
            language: language.into(),
        }
    }

    /// Description with language info.
    pub fn description_with_language(language: &str) -> String {
        format!(
            "Code represented in a string, specified in the `code` field. If this is an output field, the code \
             field should follow the markdown code block format, e.g.\n```{}\n{{code}}\n```\n\
             Programming language: {}",
            language.to_lowercase(),
            language
        )
    }
}

impl AdapterType for Code {
    fn format(&self) -> AdapterTypeOutput {
        AdapterTypeOutput::Text(self.code.clone())
    }

    fn description() -> String {
        Self::description_with_language("python")
    }

    /// Override serialize to return raw code (no custom type markers).
    fn serialize(&self) -> String {
        self.code.clone()
    }
}

impl fmt::Display for Code {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code)
    }
}

/// Extract code from markdown code blocks, stripping language identifier.
fn filter_code(code: &str) -> String {
    // Case 1: ```language\n{code}\n```
    let re1 = regex::Regex::new(r"```(?:[^\n]*)\n([\s\S]*?)```").unwrap();
    if let Some(caps) = re1.captures(code) {
        return caps[1].trim().to_string();
    }

    // Case 2: ```{code}``` (no language, possibly single-line)
    let re2 = regex::Regex::new(r"```([\s\S]*?)```").unwrap();
    if let Some(caps) = re2.captures(code) {
        return caps[1].trim().to_string();
    }

    code.to_string()
}

// ============================================================================
// Reasoning
// ============================================================================

/// Reasoning type — str-like wrapper for LM reasoning content.
#[derive(Debug, Clone)]
pub struct Reasoning {
    pub content: String,
}

impl Reasoning {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn contains(&self, s: &str) -> bool {
        self.content.contains(s)
    }

    pub fn trim(&self) -> &str {
        self.content.trim()
    }

    pub fn to_lowercase(&self) -> String {
        self.content.to_lowercase()
    }

    pub fn to_uppercase(&self) -> String {
        self.content.to_uppercase()
    }

    /// Parse reasoning from LM response if available.
    pub fn from_lm_response(response: &HashMap<String, JsonValue>) -> Option<Self> {
        response
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| Self::new(s))
    }
}

impl AdapterType for Reasoning {
    fn format(&self) -> AdapterTypeOutput {
        AdapterTypeOutput::Text(self.content.clone())
    }

    fn serialize(&self) -> String {
        self.content.clone()
    }
}

impl fmt::Display for Reasoning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.content)
    }
}

// ============================================================================
// Document (experimental — for Anthropic Citations API)
// ============================================================================

/// Allowed media types for Document content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentMediaType {
    TextPlain,
    ApplicationPdf,
}

impl DocumentMediaType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentMediaType::TextPlain => "text/plain",
            DocumentMediaType::ApplicationPdf => "application/pdf",
        }
    }
}

impl fmt::Display for DocumentMediaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A document type for providing content that can be cited by language models.
///
/// Represents documents for citation-enabled responses, particularly useful
/// with Anthropic's Citations API. Marked @experimental in Python DSPy (v3.0.4).
#[derive(Debug, Clone)]
pub struct Document {
    pub data: String,
    pub title: Option<String>,
    pub media_type: DocumentMediaType,
    pub context: Option<String>,
}

impl Document {
    /// Create a Document with just data content.
    pub fn new(data: impl Into<String>) -> Self {
        Self {
            data: data.into(),
            title: None,
            media_type: DocumentMediaType::TextPlain,
            context: None,
        }
    }

    /// Create a Document with all fields.
    pub fn with_options(
        data: impl Into<String>,
        title: Option<String>,
        media_type: DocumentMediaType,
        context: Option<String>,
    ) -> Self {
        Self {
            data: data.into(),
            title,
            media_type,
            context,
        }
    }
}

impl AdapterType for Document {
    fn format(&self) -> AdapterTypeOutput {
        let mut source = serde_json::Map::new();
        source.insert("type".into(), JsonValue::String("text".into()));
        source.insert(
            "media_type".into(),
            JsonValue::String(self.media_type.as_str().into()),
        );
        source.insert("data".into(), JsonValue::String(self.data.clone()));

        let mut citations = serde_json::Map::new();
        citations.insert("enabled".into(), JsonValue::Bool(true));

        let mut block = ContentBlock::new();
        block.insert("type".into(), JsonValue::String("document".into()));
        block.insert("source".into(), JsonValue::Object(source));
        block.insert("citations".into(), JsonValue::Object(citations));

        if let Some(ref title) = self.title {
            block.insert("title".into(), JsonValue::String(title.clone()));
        }
        if let Some(ref ctx) = self.context {
            block.insert("context".into(), JsonValue::String(ctx.clone()));
        }

        AdapterTypeOutput::Blocks(vec![block])
    }

    fn description() -> String {
        "A document containing text content that can be referenced and cited. \
         Include the full text content and optionally a title for proper referencing."
            .into()
    }
}

impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let title_part = match &self.title {
            Some(t) => format!("'{}': ", t),
            None => String::new(),
        };
        write!(f, "Document({}{} chars)", title_part, self.data.len())
    }
}

// ============================================================================
// Citations (experimental — for Anthropic Citations API)
// ============================================================================

/// Individual citation with character location information.
#[derive(Debug, Clone)]
pub struct Citation {
    pub citation_type: String,
    pub cited_text: String,
    pub document_index: usize,
    pub document_title: Option<String>,
    pub start_char_index: usize,
    pub end_char_index: usize,
    pub supported_text: Option<String>,
}

impl Citation {
    /// Create a Citation with required fields.
    pub fn new(
        cited_text: impl Into<String>,
        document_index: usize,
        start_char_index: usize,
        end_char_index: usize,
    ) -> Self {
        Self {
            citation_type: "char_location".into(),
            cited_text: cited_text.into(),
            document_index,
            document_title: None,
            start_char_index,
            end_char_index,
            supported_text: None,
        }
    }

    /// Create a Citation with all fields.
    pub fn with_options(
        citation_type: impl Into<String>,
        cited_text: impl Into<String>,
        document_index: usize,
        document_title: Option<String>,
        start_char_index: usize,
        end_char_index: usize,
        supported_text: Option<String>,
    ) -> Self {
        Self {
            citation_type: citation_type.into(),
            cited_text: cited_text.into(),
            document_index,
            document_title,
            start_char_index,
            end_char_index,
            supported_text,
        }
    }

    /// Format citation as a JSON-compatible map.
    pub fn format(&self) -> HashMap<String, JsonValue> {
        let mut result = HashMap::new();
        result.insert("type".into(), JsonValue::String(self.citation_type.clone()));
        result.insert(
            "cited_text".into(),
            JsonValue::String(self.cited_text.clone()),
        );
        result.insert(
            "document_index".into(),
            JsonValue::Number(serde_json::Number::from(self.document_index)),
        );
        result.insert(
            "start_char_index".into(),
            JsonValue::Number(serde_json::Number::from(self.start_char_index)),
        );
        result.insert(
            "end_char_index".into(),
            JsonValue::Number(serde_json::Number::from(self.end_char_index)),
        );
        if let Some(ref title) = self.document_title {
            result.insert(
                "document_title".into(),
                JsonValue::String(title.clone()),
            );
        }
        if let Some(ref text) = self.supported_text {
            result.insert(
                "supported_text".into(),
                JsonValue::String(text.clone()),
            );
        }
        result
    }
}

/// Citations extracted from an LM response with source references.
///
/// Container for citation objects returned by models that support citation
/// extraction (for instance, Anthropic's Citations API via LiteLLM).
/// Marked @experimental in Python DSPy (v3.0.4).
#[derive(Debug, Clone)]
pub struct Citations {
    pub citations: Vec<Citation>,
}

impl Citations {
    pub fn new(citations: Vec<Citation>) -> Self {
        Self { citations }
    }

    /// Create Citations from a list of JSON values (dicts).
    pub fn from_json_list(dicts: &[JsonValue]) -> Self {
        let citations = dicts
            .iter()
            .filter_map(|v| {
                let obj = v.as_object()?;
                let cited_text = obj.get("cited_text")?.as_str()?.to_string();
                let document_index = obj.get("document_index")?.as_u64()? as usize;
                let start_char_index = obj.get("start_char_index")?.as_u64()? as usize;
                let end_char_index = obj.get("end_char_index")?.as_u64()? as usize;
                let citation_type = obj
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("char_location")
                    .to_string();
                let document_title = obj
                    .get("document_title")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                let supported_text = obj
                    .get("supported_text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                Some(Citation {
                    citation_type,
                    cited_text,
                    document_index,
                    document_title,
                    start_char_index,
                    end_char_index,
                    supported_text,
                })
            })
            .collect();
        Self { citations }
    }

    /// Parse citations from an LM response if present.
    pub fn parse_lm_response(response: &HashMap<String, JsonValue>) -> Option<Self> {
        let arr = response.get("citations")?.as_array()?;
        Some(Self::from_json_list(arr))
    }

    pub fn len(&self) -> usize {
        self.citations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.citations.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Citation> {
        self.citations.iter()
    }
}

impl AdapterType for Citations {
    fn format(&self) -> AdapterTypeOutput {
        let blocks: Vec<ContentBlock> = self
            .citations
            .iter()
            .map(|c| c.format())
            .collect();
        AdapterTypeOutput::Blocks(blocks)
    }

    fn description() -> String {
        "Citations with quoted text and source references. \
         Include the exact text being cited and information about its source."
            .into()
    }
}

impl fmt::Display for Citations {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Citations({} citations)", self.citations.len())
    }
}

// ============================================================================
// Message content splitting
// ============================================================================

/// A message with role and content (string or content blocks).
#[derive(Debug, Clone)]
pub struct TypedMessage {
    pub role: String,
    pub content: MessageContent,
}

/// Message content — either a plain string or an array of content blocks.
#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Split user message content around custom type markers into content blocks.
pub fn split_message_content_for_custom_types(messages: &mut [TypedMessage]) {
    for message in messages.iter_mut() {
        if message.role != "user" {
            continue;
        }
        let text = match &message.content {
            MessageContent::Text(t) => t.clone(),
            MessageContent::Blocks(_) => continue,
        };

        let pattern = format!(
            "{}([\\s\\S]*?){}",
            regex::escape(CUSTOM_TYPE_START),
            regex::escape(CUSTOM_TYPE_END)
        );
        let re = regex::Regex::new(&pattern).unwrap();

        let mut result: Vec<ContentBlock> = Vec::new();
        let mut last_end = 0;
        let mut found_any = false;

        for caps in re.captures_iter(&text) {
            found_any = true;
            let full_match = caps.get(0).unwrap();
            let start = full_match.start();

            // Text before this custom type
            if start > last_end {
                let mut block = ContentBlock::new();
                block.insert("type".into(), JsonValue::String("text".into()));
                block.insert(
                    "text".into(),
                    JsonValue::String(text[last_end..start].into()),
                );
                result.push(block);
            }

            // Parse the JSON content
            let custom_content = caps[1].trim();
            if let Ok(parsed) = serde_json::from_str::<JsonValue>(custom_content) {
                if let Some(arr) = parsed.as_array() {
                    for item in arr {
                        if let Some(obj) = item.as_object() {
                            let block: ContentBlock = obj
                                .iter()
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect();
                            result.push(block);
                        }
                    }
                } else {
                    let mut block = ContentBlock::new();
                    block.insert("type".into(), JsonValue::String("text".into()));
                    block.insert(
                        "text".into(),
                        JsonValue::String(custom_content.into()),
                    );
                    result.push(block);
                }
            } else {
                let mut block = ContentBlock::new();
                block.insert("type".into(), JsonValue::String("text".into()));
                block.insert(
                    "text".into(),
                    JsonValue::String(custom_content.into()),
                );
                result.push(block);
            }

            last_end = full_match.end();
        }

        if !found_any {
            continue;
        }

        // Remaining text after last match
        if last_end < text.len() {
            let mut block = ContentBlock::new();
            block.insert("type".into(), JsonValue::String("text".into()));
            block.insert("text".into(), JsonValue::String(text[last_end..].into()));
            result.push(block);
        }

        message.content = MessageContent::Blocks(result);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Image --

    #[test]
    fn image_format_returns_content_block() {
        let img = Image::new("https://example.com/photo.jpg");
        match img.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "image_url");
                let inner = blocks[0]["image_url"].as_object().unwrap();
                assert_eq!(inner["url"], "https://example.com/photo.jpg");
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn image_serialize_wraps_in_markers() {
        let img = Image::new("https://example.com/photo.jpg");
        let s = img.serialize();
        assert!(s.contains(CUSTOM_TYPE_START));
        assert!(s.contains(CUSTOM_TYPE_END));
        assert!(s.contains("image_url"));
    }

    #[test]
    fn image_from_bytes_creates_data_uri() {
        let bytes = vec![0x89, 0x50, 0x4e, 0x47];
        let img = Image::from_bytes(&bytes, "image/png");
        assert!(img.url.starts_with("data:image/png;base64,"));
        // Verify round-trip
        let b64 = img.url.split(',').nth(1).unwrap();
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap();
        assert_eq!(decoded, bytes);
    }

    // -- Audio --

    #[test]
    fn audio_format_returns_content_block() {
        let audio = Audio::new("base64data", "wav");
        match audio.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "input_audio");
                let inner = blocks[0]["input_audio"].as_object().unwrap();
                assert_eq!(inner["data"], "base64data");
                assert_eq!(inner["format"], "wav");
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn audio_from_bytes() {
        let bytes = vec![0x52, 0x49, 0x46, 0x46];
        let audio = Audio::from_bytes(&bytes, "wav");
        assert_eq!(audio.audio_format, "wav");
        use base64::Engine;
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&audio.data)
            .unwrap();
        assert_eq!(decoded, bytes);
    }

    #[test]
    fn audio_from_data_uri() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"test audio");
        let uri = format!("data:audio/wav;base64,{}", b64);
        let audio = Audio::from_data_uri(&uri);
        assert_eq!(audio.audio_format, "wav");
        assert_eq!(audio.data, b64);
    }

    #[test]
    fn audio_from_data_uri_strips_x_prefix() {
        use base64::Engine;
        let b64 = base64::engine::general_purpose::STANDARD.encode(b"test");
        let uri = format!("data:audio/x-wav;base64,{}", b64);
        let audio = Audio::from_data_uri(&uri);
        assert_eq!(audio.audio_format, "wav");
    }

    // -- DSPyFile --

    #[test]
    fn dspy_file_format_with_data() {
        let file = DSPyFile::new(
            Some("data:application/pdf;base64,abc123".into()),
            None,
            Some("test.pdf".into()),
        );
        match file.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "file");
                let inner = blocks[0]["file"].as_object().unwrap();
                assert_eq!(
                    inner["file_data"],
                    "data:application/pdf;base64,abc123"
                );
                assert_eq!(inner["filename"], "test.pdf");
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn dspy_file_from_file_id() {
        let file = DSPyFile::from_file_id("file-123", Some("doc.pdf".into()));
        match file.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                let inner = blocks[0]["file"].as_object().unwrap();
                assert_eq!(inner["file_id"], "file-123");
                assert_eq!(inner["filename"], "doc.pdf");
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn dspy_file_from_bytes() {
        let file = DSPyFile::from_bytes(b"content", Some("test.txt"), "text/plain");
        assert!(file.file_data.as_ref().unwrap().starts_with("data:text/plain;base64,"));
        assert_eq!(file.filename.as_deref(), Some("test.txt"));
    }

    #[test]
    #[should_panic(expected = "DSPyFile must have at least one of")]
    fn dspy_file_empty_panics() {
        DSPyFile::new(None, None, None);
    }

    // -- History --

    #[test]
    fn history_stores_messages() {
        let mut msg = HashMap::new();
        msg.insert("role".into(), JsonValue::String("user".into()));
        msg.insert("content".into(), JsonValue::String("Hello".into()));
        let history = History::new(vec![msg.clone()]);
        assert_eq!(history.messages.len(), 1);
    }

    #[test]
    fn history_to_string_is_json() {
        let mut msg = HashMap::new();
        msg.insert("role".into(), JsonValue::String("user".into()));
        let history = History::new(vec![msg]);
        let s = history.to_string();
        let parsed: Vec<HashMap<String, JsonValue>> = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.len(), 1);
    }

    // -- Code --

    #[test]
    fn code_format_returns_raw_code() {
        let code = Code::new("print('hello')", "python");
        match code.format() {
            AdapterTypeOutput::Text(s) => assert_eq!(s, "print('hello')"),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn code_strips_markdown_with_language() {
        let code = Code::new("```python\nprint('hello')\n```", "python");
        assert_eq!(code.code, "print('hello')");
    }

    #[test]
    fn code_strips_markdown_without_language() {
        let code = Code::new("```\nsome code\n```", "python");
        assert_eq!(code.code, "some code");
    }

    #[test]
    fn code_preserves_plain_code() {
        let code = Code::new("x = 1\ny = 2", "python");
        assert_eq!(code.code, "x = 1\ny = 2");
    }

    #[test]
    fn code_serialize_no_markers() {
        let code = Code::new("x = 1", "python");
        let s = code.serialize();
        assert_eq!(s, "x = 1");
        assert!(!s.contains(CUSTOM_TYPE_START));
    }

    #[test]
    fn code_description_includes_language() {
        let desc = Code::description_with_language("javascript");
        assert!(desc.contains("javascript"));
    }

    // -- Reasoning --

    #[test]
    fn reasoning_format_returns_content() {
        let r = Reasoning::new("Step 1: Think carefully");
        match r.format() {
            AdapterTypeOutput::Text(s) => assert_eq!(s, "Step 1: Think carefully"),
            _ => panic!("Expected Text"),
        }
    }

    #[test]
    fn reasoning_serialize_no_markers() {
        let r = Reasoning::new("reasoning text");
        let s = r.serialize();
        assert_eq!(s, "reasoning text");
        assert!(!s.contains(CUSTOM_TYPE_START));
    }

    #[test]
    fn reasoning_str_methods() {
        let r = Reasoning::new("  Hello World  ");
        assert_eq!(r.len(), 15);
        assert!(!r.is_empty());
        assert!(r.contains("World"));
        assert!(!r.contains("xyz"));
        assert_eq!(r.trim(), "Hello World");
        assert_eq!(r.to_lowercase(), "  hello world  ");
        assert_eq!(r.to_uppercase(), "  HELLO WORLD  ");
    }

    #[test]
    fn reasoning_from_lm_response() {
        let mut resp = HashMap::new();
        resp.insert(
            "reasoning_content".into(),
            JsonValue::String("I think therefore I am".into()),
        );
        let r = Reasoning::from_lm_response(&resp);
        assert!(r.is_some());
        assert_eq!(r.unwrap().content, "I think therefore I am");
    }

    #[test]
    fn reasoning_from_lm_response_returns_none() {
        let mut resp = HashMap::new();
        resp.insert("text".into(), JsonValue::String("just text".into()));
        assert!(Reasoning::from_lm_response(&resp).is_none());
    }

    // -- Document --

    #[test]
    fn document_from_string() {
        let doc = Document::new("Hello world");
        assert_eq!(doc.data, "Hello world");
        assert_eq!(doc.media_type, DocumentMediaType::TextPlain);
        assert!(doc.title.is_none());
        assert!(doc.context.is_none());
    }

    #[test]
    fn document_with_all_fields() {
        let doc = Document::with_options(
            "The Earth orbits the Sun.",
            Some("Astronomy Facts".into()),
            DocumentMediaType::ApplicationPdf,
            Some("Science textbook".into()),
        );
        assert_eq!(doc.data, "The Earth orbits the Sun.");
        assert_eq!(doc.title.as_deref(), Some("Astronomy Facts"));
        assert_eq!(doc.media_type, DocumentMediaType::ApplicationPdf);
        assert_eq!(doc.context.as_deref(), Some("Science textbook"));
    }

    #[test]
    fn document_format_returns_document_block() {
        let doc = Document::with_options(
            "Water boils at 100C.",
            Some("Physics".into()),
            DocumentMediaType::TextPlain,
            None,
        );
        match doc.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["type"], "document");
                let source = blocks[0]["source"].as_object().unwrap();
                assert_eq!(source["type"], "text");
                assert_eq!(source["media_type"], "text/plain");
                assert_eq!(source["data"], "Water boils at 100C.");
                let cit = blocks[0]["citations"].as_object().unwrap();
                assert_eq!(cit["enabled"], true);
                assert_eq!(blocks[0]["title"], "Physics");
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn document_format_omits_optional_fields() {
        let doc = Document::new("Just data");
        match doc.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert!(blocks[0].get("title").is_none());
                assert!(blocks[0].get("context").is_none());
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn document_display_with_title() {
        let doc = Document::with_options(
            "Hello",
            Some("Greeting".into()),
            DocumentMediaType::TextPlain,
            None,
        );
        assert_eq!(format!("{}", doc), "Document('Greeting': 5 chars)");
    }

    #[test]
    fn document_display_without_title() {
        let doc = Document::new("Hello");
        assert_eq!(format!("{}", doc), "Document(5 chars)");
    }

    #[test]
    fn document_serialize_wraps_in_markers() {
        let doc = Document::new("test");
        let s = doc.serialize();
        assert!(s.contains(CUSTOM_TYPE_START));
        assert!(s.contains(CUSTOM_TYPE_END));
        assert!(s.contains("document"));
    }

    #[test]
    fn document_description_non_empty() {
        let desc = Document::description();
        assert!(!desc.is_empty());
        assert!(desc.contains("document"));
    }

    // -- Citation & Citations --

    #[test]
    fn citation_new_with_required_fields() {
        let c = Citation::new("The sky is blue", 0, 0, 15);
        assert_eq!(c.citation_type, "char_location");
        assert_eq!(c.cited_text, "The sky is blue");
        assert_eq!(c.document_index, 0);
        assert_eq!(c.start_char_index, 0);
        assert_eq!(c.end_char_index, 15);
        assert!(c.document_title.is_none());
        assert!(c.supported_text.is_none());
    }

    #[test]
    fn citation_with_all_fields() {
        let c = Citation::with_options(
            "custom",
            "quote",
            1,
            Some("Doc Title".into()),
            10,
            20,
            Some("full sentence".into()),
        );
        assert_eq!(c.citation_type, "custom");
        assert_eq!(c.document_title.as_deref(), Some("Doc Title"));
        assert_eq!(c.supported_text.as_deref(), Some("full sentence"));
    }

    #[test]
    fn citation_format_includes_required_and_optional() {
        let c = Citation::with_options(
            "char_location",
            "test",
            0,
            Some("Title".into()),
            0,
            4,
            Some("test sentence".into()),
        );
        let f = c.format();
        assert_eq!(f["type"], JsonValue::String("char_location".into()));
        assert_eq!(f["cited_text"], JsonValue::String("test".into()));
        assert_eq!(f["document_index"], JsonValue::Number(0.into()));
        assert_eq!(f["start_char_index"], JsonValue::Number(0.into()));
        assert_eq!(f["end_char_index"], JsonValue::Number(4.into()));
        assert_eq!(f["document_title"], JsonValue::String("Title".into()));
        assert_eq!(
            f["supported_text"],
            JsonValue::String("test sentence".into())
        );
    }

    #[test]
    fn citation_format_omits_optional_when_none() {
        let c = Citation::new("text", 0, 0, 4);
        let f = c.format();
        assert!(!f.contains_key("document_title"));
        assert!(!f.contains_key("supported_text"));
    }

    #[test]
    fn citations_from_json_list() {
        let dicts = vec![
            serde_json::json!({
                "cited_text": "sky is blue",
                "document_index": 0,
                "document_title": "Weather",
                "start_char_index": 0,
                "end_char_index": 11
            }),
            serde_json::json!({
                "cited_text": "water is wet",
                "document_index": 1,
                "start_char_index": 5,
                "end_char_index": 17
            }),
        ];
        let citations = Citations::from_json_list(&dicts);
        assert_eq!(citations.len(), 2);
        assert_eq!(citations.citations[0].cited_text, "sky is blue");
        assert_eq!(
            citations.citations[0].document_title.as_deref(),
            Some("Weather")
        );
        assert!(citations.citations[1].document_title.is_none());
    }

    #[test]
    fn citations_parse_lm_response() {
        let mut resp = HashMap::new();
        resp.insert(
            "citations".into(),
            serde_json::json!([
                {
                    "cited_text": "test",
                    "document_index": 0,
                    "start_char_index": 0,
                    "end_char_index": 4
                }
            ]),
        );
        let citations = Citations::parse_lm_response(&resp);
        assert!(citations.is_some());
        assert_eq!(citations.unwrap().len(), 1);
    }

    #[test]
    fn citations_parse_lm_response_returns_none() {
        let resp = HashMap::new();
        assert!(Citations::parse_lm_response(&resp).is_none());

        let mut resp2 = HashMap::new();
        resp2.insert("text".into(), JsonValue::String("hello".into()));
        assert!(Citations::parse_lm_response(&resp2).is_none());
    }

    #[test]
    fn citations_format_returns_blocks() {
        let c = Citations::new(vec![Citation::new("test", 0, 0, 4)]);
        match c.format() {
            AdapterTypeOutput::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(
                    blocks[0]["cited_text"],
                    JsonValue::String("test".into())
                );
            }
            _ => panic!("Expected Blocks"),
        }
    }

    #[test]
    fn citations_display() {
        let c = Citations::new(vec![
            Citation::new("a", 0, 0, 1),
            Citation::new("b", 1, 0, 1),
        ]);
        assert_eq!(format!("{}", c), "Citations(2 citations)");
    }

    #[test]
    fn citations_serialize_wraps_in_markers() {
        let c = Citations::new(vec![Citation::new("test", 0, 0, 4)]);
        let s = c.serialize();
        assert!(s.contains(CUSTOM_TYPE_START));
        assert!(s.contains(CUSTOM_TYPE_END));
    }

    #[test]
    fn citations_description_non_empty() {
        let desc = Citations::description();
        assert!(!desc.is_empty());
    }

    #[test]
    fn citations_iter() {
        let c = Citations::new(vec![
            Citation::new("a", 0, 0, 1),
            Citation::new("b", 1, 0, 1),
        ]);
        let texts: Vec<&str> = c.iter().map(|c| c.cited_text.as_str()).collect();
        assert_eq!(texts, vec!["a", "b"]);
    }

    // -- splitMessageContentForCustomTypes --

    #[test]
    fn split_leaves_non_user_unchanged() {
        let mut messages = vec![TypedMessage {
            role: "assistant".into(),
            content: MessageContent::Text("hello".into()),
        }];
        split_message_content_for_custom_types(&mut messages);
        match &messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("Should remain Text"),
        }
    }

    #[test]
    fn split_leaves_user_without_types_unchanged() {
        let mut messages = vec![TypedMessage {
            role: "user".into(),
            content: MessageContent::Text("just text".into()),
        }];
        split_message_content_for_custom_types(&mut messages);
        match &messages[0].content {
            MessageContent::Text(s) => assert_eq!(s, "just text"),
            _ => panic!("Should remain Text"),
        }
    }

    #[test]
    fn split_user_message_with_embedded_type() {
        let img_json = serde_json::json!([{"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}]);
        let content = format!(
            "Look at this: {}{}{} What do you see?",
            CUSTOM_TYPE_START, img_json, CUSTOM_TYPE_END
        );
        let mut messages = vec![TypedMessage {
            role: "user".into(),
            content: MessageContent::Text(content),
        }];
        split_message_content_for_custom_types(&mut messages);

        match &messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 3);
                assert_eq!(blocks[0]["text"], "Look at this: ");
                assert_eq!(blocks[1]["type"], "image_url");
                assert_eq!(blocks[2]["text"], " What do you see?");
            }
            _ => panic!("Should be Blocks"),
        }
    }

    #[test]
    fn split_multiple_custom_types() {
        let img1 = serde_json::json!([{"type": "image_url", "image_url": {"url": "a"}}]);
        let img2 = serde_json::json!([{"type": "image_url", "image_url": {"url": "b"}}]);
        let content = format!(
            "{}{}{} and {}{}{}",
            CUSTOM_TYPE_START, img1, CUSTOM_TYPE_END,
            CUSTOM_TYPE_START, img2, CUSTOM_TYPE_END
        );
        let mut messages = vec![TypedMessage {
            role: "user".into(),
            content: MessageContent::Text(content),
        }];
        split_message_content_for_custom_types(&mut messages);

        match &messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 3);
                assert_eq!(blocks[0]["type"], "image_url");
                assert_eq!(blocks[1]["text"], " and ");
                assert_eq!(blocks[2]["type"], "image_url");
            }
            _ => panic!("Should be Blocks"),
        }
    }

    #[test]
    fn split_handles_invalid_json_gracefully() {
        let content = format!("{}not json{}", CUSTOM_TYPE_START, CUSTOM_TYPE_END);
        let mut messages = vec![TypedMessage {
            role: "user".into(),
            content: MessageContent::Text(content),
        }];
        split_message_content_for_custom_types(&mut messages);

        match &messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0]["text"], "not json");
            }
            _ => panic!("Should be Blocks"),
        }
    }

    #[test]
    fn end_to_end_image_serialize_then_split() {
        let img = Image::new("https://example.com/photo.jpg");
        let content = format!("Describe this image: {}", img.serialize());
        let mut messages = vec![TypedMessage {
            role: "user".into(),
            content: MessageContent::Text(content),
        }];
        split_message_content_for_custom_types(&mut messages);

        match &messages[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert_eq!(blocks[0]["text"], "Describe this image: ");
                assert_eq!(blocks[1]["type"], "image_url");
                let inner = blocks[1]["image_url"].as_object().unwrap();
                assert_eq!(inner["url"], "https://example.com/photo.jpg");
            }
            _ => panic!("Should be Blocks"),
        }
    }
}
