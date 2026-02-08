//! LM — language model trait, types, history, and cache integration.
//! Python equivalent: dspy/clients/lm.py + dspy/clients/base_lm.py

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::sync::Arc;

use crate::cache::{Cache, CacheConfig};
use crate::settings::{get_settings, CacheSetting};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMConfig {
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub n: Option<u32>,
}

impl LMConfig {
    pub fn new(model: &str) -> Self {
        Self {
            model: model.to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            n: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMResponse {
    pub text: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self { role: "system".to_string(), content: content.to_string() }
    }
    pub fn user(content: &str) -> Self {
        Self { role: "user".to_string(), content: content.to_string() }
    }
    pub fn assistant(content: &str) -> Self {
        Self { role: "assistant".to_string(), content: content.to_string() }
    }
}

/// History entry for an LM call.
#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntry {
    pub messages: Vec<Message>,
    pub response: Vec<LMResponse>,
    pub model: String,
    pub cache_hit: bool,
    pub timestamp: String,
}

#[async_trait]
pub trait LM: Send + Sync {
    async fn call(&self, messages: &[Message], config: &LMConfig) -> crate::error::Result<Vec<LMResponse>>;
    fn model(&self) -> &str;
    fn config(&self) -> &LMConfig;
    fn dump_state(&self) -> serde_json::Value;
}

// ============================================================
// Global history
// ============================================================

const MAX_GLOBAL_HISTORY: usize = 10000;

thread_local! {
    static GLOBAL_HISTORY: RefCell<Vec<HistoryEntry>> = RefCell::new(Vec::new());
}

/// Inspect global LM call history (last n entries).
/// Matches Python DSPy's dspy.inspect_history().
pub fn inspect_history(n: usize) -> Vec<HistoryEntry> {
    GLOBAL_HISTORY.with(|h| {
        let h = h.borrow();
        let start = h.len().saturating_sub(n);
        h[start..].to_vec()
    })
}

/// Clear global history. Primarily for testing.
pub fn clear_history() {
    GLOBAL_HISTORY.with(|h| {
        h.borrow_mut().clear();
    });
}

fn record_global_history(entry: HistoryEntry) {
    GLOBAL_HISTORY.with(|h| {
        let mut h = h.borrow_mut();
        if h.len() >= MAX_GLOBAL_HISTORY {
            h.remove(0);
        }
        h.push(entry);
    });
}

// ============================================================
// Global cache
// ============================================================

thread_local! {
    static GLOBAL_CACHE: RefCell<Option<Arc<Cache>>> = RefCell::new(None);
    static CONFIGURED_CACHE: RefCell<Option<Arc<Cache>>> = RefCell::new(None);
}

fn get_or_init_global_cache() -> Arc<Cache> {
    GLOBAL_CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if let Some(ref cache) = *c {
            cache.clone()
        } else {
            let cache = Arc::new(Cache::default_memory_only());
            *c = Some(cache.clone());
            cache
        }
    })
}

/// Configure the global cache instance.
/// Matches Python DSPy's dspy.cache.configure_cache().
pub fn configure_cache(config: CacheConfig) {
    let cache = Arc::new(Cache::new(config));
    CONFIGURED_CACHE.with(|c| {
        *c.borrow_mut() = Some(cache);
    });
}

/// Reset the global cache. Primarily for testing.
pub fn reset_global_cache() {
    CONFIGURED_CACHE.with(|c| {
        *c.borrow_mut() = None;
    });
    GLOBAL_CACHE.with(|c| {
        *c.borrow_mut() = None;
    });
}

fn get_effective_cache() -> Arc<Cache> {
    CONFIGURED_CACHE.with(|c| {
        if let Some(ref cache) = *c.borrow() {
            return cache.clone();
        }
        get_or_init_global_cache()
    })
}

// ============================================================
// call_with_cache — wrapper that adds cache/history/usage to LM calls
// ============================================================

/// Call an LM with cache, history recording, and usage tracking.
/// Use this instead of calling `lm.call()` directly to get Python DSPy parity.
pub async fn call_with_cache(
    lm: &dyn LM,
    messages: &[Message],
    config: &LMConfig,
    use_cache: bool,
) -> crate::error::Result<Vec<LMResponse>> {
    let settings = get_settings();

    // Resolve effective cache
    let cache_instance: Option<Arc<Cache>> = if !use_cache {
        None
    } else {
        match &settings.cache {
            Some(CacheSetting::Disabled) => None,
            Some(CacheSetting::Instance(c)) => Some(c.clone()),
            None => Some(get_effective_cache()),
        }
    };

    // Build cache request
    let cache_request = cache_instance.as_ref().map(|_| {
        let msg_json: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": &m.role, "content": &m.content}))
            .collect();
        serde_json::json!({
            "model": lm.model(),
            "messages": msg_json,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        })
    });

    // Check cache
    if let (Some(ref cache), Some(ref request)) = (&cache_instance, &cache_request) {
        if let Some(cached_value) = cache.get(request, None) {
            // Deserialize cached responses
            if let Ok(responses) = serde_json::from_value::<Vec<LMResponse>>(cached_value) {
                // Record history as cache hit
                if !settings.disable_history {
                    let entry = HistoryEntry {
                        messages: messages.to_vec(),
                        response: responses.clone(),
                        model: lm.model().to_string(),
                        cache_hit: true,
                        timestamp: chrono_timestamp(),
                    };
                    record_global_history(entry);
                }
                return Ok(responses);
            }
        }
    }

    // Call LM
    let responses = lm.call(messages, config).await?;

    // Store in cache
    if let (Some(ref cache), Some(ref request)) = (&cache_instance, &cache_request) {
        if let Ok(value) = serde_json::to_value(&responses) {
            cache.put(request, &value, None, true);
        }
    }

    // Track usage (only on non-cache-hit)
    if let Some(ref tracker) = settings.usage_tracker {
        if let Ok(mut tracker) = tracker.lock() {
            for r in &responses {
                if let Some(ref usage) = r.usage {
                    tracker.add_usage(
                        lm.model(),
                        serde_json::json!({
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                        }),
                    );
                }
            }
        }
    }

    // Record history
    if !settings.disable_history {
        let entry = HistoryEntry {
            messages: messages.to_vec(),
            response: responses.clone(),
            model: lm.model().to_string(),
            cache_hit: false,
            timestamp: chrono_timestamp(),
        };
        record_global_history(entry);
    }

    Ok(responses)
}

fn chrono_timestamp() -> String {
    // Simple ISO timestamp without chrono dependency
    use std::time::SystemTime;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::{configure, reset_settings, Settings};
    use crate::usage_tracker::UsageTracker;
    use std::sync::Mutex;

    struct MockLM {
        responses: Vec<LMResponse>,
        call_count: std::sync::atomic::AtomicU32,
        cfg: LMConfig,
    }

    impl MockLM {
        fn new(responses: Vec<LMResponse>) -> Self {
            Self {
                responses,
                call_count: std::sync::atomic::AtomicU32::new(0),
                cfg: LMConfig::new("test-model"),
            }
        }

        fn calls(&self) -> u32 {
            self.call_count.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(&self, _messages: &[Message], _config: &LMConfig) -> crate::error::Result<Vec<LMResponse>> {
            let idx = self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst) as usize;
            let idx = idx.min(self.responses.len() - 1);
            Ok(vec![self.responses[idx].clone()])
        }

        fn model(&self) -> &str { "test-model" }
        fn config(&self) -> &LMConfig { &self.cfg }
        fn dump_state(&self) -> serde_json::Value { serde_json::json!({}) }
    }

    #[tokio::test]
    async fn test_call_with_cache_hit() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse { text: "hello".to_string(), usage: None }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        // First call — cache miss
        let r1 = call_with_cache(&lm, &messages, &config, true).await.unwrap();
        assert_eq!(r1[0].text, "hello");
        assert_eq!(lm.calls(), 1);

        // Second call — cache hit
        let r2 = call_with_cache(&lm, &messages, &config, true).await.unwrap();
        assert_eq!(r2[0].text, "hello");
        assert_eq!(lm.calls(), 1); // not called again
    }

    #[tokio::test]
    async fn test_cache_disabled() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![
            LMResponse { text: "first".to_string(), usage: None },
            LMResponse { text: "second".to_string(), usage: None },
        ]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        let r1 = call_with_cache(&lm, &messages, &config, false).await.unwrap();
        assert_eq!(r1[0].text, "first");
        let r2 = call_with_cache(&lm, &messages, &config, false).await.unwrap();
        assert_eq!(r2[0].text, "second");
        assert_eq!(lm.calls(), 2);
    }

    #[tokio::test]
    async fn test_settings_cache_disabled() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![
            LMResponse { text: "a".to_string(), usage: None },
            LMResponse { text: "b".to_string(), usage: None },
        ]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        configure(Settings::new().with_cache_disabled());
        let r1 = call_with_cache(&lm, &messages, &config, true).await.unwrap();
        assert_eq!(r1[0].text, "a");
        let r2 = call_with_cache(&lm, &messages, &config, true).await.unwrap();
        assert_eq!(r2[0].text, "b");
        assert_eq!(lm.calls(), 2);
        reset_settings();
    }

    #[tokio::test]
    async fn test_history_recording() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse { text: "hello".to_string(), usage: None }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        call_with_cache(&lm, &messages, &config, false).await.unwrap();

        let history = inspect_history(1);
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].model, "test-model");
        assert!(!history[0].cache_hit);
    }

    #[tokio::test]
    async fn test_history_cache_hit_recorded() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse { text: "hello".to_string(), usage: None }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        call_with_cache(&lm, &messages, &config, true).await.unwrap(); // miss
        call_with_cache(&lm, &messages, &config, true).await.unwrap(); // hit

        let history = inspect_history(2);
        assert_eq!(history.len(), 2);
        assert!(!history[0].cache_hit);
        assert!(history[1].cache_hit);
    }

    #[tokio::test]
    async fn test_disable_history() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse { text: "hello".to_string(), usage: None }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        configure(Settings::new().with_disable_history(true));
        call_with_cache(&lm, &messages, &config, false).await.unwrap();
        assert_eq!(inspect_history(10).len(), 0);
        reset_settings();
    }

    #[tokio::test]
    async fn test_usage_tracking() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse {
            text: "hello".to_string(),
            usage: Some(Usage { prompt_tokens: 100, completion_tokens: 50 }),
        }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        let tracker = Arc::new(Mutex::new(UsageTracker::new()));
        configure(Settings::new().with_usage_tracker(tracker.clone()));
        call_with_cache(&lm, &messages, &config, false).await.unwrap();
        reset_settings();

        let t = tracker.lock().unwrap();
        let totals = t.get_total_tokens();
        assert!(totals.contains_key("test-model"));
        assert_eq!(totals["test-model"]["prompt_tokens"], 100.0);
    }

    #[tokio::test]
    async fn test_cache_hit_skips_usage() {
        reset_settings();
        clear_history();
        reset_global_cache();

        let lm = MockLM::new(vec![LMResponse {
            text: "hello".to_string(),
            usage: Some(Usage { prompt_tokens: 100, completion_tokens: 50 }),
        }]);
        let messages = vec![Message::user("hi")];
        let config = LMConfig::new("test-model");

        let tracker = Arc::new(Mutex::new(UsageTracker::new()));
        configure(Settings::new().with_usage_tracker(tracker.clone()));
        call_with_cache(&lm, &messages, &config, true).await.unwrap(); // miss
        call_with_cache(&lm, &messages, &config, true).await.unwrap(); // hit
        reset_settings();

        let t = tracker.lock().unwrap();
        let totals = t.get_total_tokens();
        // Only one usage entry (from the cache miss)
        assert_eq!(totals["test-model"]["prompt_tokens"], 100.0);
    }

    #[test]
    fn test_inspect_history_last_n() {
        clear_history();

        for i in 0..5 {
            record_global_history(HistoryEntry {
                messages: vec![Message::user(&format!("msg{}", i))],
                response: vec![LMResponse { text: format!("resp{}", i), usage: None }],
                model: "m".to_string(),
                cache_hit: false,
                timestamp: "0".to_string(),
            });
        }

        let last2 = inspect_history(2);
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0].messages[0].content, "msg3");
        assert_eq!(last2[1].messages[0].content, "msg4");
    }
}
