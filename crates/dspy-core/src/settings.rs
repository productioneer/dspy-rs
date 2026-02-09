//! Settings — global + context-local configuration.
//! Python equivalent: dspy/dsp/settings.py
//!
//! Uses thread-local storage for synchronous access within async tasks.
//! `configure()` sets global defaults, `with_settings()` provides scoped overrides.

use crate::adapter::Adapter;
use crate::cache::Cache;
use crate::lm::LM;
use crate::predict::Trace;
use crate::streaming::StreamListener;
use crate::usage_tracker::UsageTracker;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Callback for receiving streaming values (chunks, status messages, predictions).
pub type SendStreamFn = Arc<dyn Fn(crate::streaming::StreamValue) + Send + Sync>;

/// Global settings (default LM, adapter, etc.)
#[derive(Clone)]
pub struct Settings {
    pub lm: Option<Arc<dyn LM>>,
    pub adapter: Option<Arc<dyn Adapter>>,
    pub trace: Option<Arc<Mutex<Vec<Trace>>>>,
    /// Global cache instance. None = use default global cache.
    pub cache: Option<CacheSetting>,
    /// Usage tracker for the current context. Set via with_settings.
    pub usage_tracker: Option<Arc<Mutex<UsageTracker>>>,
    /// Disable LM call history recording. Default: false
    pub disable_history: bool,
    /// Max entries per LM history array. Default: 10000
    pub max_history_size: usize,
    /// Stream sender for routing LM chunks. Set by streamify().
    pub send_stream: Option<SendStreamFn>,
    /// Active stream listeners for the current context.
    pub stream_listeners: Option<Vec<Arc<Mutex<StreamListener>>>>,
    /// The Predict module currently making a call. Used for stream chunk tagging.
    /// Stored as a raw pointer-based ID since Predict types are heterogeneous.
    pub caller_predict_id: Option<usize>,
}

/// Cache setting — either a specific cache instance or disabled.
#[derive(Clone)]
pub enum CacheSetting {
    /// Use a specific cache instance
    Instance(Arc<Cache>),
    /// Caching is explicitly disabled
    Disabled,
}

impl Default for Settings {
    fn default() -> Self {
        Self::new()
    }
}

impl Settings {
    pub fn new() -> Self {
        Self {
            lm: None,
            adapter: None,
            trace: None,
            cache: None,
            usage_tracker: None,
            disable_history: false,
            max_history_size: 10000,
            send_stream: None,
            stream_listeners: None,
            caller_predict_id: None,
        }
    }

    pub fn with_lm(mut self, lm: Arc<dyn LM>) -> Self {
        self.lm = Some(lm);
        self
    }

    pub fn with_adapter(mut self, adapter: Arc<dyn Adapter>) -> Self {
        self.adapter = Some(adapter);
        self
    }

    pub fn with_trace(mut self) -> Self {
        self.trace = Some(Arc::new(Mutex::new(Vec::new())));
        self
    }

    pub fn with_cache(mut self, cache: Arc<Cache>) -> Self {
        self.cache = Some(CacheSetting::Instance(cache));
        self
    }

    pub fn with_cache_disabled(mut self) -> Self {
        self.cache = Some(CacheSetting::Disabled);
        self
    }

    pub fn with_usage_tracker(mut self, tracker: Arc<Mutex<UsageTracker>>) -> Self {
        self.usage_tracker = Some(tracker);
        self
    }

    pub fn with_disable_history(mut self, disable: bool) -> Self {
        self.disable_history = disable;
        self
    }

    pub fn with_max_history_size(mut self, size: usize) -> Self {
        self.max_history_size = size;
        self
    }

    pub fn with_send_stream(mut self, send_stream: SendStreamFn) -> Self {
        self.send_stream = Some(send_stream);
        self
    }

    pub fn with_stream_listeners(mut self, listeners: Vec<Arc<Mutex<StreamListener>>>) -> Self {
        self.stream_listeners = Some(listeners);
        self
    }
}

// Thread-local settings stack
thread_local! {
    static SETTINGS_STACK: RefCell<Vec<Settings>> = RefCell::new(Vec::new());
    static GLOBAL_SETTINGS: RefCell<Settings> = RefCell::new(Settings::new());
}

/// Set global defaults
pub fn configure(settings: Settings) {
    GLOBAL_SETTINGS.with(|gs| {
        *gs.borrow_mut() = settings;
    });
}

/// Get current effective settings (top of stack, or global)
pub fn get_settings() -> Settings {
    SETTINGS_STACK.with(|stack| {
        let stack = stack.borrow();
        if let Some(top) = stack.last() {
            top.clone()
        } else {
            GLOBAL_SETTINGS.with(|gs| gs.borrow().clone())
        }
    })
}

/// Run a closure with scoped settings override
pub fn with_settings<F, T>(settings: Settings, f: F) -> T
where
    F: FnOnce() -> T,
{
    SETTINGS_STACK.with(|stack| {
        stack.borrow_mut().push(settings);
    });
    let result = f();
    SETTINGS_STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
    result
}

/// Reset all settings (for test isolation)
pub fn reset_settings() {
    SETTINGS_STACK.with(|stack| {
        stack.borrow_mut().clear();
    });
    GLOBAL_SETTINGS.with(|gs| {
        *gs.borrow_mut() = Settings::new();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        reset_settings();
        let s = get_settings();
        assert!(s.lm.is_none());
        assert!(s.adapter.is_none());
        assert!(s.cache.is_none());
        assert!(s.usage_tracker.is_none());
        assert!(!s.disable_history);
        assert_eq!(s.max_history_size, 10000);
    }

    #[test]
    fn test_configure_global() {
        reset_settings();
        // No real LM to test with, just verify the API works
        configure(Settings::new());
        let s = get_settings();
        assert!(s.lm.is_none());
    }

    #[test]
    fn test_scoped_settings() {
        reset_settings();
        let outer = get_settings();
        assert!(outer.trace.is_none());

        with_settings(Settings::new().with_trace(), || {
            let inner = get_settings();
            assert!(inner.trace.is_some());
        });

        let after = get_settings();
        assert!(after.trace.is_none());
    }

    #[test]
    fn test_nested_scoped_settings() {
        reset_settings();
        with_settings(Settings::new().with_trace(), || {
            let s1 = get_settings();
            assert!(s1.trace.is_some());

            with_settings(Settings::new(), || {
                let s2 = get_settings();
                assert!(s2.trace.is_none());
            });

            let s3 = get_settings();
            assert!(s3.trace.is_some());
        });
    }

    #[test]
    fn test_cache_setting_disabled() {
        reset_settings();
        with_settings(Settings::new().with_cache_disabled(), || {
            let s = get_settings();
            assert!(matches!(s.cache, Some(CacheSetting::Disabled)));
        });
    }

    #[test]
    fn test_usage_tracker_setting() {
        reset_settings();
        let tracker = Arc::new(Mutex::new(UsageTracker::new()));
        with_settings(Settings::new().with_usage_tracker(tracker.clone()), || {
            let s = get_settings();
            assert!(s.usage_tracker.is_some());
        });
    }

    #[test]
    fn test_history_settings() {
        reset_settings();
        with_settings(
            Settings::new()
                .with_disable_history(true)
                .with_max_history_size(100),
            || {
                let s = get_settings();
                assert!(s.disable_history);
                assert_eq!(s.max_history_size, 100);
            },
        );
    }
}
