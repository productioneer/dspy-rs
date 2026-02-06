//! Settings — global + context-local configuration.
//! Python equivalent: dspy/dsp/settings.py
//!
//! Uses thread-local storage for synchronous access within async tasks.
//! `configure()` sets global defaults, `with_settings()` provides scoped overrides.

use crate::adapter::Adapter;
use crate::lm::LM;
use crate::predict::Trace;
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

/// Global settings (default LM, adapter, etc.)
#[derive(Clone)]
pub struct Settings {
    pub lm: Option<Arc<dyn LM>>,
    pub adapter: Option<Arc<dyn Adapter>>,
    pub trace: Option<Arc<Mutex<Vec<Trace>>>>,
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
}
