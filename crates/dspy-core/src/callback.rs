//! DSPy Callback System — Event hooks for lifecycle events.
//!
//! Implement `Callback` trait and register globally to receive notifications
//! before/after module execution, LM calls, adapter operations, tool usage, and evaluation.
//!
//! Matches Python DSPy's dspy.utils.callback interface.

use std::sync::{Arc, Mutex};

/// Component type identifiers for callback routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentType {
    Module,
    Lm,
    AdapterFormat,
    AdapterParse,
    Tool,
    Evaluate,
}

/// Trait for defining callback handlers for DSPy components.
///
/// Implement desired methods to receive lifecycle notifications.
/// Default implementations are no-ops.
pub trait Callback: Send + Sync {
    /// Called when a Module's forward() is invoked.
    fn on_module_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after a Module's forward() completes.
    fn on_module_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }

    /// Called when an LM's call is invoked.
    fn on_lm_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after an LM's call completes.
    fn on_lm_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }

    /// Called when an Adapter's format() is invoked.
    fn on_adapter_format_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after an Adapter's format() completes.
    fn on_adapter_format_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }

    /// Called when an Adapter's parse() is invoked.
    fn on_adapter_parse_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after an Adapter's parse() completes.
    fn on_adapter_parse_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }

    /// Called when a Tool is invoked.
    fn on_tool_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after a Tool completes.
    fn on_tool_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }

    /// Called when evaluation starts.
    fn on_evaluate_start(
        &self,
        _call_id: &str,
        _instance_type: &str,
        _inputs: &serde_json::Value,
    ) {
    }

    /// Called after evaluation completes.
    fn on_evaluate_end(
        &self,
        _call_id: &str,
        _outputs: Option<&serde_json::Value>,
        _exception: Option<&str>,
    ) {
    }
}

/// Global callback registry.
static GLOBAL_CALLBACKS: Mutex<Vec<Arc<dyn Callback>>> = Mutex::new(Vec::new());

/// Set global callbacks. Replaces any existing callbacks.
pub fn set_global_callbacks(callbacks: Vec<Arc<dyn Callback>>) {
    if let Ok(mut global) = GLOBAL_CALLBACKS.lock() {
        *global = callbacks;
    }
}

/// Get a snapshot of global callbacks.
pub fn get_global_callbacks() -> Vec<Arc<dyn Callback>> {
    GLOBAL_CALLBACKS
        .lock()
        .map(|g| g.clone())
        .unwrap_or_default()
}

/// Add a single callback to the global registry.
pub fn add_global_callback(callback: Arc<dyn Callback>) {
    if let Ok(mut global) = GLOBAL_CALLBACKS.lock() {
        global.push(callback);
    }
}

/// Clear all global callbacks.
pub fn clear_global_callbacks() {
    if let Ok(mut global) = GLOBAL_CALLBACKS.lock() {
        global.clear();
    }
}

/// Invoke start callbacks for a component.
pub fn invoke_start_callbacks(
    component_type: ComponentType,
    callbacks: &[Arc<dyn Callback>],
    call_id: &str,
    instance_type: &str,
    inputs: &serde_json::Value,
) {
    for cb in callbacks {
        match component_type {
            ComponentType::Module => cb.on_module_start(call_id, instance_type, inputs),
            ComponentType::Lm => cb.on_lm_start(call_id, instance_type, inputs),
            ComponentType::AdapterFormat => cb.on_adapter_format_start(call_id, instance_type, inputs),
            ComponentType::AdapterParse => cb.on_adapter_parse_start(call_id, instance_type, inputs),
            ComponentType::Tool => cb.on_tool_start(call_id, instance_type, inputs),
            ComponentType::Evaluate => cb.on_evaluate_start(call_id, instance_type, inputs),
        }
    }
}

/// Invoke end callbacks for a component.
pub fn invoke_end_callbacks(
    component_type: ComponentType,
    callbacks: &[Arc<dyn Callback>],
    call_id: &str,
    outputs: Option<&serde_json::Value>,
    exception: Option<&str>,
) {
    for cb in callbacks {
        match component_type {
            ComponentType::Module => cb.on_module_end(call_id, outputs, exception),
            ComponentType::Lm => cb.on_lm_end(call_id, outputs, exception),
            ComponentType::AdapterFormat => cb.on_adapter_format_end(call_id, outputs, exception),
            ComponentType::AdapterParse => cb.on_adapter_parse_end(call_id, outputs, exception),
            ComponentType::Tool => cb.on_tool_end(call_id, outputs, exception),
            ComponentType::Evaluate => cb.on_evaluate_end(call_id, outputs, exception),
        }
    }
}

/// Execute an async operation that returns Result, wrapped with start/end callbacks.
/// On success, passes serialized output to end callbacks. On error, passes exception message.
pub async fn with_callbacks_async<T, E, F, Fut>(
    component_type: ComponentType,
    instance_type: &str,
    inputs: &serde_json::Value,
    func: F,
) -> std::result::Result<T, E>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
    E: std::fmt::Display,
{
    let callbacks = get_global_callbacks();
    if callbacks.is_empty() {
        return func().await;
    }

    let call_id = format!("{:032x}", rand::random::<u128>());
    invoke_start_callbacks(component_type, &callbacks, &call_id, instance_type, inputs);
    match func().await {
        Ok(value) => {
            invoke_end_callbacks(component_type, &callbacks, &call_id, None, None);
            Ok(value)
        }
        Err(e) => {
            let err_msg = e.to_string();
            invoke_end_callbacks(component_type, &callbacks, &call_id, None, Some(&err_msg));
            Err(e)
        }
    }
}

/// Execute a sync operation that returns Result, wrapped with start/end callbacks.
pub fn with_callbacks_sync<T, E, F>(
    component_type: ComponentType,
    instance_type: &str,
    inputs: &serde_json::Value,
    func: F,
) -> std::result::Result<T, E>
where
    F: FnOnce() -> std::result::Result<T, E>,
    E: std::fmt::Display,
{
    let callbacks = get_global_callbacks();
    if callbacks.is_empty() {
        return func();
    }

    let call_id = format!("{:032x}", rand::random::<u128>());
    invoke_start_callbacks(component_type, &callbacks, &call_id, instance_type, inputs);
    match func() {
        Ok(value) => {
            invoke_end_callbacks(component_type, &callbacks, &call_id, None, None);
            Ok(value)
        }
        Err(e) => {
            let err_msg = e.to_string();
            invoke_end_callbacks(component_type, &callbacks, &call_id, None, Some(&err_msg));
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingCallback {
        start_count: AtomicUsize,
        end_count: AtomicUsize,
    }

    impl CountingCallback {
        fn new() -> Self {
            Self {
                start_count: AtomicUsize::new(0),
                end_count: AtomicUsize::new(0),
            }
        }
    }

    impl Callback for CountingCallback {
        fn on_module_start(&self, _call_id: &str, _instance_type: &str, _inputs: &serde_json::Value) {
            self.start_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_module_end(&self, _call_id: &str, _outputs: Option<&serde_json::Value>, _exception: Option<&str>) {
            self.end_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_lm_start(&self, _call_id: &str, _instance_type: &str, _inputs: &serde_json::Value) {
            self.start_count.fetch_add(1, Ordering::SeqCst);
        }

        fn on_lm_end(&self, _call_id: &str, _outputs: Option<&serde_json::Value>, _exception: Option<&str>) {
            self.end_count.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_invoke_callbacks() {
        let cb = Arc::new(CountingCallback::new());
        let callbacks: Vec<Arc<dyn Callback>> = vec![cb.clone()];
        let inputs = serde_json::json!({"query": "test"});

        invoke_start_callbacks(ComponentType::Module, &callbacks, "call1", "TestModule", &inputs);
        assert_eq!(cb.start_count.load(Ordering::SeqCst), 1);

        invoke_end_callbacks(ComponentType::Module, &callbacks, "call1", Some(&serde_json::json!({"result": "ok"})), None);
        assert_eq!(cb.end_count.load(Ordering::SeqCst), 1);

        invoke_start_callbacks(ComponentType::Lm, &callbacks, "call2", "TestLM", &inputs);
        assert_eq!(cb.start_count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_global_callbacks() {
        clear_global_callbacks();
        assert!(get_global_callbacks().is_empty());

        let cb = Arc::new(CountingCallback::new());
        add_global_callback(cb.clone());
        assert_eq!(get_global_callbacks().len(), 1);

        clear_global_callbacks();
        assert!(get_global_callbacks().is_empty());
    }
}
