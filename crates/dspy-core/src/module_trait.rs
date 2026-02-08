//! Module trait — composable program unit.
//! Python equivalent: dspy/primitives/module.py

use crate::callback::{with_callbacks_async, ComponentType};
use crate::error::Result;
use crate::example::Example;
use crate::lm::LM;
use crate::predict::Predict;
use crate::prediction::Prediction;
use async_trait::async_trait;
use std::sync::Arc;

#[async_trait]
pub trait Module: Send + Sync {
    async fn forward(&self, args: &Example) -> Result<Prediction>;

    /// Returns the type name for this module (used in callback reporting).
    /// Override in implementations to report the actual struct name.
    fn module_type_name(&self) -> &str {
        "Module"
    }

    /// Call this module with callbacks. This is the standard entry point
    /// (equivalent to Python's Module.__call__).
    async fn call(&self, args: &Example) -> Result<Prediction> {
        let inputs = serde_json::to_value(args).unwrap_or_default();
        with_callbacks_async(
            ComponentType::Module,
            self.module_type_name(),
            &inputs,
            || self.forward(args),
        )
        .await
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        vec![]
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![]
    }

    fn set_lm(&mut self, lm: Arc<dyn LM>) {
        for (_, pred) in self.named_predictors_mut() {
            pred.set_lm(lm.clone());
        }
    }

    fn deep_copy(&self) -> Box<dyn Module>;

    fn dump_state(&self) -> serde_json::Value {
        serde_json::json!({})
    }

    fn load_state(&mut self, _state: &serde_json::Value) -> Result<()> {
        Ok(())
    }

    /// Save the module state to a JSON file.
    /// Only saves parameter state (demos, instructions, signatures).
    /// To restore, create the same program and call load().
    fn save(&self, path: &str) -> Result<()> {
        if !path.ends_with(".json") {
            return Err(crate::error::DspyError::Other(format!(
                "`path` must end with `.json`, but received: {}",
                path
            )));
        }

        let mut state = self.dump_state();
        let metadata = serde_json::json!({
            "dependency_versions": { "dspy_rs": "1.0.0" }
        });
        if let Some(obj) = state.as_object_mut() {
            obj.insert("metadata".to_string(), metadata);
        }

        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    crate::error::DspyError::Other(format!("Failed to create directory: {}", e))
                })?;
            }
        }

        let json = serde_json::to_string_pretty(&state).map_err(|e| {
            crate::error::DspyError::Other(format!("Failed to serialize state: {}", e))
        })?;
        std::fs::write(path, json + "\n")
            .map_err(|e| crate::error::DspyError::Other(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Load module state from a JSON file.
    /// The file must have been created by save().
    fn load(&mut self, path: &str) -> Result<()> {
        if !path.ends_with(".json") {
            return Err(crate::error::DspyError::Other(format!(
                "`path` must end with `.json`, but received: {}",
                path
            )));
        }

        let raw = std::fs::read_to_string(path)
            .map_err(|e| crate::error::DspyError::Other(format!("Failed to read file: {}", e)))?;
        let state: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| crate::error::DspyError::Other(format!("Failed to parse JSON: {}", e)))?;

        // Check version metadata if present
        if let Some(metadata) = state.get("metadata") {
            if let Some(versions) = metadata.get("dependency_versions") {
                if let Some(version) = versions.get("dspy_rs").and_then(|v| v.as_str()) {
                    if version != "1.0.0" {
                        eprintln!(
                            "Warning: Version mismatch: saved with dspy_rs=={}, current is dspy_rs==1.0.0",
                            version
                        );
                    }
                }
            }
        }

        self.load_state(&state)
    }
}
