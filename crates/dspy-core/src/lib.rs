//! DSPy Core — primitives, signatures, modules, predict, adapters, evaluate
//!
//! Full-parity port of Python DSPy v3.1.3 core infrastructure.

pub mod error;
pub mod value;
pub mod example;
pub mod prediction;
pub mod signature;
pub mod settings;
pub mod lm;
pub mod adapter;
pub mod module_trait;
pub mod predict;
pub mod chain_of_thought;
pub mod evaluate;
pub mod knn;

// Re-exports
pub use error::{DspyError, Result};
pub use value::Value;
pub use example::Example;
pub use prediction::Prediction;
pub use signature::{Signature, FieldDef, FieldType, FieldUpdate, input_field, output_field, SignatureBuilder};
pub use settings::{Settings, configure, get_settings, with_settings, reset_settings};
pub use lm::{LM, LMConfig, LMResponse, Message, Usage};
pub use adapter::{Adapter, ChatAdapter};
pub use module_trait::Module;
pub use predict::{Predict, Trace};
pub use chain_of_thought::ChainOfThought;
pub use evaluate::{Evaluate, EvaluateConfig, EvaluationResult, Metric};
pub use knn::{KNN, Embedder};
