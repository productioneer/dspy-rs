use thiserror::Error;

#[derive(Error, Debug)]
pub enum DspyError {
    #[error("Missing required input field: {0}")]
    MissingField(String),

    #[error("Invalid signature format: {0}")]
    InvalidSignature(String),

    #[error("LM call failed: {0}")]
    LMError(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Module error: {0}")]
    ModuleError(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DspyError>;
