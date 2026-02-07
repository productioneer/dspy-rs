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
pub mod adapter_types;
pub mod json_adapter;
pub mod xml_adapter;
pub mod two_step_adapter;
pub mod module_trait;
pub mod predict;
pub mod chain_of_thought;
pub mod aggregation;
pub mod best_of_n;
pub mod multi_chain_comparison;
pub mod parallel;
pub mod refine;
pub mod tool;
pub mod react;
pub mod evaluate;
pub mod knn;
pub mod claude_lm;
pub mod codex_lm;
pub mod interpreter;
pub mod mock_interpreter;
pub mod repl_types;
pub mod program_of_thought;
pub mod code_act;
pub mod rlm;
pub mod finetune_types;
pub mod provider;
pub mod bootstrap_trace;

// Re-exports
pub use error::{DspyError, Result};
pub use value::Value;
pub use example::Example;
pub use prediction::Prediction;
pub use signature::{Signature, FieldDef, FieldType, FieldUpdate, input_field, output_field, SignatureBuilder};
pub use settings::{Settings, configure, get_settings, with_settings, reset_settings};
pub use lm::{LM, LMConfig, LMResponse, Message, Usage};
pub use adapter::{Adapter, ChatAdapter};
pub use adapter_types::{
    AdapterType, AdapterTypeOutput, Image, Audio, DSPyFile, History,
    Code as CodeType, Reasoning, ContentBlock,
    TypedMessage, MessageContent,
    CUSTOM_TYPE_START, CUSTOM_TYPE_END,
    split_message_content_for_custom_types,
};
pub use json_adapter::JSONAdapter;
pub use xml_adapter::XMLAdapter;
pub use two_step_adapter::TwoStepAdapter;
pub use module_trait::Module;
pub use predict::{Predict, Trace};
pub use chain_of_thought::ChainOfThought;
pub use aggregation::majority;
pub use best_of_n::BestOfN;
pub use multi_chain_comparison::MultiChainComparison;
pub use parallel::{parallel_execute, ParallelConfig, ParallelResult};
pub use refine::Refine;
pub use tool::{Tool, ToolArg, ToolCall, ToolCalls};
pub use react::{ReAct, ReActOptions};
pub use evaluate::{Evaluate, EvaluateConfig, EvaluationResult, Metric};
pub use knn::{KNN, Embedder};
pub use claude_lm::{ClaudeLM, ClaudeLMConfig};
pub use codex_lm::{CodexLM, CodexLMConfig};
pub use interpreter::{CodeInterpreter, CodeInterpreterError, ExecutionResult, FinalOutput, InterpreterTool, OutputFieldDef};
pub use mock_interpreter::{MockInterpreter, MockResponse, MockErrorType};
pub use repl_types::{REPLVariable, REPLEntry, REPLHistory, create_repl_variable, format_repl_variable};
pub use program_of_thought::ProgramOfThought;
pub use code_act::CodeAct;
pub use rlm::RLM;
pub use finetune_types::{
    TrainingStatus, TrainDataFormat, TrainingMessage,
    GRPOChatData, GRPOGroup, GRPOStatus,
    infer_data_format, validate_data_format,
};
pub use provider::{TrainingJob, ReinforceJob, Provider};
pub use bootstrap_trace::{
    FailedPrediction, TraceEntry, TraceData,
    BootstrapTraceOptions, bootstrap_trace_data,
};
