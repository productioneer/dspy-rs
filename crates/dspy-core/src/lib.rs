//! DSPy Core — primitives, signatures, modules, predict, adapters, evaluate
//!
//! Full-parity port of Python DSPy v3.1.3 core infrastructure.

pub mod adapter;
pub mod adapter_types;
pub mod aggregation;
pub mod best_of_n;
pub mod bootstrap_trace;
pub mod cache;
pub mod callback;
pub mod chain_of_thought;
pub mod claude_lm;
pub mod code_act;
pub mod codex_lm;
pub mod dataset;
pub mod embedder;
pub mod error;
pub mod evaluate;
pub mod example;
pub mod finetune_types;
pub mod interpreter;
pub mod json_adapter;
pub mod knn;
pub mod lm;
pub mod mock_interpreter;
pub mod module_trait;
pub mod multi_chain_comparison;
pub mod parallel;
pub mod parallelizer;
pub mod predict;
pub mod prediction;
pub mod program_of_thought;
pub mod provider;
pub mod react;
pub mod refine;
pub mod repl_types;
pub mod retrieve;
pub mod rlm;
pub mod sandboxed_interpreter;
pub mod settings;
pub mod signature;
pub mod streaming;
pub mod tool;
pub mod two_step_adapter;
pub mod usage_tracker;
pub mod value;
pub mod xml_adapter;

#[cfg(test)]
mod golden_trace_tests;

// Re-exports
pub use adapter::{Adapter, ChatAdapter};
pub use adapter_types::{
    split_message_content_for_custom_types, AdapterType, AdapterTypeOutput, Audio, Citation,
    Citations, Code as CodeType, ContentBlock, DSPyFile, Document, DocumentMediaType, History,
    Image, MessageContent, Reasoning, TypedMessage, CUSTOM_TYPE_END, CUSTOM_TYPE_START,
};
pub use aggregation::majority;
pub use best_of_n::BestOfN;
pub use bootstrap_trace::{
    bootstrap_trace_data, BootstrapTraceOptions, FailedPrediction, TraceData, TraceEntry,
};
pub use cache::{Cache, CacheConfig};
pub use callback::{
    add_global_callback, clear_global_callbacks, get_global_callbacks, invoke_end_callbacks,
    invoke_start_callbacks, set_global_callbacks, Callback, ComponentType,
};
pub use chain_of_thought::ChainOfThought;
pub use claude_lm::{ClaudeLM, ClaudeLMConfig};
pub use code_act::CodeAct;
pub use codex_lm::{CodexLM, CodexLMConfig};
pub use dataset::{Dataset as DatasetBase, DatasetConfig};
pub use embedder::{Embedder as EmbedderClient, EmbeddingFunction};
pub use error::{DspyError, Result};
pub use evaluate::{Evaluate, EvaluateConfig, EvaluationResult, Metric};
pub use example::Example;
pub use finetune_types::{
    infer_data_format, validate_data_format, GRPOChatData, GRPOGroup, GRPOStatus, TrainDataFormat,
    TrainingMessage, TrainingStatus,
};
pub use interpreter::{
    CodeInterpreter, CodeInterpreterError, ExecutionResult, FinalOutput, InterpreterTool,
    OutputFieldDef,
};
pub use json_adapter::JSONAdapter;
pub use knn::{Embedder, KNN};
pub use lm::{
    call_with_cache, clear_history, configure_cache, inspect_history, reset_global_cache,
    HistoryEntry, LMConfig, LMResponse, Message, Usage, LM,
};
pub use mock_interpreter::{MockErrorType, MockInterpreter, MockResponse};
pub use module_trait::Module;
pub use multi_chain_comparison::MultiChainComparison;
pub use parallel::{parallel_execute, ParallelConfig, ParallelResult};
pub use parallelizer::{ParallelExecutor, ParallelExecutorConfig};
pub use predict::{Predict, Trace};
pub use prediction::Prediction;
pub use program_of_thought::ProgramOfThought;
pub use provider::{Provider, ReinforceJob, TrainingJob};
pub use react::{ReAct, ReActOptions};
pub use refine::Refine;
pub use repl_types::{
    create_repl_variable, format_repl_variable, REPLEntry, REPLHistory, REPLVariable,
};
pub use retrieve::{get_global_retriever, set_global_retriever, Retrieve, RetrieverModule};
pub use rlm::RLM;
pub use sandboxed_interpreter::{SandboxedInterpreter, SandboxedInterpreterOptions};
pub use settings::{
    configure, get_settings, reset_settings, with_settings, SendStreamFn, Settings,
};
pub use signature::{
    input_field, output_field, FieldDef, FieldType, FieldUpdate, Signature, SignatureBuilder,
};
pub use streaming::{
    streamify, AdapterType as StreamAdapterType, StatusMessage, StatusMessageProvider,
    StreamListener, StreamResponse, StreamValue, StreamifyOptions,
};
pub use tool::{Tool, ToolArg, ToolCall, ToolCalls};
pub use two_step_adapter::TwoStepAdapter;
pub use usage_tracker::UsageTracker;
pub use value::Value;
pub use xml_adapter::XMLAdapter;
