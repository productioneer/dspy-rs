//! DSPy Optimizers — search algorithms over prompt components.
//!
//! Provides optimizers for improving DSPy programs:
//! - LabeledFewShot: assigns random labeled examples as demos
//! - BootstrapFewShot: generates synthetic demos by running teacher on trainset
//! - BootstrapFewShotWithRandomSearch: multiple bootstrap rounds, picks best
//! - COPRO: iterative instruction optimization via LLM meta-prompting
//! - MIPROv2: Bayesian optimization over instructions+demos (TPE + proposer)
//! - SIMBA: stochastic mini-batch ascent with softmax-weighted program selection
//! - KNNFewShot: dynamic k-nearest neighbor demo selection at inference time
//! - Ensemble: combines multiple programs via majority vote
//! - GEPA: generalized evolutionary prompt adaptation via reflective mutation
//! - BootstrapFinetune: finetune LMs using bootstrapped trace data
//! - GRPO: group relative policy optimization for online RL training
//! - BetterTogether: composes prompt and weight optimization strategies
//! - AvatarOptimizer: iterative instruction refinement for tool-using agents
//! - BootstrapFewShotWithOptuna: demo selection using TPE Bayesian optimization

pub mod avatar_optimizer;
pub mod better_together;
pub mod bootstrap_few_shot;
pub mod bootstrap_fewshot_optuna;
pub mod bootstrap_finetune;
pub mod copro;
pub mod ensemble;
pub mod gepa;
pub mod grpo;
pub mod infer_rules;
pub mod knn_fewshot;
pub mod labeled_few_shot;
pub mod mipro_v2;
pub mod random_search;
pub mod simba;

// Re-exports
pub use avatar_optimizer::{AvatarOptimizer, AvatarOptimizerConfig, EvalResult, OptimizeDirection};
pub use better_together::{BetterTogether, BetterTogetherConfig};
pub use bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};
pub use bootstrap_fewshot_optuna::{BootstrapFewShotWithOptuna, BootstrapFewShotWithOptunaConfig};
pub use bootstrap_finetune::{BootstrapFinetune, BootstrapFinetuneConfig};
pub use copro::{COPROConfig, COPRO};
pub use ensemble::{EnsembleConfig, EnsembleModule, ReduceFn};
pub use gepa::{
    CandidateSelection, ComponentSelection, GEPABudget, GEPAConfig, GEPAFeedbackMetric,
    GEPAMetricResult, GEPAResult, ProposalFn, ReflectiveExample, ScoreWithFeedback, GEPA,
};
pub use grpo::{GRPOConfig, GRPO};
pub use infer_rules::{InferRules, InferRulesConfig};
pub use knn_fewshot::{KNNCompiledProgram, KNNFewShot, KNNFewShotConfig};
pub use labeled_few_shot::LabeledFewShot;
pub use mipro_v2::{AutoMode, MIPROv2, MIPROv2CompileOptions, MIPROv2Config, MIPROv2Result};
pub use random_search::{BootstrapFewShotWithRandomSearch, RandomSearchConfig};
pub use simba::{SIMBAConfig, SIMBAResult, SIMBA};
