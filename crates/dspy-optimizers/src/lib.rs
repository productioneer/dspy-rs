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

pub mod labeled_few_shot;
pub mod bootstrap_few_shot;
pub mod random_search;
pub mod copro;
pub mod ensemble;
pub mod mipro_v2;
pub mod simba;
pub mod knn_fewshot;
pub mod infer_rules;
pub mod gepa;
pub mod bootstrap_finetune;
pub mod grpo;
pub mod better_together;
pub mod avatar_optimizer;
pub mod bootstrap_fewshot_optuna;

// Re-exports
pub use labeled_few_shot::LabeledFewShot;
pub use bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};
pub use random_search::{BootstrapFewShotWithRandomSearch, RandomSearchConfig};
pub use copro::{COPRO, COPROConfig};
pub use ensemble::{EnsembleModule, EnsembleConfig, ReduceFn};
pub use mipro_v2::{MIPROv2, MIPROv2Config, MIPROv2CompileOptions, AutoMode, MIPROv2Result};
pub use simba::{SIMBA, SIMBAConfig, SIMBAResult};
pub use knn_fewshot::{KNNFewShot, KNNFewShotConfig, KNNCompiledProgram};
pub use infer_rules::{InferRules, InferRulesConfig};
pub use gepa::{GEPA, GEPAConfig, GEPABudget, GEPAResult, GEPAFeedbackMetric, GEPAMetricResult, ScoreWithFeedback, CandidateSelection, ComponentSelection, ProposalFn, ReflectiveExample};
pub use bootstrap_finetune::{BootstrapFinetune, BootstrapFinetuneConfig};
pub use grpo::{GRPO, GRPOConfig};
pub use better_together::{BetterTogether, BetterTogetherConfig};
pub use avatar_optimizer::{AvatarOptimizer, AvatarOptimizerConfig, OptimizeDirection, EvalResult};
pub use bootstrap_fewshot_optuna::{BootstrapFewShotWithOptuna, BootstrapFewShotWithOptunaConfig};
