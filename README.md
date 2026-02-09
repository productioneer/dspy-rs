# dspy-rs

> **v0.1 — Experimental Release**
>
> This is pre-production software. It has been tested (367 tests passing) but has not been battle-tested in production environments. Do not use blindly in production. Verify any behavior you depend on with your own eyes. We tried our best, but we make no guarantees that everything works correctly. APIs may change without notice.
>
> **Built by AI.** This codebase was written and maintained primarily by AI agents (Claude and Codex), with human direction and review. We believe it to be useful, safe, and secure — but verify for yourself. Issues and pull requests are welcome.

Full-parity Rust port of [DSPy](https://github.com/stanfordnlp/dspy) v3.1.3 — the framework for programming, not prompting, language models.

## What's Included

### Crates

| Crate | Description |
|---|---|
| **dspy-core** | Signatures, modules, predict, adapters, LM trait, evaluate, caching |
| **dspy-optimizers** | All optimizers: BootstrapFewShot, MIPROv2, GEPA, SIMBA, GRPO, etc. |
| **dspy-tpe** | Tree-structured Parzen Estimator for Bayesian optimization |
| **dspy-propose** | Proposer system for instruction generation (used by MIPROv2) |

### Core (`dspy-core`)
- **Signature** — typed input/output field declarations with parser
- **Example / Prediction** — structured data containers for demos and outputs
- **Module trait** — `forward()`, `named_predictors()`, `deep_copy()`, `dump_state()` / `load_state()`
- **LM trait** — async language model interface with caching, history, usage tracking, and streaming hooks
- **Evaluate** — batch evaluation with metrics and parallel execution

### Predict Modules
- **Predict** — core prediction with demos, temperature auto-adjustment
- **ChainOfThought** — adds reasoning before output
- **BestOfN** — generate N, pick best by reward
- **MultiChainComparison** — compare multiple reasoning chains
- **Refine** — iterative self-refinement
- **ReAct** — reasoning + action with tool use
- **ProgramOfThought** — code generation and execution
- **CodeAct** — agentic code execution with REPL
- **RLM** — retrieval-augmented language model
- **Parallel** — concurrent predict calls
- **Retrieve** — pluggable retrieval module
- **Tool / ToolCall** — structured tool definitions

### Adapters
- **ChatAdapter** — chat message formatting with `[[ ## field ## ]]` markers and native response type extraction
- **JSONAdapter** — JSON output parsing
- **XMLAdapter** — XML output parsing
- **TwoStepAdapter** — two-pass extraction

### Adapter Types
`Image`, `Audio`, `DspyFile`, `History`, `Code`, `Reasoning`, `Document`, `Citation`, `Citations` — rich types for multimodal and structured content

### Optimizers (`dspy-optimizers`)
| Optimizer | Description |
|---|---|
| **LabeledFewShot** | Select demos from labeled examples |
| **BootstrapFewShot** | Generate demos via bootstrapped execution |
| **BootstrapFewShotWithRandomSearch** | Bootstrap + random search over demo sets |
| **BootstrapFewShotWithOptuna** | Bootstrap backed by Optuna-style trial search |
| **COPRO** | Collaborative Prompt Optimization — LM-proposed instructions |
| **MIPROv2** | Multi-prompt Instruction Proposal Optimizer with TPE + minibatch eval |
| **SIMBA** | Softmax selection, Poisson demo dropping, rule generation |
| **GEPA** | Generalized Efficient Prompt Approximation — state-of-the-art |
| **KNNFewShot** | k-nearest-neighbor demo selection at inference time |
| **InferRules** | Extract reusable rules from execution traces |
| **Ensemble** | Combine multiple program variants |
| **AvatarOptimizer** | Persona-based optimization |
| **BootstrapFinetune** | Bootstrap training data then finetune the LM |
| **GRPO** | Group Relative Policy Optimization (RL-based weight training) |
| **BetterTogether** | Joint optimization of prompts and finetunes |

### Infrastructure
- **TPE** (`dspy-tpe`) — Parzen estimator, kernel density estimation, startup random phase
- **Proposer** (`dspy-propose`) — GroundedProposer for instruction generation
- **KNN** — k-nearest-neighbor retriever with pluggable embedding
- **Cache** — disk-backed caching with TTL and eviction
- **Embedder** — embedding client with caching support
- **Callbacks** — full observability (module, LM, adapter events)
- **Streaming** — `streamify()`, `StatusMessageProvider`, stream sender plumbing
- **Provider / TrainingJob / ReinforceJob** — finetuning infrastructure

## Quick Start

```bash
cargo build
cargo test
```

```rust
use dspy_core::{configure, ChainOfThought, Signature, Example};
use dspy_core::module_trait::Module;
use std::sync::Arc;

// Implement the LM trait for your provider
use dspy_core::lm::{LM, LMConfig, LMResponse, Message};
use async_trait::async_trait;

struct MyLM;

#[async_trait]
impl LM for MyLM {
    async fn forward(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<Vec<LMResponse>> {
        // Call your LLM provider here
        Ok(vec![LMResponse::new("response text", None)])
    }

    fn model_name(&self) -> &str { "my-model" }
    fn clone_box(&self) -> Box<dyn LM> { Box::new(MyLM) }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    configure(Arc::new(MyLM), None);

    let sig = Signature::from_string("question -> answer")?;
    let cot = ChainOfThought::new(sig);
    let result = cot.forward(&Example::from([("question", "What is DSPy?")])).await?;
    println!("{}", result.get("answer").unwrap());
    Ok(())
}
```

## Implementing a Custom LM

Implement the `LM` trait:

```rust
use dspy_core::lm::{LM, LMConfig, LMResponse, Message};
use async_trait::async_trait;

struct CustomLM {
    model: String,
}

#[async_trait]
impl LM for CustomLM {
    async fn forward(
        &self,
        messages: &[Message],
        config: &LMConfig,
    ) -> anyhow::Result<Vec<LMResponse>> {
        // config.temperature, config.max_tokens, config.n are available
        // Return a Vec<LMResponse> (length = config.n)
        Ok(vec![LMResponse::new("response", None)])
    }

    fn model_name(&self) -> &str { &self.model }
    fn clone_box(&self) -> Box<dyn LM> {
        Box::new(CustomLM { model: self.model.clone() })
    }
}
```

The `BaseLM` wrapper (used via `configure()`) handles caching, history tracking, usage aggregation, and streaming hooks automatically.

## Callbacks

Observe every level of execution:

```rust
use dspy_core::callback::{Callback, set_global_callbacks};
use std::sync::Arc;

struct LoggingCallback;

impl Callback for LoggingCallback {
    fn on_module_start(&self, call_id: &str, instance_type: &str, inputs: &serde_json::Value) {
        println!("[{}] start: {}", instance_type, inputs);
    }
    fn on_module_end(&self, call_id: &str, outputs: Option<&serde_json::Value>, exception: Option<&str>) {
        match exception {
            Some(e) => println!("  error: {}", e),
            None => println!("  done: {:?}", outputs),
        }
    }
    fn on_lm_start(&self, call_id: &str, instance_type: &str, inputs: &serde_json::Value) {
        println!("  LM call ({})", instance_type);
    }
    fn on_lm_end(&self, call_id: &str, outputs: Option<&serde_json::Value>, exception: Option<&str>) {
        println!("  LM done");
    }
}

set_global_callbacks(vec![Arc::new(LoggingCallback)]);
```

## Running Tests

```bash
cargo test                  # all 367 tests
cargo test -p dspy-core     # core crate only
cargo test -p dspy-optimizers  # optimizers only
```

## Architecture

DSPy programs are composable modules, each containing `Predict` instances. Signatures define typed I/O contracts. Adapters format signatures into LM prompts and parse responses. Optimizers search instruction/demo space to maximize a metric.

```
Module (ChainOfThought, ReAct, etc.)
  └── Predict (core prediction unit)
        ├── Signature (typed I/O contract)
        ├── Adapter (prompt formatting / response parsing)
        └── LM (language model backend)
```

### Workspace Structure

```
dspy-rs/
├── crates/
│   ├── dspy-core/        # Core: signatures, modules, predict, adapters, LM, evaluate
│   ├── dspy-optimizers/  # All optimizers
│   ├── dspy-tpe/         # TPE sampler (standalone, minimal deps)
│   └── dspy-propose/     # Proposer system for MIPROv2
├── Cargo.toml            # Workspace root
└── Cargo.lock
```

## License

MIT
