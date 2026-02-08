//! MIPROv2 — Multi-prompt Instruction Proposal Optimizer v2.
//!
//! The most sophisticated "standard" optimizer in DSPy. Combines:
//! 1. Bootstrap few-shot example generation (reuses BootstrapFewShot)
//! 2. LLM-driven instruction proposal (uses GroundedProposer)
//! 3. Bayesian optimization over instruction+demo combinations (uses TPE)
//!
//! Python equivalent: dspy/teleprompt/mipro_optimizer_v2.py

use dspy_core::{Evaluate, EvaluateConfig, Example, Metric, Module, LM};
use dspy_propose::{GroundedProposer, ProposerConfig};
use dspy_tpe::{Direction, Study, TPESampler};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::HashMap;
use std::sync::Arc;

use crate::bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};

/// Auto-run presets matching Python DSPy
#[derive(Debug, Clone, Copy)]
pub enum AutoMode {
    Light,
    Medium,
    Heavy,
}

impl AutoMode {
    fn n(&self) -> usize {
        match self {
            AutoMode::Light => 6,
            AutoMode::Medium => 12,
            AutoMode::Heavy => 18,
        }
    }

    fn val_size(&self) -> usize {
        match self {
            AutoMode::Light => 100,
            AutoMode::Medium => 300,
            AutoMode::Heavy => 1000,
        }
    }
}

const MIN_MINIBATCH_SIZE: usize = 50;

/// Configuration for MIPROv2
pub struct MIPROv2Config {
    pub metric: Metric,
    pub prompt_model: Arc<dyn LM>,
    pub task_model: Option<Arc<dyn LM>>,
    pub auto: Option<AutoMode>,
    pub num_candidates: Option<usize>,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_errors: usize,
    pub seed: u64,
    pub init_temperature: f64,
    pub metric_threshold: Option<f64>,
}

impl MIPROv2Config {
    pub fn new(metric: Metric, prompt_model: Arc<dyn LM>) -> Self {
        Self {
            metric,
            prompt_model,
            task_model: None,
            auto: Some(AutoMode::Light),
            num_candidates: None,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 4,
            max_errors: 5,
            seed: 9,
            init_temperature: 1.0,
            metric_threshold: None,
        }
    }
}

/// Compile-time options
pub struct MIPROv2CompileOptions {
    pub num_trials: Option<usize>,
    pub max_bootstrapped_demos: Option<usize>,
    pub max_labeled_demos: Option<usize>,
    pub seed: Option<u64>,
    pub minibatch: bool,
    pub minibatch_size: usize,
    pub minibatch_full_eval_steps: usize,
    pub program_aware_proposer: bool,
    pub data_aware_proposer: bool,
    pub view_data_batch_size: usize,
    pub tip_aware_proposer: bool,
    pub fewshot_aware_proposer: bool,
}

impl Default for MIPROv2CompileOptions {
    fn default() -> Self {
        Self {
            num_trials: None,
            max_bootstrapped_demos: None,
            max_labeled_demos: None,
            seed: None,
            minibatch: true,
            minibatch_size: 35,
            minibatch_full_eval_steps: 5,
            program_aware_proposer: true,
            data_aware_proposer: true,
            view_data_batch_size: 10,
            tip_aware_proposer: true,
            fewshot_aware_proposer: true,
        }
    }
}

/// MIPROv2 optimizer
pub struct MIPROv2 {
    config: MIPROv2Config,
}

/// Result of MIPROv2 optimization with metadata
pub struct MIPROv2Result {
    pub program: Box<dyn Module>,
    pub score: f64,
    pub trial_logs: HashMap<usize, TrialLog>,
}

#[derive(Debug, Clone)]
pub struct TrialLog {
    pub score: f64,
    pub is_full_eval: bool,
    pub params: HashMap<String, usize>,
}

impl MIPROv2 {
    pub fn new(config: MIPROv2Config) -> Self {
        Self { config }
    }

    /// Compile: optimize the student program
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        valset: Option<&[Example]>,
        teacher: Option<&dyn Module>,
        options: MIPROv2CompileOptions,
    ) -> dspy_core::Result<MIPROv2Result> {
        let seed = options.seed.unwrap_or(self.config.seed);
        let mut rng = StdRng::seed_from_u64(seed);

        let max_bootstrapped = options
            .max_bootstrapped_demos
            .unwrap_or(self.config.max_bootstrapped_demos);
        let max_labeled = options
            .max_labeled_demos
            .unwrap_or(self.config.max_labeled_demos);

        let zeroshot = max_bootstrapped == 0 && max_labeled == 0;

        // Split trainset into train/val if no valset provided
        let (train_data, val_data) = if let Some(vs) = valset {
            (trainset.to_vec(), vs.to_vec())
        } else {
            split_train_val(trainset, &mut rng)
        };

        // Determine hyperparameters from auto mode
        let (
            num_trials,
            effective_val,
            use_minibatch,
            num_instruct_candidates,
            num_fewshot_candidates,
        ) = self.resolve_hyperparams(
            student,
            &val_data,
            options.num_trials,
            options.minibatch,
            zeroshot,
            &mut rng,
        );

        // Step 1: Bootstrap few-shot examples
        let demo_candidates = if !zeroshot {
            Some(
                self.bootstrap_demo_candidates(
                    student,
                    &train_data,
                    teacher,
                    num_fewshot_candidates,
                    max_bootstrapped,
                    max_labeled,
                    seed,
                    &mut rng,
                )
                .await,
            )
        } else {
            None
        };

        // Step 2: Propose instruction candidates
        let instruction_candidates = self
            .propose_instruction_candidates(
                student,
                &train_data,
                demo_candidates.as_deref(),
                num_instruct_candidates,
                seed,
                &options,
            )
            .await;

        // Step 3: Bayesian optimization over instruction+demo combinations
        let result = self
            .optimize_with_tpe(
                student,
                &instruction_candidates,
                demo_candidates.as_deref(),
                &effective_val,
                num_trials,
                use_minibatch,
                options.minibatch_size,
                options.minibatch_full_eval_steps,
                seed,
                &mut rng,
            )
            .await?;

        Ok(result)
    }

    fn resolve_hyperparams(
        &self,
        student: &dyn Module,
        valset: &[Example],
        num_trials: Option<usize>,
        minibatch: bool,
        zeroshot: bool,
        rng: &mut StdRng,
    ) -> (usize, Vec<Example>, bool, usize, usize) {
        match self.config.auto {
            Some(auto_mode) => {
                let n = auto_mode.n();
                let val_size = auto_mode.val_size().min(valset.len());

                let effective_val: Vec<Example> = if val_size < valset.len() {
                    let mut indices: Vec<usize> = (0..valset.len()).collect();
                    indices.shuffle(rng);
                    indices[..val_size]
                        .iter()
                        .map(|&i| valset[i].clone())
                        .collect()
                } else {
                    valset.to_vec()
                };

                let use_minibatch = effective_val.len() > MIN_MINIBATCH_SIZE;
                let num_instruct = if zeroshot { n } else { n / 2 };
                let num_fewshot = n;
                let num_vars = student.named_predictors().len();
                let effective_vars = if zeroshot { num_vars } else { num_vars * 2 };
                let computed_trials = ((2.0 * effective_vars as f64 * (n as f64).log2())
                    .max(1.5 * n as f64)) as usize;

                (
                    computed_trials,
                    effective_val,
                    use_minibatch,
                    num_instruct,
                    num_fewshot,
                )
            }
            None => {
                let n = self.config.num_candidates.unwrap_or(6);
                let trials = num_trials.unwrap_or(n * 2);
                let use_minibatch = minibatch && valset.len() > MIN_MINIBATCH_SIZE;
                let num_instruct = if zeroshot { n } else { n / 2 };
                (trials, valset.to_vec(), use_minibatch, num_instruct, n)
            }
        }
    }

    /// Bootstrap N sets of demo candidates for each predictor using BootstrapFewShot.
    /// Matches Python's create_n_fewshot_demo_sets: runs BootstrapFewShot multiple times
    /// with different seeds to collect diverse demo sets.
    async fn bootstrap_demo_candidates(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
        num_sets: usize,
        max_bootstrapped: usize,
        max_labeled: usize,
        seed: u64,
        rng: &mut StdRng,
    ) -> Vec<Vec<Vec<Example>>> {
        let predictors = student.named_predictors();
        let num_predictors = predictors.len();

        // Each set: run BootstrapFewShot with a different seed, collect demos per predictor
        let mut all_sets: Vec<Vec<Vec<Example>>> = vec![Vec::new(); num_predictors];

        for set_i in 0..num_sets {
            let set_seed = seed.wrapping_add(set_i as u64);

            // Shuffle trainset for this set
            let mut shuffled_train = trainset.to_vec();
            let mut set_rng = StdRng::seed_from_u64(set_seed);
            shuffled_train.shuffle(&mut set_rng);

            // Run BootstrapFewShot to collect traces
            let config = BootstrapFewShotConfig {
                metric: self.config.metric.clone(),
                metric_threshold: self.config.metric_threshold,
                max_bootstrapped_demos: max_bootstrapped,
                max_labeled_demos: max_labeled,
                max_rounds: 1,
                max_errors: self.config.max_errors,
            };
            let bootstrap = BootstrapFewShot::new(config);

            match bootstrap.compile(student, &shuffled_train, teacher).await {
                Ok(compiled) => {
                    // Extract demos from each predictor of the compiled program
                    let compiled_preds = compiled.named_predictors();
                    for (pred_i, (_name, pred)) in compiled_preds.iter().enumerate() {
                        if pred_i < num_predictors {
                            all_sets[pred_i].push(pred.demos.clone());
                        }
                    }
                }
                Err(_) => {
                    // Fallback: random labeled examples for each predictor
                    for pred_i in 0..num_predictors {
                        let num_labeled = max_labeled.min(shuffled_train.len());
                        let demos: Vec<Example> = shuffled_train[..num_labeled].to_vec();
                        all_sets[pred_i].push(demos);
                    }
                }
            }
        }

        // Shuffle demo sets for variety
        for pred_demos in &mut all_sets {
            pred_demos.shuffle(rng);
        }

        all_sets
    }

    /// Propose instruction candidates using GroundedProposer
    async fn propose_instruction_candidates(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        demo_candidates: Option<&[Vec<Vec<Example>>]>,
        num_candidates: usize,
        seed: u64,
        options: &MIPROv2CompileOptions,
    ) -> HashMap<usize, Vec<String>> {
        let config = ProposerConfig {
            program_aware: options.program_aware_proposer,
            use_dataset_summary: options.data_aware_proposer,
            use_task_demos: options.fewshot_aware_proposer,
            use_tip: options.tip_aware_proposer,
            set_tip_randomly: options.tip_aware_proposer,
            view_data_batch_size: options.view_data_batch_size,
            init_temperature: self.config.init_temperature,
            ..Default::default()
        };

        let mut proposer = GroundedProposer::new(self.config.prompt_model.clone(), config, seed);

        let mut instructions = proposer
            .propose_instructions_for_program(student, trainset, demo_candidates, num_candidates)
            .await;

        // Prepend the original instruction as candidate 0 for each predictor
        let predictors = student.named_predictors();
        for (i, (_name, pred)) in predictors.iter().enumerate() {
            let original = pred.signature.instructions().to_string();
            if let Some(candidates) = instructions.get_mut(&i) {
                candidates.insert(0, original);
            } else {
                instructions.insert(i, vec![original]);
            }
        }

        instructions
    }

    /// TPE-based optimization over instruction+demo combinations
    async fn optimize_with_tpe(
        &self,
        student: &dyn Module,
        instruction_candidates: &HashMap<usize, Vec<String>>,
        demo_candidates: Option<&[Vec<Vec<Example>>]>,
        valset: &[Example],
        num_trials: usize,
        use_minibatch: bool,
        minibatch_size: usize,
        full_eval_steps: usize,
        seed: u64,
        rng: &mut StdRng,
    ) -> dspy_core::Result<MIPROv2Result> {
        let sampler = TPESampler::new(seed).with_n_startup_trials(
            // Startup with random for first ~sqrt(num_trials) trials
            ((num_trials as f64).sqrt().ceil() as usize)
                .max(3)
                .min(num_trials),
        );
        let mut study = Study::new(Direction::Maximize, sampler);

        // Evaluate the default program first
        let default_program = student.deep_copy();
        let default_score = self.evaluate_program(&*default_program, valset).await?;

        // Add default as baseline trial
        let mut default_params = HashMap::new();
        let num_predictors = student.named_predictors().len();
        for i in 0..num_predictors {
            default_params.insert(format!("{i}_instruction"), 0);
            if demo_candidates.is_some() {
                default_params.insert(format!("{i}_demos"), 0);
            }
        }
        let baseline = study.create_trial(default_params, HashMap::new(), default_score);
        study.add_trial(baseline);

        let mut best_score = default_score;
        let mut best_program = student.deep_copy();
        let mut trial_logs = HashMap::new();

        trial_logs.insert(
            0,
            TrialLog {
                score: default_score,
                is_full_eval: true,
                params: HashMap::new(),
            },
        );

        // Track minibatch scores for full evaluation selection
        let mut minibatch_scores: Vec<(f64, HashMap<String, usize>)> = Vec::new();

        // Run optimization trials
        for trial_i in 0..num_trials {
            // Use the study to suggest parameters via ask/tell API
            let mut chosen_params = HashMap::new();
            for i in 0..num_predictors {
                let n_instructions = instruction_candidates.get(&i).map(|v| v.len()).unwrap_or(1);
                let inst_idx =
                    study.suggest_categorical(&format!("{i}_instruction"), n_instructions);
                chosen_params.insert(format!("{i}_instruction"), inst_idx);

                if let Some(dc) = demo_candidates {
                    if i < dc.len() {
                        let n_demos = dc[i].len().max(1);
                        let demo_idx = study.suggest_categorical(&format!("{i}_demos"), n_demos);
                        chosen_params.insert(format!("{i}_demos"), demo_idx);
                    }
                }
            }

            // Apply the chosen parameters to a candidate program
            let mut candidate_program = student.deep_copy();
            let predictors = candidate_program.named_predictors_mut();
            for (i, (_name, pred)) in predictors.into_iter().enumerate() {
                // Apply instruction
                if let Some(&inst_idx) = chosen_params.get(&format!("{i}_instruction")) {
                    if let Some(candidates) = instruction_candidates.get(&i) {
                        if inst_idx < candidates.len() {
                            pred.signature =
                                pred.signature.with_instructions(&candidates[inst_idx]);
                        }
                    }
                }

                // Apply demos
                if let Some(dc) = demo_candidates {
                    if let Some(&demo_idx) = chosen_params.get(&format!("{i}_demos")) {
                        if i < dc.len() && demo_idx < dc[i].len() {
                            pred.demos = dc[i][demo_idx].clone();
                        }
                    }
                }
            }

            // Evaluate (minibatch or full)
            let eval_set = if use_minibatch {
                let size = minibatch_size.min(valset.len());
                let mut indices: Vec<usize> = (0..valset.len()).collect();
                indices.shuffle(rng);
                indices[..size].iter().map(|&i| valset[i].clone()).collect()
            } else {
                valset.to_vec()
            };

            let score = self
                .evaluate_program(&*candidate_program, &eval_set)
                .await?;

            // Record the trial with the REAL score so TPE can learn
            study.record_trial(chosen_params.clone(), score);

            if !use_minibatch && score > best_score {
                best_score = score;
                best_program = candidate_program;
            }

            if use_minibatch {
                minibatch_scores.push((score, chosen_params.clone()));
            }

            trial_logs.insert(
                trial_i + 1,
                TrialLog {
                    score,
                    is_full_eval: !use_minibatch,
                    params: chosen_params,
                },
            );

            // Minibatch full evaluation at intervals
            if use_minibatch
                && full_eval_steps > 0
                && ((trial_i + 1) % full_eval_steps == 0 || trial_i == num_trials - 1)
            {
                // Find best-averaging param combo from minibatch trials
                if let Some(best_combo) = self.best_averaging_combo(&minibatch_scores) {
                    // Build program with these params and do full eval
                    let mut full_eval_program = student.deep_copy();
                    let predictors = full_eval_program.named_predictors_mut();
                    for (i, (_name, pred)) in predictors.into_iter().enumerate() {
                        if let Some(&inst_idx) = best_combo.get(&format!("{i}_instruction")) {
                            if let Some(candidates) = instruction_candidates.get(&i) {
                                if inst_idx < candidates.len() {
                                    pred.signature =
                                        pred.signature.with_instructions(&candidates[inst_idx]);
                                }
                            }
                        }
                        if let Some(dc) = demo_candidates {
                            if let Some(&demo_idx) = best_combo.get(&format!("{i}_demos")) {
                                if i < dc.len() && demo_idx < dc[i].len() {
                                    pred.demos = dc[i][demo_idx].clone();
                                }
                            }
                        }
                    }

                    let full_score = self.evaluate_program(&*full_eval_program, valset).await?;

                    // Record full eval as a trial so TPE can learn from it
                    study.record_trial(best_combo.clone(), full_score);

                    if full_score > best_score {
                        best_score = full_score;
                        best_program = full_eval_program;
                    }
                }
            }
        }

        Ok(MIPROv2Result {
            program: best_program,
            score: best_score,
            trial_logs,
        })
    }

    /// Find the param combo with the highest average score from minibatch trials
    fn best_averaging_combo(
        &self,
        scores: &[(f64, HashMap<String, usize>)],
    ) -> Option<HashMap<String, usize>> {
        if scores.is_empty() {
            return None;
        }

        // Group by param combo key
        let mut combo_scores: HashMap<String, (f64, usize, HashMap<String, usize>)> =
            HashMap::new();
        for (score, params) in scores {
            let key = Self::param_key(params);
            let entry = combo_scores.entry(key).or_insert((0.0, 0, params.clone()));
            entry.0 += score;
            entry.1 += 1;
        }

        // Find highest average
        combo_scores
            .into_values()
            .max_by(|a, b| {
                let avg_a = a.0 / a.1 as f64;
                let avg_b = b.0 / b.1 as f64;
                avg_a.partial_cmp(&avg_b).unwrap()
            })
            .map(|(_, _, params)| params)
    }

    fn param_key(params: &HashMap<String, usize>) -> String {
        let mut keys: Vec<_> = params.iter().collect();
        keys.sort_by_key(|(k, _)| (*k).clone());
        keys.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    async fn evaluate_program(
        &self,
        program: &dyn Module,
        valset: &[Example],
    ) -> dspy_core::Result<f64> {
        let evaluate = Evaluate::new(
            valset.to_vec(),
            self.config.metric.clone(),
            EvaluateConfig {
                max_errors: self.config.max_errors,
                ..Default::default()
            },
        );
        let result = evaluate.run(program).await?;
        Ok(result.score)
    }
}

fn split_train_val(data: &[Example], rng: &mut StdRng) -> (Vec<Example>, Vec<Example>) {
    if data.len() < 2 {
        return (data.to_vec(), data.to_vec());
    }

    let val_size = (data.len() as f64 * 0.8).ceil() as usize;
    let val_size = val_size.min(1000).max(1);
    let cutoff = data.len().saturating_sub(val_size);

    let mut indices: Vec<usize> = (0..data.len()).collect();
    indices.shuffle(rng);

    let train: Vec<Example> = indices[..cutoff].iter().map(|&i| data[i].clone()).collect();
    let val: Vec<Example> = indices[cutoff..].iter().map(|&i| data[i].clone()).collect();

    (train, val)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{LMConfig, LMResponse, Message, Predict, Prediction, Signature};

    struct MockLM {
        answer: String,
        config: LMConfig,
    }

    impl MockLM {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(
            &self,
            messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            // Check if this is a proposer call (looking for "proposed_instruction" in output)
            let is_proposer = messages.iter().any(|m| {
                m.content.contains("proposed_instruction")
                    || m.content.contains("PROPOSED INSTRUCTION")
            });

            let text = if is_proposer {
                "[[ ## proposed_instruction ## ]]\nImproved: answer accurately and completely"
                    .to_string()
            } else {
                format!("[[ ## answer ## ]]\n{}", self.answer)
            };

            Ok(vec![LMResponse { text, usage: None }])
        }
        fn model(&self) -> &str {
            "mock"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    struct SimpleModule {
        predict: Predict,
    }

    impl SimpleModule {
        fn new(sig: Signature) -> Self {
            Self {
                predict: Predict::new(sig),
            }
        }
        fn with_lm(mut self, lm: Arc<dyn LM>) -> Self {
            self.predict.set_lm(lm);
            self
        }
    }

    #[async_trait]
    impl Module for SimpleModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
            self.predict.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(SimpleModule {
                predict: self.predict.clone(),
            })
        }
    }

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}").as_str())
                    .field("answer", "42")
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[tokio::test]
    async fn test_mipro_basic() {
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Answer the question");
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let config = MIPROv2Config {
            metric: metric.clone(),
            prompt_model: lm.clone(),
            auto: Some(AutoMode::Light),
            ..MIPROv2Config::new(metric, lm)
        };

        let optimizer = MIPROv2::new(config);
        let trainset = make_trainset(20);

        let result = optimizer
            .compile(
                &student,
                &trainset,
                None,
                None,
                MIPROv2CompileOptions::default(),
            )
            .await
            .unwrap();

        assert!(result.score >= 0.0);
        assert!(!result.trial_logs.is_empty());
    }

    #[tokio::test]
    async fn test_mipro_with_valset() {
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let config = MIPROv2Config::new(metric, lm);
        let optimizer = MIPROv2::new(config);

        let trainset = make_trainset(15);
        let valset = make_trainset(5);

        let result = optimizer
            .compile(
                &student,
                &trainset,
                Some(&valset),
                None,
                MIPROv2CompileOptions {
                    num_trials: Some(5),
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        assert!(result.score >= 0.0);
        assert!(result.trial_logs.len() >= 5);
    }

    #[tokio::test]
    async fn test_mipro_zeroshot() {
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: Metric = Arc::new(|_, _| 1.0);

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let config = MIPROv2Config {
            max_bootstrapped_demos: 0,
            max_labeled_demos: 0,
            ..MIPROv2Config::new(metric, lm)
        };

        let optimizer = MIPROv2::new(config);
        let trainset = make_trainset(10);

        let result = optimizer
            .compile(
                &student,
                &trainset,
                None,
                None,
                MIPROv2CompileOptions::default(),
            )
            .await
            .unwrap();

        // Zero-shot means no demo candidates
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_split_train_val() {
        let mut rng = StdRng::seed_from_u64(42);
        let data = make_trainset(100);
        let (train, val) = split_train_val(&data, &mut rng);

        assert!(train.len() + val.len() == 100);
        assert!(!val.is_empty());
        assert!(!train.is_empty());
    }

    #[test]
    fn test_split_train_val_small() {
        let mut rng = StdRng::seed_from_u64(42);
        let data = make_trainset(2);
        let (train, val) = split_train_val(&data, &mut rng);

        assert!(train.len() + val.len() == 2);
    }
}
