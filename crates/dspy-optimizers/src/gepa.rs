//! GEPA — Generalized Evolutionary Prompt Adaptation.
//!
//! An evolutionary optimizer that uses reflective mutation to evolve
//! text components (instructions) of DSPy modules. GEPA captures full
//! traces, reflects on predictor behaviour, and proposes improved
//! instructions via a reflection LM.
//!
//! This is a self-contained port of both the DSPy GEPA wrapper and the
//! core `gepa` optimization engine (Python gepa package v0.0.24).
//!
//! Python equivalent: dspy/teleprompt/gepa/

use dspy_core::{Example, LM, LMConfig, Module, Prediction};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A candidate maps predictor name -> instruction text.
pub type Candidate = HashMap<String, String>;

/// Score with textual feedback for reflective updates.
#[derive(Debug, Clone)]
pub struct ScoreWithFeedback {
    pub score: f64,
    pub feedback: String,
}

/// GEPA feedback metric. Returns either a numeric score or a ScoreWithFeedback.
pub type GEPAFeedbackMetric =
    Arc<dyn Fn(&Example, &Prediction) -> GEPAMetricResult + Send + Sync>;

/// Return type of GEPA feedback metric.
pub enum GEPAMetricResult {
    Score(f64),
    WithFeedback(ScoreWithFeedback),
}

impl GEPAMetricResult {
    pub fn score(&self) -> f64 {
        match self {
            GEPAMetricResult::Score(s) => *s,
            GEPAMetricResult::WithFeedback(sf) => sf.score,
        }
    }

    pub fn feedback(&self) -> Option<&str> {
        match self {
            GEPAMetricResult::Score(_) => None,
            GEPAMetricResult::WithFeedback(sf) => Some(&sf.feedback),
        }
    }
}

/// Custom instruction proposer function type.
pub type ProposalFn = Arc<
    dyn Fn(
            &Candidate,
            &HashMap<String, Vec<ReflectiveExample>>,
            &[String],
        ) -> Candidate
        + Send
        + Sync,
>;

/// Single entry in a reflective dataset.
#[derive(Debug, Clone)]
pub struct ReflectiveExample {
    pub inputs: HashMap<String, String>,
    pub generated_outputs: HashMap<String, String>,
    pub feedback: String,
}

/// Evaluation batch result.
struct EvaluationBatch {
    #[allow(dead_code)]
    outputs: Vec<Option<Prediction>>,
    scores: Vec<f64>,
    trajectories: Option<Vec<TrajectoryEntry>>,
}

/// Single trajectory entry from traced evaluation.
struct TrajectoryEntry {
    example: Example,
    prediction: Option<Prediction>,
    score: f64,
}

// ---------------------------------------------------------------------------
// GEPAResult
// ---------------------------------------------------------------------------

/// Detailed results from a GEPA optimization run.
pub struct GEPAResult {
    pub program: Box<dyn Module>,
    pub candidates: Vec<Candidate>,
    pub parents: Vec<Vec<Option<usize>>>,
    pub val_aggregate_scores: Vec<f64>,
    pub total_metric_calls: usize,
    pub num_full_val_evals: usize,
    pub seed: u64,
    pub best_idx: usize,
}

// ---------------------------------------------------------------------------
// Auto-budget presets
// ---------------------------------------------------------------------------

fn auto_run_n(preset: &str) -> usize {
    match preset {
        "light" => 6,
        "medium" => 12,
        "heavy" => 18,
        _ => 6,
    }
}

// ---------------------------------------------------------------------------
// Instruction Proposal Signature
// ---------------------------------------------------------------------------

const DEFAULT_INSTRUCTION_PROMPT_TEMPLATE: &str = r#"I provided an assistant with the following instructions to perform a task for me:
```
<curr_instructions>
```

The following are examples of different task inputs provided to the assistant along with the assistant's response for each of them, and some feedback on how the assistant's response could be better:
```
<inputs_outputs_feedback>
```

Your task is to write a new instruction for the assistant.

Read the inputs carefully and identify the input format and infer detailed task description about the task I wish to solve with the assistant.

Read all the assistant responses and the corresponding feedback. Identify all niche and domain specific factual information about the task and include it in the instruction, as a lot of it may not be available to the assistant in the future. The assistant may have utilized a generalizable strategy to solve the task, if so, include that in the instruction as well.

Provide the new instructions within ``` blocks."#;

fn render_reflective_dataset(samples: &[ReflectiveExample]) -> String {
    let mut out = String::new();
    for (i, sample) in samples.iter().enumerate() {
        out.push_str(&format!("# Example {}\n", i + 1));
        out.push_str("## Inputs\n");
        for (k, v) in &sample.inputs {
            out.push_str(&format!("### {}\n{}\n\n", k, v.trim()));
        }
        out.push_str("## Generated Outputs\n");
        for (k, v) in &sample.generated_outputs {
            out.push_str(&format!("### {}\n{}\n\n", k, v.trim()));
        }
        out.push_str("## Feedback\n");
        out.push_str(sample.feedback.trim());
        out.push_str("\n\n");
    }
    out
}

fn build_instruction_prompt(
    current_instruction: &str,
    dataset_with_feedback: &[ReflectiveExample],
) -> String {
    let rendered = render_reflective_dataset(dataset_with_feedback);
    DEFAULT_INSTRUCTION_PROMPT_TEMPLATE
        .replace("<curr_instructions>", current_instruction)
        .replace("<inputs_outputs_feedback>", &rendered)
}

fn extract_instruction_from_lm_output(lm_out: &str) -> String {
    let first = lm_out.find("```");
    let last = lm_out.rfind("```");

    match (first, last) {
        (Some(start), Some(end)) if start < end => {
            let after_first = start + 3;
            let mut content = &lm_out[after_first..end];
            // Skip language tag on first line
            if let Some(nl) = content.find('\n') {
                let first_line = &content[..nl];
                if !first_line.contains(' ') && first_line.len() < 20 {
                    content = &content[nl + 1..];
                }
            }
            content.trim().to_string()
        }
        _ => {
            let stripped = lm_out.trim();
            if stripped.starts_with("```") {
                // Strip opening fence
                let rest = stripped.trim_start_matches("```");
                if let Some(nl) = rest.find('\n') {
                    return rest[nl + 1..].trim().to_string();
                }
                return rest.trim().to_string();
            }
            if stripped.ends_with("```") {
                return stripped[..stripped.len() - 3].trim().to_string();
            }
            stripped.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// GEPAState
// ---------------------------------------------------------------------------

struct GEPAState {
    program_candidates: Vec<Candidate>,
    parent_program_for_candidate: Vec<Vec<Option<usize>>>,
    prog_candidate_val_subscores: Vec<BTreeMap<usize, f64>>,
    pareto_front_valset: BTreeMap<usize, f64>,
    program_at_pareto_front_valset: BTreeMap<usize, BTreeSet<usize>>,
    list_of_named_predictors: Vec<String>,
    named_predictor_id_to_update_next: Vec<usize>,
    num_metric_calls_by_discovery: Vec<usize>,
    i: i64,
    num_full_ds_evals: usize,
    total_num_evals: usize,
}

impl GEPAState {
    fn new(seed_candidate: Candidate, base_scores_by_val_id: &BTreeMap<usize, f64>) -> Self {
        let pred_names: Vec<String> = seed_candidate.keys().cloned().collect();

        let mut program_at_pareto = BTreeMap::new();
        for &val_id in base_scores_by_val_id.keys() {
            let mut set = BTreeSet::new();
            set.insert(0);
            program_at_pareto.insert(val_id, set);
        }

        Self {
            program_candidates: vec![seed_candidate],
            parent_program_for_candidate: vec![vec![None]],
            prog_candidate_val_subscores: vec![base_scores_by_val_id.clone()],
            pareto_front_valset: base_scores_by_val_id.clone(),
            program_at_pareto_front_valset: program_at_pareto,
            list_of_named_predictors: pred_names,
            named_predictor_id_to_update_next: vec![0],
            num_metric_calls_by_discovery: vec![0],
            i: -1,
            num_full_ds_evals: 0,
            total_num_evals: 0,
        }
    }

    fn get_program_average_val_subset(&self, program_idx: usize) -> (f64, usize) {
        let scores = &self.prog_candidate_val_subscores[program_idx];
        if scores.is_empty() {
            return (f64::NEG_INFINITY, 0);
        }
        let sum: f64 = scores.values().sum();
        (sum / scores.len() as f64, scores.len())
    }

    fn program_full_scores_val_set(&self) -> Vec<f64> {
        (0..self.program_candidates.len())
            .map(|idx| self.get_program_average_val_subset(idx).0)
            .collect()
    }

    fn get_pareto_front_program_ids(&self) -> BTreeSet<usize> {
        let mut ids = BTreeSet::new();
        for front in self.program_at_pareto_front_valset.values() {
            for &id in front {
                ids.insert(id);
            }
        }
        ids
    }

    fn update_state_with_new_program(
        &mut self,
        parent_program_idx: &[usize],
        new_program: Candidate,
        scores_by_val_id: BTreeMap<usize, f64>,
        num_metric_calls_by_discovery: usize,
    ) -> usize {
        let new_idx = self.program_candidates.len();
        self.program_candidates.push(new_program);
        self.num_metric_calls_by_discovery
            .push(num_metric_calls_by_discovery);

        let max_pred_id = parent_program_idx
            .iter()
            .map(|&p| self.named_predictor_id_to_update_next[p])
            .max()
            .unwrap_or(0);
        self.named_predictor_id_to_update_next.push(max_pred_id);
        self.parent_program_for_candidate
            .push(parent_program_idx.iter().map(|&p| Some(p)).collect());
        self.prog_candidate_val_subscores
            .push(scores_by_val_id.clone());

        for (&val_id, &score) in &scores_by_val_id {
            let prev_score = self
                .pareto_front_valset
                .get(&val_id)
                .copied()
                .unwrap_or(f64::NEG_INFINITY);
            if score > prev_score {
                self.pareto_front_valset.insert(val_id, score);
                let mut set = BTreeSet::new();
                set.insert(new_idx);
                self.program_at_pareto_front_valset.insert(val_id, set);
            } else if (score - prev_score).abs() < f64::EPSILON {
                self.program_at_pareto_front_valset
                    .entry(val_id)
                    .or_insert_with(BTreeSet::new)
                    .insert(new_idx);
            }
        }

        new_idx
    }
}

// ---------------------------------------------------------------------------
// Batch Sampler
// ---------------------------------------------------------------------------

struct EpochShuffledBatchSampler {
    minibatch_size: usize,
    shuffled_ids: Vec<usize>,
    epoch: i64,
    id_freqs: HashMap<usize, usize>,
    last_trainset_size: usize,
    rng: StdRng,
}

impl EpochShuffledBatchSampler {
    fn new(minibatch_size: usize, seed: u64) -> Self {
        Self {
            minibatch_size,
            shuffled_ids: Vec::new(),
            epoch: -1,
            id_freqs: HashMap::new(),
            last_trainset_size: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    fn update_shuffled(&mut self, trainset_size: usize) {
        self.last_trainset_size = trainset_size;
        if trainset_size == 0 {
            self.shuffled_ids.clear();
            self.id_freqs.clear();
            return;
        }

        self.shuffled_ids = (0..trainset_size).collect();
        shuffle_vec(&mut self.shuffled_ids, &mut self.rng);

        self.id_freqs.clear();
        for &id in &self.shuffled_ids {
            *self.id_freqs.entry(id).or_insert(0) += 1;
        }

        let remainder = trainset_size % self.minibatch_size;
        let num_to_pad = if remainder != 0 {
            self.minibatch_size - remainder
        } else {
            0
        };
        for _ in 0..num_to_pad {
            let least_freq_id = *self
                .id_freqs
                .iter()
                .min_by_key(|(_, &freq)| freq)
                .map(|(id, _)| id)
                .unwrap_or(&0);
            self.shuffled_ids.push(least_freq_id);
            *self.id_freqs.entry(least_freq_id).or_insert(0) += 1;
        }
    }

    fn next_minibatch_ids(&mut self, trainset_size: usize, iteration: usize) -> Vec<usize> {
        assert!(trainset_size > 0, "Empty trainset");

        let base_idx = iteration * self.minibatch_size;
        let curr_epoch = if self.epoch == -1 {
            0
        } else {
            let len = self.shuffled_ids.len().max(1);
            (base_idx / len) as i64
        };

        let needs_refresh = self.shuffled_ids.is_empty()
            || trainset_size != self.last_trainset_size
            || curr_epoch > self.epoch;

        if needs_refresh {
            self.epoch = curr_epoch;
            self.update_shuffled(trainset_size);
        }

        let idx = base_idx % self.shuffled_ids.len();
        let end = (idx + self.minibatch_size).min(self.shuffled_ids.len());
        self.shuffled_ids[idx..end].to_vec()
    }
}

// ---------------------------------------------------------------------------
// Candidate Selectors
// ---------------------------------------------------------------------------

fn select_from_pareto_front(state: &GEPAState, rng: &mut StdRng) -> usize {
    let front_ids = state.get_pareto_front_program_ids();
    if front_ids.is_empty() {
        return 0;
    }

    let scores = state.program_full_scores_val_set();
    let ids: Vec<usize> = front_ids.into_iter().collect();
    let total: f64 = ids.iter().map(|&id| scores[id].max(0.001)).sum();

    let mut r = rng.gen::<f64>() * total;
    for &id in &ids {
        r -= scores[id].max(0.001);
        if r <= 0.0 {
            return id;
        }
    }
    *ids.last().unwrap()
}

fn select_current_best(state: &GEPAState) -> usize {
    let scores = state.program_full_scores_val_set();
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Component Selectors
// ---------------------------------------------------------------------------

fn round_robin_component_selector(state: &mut GEPAState, candidate_idx: usize) -> Vec<String> {
    let pid = state.named_predictor_id_to_update_next[candidate_idx];
    let num_preds = state.list_of_named_predictors.len();
    state.named_predictor_id_to_update_next[candidate_idx] = (pid + 1) % num_preds;
    vec![state.list_of_named_predictors[pid].clone()]
}

fn all_component_selector(candidate: &Candidate) -> Vec<String> {
    candidate.keys().cloned().collect()
}

// ---------------------------------------------------------------------------
// Merge Logic
// ---------------------------------------------------------------------------

struct MergeState {
    use_merge: bool,
    max_merge_invocations: usize,
    merges_due: usize,
    total_merges_tested: usize,
    last_iter_found_new_program: bool,
    merges_performed: HashSet<String>,
}

struct MergeProposal {
    candidate: Candidate,
    parent_program_ids: Vec<usize>,
    subsample_indices: Vec<usize>,
    subsample_scores_before: Vec<f64>,
    subsample_scores_after: Vec<f64>,
}

fn find_common_ancestor_pair(
    rng: &mut StdRng,
    parent_list: &[Vec<Option<usize>>],
    program_indexes: &[usize],
    merges_performed: &HashSet<String>,
    agg_scores: &[f64],
    program_candidates: &[Candidate],
    max_attempts: usize,
) -> Option<(usize, usize, usize)> {
    fn get_ancestors(node: usize, parent_list: &[Vec<Option<usize>>]) -> HashSet<usize> {
        let mut found = HashSet::new();
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            for parent in &parent_list[n] {
                if let Some(p) = parent {
                    if found.insert(*p) {
                        stack.push(*p);
                    }
                }
            }
        }
        found
    }

    for _ in 0..max_attempts {
        if program_indexes.len() < 2 {
            return None;
        }
        let idx1 = rng.gen_range(0..program_indexes.len());
        let mut idx2 = rng.gen_range(0..program_indexes.len());
        while idx2 == idx1 && program_indexes.len() > 1 {
            idx2 = rng.gen_range(0..program_indexes.len());
        }
        let (mut i, mut j) = (program_indexes[idx1], program_indexes[idx2]);
        if i == j {
            continue;
        }
        if j < i {
            std::mem::swap(&mut i, &mut j);
        }

        let ancestors_i = get_ancestors(i, parent_list);
        let ancestors_j = get_ancestors(j, parent_list);
        if ancestors_i.contains(&j) || ancestors_j.contains(&i) {
            continue;
        }

        let common: Vec<usize> = ancestors_i
            .iter()
            .filter(|a| ancestors_j.contains(a))
            .copied()
            .collect();

        let valid: Vec<usize> = common
            .into_iter()
            .filter(|&ancestor| {
                let key = format!("{},{},{}", i, j, ancestor);
                if merges_performed.contains(&key) {
                    return false;
                }
                if agg_scores[ancestor] > agg_scores[i]
                    || agg_scores[ancestor] > agg_scores[j]
                {
                    return false;
                }
                // Check desirable predictors
                let pred_names: Vec<&String> = program_candidates[ancestor].keys().collect();
                for pn in &pred_names {
                    let pred_anc = &program_candidates[ancestor][*pn];
                    let pred_i = &program_candidates[i][*pn];
                    let pred_j = &program_candidates[j][*pn];
                    if (pred_anc == pred_i || pred_anc == pred_j) && pred_i != pred_j {
                        return true;
                    }
                }
                false
            })
            .collect();

        if !valid.is_empty() {
            let total_w: f64 = valid.iter().map(|&a| agg_scores[a].max(0.001)).sum();
            let mut r = rng.gen::<f64>() * total_w;
            for &a in &valid {
                r -= agg_scores[a].max(0.001);
                if r <= 0.0 {
                    return Some((i, j, a));
                }
            }
            return Some((i, j, *valid.last().unwrap()));
        }
    }
    None
}

fn attempt_merge(
    state: &GEPAState,
    merge_state: &mut MergeState,
    rng: &mut StdRng,
) -> Option<MergeProposal> {
    let front_ids: Vec<usize> = state.get_pareto_front_program_ids().into_iter().collect();
    let agg_scores = state.program_full_scores_val_set();

    let result = find_common_ancestor_pair(
        rng,
        &state.parent_program_for_candidate,
        &front_ids,
        &merge_state.merges_performed,
        &agg_scores,
        &state.program_candidates,
        10,
    )?;

    let (id1, id2, ancestor) = result;
    merge_state
        .merges_performed
        .insert(format!("{},{},{}", id1, id2, ancestor));

    // Construct merged program
    let mut new_program = state.program_candidates[ancestor].clone();
    let pred_names: Vec<String> = new_program.keys().cloned().collect();
    for pn in &pred_names {
        let pred_anc = &state.program_candidates[ancestor][pn];
        let pred_id1 = &state.program_candidates[id1][pn];
        let pred_id2 = &state.program_candidates[id2][pn];
        if (pred_anc == pred_id1 || pred_anc == pred_id2) && pred_id1 != pred_id2 {
            let same_as_ancestor_id = if pred_anc == pred_id1 { 1 } else { 2 };
            new_program.insert(
                pn.clone(),
                if same_as_ancestor_id == 1 {
                    state.program_candidates[id2][pn].clone()
                } else {
                    state.program_candidates[id1][pn].clone()
                },
            );
        } else if pred_anc != pred_id1 && pred_anc != pred_id2 {
            let pick = if agg_scores[id1] > agg_scores[id2] {
                &state.program_candidates[id1][pn]
            } else if agg_scores[id2] > agg_scores[id1] {
                &state.program_candidates[id2][pn]
            } else if rng.gen::<f64>() < 0.5 {
                &state.program_candidates[id1][pn]
            } else {
                &state.program_candidates[id2][pn]
            };
            new_program.insert(pn.clone(), pick.clone());
        } else {
            new_program.insert(pn.clone(), state.program_candidates[id1][pn].clone());
        }
    }

    // Select subsample IDs
    let scores1 = &state.prog_candidate_val_subscores[id1];
    let scores2 = &state.prog_candidate_val_subscores[id2];
    let common_ids: Vec<usize> = scores1
        .keys()
        .filter(|k| scores2.contains_key(k))
        .copied()
        .collect();
    if common_ids.len() < 5 {
        return None;
    }

    let p1: Vec<usize> = common_ids
        .iter()
        .filter(|&&id| scores1.get(&id).unwrap_or(&0.0) > scores2.get(&id).unwrap_or(&0.0))
        .copied()
        .collect();
    let p2: Vec<usize> = common_ids
        .iter()
        .filter(|&&id| scores2.get(&id).unwrap_or(&0.0) > scores1.get(&id).unwrap_or(&0.0))
        .copied()
        .collect();
    let p3: Vec<usize> = common_ids
        .iter()
        .filter(|&&id| !p1.contains(&id) && !p2.contains(&id))
        .copied()
        .collect();

    let num_subsample = 5;
    let n_each = (num_subsample / 3).max(1);
    let mut selected: Vec<usize> = Vec::new();
    for bucket in [&p1, &p2, &p3] {
        if selected.len() >= num_subsample {
            break;
        }
        let available: Vec<usize> = bucket
            .iter()
            .filter(|id| !selected.contains(id))
            .copied()
            .collect();
        let take = available.len().min(n_each).min(num_subsample - selected.len());
        if take > 0 {
            let mut shuffled = available;
            shuffle_vec(&mut shuffled, rng);
            selected.extend(&shuffled[..take]);
        }
    }
    let remaining = num_subsample.saturating_sub(selected.len());
    if remaining > 0 {
        let mut unused: Vec<usize> = common_ids
            .iter()
            .filter(|id| !selected.contains(id))
            .copied()
            .collect();
        shuffle_vec(&mut unused, rng);
        selected.extend(&unused[..remaining.min(unused.len())]);
    }

    let subsample_ids: Vec<usize> = selected.into_iter().take(num_subsample).collect();
    let id1_sub_scores: Vec<f64> = subsample_ids
        .iter()
        .map(|id| *scores1.get(id).unwrap_or(&0.0))
        .collect();
    let id2_sub_scores: Vec<f64> = subsample_ids
        .iter()
        .map(|id| *scores2.get(id).unwrap_or(&0.0))
        .collect();

    Some(MergeProposal {
        candidate: new_program,
        parent_program_ids: vec![id1, id2],
        subsample_indices: subsample_ids,
        subsample_scores_before: vec![
            id1_sub_scores.iter().sum(),
            id2_sub_scores.iter().sum(),
        ],
        subsample_scores_after: Vec::new(),
    })
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn shuffle_vec<T>(arr: &mut [T], rng: &mut StdRng) {
    for i in (1..arr.len()).rev() {
        let j = rng.gen_range(0..=i);
        arr.swap(i, j);
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Budget specification for GEPA.
pub enum GEPABudget {
    /// Auto preset: "light", "medium", "heavy".
    Auto(String),
    /// Maximum number of full evaluations.
    MaxFullEvals(usize),
    /// Maximum number of individual metric calls.
    MaxMetricCalls(usize),
}

/// Candidate selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateSelection {
    Pareto,
    CurrentBest,
}

/// Component selector strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentSelection {
    RoundRobin,
    All,
}

/// Configuration for GEPA optimizer.
pub struct GEPAConfig {
    pub metric: GEPAFeedbackMetric,
    pub budget: GEPABudget,
    pub reflection_minibatch_size: usize,
    pub candidate_selection_strategy: CandidateSelection,
    pub reflection_lm: Option<Arc<dyn LM>>,
    pub skip_perfect_score: bool,
    pub instruction_proposer: Option<ProposalFn>,
    pub component_selector: ComponentSelection,
    pub use_merge: bool,
    pub max_merge_invocations: usize,
    pub num_threads: usize,
    pub failure_score: f64,
    pub perfect_score: f64,
    pub track_stats: bool,
    pub seed: u64,
}

impl GEPAConfig {
    pub fn new(metric: GEPAFeedbackMetric, budget: GEPABudget) -> Self {
        Self {
            metric,
            budget,
            reflection_minibatch_size: 3,
            candidate_selection_strategy: CandidateSelection::Pareto,
            reflection_lm: None,
            skip_perfect_score: true,
            instruction_proposer: None,
            component_selector: ComponentSelection::RoundRobin,
            use_merge: true,
            max_merge_invocations: 5,
            num_threads: 2,
            failure_score: 0.0,
            perfect_score: 1.0,
            track_stats: false,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// GEPA Optimizer
// ---------------------------------------------------------------------------

pub struct GEPA {
    config: GEPAConfig,
}

impl GEPA {
    pub fn new(config: GEPAConfig) -> Self {
        assert!(
            config.reflection_lm.is_some() || config.instruction_proposer.is_some(),
            "GEPA requires either a reflectionLm or a custom instruction proposer.",
        );
        Self { config }
    }

    /// Auto-budget calculation matching Python GEPA.
    fn auto_budget(
        &self,
        num_preds: usize,
        num_candidates: usize,
        valset_size: usize,
    ) -> usize {
        let minibatch_size = 35;
        let full_eval_steps = 5;

        let num_trials = (2 * num_preds * 2 * ((num_candidates as f64).log2() as usize))
            .max((1.5 * num_candidates as f64) as usize);
        let mut total = valset_size; // initial full eval
        total += num_candidates * 5; // bootstrapping
        total += num_trials * minibatch_size;
        if num_trials == 0 {
            return total;
        }
        let periodic_fulls = (num_trials + 1) / full_eval_steps + 1;
        let extra_final = if num_trials < full_eval_steps { 1 } else { 0 };
        total += (periodic_fulls + extra_final) * valset_size;
        total
    }

    /// Compile: run GEPA optimization on the student module.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        valset: Option<&[Example]>,
    ) -> dspy_core::Result<GEPAResult> {
        assert!(!trainset.is_empty(), "Trainset must be provided and non-empty");

        let effective_valset = valset.unwrap_or(trainset);

        // Resolve budget
        let max_metric_calls = match &self.config.budget {
            GEPABudget::Auto(preset) => {
                let num_preds = student.named_predictors().len();
                let n = auto_run_n(preset);
                self.auto_budget(num_preds, n, effective_valset.len())
            }
            GEPABudget::MaxFullEvals(max_full) => {
                max_full * (trainset.len() + effective_valset.len())
            }
            GEPABudget::MaxMetricCalls(max_calls) => *max_calls,
        };

        let mut rng = StdRng::seed_from_u64(self.config.seed);

        // Build seed candidate
        let mut seed_candidate: Candidate = HashMap::new();
        for (name, pred) in student.named_predictors() {
            seed_candidate.insert(name.to_string(), pred.signature.instructions().to_string());
        }

        // Evaluate seed candidate on full valset
        let seed_scores = self
            .evaluate_candidate(student, &seed_candidate, effective_valset, false)
            .await;

        let mut seed_scores_by_val_id = BTreeMap::new();
        for (i, &score) in seed_scores.scores.iter().enumerate() {
            seed_scores_by_val_id.insert(i, score);
        }

        // Initialize state
        let mut state = GEPAState::new(seed_candidate, &seed_scores_by_val_id);
        state.num_full_ds_evals = 1;
        state.total_num_evals = effective_valset.len();

        // Initialize batch sampler
        let mut batch_sampler =
            EpochShuffledBatchSampler::new(self.config.reflection_minibatch_size, self.config.seed);

        // Initialize merge state
        let mut merge_state = MergeState {
            use_merge: self.config.use_merge,
            max_merge_invocations: self.config.max_merge_invocations,
            merges_due: 0,
            total_merges_tested: 0,
            last_iter_found_new_program: false,
            merges_performed: HashSet::new(),
        };

        // Main optimization loop
        while state.total_num_evals < max_metric_calls {
            state.i += 1;

            // 1) Attempt merge if scheduled
            if merge_state.use_merge
                && merge_state.merges_due > 0
                && merge_state.last_iter_found_new_program
            {
                let merge_proposal = attempt_merge(&state, &mut merge_state, &mut rng);
                merge_state.last_iter_found_new_program = false;

                if let Some(mut proposal) = merge_proposal {
                    let merge_batch: Vec<Example> = proposal
                        .subsample_indices
                        .iter()
                        .map(|&i| effective_valset[i].clone())
                        .collect();
                    let merge_eval = self
                        .evaluate_candidate(student, &proposal.candidate, &merge_batch, false)
                        .await;
                    state.total_num_evals += proposal.subsample_indices.len();
                    proposal.subsample_scores_after = merge_eval.scores.clone();

                    let new_sum: f64 = merge_eval.scores.iter().sum();
                    let parent_max = proposal
                        .subsample_scores_before
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    if new_sum >= parent_max {
                        self.full_eval_and_add(
                            student,
                            proposal.candidate,
                            &mut state,
                            effective_valset,
                            &proposal.parent_program_ids,
                        )
                        .await;
                        merge_state.merges_due -= 1;
                        merge_state.total_merges_tested += 1;
                        continue;
                    }
                }
                continue;
            }

            if merge_state.use_merge {
                merge_state.last_iter_found_new_program = false;
            }

            // 2) Reflective mutation
            let candidate_idx = match self.config.candidate_selection_strategy {
                CandidateSelection::Pareto => select_from_pareto_front(&state, &mut rng),
                CandidateSelection::CurrentBest => select_current_best(&state),
            };
            let curr_prog = state.program_candidates[candidate_idx].clone();

            // Sample minibatch from trainset
            let subsample_ids =
                batch_sampler.next_minibatch_ids(trainset.len(), state.i as usize);
            let minibatch: Vec<Example> =
                subsample_ids.iter().map(|&i| trainset[i].clone()).collect();

            // Evaluate current program with traces
            let eval_curr = self
                .evaluate_candidate(student, &curr_prog, &minibatch, true)
                .await;
            state.total_num_evals += subsample_ids.len();

            if eval_curr.trajectories.is_none()
                || eval_curr
                    .trajectories
                    .as_ref()
                    .map_or(true, |t| t.is_empty())
            {
                continue;
            }

            if self.config.skip_perfect_score
                && eval_curr
                    .scores
                    .iter()
                    .all(|&s| s >= self.config.perfect_score)
            {
                continue;
            }

            // Select components to update
            let components_to_update = match self.config.component_selector {
                ComponentSelection::RoundRobin => {
                    round_robin_component_selector(&mut state, candidate_idx)
                }
                ComponentSelection::All => all_component_selector(&curr_prog),
            };

            // Build reflective dataset and propose new texts
            let dataset = match self.make_reflective_dataset(
                student,
                &curr_prog,
                &eval_curr,
                &minibatch,
                &components_to_update,
            ) {
                Ok(d) => d,
                Err(_) => continue,
            };

            let new_texts = match self
                .propose_new_texts(&curr_prog, &dataset, &components_to_update)
                .await
            {
                Ok(texts) => texts,
                Err(_) => continue,
            };

            // Create new candidate
            let mut new_candidate = curr_prog.clone();
            for (pname, text) in &new_texts {
                new_candidate.insert(pname.clone(), text.clone());
            }

            // Evaluate new candidate on same minibatch
            let eval_new = self
                .evaluate_candidate(student, &new_candidate, &minibatch, false)
                .await;
            state.total_num_evals += subsample_ids.len();

            let old_sum: f64 = eval_curr.scores.iter().sum();
            let new_sum: f64 = eval_new.scores.iter().sum();
            if new_sum <= old_sum {
                continue;
            }

            // Accept: full eval and add to state
            self.full_eval_and_add(
                student,
                new_candidate,
                &mut state,
                effective_valset,
                &[candidate_idx],
            )
            .await;

            // Schedule merge
            if merge_state.use_merge {
                merge_state.last_iter_found_new_program = true;
                if merge_state.total_merges_tested < merge_state.max_merge_invocations {
                    merge_state.merges_due += 1;
                }
            }
        }

        // Find best candidate
        let scores = state.program_full_scores_val_set();
        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Build result program
        let best_candidate = &state.program_candidates[best_idx];
        let result_program = self.build_program(student, best_candidate);

        Ok(GEPAResult {
            program: result_program,
            candidates: state.program_candidates,
            parents: state.parent_program_for_candidate,
            val_aggregate_scores: scores,
            total_metric_calls: state.total_num_evals,
            num_full_val_evals: state.num_full_ds_evals,
            seed: self.config.seed,
            best_idx,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn build_program(&self, student: &dyn Module, candidate: &Candidate) -> Box<dyn Module> {
        let mut prog = student.deep_copy();
        for (name, pred) in prog.named_predictors_mut() {
            if let Some(instr) = candidate.get(name) {
                pred.signature = pred.signature.with_instructions(instr);
            }
        }
        prog
    }

    async fn evaluate_candidate(
        &self,
        student: &dyn Module,
        candidate: &Candidate,
        batch: &[Example],
        capture_traces: bool,
    ) -> EvaluationBatch {
        let program = self.build_program(student, candidate);
        let mut scores = Vec::new();
        let mut outputs = Vec::new();
        let mut trajectories: Option<Vec<TrajectoryEntry>> = if capture_traces {
            Some(Vec::new())
        } else {
            None
        };

        for example in batch {
            let inputs = example.inputs();
            match program.call(&inputs).await {
                Ok(prediction) => {
                    let score_result = (self.config.metric)(example, &prediction);
                    let score = score_result.score();
                    scores.push(score);
                    outputs.push(Some(prediction.clone()));
                    if let Some(ref mut trajs) = trajectories {
                        trajs.push(TrajectoryEntry {
                            example: example.clone(),
                            prediction: Some(prediction),
                            score,
                        });
                    }
                }
                Err(_) => {
                    scores.push(self.config.failure_score);
                    outputs.push(None);
                    if let Some(ref mut trajs) = trajectories {
                        trajs.push(TrajectoryEntry {
                            example: example.clone(),
                            prediction: None,
                            score: self.config.failure_score,
                        });
                    }
                }
            }
        }

        EvaluationBatch {
            outputs,
            scores,
            trajectories,
        }
    }

    async fn full_eval_and_add(
        &self,
        student: &dyn Module,
        new_program: Candidate,
        state: &mut GEPAState,
        valset: &[Example],
        parent_program_idx: &[usize],
    ) -> usize {
        let num_metric_calls_by_discovery = state.total_num_evals;

        let eval_result = self
            .evaluate_candidate(student, &new_program, valset, false)
            .await;
        state.num_full_ds_evals += 1;
        state.total_num_evals += valset.len();

        let mut scores_by_val_id = BTreeMap::new();
        for (i, &score) in eval_result.scores.iter().enumerate() {
            scores_by_val_id.insert(i, score);
        }

        state.update_state_with_new_program(
            parent_program_idx,
            new_program,
            scores_by_val_id,
            num_metric_calls_by_discovery,
        )
    }

    fn make_reflective_dataset(
        &self,
        student: &dyn Module,
        candidate: &Candidate,
        eval_batch: &EvaluationBatch,
        _batch: &[Example],
        components_to_update: &[String],
    ) -> dspy_core::Result<HashMap<String, Vec<ReflectiveExample>>> {
        let program = self.build_program(student, candidate);
        let mut result: HashMap<String, Vec<ReflectiveExample>> = HashMap::new();

        for pred_name in components_to_update {
            let mut found_pred = false;
            for (name, _) in program.named_predictors() {
                if name == pred_name {
                    found_pred = true;
                    break;
                }
            }
            if !found_pred {
                continue;
            }

            let mut items = Vec::new();
            if let Some(ref trajs) = eval_batch.trajectories {
                for traj in trajs {
                    let mut inputs = HashMap::new();
                    let example = &traj.example;
                    for (k, v) in example.to_map() {
                        inputs.insert(k.clone(), v.to_string().trim_matches('"').to_string());
                    }

                    let mut gen_outputs = HashMap::new();
                    if let Some(ref prediction) = traj.prediction {
                        for (k, v) in prediction.example.to_map() {
                            gen_outputs
                                .insert(k.clone(), v.to_string().trim_matches('"').to_string());
                        }
                    }

                    // Get feedback
                    let feedback = if let Some(ref prediction) = traj.prediction {
                        let feedback_result = (self.config.metric)(example, prediction);
                        match feedback_result.feedback() {
                            Some(fb) => fb.to_string(),
                            None => format!(
                                "This trajectory got a score of {:.2}.",
                                feedback_result.score()
                            ),
                        }
                    } else {
                        format!(
                            "This trajectory got a score of {:.2}.",
                            traj.score
                        )
                    };

                    items.push(ReflectiveExample {
                        inputs,
                        generated_outputs: gen_outputs,
                        feedback,
                    });
                }
            }

            if !items.is_empty() {
                result.insert(pred_name.clone(), items);
            }
        }

        if result.is_empty() {
            return Err(dspy_core::DspyError::OptimizationError(
                "No valid reflective examples found.".to_string(),
            ));
        }

        Ok(result)
    }

    async fn propose_new_texts(
        &self,
        candidate: &Candidate,
        reflective_dataset: &HashMap<String, Vec<ReflectiveExample>>,
        components_to_update: &[String],
    ) -> dspy_core::Result<Candidate> {
        if let Some(ref proposer) = self.config.instruction_proposer {
            return Ok(proposer(candidate, reflective_dataset, components_to_update));
        }

        let reflection_lm = self.config.reflection_lm.as_ref().ok_or_else(|| {
            dspy_core::DspyError::LMError(
                "reflectionLm is required when no custom proposer is set.".to_string(),
            )
        })?;

        let mut results = Candidate::new();
        for name in components_to_update {
            let dataset = match reflective_dataset.get(name) {
                Some(d) if !d.is_empty() => d,
                _ => continue,
            };

            let base_instruction = candidate.get(name).map(|s| s.as_str()).unwrap_or("");
            let prompt = build_instruction_prompt(base_instruction, dataset);

            let config = LMConfig {
                temperature: Some(1.0),
                ..LMConfig::new(reflection_lm.model())
            };
            let messages = vec![dspy_core::Message::user(&prompt)];

            match reflection_lm.call(&messages, &config).await {
                Ok(responses) => {
                    let text = responses
                        .first()
                        .map(|r| r.text.as_str())
                        .unwrap_or("");
                    results.insert(name.clone(), extract_instruction_from_lm_output(text));
                }
                Err(_) => continue,
            }
        }

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{LMConfig, LMResponse, Message, Predict, Signature};

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
            // Check if this is a reflection call
            let is_reflection = messages
                .iter()
                .any(|m| m.content.contains("new instruction"));
            let text = if is_reflection {
                format!("```\nImproved instruction: {}\n```", self.answer)
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

    struct TwoPredModule {
        pred1: Predict,
        pred2: Predict,
    }

    impl TwoPredModule {
        fn new(sig1: Signature, sig2: Signature) -> Self {
            Self {
                pred1: Predict::new(sig1),
                pred2: Predict::new(sig2),
            }
        }
        fn with_lm(mut self, lm: Arc<dyn LM>) -> Self {
            self.pred1.set_lm(lm.clone());
            self.pred2.set_lm(lm);
            self
        }
    }

    #[async_trait]
    impl Module for TwoPredModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
            self.pred1.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("pred1", &self.pred1), ("pred2", &self.pred2)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("pred1", &mut self.pred1), ("pred2", &mut self.pred2)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(TwoPredModule {
                pred1: self.pred1.clone(),
                pred2: self.pred2.clone(),
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

    #[test]
    fn test_construction_requires_lm_or_proposer() {
        let metric: GEPAFeedbackMetric = Arc::new(|_, _| GEPAMetricResult::Score(1.0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            GEPA::new(GEPAConfig {
                metric,
                budget: GEPABudget::MaxMetricCalls(100),
                reflection_lm: None,
                instruction_proposer: None,
                ..GEPAConfig::new(
                    Arc::new(|_, _| GEPAMetricResult::Score(1.0)),
                    GEPABudget::MaxMetricCalls(100),
                )
            });
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_construction_with_reflection_lm() {
        let metric: GEPAFeedbackMetric = Arc::new(|_, _| GEPAMetricResult::Score(1.0));
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("test"));
        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(100),
            reflection_lm: Some(lm),
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(1.0)),
                GEPABudget::MaxMetricCalls(100),
            )
        });
        assert!(gepa.config.reflection_lm.is_some());
    }

    #[test]
    fn test_construction_with_proposer() {
        let metric: GEPAFeedbackMetric = Arc::new(|_, _| GEPAMetricResult::Score(1.0));
        let proposer: ProposalFn = Arc::new(|candidate, _, _| candidate.clone());
        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(100),
            instruction_proposer: Some(proposer),
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(1.0)),
                GEPABudget::MaxMetricCalls(100),
            )
        });
        assert!(gepa.config.instruction_proposer.is_some());
    }

    #[test]
    fn test_extract_instruction_from_lm_output() {
        assert_eq!(
            extract_instruction_from_lm_output("```\nNew instruction text here\n```"),
            "New instruction text here"
        );
        assert_eq!(
            extract_instruction_from_lm_output("```python\nNew instruction\n```"),
            "New instruction"
        );
        assert_eq!(
            extract_instruction_from_lm_output("Just plain text"),
            "Just plain text"
        );
    }

    #[test]
    fn test_gepa_state_basic() {
        let mut seed = Candidate::new();
        seed.insert("predict".to_string(), "Answer the question.".to_string());

        let mut base_scores = BTreeMap::new();
        base_scores.insert(0, 0.5);
        base_scores.insert(1, 0.7);
        base_scores.insert(2, 0.3);

        let state = GEPAState::new(seed, &base_scores);
        assert_eq!(state.program_candidates.len(), 1);
        assert_eq!(state.pareto_front_valset.len(), 3);
        let (avg, count) = state.get_program_average_val_subset(0);
        assert_eq!(count, 3);
        assert!((avg - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_gepa_state_update() {
        let mut seed = Candidate::new();
        seed.insert("predict".to_string(), "Original".to_string());

        let mut base_scores = BTreeMap::new();
        base_scores.insert(0, 0.5);
        base_scores.insert(1, 0.5);

        let mut state = GEPAState::new(seed, &base_scores);

        let mut new_prog = Candidate::new();
        new_prog.insert("predict".to_string(), "Improved".to_string());

        let mut new_scores = BTreeMap::new();
        new_scores.insert(0, 0.8);
        new_scores.insert(1, 0.6);

        let new_idx = state.update_state_with_new_program(&[0], new_prog, new_scores, 10);
        assert_eq!(new_idx, 1);
        assert_eq!(state.program_candidates.len(), 2);
        assert_eq!(state.pareto_front_valset[&0], 0.8);
    }

    #[test]
    fn test_batch_sampler() {
        let mut sampler = EpochShuffledBatchSampler::new(3, 42);
        let ids = sampler.next_minibatch_ids(10, 0);
        assert_eq!(ids.len(), 3);
        // All ids should be in range
        for &id in &ids {
            assert!(id < 10);
        }
    }

    #[tokio::test]
    async fn test_compile_minimal_budget() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(0.5));

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(20),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.5)),
                GEPABudget::MaxMetricCalls(20),
            )
        });

        let trainset = make_trainset(3);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
        assert!(result.total_metric_calls > 0);
    }

    #[tokio::test]
    async fn test_compile_with_custom_proposer() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(0.5));
        let proposer: ProposalFn = Arc::new(|candidate, _, components| {
            let mut result = Candidate::new();
            for c in components {
                let base = candidate.get(c).cloned().unwrap_or_default();
                result.insert(c.clone(), format!("{} [improved]", base));
            }
            result
        });

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm);

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(30),
            instruction_proposer: Some(proposer),
            reflection_minibatch_size: 2,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.5)),
                GEPABudget::MaxMetricCalls(30),
            )
        });

        let trainset = make_trainset(3);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_compile_with_valset() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(0.7));

        let sig = Signature::from_string("q -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(25),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.7)),
                GEPABudget::MaxMetricCalls(25),
            )
        });

        let trainset = make_trainset(2);
        let valset = make_trainset(3);
        let result = gepa
            .compile(&student, &trainset, Some(&valset))
            .await
            .unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_compile_skips_perfect_scores() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(1.0));

        let sig = Signature::from_string("q -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(15),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            skip_perfect_score: true,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(1.0)),
                GEPABudget::MaxMetricCalls(15),
            )
        });

        let trainset = make_trainset(2);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_compile_two_predictors_round_robin() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(0.6));

        let sig1 = Signature::from_string("question -> answer").unwrap();
        let sig2 = Signature::from_string("answer -> summary").unwrap();
        let student = TwoPredModule::new(sig1, sig2).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(30),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            component_selector: ComponentSelection::RoundRobin,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.6)),
                GEPABudget::MaxMetricCalls(30),
            )
        });

        let trainset = make_trainset(3);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_compile_current_best_selection() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: GEPAFeedbackMetric =
            Arc::new(|_, _| GEPAMetricResult::Score(0.5));

        let sig = Signature::from_string("q -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(15),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            candidate_selection_strategy: CandidateSelection::CurrentBest,
            use_merge: false,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.5)),
                GEPABudget::MaxMetricCalls(15),
            )
        });

        let trainset = make_trainset(2);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[tokio::test]
    async fn test_compile_with_merge() {
        dspy_core::reset_settings();
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let call_count = std::sync::atomic::AtomicUsize::new(0);
        let metric: GEPAFeedbackMetric = Arc::new(move |_, _| {
            let count = call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            GEPAMetricResult::Score(0.3 + (count as f64 * 0.02).min(0.65))
        });

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::MaxMetricCalls(50),
            reflection_lm: Some(lm),
            reflection_minibatch_size: 2,
            use_merge: true,
            max_merge_invocations: 3,
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(0.5)),
                GEPABudget::MaxMetricCalls(50),
            )
        });

        let trainset = make_trainset(5);
        let result = gepa.compile(&student, &trainset, None).await.unwrap();
        assert!(!result.candidates.is_empty());
    }

    #[test]
    fn test_auto_budget() {
        let metric: GEPAFeedbackMetric = Arc::new(|_, _| GEPAMetricResult::Score(1.0));
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("test"));
        let gepa = GEPA::new(GEPAConfig {
            metric,
            budget: GEPABudget::Auto("light".to_string()),
            reflection_lm: Some(lm),
            ..GEPAConfig::new(
                Arc::new(|_, _| GEPAMetricResult::Score(1.0)),
                GEPABudget::Auto("light".to_string()),
            )
        });
        let budget = gepa.auto_budget(1, 6, 10);
        assert!(budget > 0);
    }

    #[test]
    fn test_render_reflective_dataset() {
        let samples = vec![ReflectiveExample {
            inputs: {
                let mut m = HashMap::new();
                m.insert("question".to_string(), "What is 2+2?".to_string());
                m
            },
            generated_outputs: {
                let mut m = HashMap::new();
                m.insert("answer".to_string(), "5".to_string());
                m
            },
            feedback: "The answer should be 4, not 5.".to_string(),
        }];
        let rendered = render_reflective_dataset(&samples);
        assert!(rendered.contains("Example 1"));
        assert!(rendered.contains("What is 2+2?"));
        assert!(rendered.contains("answer should be 4"));
    }
}
