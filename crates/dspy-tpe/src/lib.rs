//! Tree-structured Parzen Estimator — Bayesian optimization for MIPROv2.
//!
//! Implements the TPE sampler algorithm focused on categorical parameter spaces,
//! which is what MIPROv2 uses for optimizing instruction and demo selections.
//!
//! Core algorithm:
//! 1. Split completed trials into "good" (above threshold) and "bad" (below)
//! 2. Model each parameter with two density estimates: l(x) for good, g(x) for bad
//! 3. Suggest parameters that maximize l(x)/g(x) ratio (Expected Improvement)

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Direction of optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Minimize,
    Maximize,
}

/// State of a trial
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrialState {
    Running,
    Complete,
    Pruned,
    Fail,
}

/// Distribution for a parameter
#[derive(Debug, Clone)]
pub enum Distribution {
    /// Categorical with N choices (0..n)
    Categorical(usize),
}

/// A completed trial with its parameters and objective value
#[derive(Debug, Clone)]
pub struct FrozenTrial {
    pub number: usize,
    pub state: TrialState,
    pub value: Option<f64>,
    pub params: HashMap<String, usize>,
    pub distributions: HashMap<String, Distribution>,
}

/// An active trial for parameter suggestion
pub struct Trial<'a> {
    pub number: usize,
    study: &'a mut Study,
    params: HashMap<String, usize>,
}

impl<'a> Trial<'a> {
    /// Suggest a categorical parameter value using TPE
    pub fn suggest_categorical(&mut self, name: &str, n_choices: usize) -> usize {
        let value = self.study.sampler.sample_categorical(
            name,
            n_choices,
            &self.study.trials,
            self.study.direction,
        );
        self.params.insert(name.to_string(), value);
        value
    }

    /// Get current params
    pub fn params(&self) -> &HashMap<String, usize> {
        &self.params
    }
}

/// TPE Sampler — the core Bayesian optimization algorithm
pub struct TPESampler {
    rng: StdRng,
    /// Prior weight for Laplace smoothing
    prior_weight: f64,
    /// Number of startup trials before using TPE (use random until then)
    n_startup_trials: usize,
    /// Gamma function: fraction of trials considered "good"
    /// Default: sqrt(n) / n, which gives approximately n_below = sqrt(n)
    gamma_fn: Box<dyn Fn(usize) -> usize + Send + Sync>,
}

impl TPESampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            prior_weight: 1.0,
            n_startup_trials: 10,
            gamma_fn: Box::new(|n: usize| {
                // Default: n_below = ceil(sqrt(n))
                let sq = (n as f64).sqrt().ceil() as usize;
                sq.max(1).min(n)
            }),
        }
    }

    pub fn with_prior_weight(mut self, weight: f64) -> Self {
        self.prior_weight = weight;
        self
    }

    pub fn with_n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Sample a categorical parameter value using TPE
    fn sample_categorical(
        &mut self,
        name: &str,
        n_choices: usize,
        trials: &[FrozenTrial],
        direction: Direction,
    ) -> usize {
        if n_choices == 0 {
            panic!("n_choices must be > 0");
        }
        if n_choices == 1 {
            return 0;
        }

        // Collect completed trials that have this parameter
        let completed: Vec<(usize, f64)> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.value.is_some())
            .filter_map(|t| {
                t.params.get(name).map(|&v| (v, t.value.unwrap()))
            })
            .collect();

        // During startup phase, sample uniformly
        if completed.len() < self.n_startup_trials {
            return self.rng.gen_range(0..n_choices);
        }

        // Split into good and bad
        let n_below = (self.gamma_fn)(completed.len());
        let (below_values, above_values) = split_trials(&completed, n_below, direction);

        // Compute l(x) and g(x) for each choice
        let l_probs = categorical_log_pdf(&below_values, n_choices, self.prior_weight);
        let g_probs = categorical_log_pdf(&above_values, n_choices, self.prior_weight);

        // EI(x) ∝ l(x) / g(x), so log_ei = log_l - log_g
        let log_ei: Vec<f64> = l_probs
            .iter()
            .zip(g_probs.iter())
            .map(|(&l, &g)| l - g)
            .collect();

        // Convert to probabilities and sample
        let max_log_ei = log_ei.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_ei.iter().map(|&v| (v - max_log_ei).exp()).collect();
        let sum_weights: f64 = weights.iter().sum();

        if sum_weights <= 0.0 {
            return self.rng.gen_range(0..n_choices);
        }

        // Weighted random choice
        let mut r = self.rng.gen::<f64>() * sum_weights;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }

        n_choices - 1
    }
}

/// Split trials into "below" (good) and "above" (bad) groups
/// For maximization: good = highest values; for minimization: good = lowest values
fn split_trials(
    trials: &[(usize, f64)],
    n_below: usize,
    direction: Direction,
) -> (Vec<usize>, Vec<usize>) {
    let mut indexed: Vec<(usize, usize, f64)> = trials
        .iter()
        .enumerate()
        .map(|(i, &(choice, value))| (i, choice, value))
        .collect();

    match direction {
        Direction::Minimize => indexed.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap()),
        Direction::Maximize => indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap()),
    }

    let below: Vec<usize> = indexed[..n_below].iter().map(|&(_, c, _)| c).collect();
    let above: Vec<usize> = indexed[n_below..].iter().map(|&(_, c, _)| c).collect();

    (below, above)
}

/// Compute log probability density for each category using Laplace smoothing
fn categorical_log_pdf(values: &[usize], n_choices: usize, prior_weight: f64) -> Vec<f64> {
    let mut counts = vec![0.0f64; n_choices];
    for &v in values {
        if v < n_choices {
            counts[v] += 1.0;
        }
    }

    let total = values.len() as f64 + prior_weight * n_choices as f64;

    counts
        .iter()
        .map(|&c| ((c + prior_weight) / total).ln())
        .collect()
}

/// A study that manages trials and optimization
pub struct Study {
    pub direction: Direction,
    sampler: TPESampler,
    trials: Vec<FrozenTrial>,
    next_trial_number: usize,
}

impl Study {
    pub fn new(direction: Direction, sampler: TPESampler) -> Self {
        Self {
            direction,
            sampler,
            trials: Vec::new(),
            next_trial_number: 0,
        }
    }

    /// Add a pre-computed trial (for injecting baseline results)
    pub fn add_trial(&mut self, trial: FrozenTrial) {
        self.trials.push(trial);
        if self.trials.last().unwrap().number >= self.next_trial_number {
            self.next_trial_number = self.trials.last().unwrap().number + 1;
        }
    }

    /// Create a frozen trial from known params and value (for baseline injection)
    pub fn create_trial(
        &self,
        params: HashMap<String, usize>,
        distributions: HashMap<String, Distribution>,
        value: f64,
    ) -> FrozenTrial {
        FrozenTrial {
            number: self.next_trial_number,
            state: TrialState::Complete,
            value: Some(value),
            params,
            distributions,
        }
    }

    /// Run optimization with a closure that evaluates each trial
    pub fn optimize<F>(&mut self, n_trials: usize, mut objective: F)
    where
        F: FnMut(&mut Trial<'_>) -> f64,
    {
        for _ in 0..n_trials {
            let trial_number = self.next_trial_number;
            self.next_trial_number += 1;

            // Create trial — we need to pass `self` mutably so the trial can access the sampler.
            // We'll collect the params after the objective runs.
            let mut trial = Trial {
                number: trial_number,
                study: self,
                params: HashMap::new(),
            };

            let value = objective(&mut trial);
            let params = trial.params.clone();

            // Record the completed trial
            // Build distributions from the params (all categorical)
            let distributions: HashMap<String, Distribution> = HashMap::new();
            self.trials.push(FrozenTrial {
                number: trial_number,
                state: TrialState::Complete,
                value: Some(value),
                params,
                distributions,
            });
        }
    }

    /// Suggest a categorical parameter value for a pending trial.
    /// This exposes the sampler for manual ask/tell workflows.
    pub fn suggest_categorical(&mut self, name: &str, n_choices: usize) -> usize {
        self.sampler.sample_categorical(name, n_choices, &self.trials, self.direction)
    }

    /// Record a completed trial with known params and score.
    /// Use this for async evaluation workflows (ask → evaluate → tell).
    pub fn record_trial(&mut self, params: HashMap<String, usize>, value: f64) {
        let number = self.next_trial_number;
        self.next_trial_number += 1;
        self.trials.push(FrozenTrial {
            number,
            state: TrialState::Complete,
            value: Some(value),
            params,
            distributions: HashMap::new(),
        });
    }

    /// Get all completed trials
    pub fn completed_trials(&self) -> Vec<&FrozenTrial> {
        self.trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect()
    }

    /// Get the best trial
    pub fn best_trial(&self) -> Option<&FrozenTrial> {
        self.completed_trials()
            .into_iter()
            .filter(|t| t.value.is_some())
            .max_by(|a, b| {
                let va = a.value.unwrap();
                let vb = b.value.unwrap();
                match self.direction {
                    Direction::Maximize => va.partial_cmp(&vb).unwrap(),
                    Direction::Minimize => vb.partial_cmp(&va).unwrap(),
                }
            })
    }

    /// Number of completed trials
    pub fn n_trials(&self) -> usize {
        self.trials.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_study_maximize_simple() {
        let sampler = TPESampler::new(42).with_n_startup_trials(3);
        let mut study = Study::new(Direction::Maximize, sampler);

        // Optimize a simple function: maximize choice value from [0, 1, 2]
        study.optimize(20, |trial| {
            let x = trial.suggest_categorical("x", 3);
            x as f64
        });

        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 2.0);
        assert_eq!(best.params["x"], 2);
    }

    #[test]
    fn test_study_minimize_simple() {
        let sampler = TPESampler::new(42).with_n_startup_trials(3);
        let mut study = Study::new(Direction::Minimize, sampler);

        study.optimize(20, |trial| {
            let x = trial.suggest_categorical("x", 5);
            x as f64
        });

        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 0.0);
        assert_eq!(best.params["x"], 0);
    }

    #[test]
    fn test_study_add_baseline_trial() {
        let sampler = TPESampler::new(42);
        let mut study = Study::new(Direction::Maximize, sampler);

        // Add a baseline trial
        let mut params = HashMap::new();
        params.insert("instruction".to_string(), 0);
        params.insert("demos".to_string(), 0);

        let baseline = study.create_trial(params, HashMap::new(), 0.75);
        study.add_trial(baseline);

        assert_eq!(study.n_trials(), 1);
        assert_eq!(study.best_trial().unwrap().value.unwrap(), 0.75);
    }

    #[test]
    fn test_study_multi_param() {
        let sampler = TPESampler::new(123).with_n_startup_trials(5);
        let mut study = Study::new(Direction::Maximize, sampler);

        // Two parameters: instruction (3 choices) and demos (4 choices)
        // Best combo is instruction=2, demos=3 (score = instruction + demos)
        study.optimize(30, |trial| {
            let inst = trial.suggest_categorical("instruction", 3);
            let demos = trial.suggest_categorical("demos", 4);
            (inst + demos) as f64
        });

        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 5.0); // 2 + 3
    }

    #[test]
    fn test_tpe_learns_pattern() {
        // After enough trials, TPE should favor the best category
        let sampler = TPESampler::new(42).with_n_startup_trials(5);
        let mut study = Study::new(Direction::Maximize, sampler);

        // Score mapping: choice 0 -> 0.1, 1 -> 0.5, 2 -> 0.9
        let scores = [0.1, 0.5, 0.9];

        study.optimize(40, |trial| {
            let x = trial.suggest_categorical("x", 3);
            scores[x]
        });

        // Count how many trials chose the best option (2) in the last 20 trials
        let completed = study.completed_trials();
        let last_20: Vec<_> = completed.iter().rev().take(20).collect();
        let best_count = last_20.iter().filter(|t| t.params["x"] == 2).count();

        // TPE should favor choice 2 significantly
        assert!(
            best_count >= 8,
            "TPE should learn to favor choice 2, but only chose it {best_count}/20 times"
        );
    }

    #[test]
    fn test_categorical_log_pdf() {
        let values = vec![0, 0, 1, 2];
        let probs = categorical_log_pdf(&values, 3, 1.0);

        // counts: [2, 1, 1], total = 4 + 3*1 = 7
        // probs: [(2+1)/7, (1+1)/7, (1+1)/7] = [3/7, 2/7, 2/7]
        let expected: Vec<f64> = vec![
            (3.0f64 / 7.0).ln(),
            (2.0f64 / 7.0).ln(),
            (2.0f64 / 7.0).ln(),
        ];

        for (i, (got, exp)) in probs.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "Mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_split_trials_maximize() {
        let trials = vec![(0, 1.0), (1, 3.0), (2, 2.0), (0, 4.0)];
        let (below, above) = split_trials(&trials, 2, Direction::Maximize);

        // Maximize: good = highest values. Sorted desc: (0,4), (1,3), (2,2), (0,1)
        // Below (good, top 2): choices [0, 1]
        // Above (bad, rest): choices [2, 0]
        assert_eq!(below, vec![0, 1]);
        assert_eq!(above, vec![2, 0]);
    }

    #[test]
    fn test_split_trials_minimize() {
        let trials = vec![(0, 3.0), (1, 1.0), (2, 2.0)];
        let (below, above) = split_trials(&trials, 1, Direction::Minimize);

        // Minimize: good = lowest values. Sorted asc: (1,1), (2,2), (0,3)
        // Below (good, top 1): [1]
        // Above (bad, rest): [2, 0]
        assert_eq!(below, vec![1]);
        assert_eq!(above, vec![2, 0]);
    }

    #[test]
    fn test_single_choice_returns_zero() {
        let sampler = TPESampler::new(42);
        let mut study = Study::new(Direction::Maximize, sampler);

        study.optimize(5, |trial| {
            let x = trial.suggest_categorical("x", 1);
            assert_eq!(x, 0, "Single choice must always return 0");
            1.0
        });
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let run = |seed: u64| -> Vec<usize> {
            let sampler = TPESampler::new(seed).with_n_startup_trials(3);
            let mut study = Study::new(Direction::Maximize, sampler);
            let mut choices = Vec::new();

            study.optimize(15, |trial| {
                let x = trial.suggest_categorical("x", 4);
                choices.push(x);
                x as f64
            });

            choices
        };

        let run1 = run(42);
        let run2 = run(42);
        assert_eq!(run1, run2, "Same seed must produce identical results");

        let run3 = run(99);
        assert_ne!(run1, run3, "Different seeds should produce different results");
    }

    #[test]
    fn test_study_with_injected_baseline() {
        let sampler = TPESampler::new(42).with_n_startup_trials(3);
        let mut study = Study::new(Direction::Maximize, sampler);

        // Inject a baseline trial showing that choice 2 scores well
        let mut baseline_params = HashMap::new();
        baseline_params.insert("x".to_string(), 2);
        let baseline = study.create_trial(baseline_params, HashMap::new(), 0.9);
        study.add_trial(baseline);

        // Now optimize — the injected baseline should influence TPE
        study.optimize(15, |trial| {
            let x = trial.suggest_categorical("x", 3);
            [0.1, 0.5, 0.9][x]
        });

        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 0.9);
    }

    #[test]
    fn test_ask_tell_api_basic() {
        // Test the ask/tell API (suggest_categorical + record_trial)
        let sampler = TPESampler::new(42).with_n_startup_trials(3);
        let mut study = Study::new(Direction::Maximize, sampler);

        let scores = [0.1, 0.5, 0.9];

        for _ in 0..20 {
            let x = study.suggest_categorical("x", 3);
            let score = scores[x];
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            study.record_trial(params, score);
        }

        assert_eq!(study.n_trials(), 20);
        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 0.9);
        assert_eq!(best.params["x"], 2);
    }

    #[test]
    fn test_ask_tell_api_multi_param() {
        // Test ask/tell with multiple parameters (like MIPROv2 uses)
        let sampler = TPESampler::new(123).with_n_startup_trials(5);
        let mut study = Study::new(Direction::Maximize, sampler);

        for _ in 0..30 {
            let inst = study.suggest_categorical("instruction", 3);
            let demos = study.suggest_categorical("demos", 4);
            let score = (inst + demos) as f64;

            let mut params = HashMap::new();
            params.insert("instruction".to_string(), inst);
            params.insert("demos".to_string(), demos);
            study.record_trial(params, score);
        }

        let best = study.best_trial().unwrap();
        assert_eq!(best.value.unwrap(), 5.0); // 2 + 3
    }

    #[test]
    fn test_ask_tell_learns_like_optimize() {
        // Verify ask/tell API produces same learning behavior as optimize()
        let scores = [0.1, 0.5, 0.9];

        // Run with optimize()
        let sampler1 = TPESampler::new(42).with_n_startup_trials(5);
        let mut study1 = Study::new(Direction::Maximize, sampler1);
        study1.optimize(40, |trial| {
            let x = trial.suggest_categorical("x", 3);
            scores[x]
        });

        // Run with ask/tell
        let sampler2 = TPESampler::new(42).with_n_startup_trials(5);
        let mut study2 = Study::new(Direction::Maximize, sampler2);
        for _ in 0..40 {
            let x = study2.suggest_categorical("x", 3);
            let mut params = HashMap::new();
            params.insert("x".to_string(), x);
            study2.record_trial(params, scores[x]);
        }

        // Both should find the same best trial value
        assert_eq!(
            study1.best_trial().unwrap().value,
            study2.best_trial().unwrap().value
        );
        assert_eq!(study1.n_trials(), study2.n_trials());
    }
}
