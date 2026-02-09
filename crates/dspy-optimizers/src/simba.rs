//! SIMBA — Stochastic Introspective Mini-Batch Ascent.
//!
//! Optimizer that uses mini-batches of training data, samples multiple program
//! trajectories per example, identifies high-variability examples, then applies
//! strategies (append demos or append LLM-generated improvement rules).
//!
//! Python equivalent: dspy/teleprompt/simba.py

use dspy_core::{Example, Metric, Module, Predict, Prediction, SignatureBuilder, LM};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;

/// Optimization strategy for SIMBA
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    AppendDemo,
    AppendRule,
}

/// Configuration for SIMBA
pub struct SIMBAConfig {
    pub metric: Metric,
    pub batch_size: usize,
    pub num_candidates: usize,
    pub max_steps: usize,
    pub max_demos: usize,
    pub prompt_model: Option<Arc<dyn LM>>,
    pub teacher_settings: Option<HashMap<String, String>>,
    pub demo_input_field_maxlen: usize,
    pub temperature_for_sampling: f64,
    pub temperature_for_candidates: f64,
}

impl SIMBAConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            batch_size: 32,
            num_candidates: 6,
            max_steps: 8,
            max_demos: 4,
            prompt_model: None,
            teacher_settings: None,
            demo_input_field_maxlen: 100_000,
            temperature_for_sampling: 0.2,
            temperature_for_candidates: 0.2,
        }
    }
}

/// SIMBA result with metadata
pub struct SIMBAResult {
    pub program: Box<dyn Module>,
    pub score: f64,
    pub candidate_programs: Vec<ScoredProgram>,
    pub trial_logs: HashMap<usize, SIMBATrialLog>,
}

pub struct ScoredProgram {
    pub score: f64,
    pub program: Box<dyn Module>,
}

#[derive(Debug, Clone)]
pub struct SIMBATrialLog {
    pub batch_baseline: f64,
    pub candidate_scores: Vec<f64>,
    pub best_score: f64,
    pub strategy_used: Option<String>,
}

/// A tracked program in SIMBA's pool
struct TrackedProgram {
    idx: usize,
    program: Box<dyn Module>,
    scores: Vec<f64>,
}

impl TrackedProgram {
    fn avg_score(&self) -> f64 {
        if self.scores.is_empty() {
            0.0
        } else {
            self.scores.iter().sum::<f64>() / self.scores.len() as f64
        }
    }
}

/// Bucket: a group of execution results for the same training example
struct Bucket {
    results: Vec<ExecutionResult>,
    max_to_min_gap: f64,
    max_score: f64,
    max_to_avg_gap: f64,
}

/// Result of executing a program on one example
struct ExecutionResult {
    score: f64,
    example: Example,
    prediction: Option<Prediction>,
}

pub struct SIMBA {
    config: SIMBAConfig,
}

impl SIMBA {
    pub fn new(config: SIMBAConfig) -> Self {
        Self { config }
    }

    /// Compile: optimize the student program using SIMBA
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        seed: u64,
    ) -> dspy_core::Result<SIMBAResult> {
        assert!(
            trainset.len() >= self.config.batch_size,
            "Trainset too small: {} < {}",
            trainset.len(),
            self.config.batch_size
        );

        let mut rng = StdRng::seed_from_u64(seed);

        let strategies: Vec<Strategy> = if self.config.max_demos > 0 {
            vec![Strategy::AppendDemo, Strategy::AppendRule]
        } else {
            vec![Strategy::AppendRule]
        };

        // Initialize program pool
        let mut programs: Vec<TrackedProgram> = Vec::new();
        let mut next_idx = 0;

        // Baseline program
        let baseline = student.deep_copy();
        programs.push(TrackedProgram {
            idx: 0,
            program: baseline,
            scores: Vec::new(),
        });
        next_idx += 1;

        let mut winning_programs: Vec<Box<dyn Module>> = vec![student.deep_copy()];
        let mut trial_logs = HashMap::new();

        // Data shuffling
        let mut data_indices: Vec<usize> = (0..trainset.len()).collect();
        data_indices.shuffle(&mut rng);
        let mut instance_idx = 0;

        for batch_idx in 0..self.config.max_steps {
            // Step 1: Get next batch
            if instance_idx + self.config.batch_size > trainset.len() {
                data_indices.shuffle(&mut rng);
                instance_idx = 0;
            }

            let batch_indices = &data_indices[instance_idx..instance_idx + self.config.batch_size];
            let batch: Vec<Example> = batch_indices.iter().map(|&i| trainset[i].clone()).collect();
            instance_idx += self.config.batch_size;

            // Get top programs for sampling
            let top_programs = self.top_k_plus_baseline(&programs, self.config.num_candidates);

            // Step 2: Execute programs on batch — sample trajectories
            let mut all_results = Vec::new();
            for _ in 0..self.config.num_candidates {
                for example in &batch {
                    let prog_idx = self.softmax_sample(
                        &programs,
                        &top_programs,
                        self.config.temperature_for_sampling,
                        &mut rng,
                    );

                    let prog = &programs.iter().find(|p| p.idx == prog_idx).unwrap().program;
                    let result = self.evaluate_single(prog.as_ref(), example).await;
                    all_results.push(result);
                }
            }

            // Step 3: Sort into buckets by variability
            let mut buckets: Vec<Bucket> = Vec::new();
            let all_scores: Vec<f64> = all_results.iter().map(|r| r.score).collect();
            let batch_baseline_score = if all_scores.is_empty() {
                0.0
            } else {
                all_scores.iter().sum::<f64>() / all_scores.len() as f64
            };

            for (ex_idx, _) in batch.iter().enumerate() {
                let mut bucket_results: Vec<&ExecutionResult> = Vec::new();
                for candidate_i in 0..self.config.num_candidates {
                    let result_idx = candidate_i * self.config.batch_size + ex_idx;
                    if result_idx < all_results.len() {
                        bucket_results.push(&all_results[result_idx]);
                    }
                }

                if bucket_results.is_empty() {
                    continue;
                }

                bucket_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                let max_score = bucket_results[0].score;
                let min_score = bucket_results.last().unwrap().score;
                let avg_score: f64 = bucket_results.iter().map(|r| r.score).sum::<f64>()
                    / bucket_results.len() as f64;

                buckets.push(Bucket {
                    results: bucket_results
                        .into_iter()
                        .map(|r| ExecutionResult {
                            score: r.score,
                            example: r.example.clone(),
                            prediction: r.prediction.clone(),
                        })
                        .collect(),
                    max_to_min_gap: max_score - min_score,
                    max_score,
                    max_to_avg_gap: max_score - avg_score,
                });
            }

            // Sort buckets by variability (most variable first)
            buckets.sort_by(|a, b| {
                let key_a = (a.max_to_min_gap, a.max_score, a.max_to_avg_gap);
                let key_b = (b.max_to_min_gap, b.max_score, b.max_to_avg_gap);
                key_b
                    .0
                    .partial_cmp(&key_a.0)
                    .unwrap()
                    .then(key_b.1.partial_cmp(&key_a.1).unwrap())
                    .then(key_b.2.partial_cmp(&key_a.2).unwrap())
            });

            let percentile_10 = percentile(&all_scores, 10.0);
            let percentile_90 = percentile(&all_scores, 90.0);

            // Step 4: Build new candidates by applying strategies
            let mut system_candidates: Vec<Box<dyn Module>> = Vec::new();

            for bucket in &buckets {
                if system_candidates.len() >= self.config.num_candidates + 1 {
                    break;
                }

                // Pick source program
                let src_prog_idx = self.softmax_sample(
                    &programs,
                    &top_programs,
                    self.config.temperature_for_candidates,
                    &mut rng,
                );
                let mut candidate = programs
                    .iter()
                    .find(|p| p.idx == src_prog_idx)
                    .unwrap()
                    .program
                    .deep_copy();

                // Drop some demos
                self.drop_random_demos(&mut *candidate, &mut rng);

                // Pick and apply a strategy
                let strategy = *strategies.choose(&mut rng).unwrap();

                match strategy {
                    Strategy::AppendDemo => {
                        if let Some(best) = bucket.results.first() {
                            if best.score > percentile_10 {
                                self.apply_append_demo(&mut *candidate, best);
                            }
                        }
                    }
                    Strategy::AppendRule => {
                        if let (Some(good), Some(bad)) =
                            (bucket.results.first(), bucket.results.last())
                        {
                            if good.score > percentile_10 && bad.score < percentile_90 {
                                self.apply_append_rule(&mut *candidate, good, bad).await;
                            }
                        }
                    }
                }

                system_candidates.push(candidate);
            }

            // Step 5: Evaluate new candidates on the batch
            let mut candidate_scores: Vec<f64> = Vec::new();
            for candidate in &system_candidates {
                let mut total = 0.0;
                for example in &batch {
                    let result = self.evaluate_single(candidate.as_ref(), example).await;
                    total += result.score;
                }
                candidate_scores.push(total / batch.len() as f64);
            }

            // Step 6: Track best
            if let Some((best_idx, _best_score)) = candidate_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            {
                winning_programs.push(system_candidates[best_idx].deep_copy());
            }

            // Step 7: Register all candidates in global pool
            for (i, candidate) in system_candidates.into_iter().enumerate() {
                next_idx += 1;
                let score = if i < candidate_scores.len() {
                    candidate_scores[i]
                } else {
                    0.0
                };
                programs.push(TrackedProgram {
                    idx: next_idx,
                    program: candidate,
                    scores: vec![score],
                });
            }

            trial_logs.insert(
                batch_idx,
                SIMBATrialLog {
                    batch_baseline: batch_baseline_score,
                    candidate_scores: candidate_scores.clone(),
                    best_score: candidate_scores
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max),
                    strategy_used: None,
                },
            );
        }

        // Final validation on full trainset
        let n_winning = winning_programs.len();
        let n_to_eval = (self.config.num_candidates + 1).min(n_winning);
        let mut eval_indices: Vec<usize> = if n_winning <= 1 {
            vec![0; n_to_eval]
        } else {
            (0..n_to_eval)
                .map(|i| (i * (n_winning - 1)) / (n_to_eval - 1).max(1))
                .collect()
        };
        eval_indices.dedup();

        let eval_programs: Vec<Box<dyn Module>> = eval_indices
            .iter()
            .map(|&i| winning_programs[i].deep_copy())
            .collect();

        let mut final_scores = Vec::new();
        for prog in &eval_programs {
            let mut total = 0.0;
            for example in trainset {
                let result = self.evaluate_single(prog.as_ref(), example).await;
                total += result.score;
            }
            final_scores.push(total / trainset.len() as f64);
        }

        let best_idx = final_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_program = eval_programs
            .into_iter()
            .enumerate()
            .find(|(i, _)| *i == best_idx)
            .map(|(_, p)| p)
            .unwrap();

        let best_score = final_scores[best_idx];

        let candidate_programs: Vec<ScoredProgram> = final_scores
            .into_iter()
            .zip(winning_programs.into_iter())
            .map(|(s, p)| ScoredProgram {
                score: s,
                program: p,
            })
            .collect();

        Ok(SIMBAResult {
            program: best_program,
            score: best_score,
            candidate_programs,
            trial_logs,
        })
    }

    /// Get top-k program indices plus baseline (0)
    fn top_k_plus_baseline(&self, programs: &[TrackedProgram], k: usize) -> Vec<usize> {
        let mut sorted: Vec<&TrackedProgram> = programs.iter().collect();
        sorted.sort_by(|a, b| b.avg_score().partial_cmp(&a.avg_score()).unwrap());

        let mut top: Vec<usize> = sorted.iter().take(k).map(|p| p.idx).collect();
        if !top.contains(&0) && !top.is_empty() {
            *top.last_mut().unwrap() = 0;
        }
        top.dedup();
        top
    }

    /// Softmax-weighted sampling over program indices
    fn softmax_sample(
        &self,
        programs: &[TrackedProgram],
        indices: &[usize],
        temperature: f64,
        rng: &mut StdRng,
    ) -> usize {
        if indices.is_empty() {
            return 0;
        }

        let scores: Vec<f64> = indices
            .iter()
            .map(|&idx| {
                programs
                    .iter()
                    .find(|p| p.idx == idx)
                    .map(|p| p.avg_score())
                    .unwrap_or(0.0)
            })
            .collect();

        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores
            .iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();
        let sum_weights: f64 = weights.iter().sum();

        if sum_weights <= 0.0 {
            return indices[rng.gen_range(0..indices.len())];
        }

        let mut r = rng.gen::<f64>() * sum_weights;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return indices[i];
            }
        }

        *indices.last().unwrap()
    }

    /// Evaluate a single program on one example
    async fn evaluate_single(&self, program: &dyn Module, example: &Example) -> ExecutionResult {
        let inputs = example.inputs();
        match program.call(&inputs).await {
            Ok(prediction) => {
                let score = (self.config.metric)(example, &prediction);
                ExecutionResult {
                    score,
                    example: example.clone(),
                    prediction: Some(prediction),
                }
            }
            Err(_) => ExecutionResult {
                score: 0.0,
                example: example.clone(),
                prediction: None,
            },
        }
    }

    /// Drop some random demos from a program's predictors using Poisson distribution
    fn drop_random_demos(&self, program: &mut dyn Module, rng: &mut StdRng) {
        let max_demos = if self.config.max_demos > 0 {
            self.config.max_demos
        } else {
            3
        };

        let predictors = program.named_predictors_mut();
        for (_name, pred) in predictors {
            let n = pred.demos.len();
            if n >= max_demos {
                // Poisson-distributed dropping, matching Python DSPy
                let lambda = n as f64 / max_demos as f64;
                let n_drops = poisson_sample(lambda, rng).min(n);
                // Remove random indices
                for _ in 0..n_drops {
                    if !pred.demos.is_empty() {
                        let idx = rng.gen_range(0..pred.demos.len());
                        pred.demos.remove(idx);
                    }
                }
            }
        }
    }

    /// Strategy: append a demo from the best execution
    fn apply_append_demo(&self, program: &mut dyn Module, best: &ExecutionResult) {
        // Add the example itself as a demo to each predictor
        let predictors = program.named_predictors_mut();
        for (_name, pred) in predictors {
            let demo = best.example.clone();
            pred.demos.push(demo);
        }
    }

    /// Strategy: append an improvement rule to instructions
    async fn apply_append_rule(
        &self,
        program: &mut dyn Module,
        good: &ExecutionResult,
        bad: &ExecutionResult,
    ) {
        // Generate a rule based on contrasting good vs bad
        let rule = if let Some(prompt_model) = &self.config.prompt_model {
            self.generate_rule_via_llm(prompt_model.clone(), good, bad)
                .await
        } else {
            self.generate_heuristic_rule(good, bad)
        };

        // Append rule to all predictor instructions
        let predictors = program.named_predictors_mut();
        for (_name, pred) in predictors {
            if !rule.is_empty() {
                let current = pred.signature.instructions().to_string();
                let new_instructions = if current.is_empty() {
                    rule.clone()
                } else {
                    format!("{current}\n\nRule: {rule}")
                };
                pred.signature = pred.signature.with_instructions(&new_instructions);
            }
        }
    }

    /// Generate improvement rule using LLM
    async fn generate_rule_via_llm(
        &self,
        prompt_model: Arc<dyn LM>,
        good: &ExecutionResult,
        bad: &ExecutionResult,
    ) -> String {
        let sig = SignatureBuilder::new()
            .instructions(
                "Given a better and worse execution of a program, generate a concise rule \
                 that would help the program produce better outputs in similar situations.",
            )
            .input_with_desc("good_score", "Score of the better execution")
            .input_with_desc("bad_score", "Score of the worse execution")
            .input_with_desc("example_input", "The input that was provided")
            .output_with_desc("improvement_rule", "A concise improvement rule")
            .build();

        let mut generator = Predict::new(sig);
        generator.set_lm(prompt_model);

        let good_score_str = format!("{:.2}", good.score);
        let bad_score_str = format!("{:.2}", bad.score);
        let example_str = format!("{:?}", good.example);
        let input = Example::new()
            .field("good_score", good_score_str.as_str())
            .field("bad_score", bad_score_str.as_str())
            .field("example_input", example_str.as_str())
            .with_inputs(&["good_score", "bad_score", "example_input"]);

        match generator.call(&input).await {
            Ok(prediction) => prediction
                .get_str("improvement_rule")
                .unwrap_or("")
                .to_string(),
            Err(_) => self.generate_heuristic_rule(good, bad),
        }
    }

    /// Generate a heuristic improvement rule
    fn generate_heuristic_rule(&self, good: &ExecutionResult, bad: &ExecutionResult) -> String {
        format!(
            "When scoring varies from {:.2} to {:.2}, ensure outputs are thorough and precise.",
            bad.score, good.score,
        )
    }
}

/// Calculate the p-th percentile of a list of scores
fn percentile(scores: &[f64], p: f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted = scores.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Sample from Poisson distribution using inverse transform method
fn poisson_sample(lambda: f64, rng: &mut StdRng) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0f64;
    loop {
        k += 1;
        p *= rng.gen::<f64>();
        if p <= l {
            return k - 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{LMConfig, LMResponse, Message, Signature};

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
            let is_rule_gen = messages
                .iter()
                .any(|m| m.content.contains("improvement_rule"));
            let text = if is_rule_gen {
                "[[ ## improvement_rule ## ]]\nBe more precise and thorough.".to_string()
            } else {
                format!("[[ ## answer ## ]]\n{}", self.answer)
            };
            Ok(vec![LMResponse::new(text, None)])
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
    async fn test_simba_basic() {
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

        let config = SIMBAConfig {
            metric,
            batch_size: 5,
            num_candidates: 3,
            max_steps: 2,
            max_demos: 2,
            prompt_model: Some(lm),
            ..SIMBAConfig::new(Arc::new(|_, _| 1.0))
        };

        let optimizer = SIMBA::new(config);
        let trainset = make_trainset(10);

        let result = optimizer.compile(&student, &trainset, 42).await.unwrap();
        assert!(result.score >= 0.0);
        assert!(!result.trial_logs.is_empty());
    }

    #[tokio::test]
    async fn test_simba_no_demos() {
        let lm: Arc<dyn LM> = Arc::new(MockLM::new("42"));
        let metric: Metric = Arc::new(|_, _| 0.5);

        let sig = Signature::from_string("question -> answer").unwrap();
        let student = SimpleModule::new(sig).with_lm(lm.clone());

        let config = SIMBAConfig {
            metric,
            batch_size: 3,
            num_candidates: 2,
            max_steps: 2,
            max_demos: 0, // No demos, only rules
            prompt_model: None,
            ..SIMBAConfig::new(Arc::new(|_, _| 0.5))
        };

        let optimizer = SIMBA::new(config);
        let trainset = make_trainset(5);

        let result = optimizer.compile(&student, &trainset, 42).await.unwrap();
        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_percentile() {
        let scores = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(percentile(&scores, 0.0), 10.0);
        assert_eq!(percentile(&scores, 50.0), 30.0);
        assert_eq!(percentile(&scores, 100.0), 50.0);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_heuristic_rule() {
        let simba = SIMBA::new(SIMBAConfig::new(Arc::new(|_, _| 1.0)));

        let good = ExecutionResult {
            score: 0.9,
            example: Example::new(),
            prediction: None,
        };
        let bad = ExecutionResult {
            score: 0.2,
            example: Example::new(),
            prediction: None,
        };

        let rule = simba.generate_heuristic_rule(&good, &bad);
        assert!(rule.contains("0.20"));
        assert!(rule.contains("0.90"));
    }

    #[test]
    fn test_poisson_sample_zero_lambda() {
        let mut rng = StdRng::seed_from_u64(42);
        // lambda = 0 should always return 0
        for _ in 0..100 {
            assert_eq!(poisson_sample(0.0, &mut rng), 0);
        }
    }

    #[test]
    fn test_poisson_sample_distribution() {
        // With lambda = 2.0, mean should be close to 2.0 over many samples
        let mut rng = StdRng::seed_from_u64(42);
        let n = 10_000;
        let sum: usize = (0..n).map(|_| poisson_sample(2.0, &mut rng)).sum();
        let mean = sum as f64 / n as f64;
        // Mean should be close to lambda=2.0 (within 10% for 10k samples)
        assert!(
            (mean - 2.0).abs() < 0.2,
            "Poisson mean should be ~2.0, got {mean}"
        );
    }

    #[test]
    fn test_poisson_sample_negative_lambda() {
        let mut rng = StdRng::seed_from_u64(42);
        // Negative lambda should return 0 (same as zero)
        assert_eq!(poisson_sample(-1.0, &mut rng), 0);
    }
}
