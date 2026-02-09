//! AvatarOptimizer: optimizes instructions for avatar/tool-using agents
//! using feedback-based iterative refinement.
//! Python equivalent: dspy/teleprompt/avatar_optimizer.py

use dspy_core::{Example, Metric, Module, Predict, Prediction, Signature};

const DEFAULT_MAX_EXAMPLES: usize = 10;

/// Result of evaluating a single example.
#[derive(Clone)]
pub struct EvalResult {
    pub example: Example,
    pub score: f64,
    pub prediction: Option<Prediction>,
}

/// Configuration for AvatarOptimizer.
pub struct AvatarOptimizerConfig {
    pub metric: Metric,
    pub max_iters: usize,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub max_positive_inputs: usize,
    pub max_negative_inputs: usize,
    pub optimize_for: OptimizeDirection,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizeDirection {
    Max,
    Min,
}

impl AvatarOptimizerConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            max_iters: 10,
            lower_bound: 0.0,
            upper_bound: 1.0,
            max_positive_inputs: DEFAULT_MAX_EXAMPLES,
            max_negative_inputs: DEFAULT_MAX_EXAMPLES,
            optimize_for: OptimizeDirection::Max,
        }
    }
}

/// AvatarOptimizer optimizes tool-using agents by:
/// 1. Running the agent on a trainset
/// 2. Separating results into positive/negative based on metric bounds
/// 3. Using a comparator to identify patterns and generate feedback
/// 4. Using feedback to generate improved instructions
/// 5. Repeating for max_iters
pub struct AvatarOptimizer {
    config: AvatarOptimizerConfig,
    comparator: Predict,
    feedback_instruction: Predict,
}

impl AvatarOptimizer {
    pub fn new(config: AvatarOptimizerConfig) -> Self {
        // Create internal predictors for comparator and feedback systems
        // Python equivalent: Comparator and FeedbackBasedInstruction signatures with detailed docstrings
        let comparator_sig = Signature::from_string(
            "instruction, actions, pos_input_with_metrics, neg_input_with_metrics -> feedback",
        )
        .unwrap()
        .with_instructions(
            "After executing the given actions on user inputs using the given instruction, \
            some inputs have yielded good results, while others have not. \
            I'll provide you the inputs along with their corresponding evaluation metrics:\n\n\
            Task:\n\
            (1) Firstly, identify and contrast the patterns of inputs that have achieved good results with those that have not.\n\
            (2) Then, review the computational logic for any inconsistencies in the previous actions.\n\
            (3) Lastly, specify the modification in tools used that can lead to improved performance on the negative inputs.",
        );
        let comparator = Predict::new(comparator_sig);

        let feedback_sig = Signature::from_string("previous_instruction, feedback -> new_instruction")
            .unwrap()
            .with_instructions(
                "There is a task that needs to be completed for which one can use multiple tools to achieve the desired outcome. \
                A group's performance was evaluated on a dataset of inputs, the inputs that did well are positive inputs, \
                and the inputs that did not do well are negative inputs.\n\n\
                You received feedback on how they can better use the tools to improve your performance on the negative inputs. \
                You have been provided with the previous instruction, that they followed to use tools to complete the task, \
                and the feedback on your performance.\n\n\
                Your task is to incorporate the feedback and generate a detailed instruction for the group to follow to improve \
                their performance on the task.\n\n\
                Make sure that the new instruction talks about how to use the tools effectively and should be no more than \
                3 paragraphs long. The previous instruction contains general guidelines that you must retain in the new instruction.",
            );
        let feedback_instruction = Predict::new(feedback_sig);

        Self {
            config,
            comparator,
            feedback_instruction,
        }
    }

    /// Process a single example through the actor and metric.
    async fn process_example(&self, actor: &dyn Module, example: &Example) -> EvalResult {
        match actor.call(example).await {
            Ok(prediction) => {
                let score = (self.config.metric)(example, &prediction);
                EvalResult {
                    example: example.clone(),
                    score,
                    prediction: Some(prediction),
                }
            }
            Err(_) => EvalResult {
                example: example.clone(),
                score: 0.0,
                prediction: None,
            },
        }
    }

    /// Evaluate actor on dataset, collecting results.
    async fn evaluate_dataset(&self, actor: &dyn Module, dataset: &[Example]) -> Vec<EvalResult> {
        let mut results = Vec::new();
        for example in dataset {
            let result = self.process_example(actor, example).await;
            results.push(result);
        }
        results
    }

    /// Get positive and negative results from evaluation.
    fn get_pos_neg_results<'a>(
        &self,
        results: &'a [EvalResult],
    ) -> dspy_core::Result<(Vec<&'a EvalResult>, Vec<&'a EvalResult>)> {
        let mut pos_inputs = Vec::new();
        let mut neg_inputs = Vec::new();

        for result in results {
            if result.score >= self.config.upper_bound {
                pos_inputs.push(result);
            } else if result.score <= self.config.lower_bound {
                neg_inputs.push(result);
            }
        }

        if pos_inputs.is_empty() {
            return Err(dspy_core::DspyError::OptimizationError(
                "No positive examples found, try lowering the upperBound or providing more training data".to_string(),
            ));
        }
        if neg_inputs.is_empty() {
            return Err(dspy_core::DspyError::OptimizationError(
                "No negative examples found, try raising the lowerBound or providing more training data".to_string(),
            ));
        }

        Ok((pos_inputs, neg_inputs))
    }

    /// Compile: iteratively optimize the actor's instructions.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
    ) -> dspy_core::Result<Box<dyn Module>> {
        let mut best_actor = student.deep_copy();
        let mut best_score = if self.config.optimize_for == OptimizeDirection::Max {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };

        for _iter in 0..self.config.max_iters {
            // Evaluate on trainset
            let results = self.evaluate_dataset(best_actor.as_ref(), trainset).await;
            let avg_score: f64 =
                results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;

            // Separate positive and negative
            let (pos_inputs, neg_inputs) = match self.get_pos_neg_results(&results) {
                Ok(pn) => pn,
                Err(_) => break, // Cannot generate feedback without both pos and neg
            };

            // Sample if too many
            let sampled_pos: Vec<_> = pos_inputs
                .into_iter()
                .take(self.config.max_positive_inputs)
                .collect();
            let sampled_neg: Vec<_> = neg_inputs
                .into_iter()
                .take(self.config.max_negative_inputs)
                .collect();

            // Get current instruction from the first predictor
            let predictors = best_actor.named_predictors();
            let current_instruction = if let Some((_, pred)) = predictors.first() {
                let state = pred.dump_state();
                state
                    .get("signature")
                    .and_then(|s| s.get("instructions"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .to_string()
            } else {
                String::new()
            };

            // Format positive/negative summaries with example inputs (Python passes full EvalResult objects)
            let pos_summary: String = sampled_pos
                .iter()
                .map(|r| {
                    let inputs = r.example.inputs();
                    let inputs_str = inputs
                        .to_map()
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({}, score={:.2})", inputs_str, r.score)
                })
                .collect::<Vec<_>>()
                .join("; ");
            let neg_summary: String = sampled_neg
                .iter()
                .map(|r| {
                    let inputs = r.example.inputs();
                    let inputs_str = inputs
                        .to_map()
                        .iter()
                        .map(|(k, v)| format!("{}={}", k, v))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({}, score={:.2})", inputs_str, r.score)
                })
                .collect::<Vec<_>>()
                .join("; ");

            // Build actions list from predictor names (Python uses actor.get_tools())
            // Since Rust doesn't have a tool system, predictor names serve as action descriptors
            let actions: String = {
                let names: Vec<&str> = best_actor
                    .named_predictors()
                    .iter()
                    .map(|(name, _)| *name)
                    .collect();
                format!("{:?}", names)
            };

            // Generate feedback using comparator
            let feedback_input = Example::new()
                .field("instruction", current_instruction.as_str())
                .field("actions", actions.as_str())
                .field("pos_input_with_metrics", pos_summary.as_str())
                .field("neg_input_with_metrics", neg_summary.as_str());

            let feedback_result = self.comparator.call(&feedback_input).await;
            let feedback = feedback_result
                .as_ref()
                .ok()
                .and_then(|p| p.get_str("feedback").map(String::from))
                .unwrap_or_default();

            // Generate new instruction from feedback
            let instr_input = Example::new()
                .field("previous_instruction", current_instruction.as_str())
                .field("feedback", feedback.as_str());

            let instr_result = self.feedback_instruction.call(&instr_input).await;
            let new_instruction = instr_result
                .as_ref()
                .ok()
                .and_then(|p| p.get_str("new_instruction").map(String::from))
                .unwrap_or_default();

            // Update if improved — apply new instruction to predictors
            let improved = match self.config.optimize_for {
                OptimizeDirection::Max => avg_score > best_score,
                OptimizeDirection::Min => avg_score < best_score,
            };

            if improved && !new_instruction.is_empty() {
                best_score = avg_score;
                // Apply the new instruction to all predictors in the actor
                for (_, pred) in best_actor.named_predictors_mut() {
                    pred.signature = pred.signature.with_instructions(&new_instruction);
                }
            }
        }

        Ok(best_actor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{Example, LMConfig, LMResponse, Message, Predict, Prediction, Signature, LM};
    use std::sync::Arc;

    struct MockLM {
        config: LMConfig,
    }

    impl MockLM {
        fn new() -> Self {
            Self {
                config: LMConfig::new("mock"),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse::new("[[ ## answer ## ]]\n42\n[[ ## completed ## ]]", None)])
        }
        fn model(&self) -> &str {
            "mock"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({"model": "mock"})
        }
    }

    struct SimpleQA {
        predict: Predict,
    }

    impl SimpleQA {
        fn new() -> Self {
            let mut predict = Predict::new(Signature::from_string("question -> answer").unwrap());
            predict.set_lm(Arc::new(MockLM::new()));
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for SimpleQA {
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
            Box::new(SimpleQA::new())
        }
    }

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .field("answer", "42")
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[test]
    fn test_avatar_config_defaults() {
        let metric: Metric = Arc::new(|_, _| 1.0);
        let config = AvatarOptimizerConfig::new(metric);
        assert_eq!(config.max_iters, 10);
        assert_eq!(config.lower_bound, 0.0);
        assert_eq!(config.upper_bound, 1.0);
        assert_eq!(config.max_positive_inputs, DEFAULT_MAX_EXAMPLES);
        assert_eq!(config.max_negative_inputs, DEFAULT_MAX_EXAMPLES);
        assert_eq!(config.optimize_for, OptimizeDirection::Max);
    }

    #[tokio::test]
    async fn test_avatar_compile_basic() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(5);

        // Metric always returns 0.5 — between bounds, so no pos/neg found
        // Avatar should handle this gracefully by breaking out
        let metric: Metric = Arc::new(|_, _| 0.5);
        let avatar = AvatarOptimizer::new(AvatarOptimizerConfig {
            metric,
            max_iters: 2,
            ..AvatarOptimizerConfig::new(Arc::new(|_, _| 0.5))
        });

        let result = avatar.compile(&student, &trainset).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_avatar_compile_with_pos_neg() {
        dspy_core::reset_settings();
        let student = SimpleQA::new();
        let trainset = make_trainset(5);

        // Metric returns 1.0 (>= upper_bound=1) for half, 0.0 (<= lower_bound=0) for rest
        let metric: Metric = Arc::new(|example: &Example, _: &Prediction| {
            let q = example.get_str("question").unwrap_or("");
            if q.contains('0') || q.contains('2') || q.contains('4') {
                1.0
            } else {
                0.0
            }
        });

        let avatar = AvatarOptimizer::new(AvatarOptimizerConfig {
            metric,
            max_iters: 1,
            ..AvatarOptimizerConfig::new(Arc::new(|_, _| 0.0))
        });

        let result = avatar.compile(&student, &trainset).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_eval_result() {
        let example = Example::new().field("q", "test");
        let result = EvalResult {
            example: example.clone(),
            score: 0.75,
            prediction: None,
        };
        assert_eq!(result.score, 0.75);
        assert!(result.prediction.is_none());
    }
}
