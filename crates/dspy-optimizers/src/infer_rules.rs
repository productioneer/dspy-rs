//! InferRules: induces natural language rules from training examples
//! and injects them into predictor instructions.
//!
//! Wraps BootstrapFewShot — first bootstraps demos, then generates
//! candidate programs with rule-augmented instructions and picks the best.
//!
//! Python equivalent: dspy/teleprompt/infer_rules.py

use async_trait::async_trait;
use dspy_core::{
    ChainOfThought, DspyError, Evaluate, EvaluateConfig, Example, Metric, Module, Predict,
    Prediction, Result, Signature,
};
use std::sync::Arc;

use crate::bootstrap_few_shot::{BootstrapFewShot, BootstrapFewShotConfig};

pub struct InferRulesConfig {
    pub metric: Metric,
    pub num_candidates: usize,
    pub num_rules: usize,
    pub num_threads: usize,
    pub max_bootstrapped_demos: usize,
    pub max_labeled_demos: usize,
    pub max_rounds: usize,
    pub max_errors: usize,
}

impl InferRulesConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            num_candidates: 10,
            num_rules: 10,
            num_threads: 2,
            max_bootstrapped_demos: 4,
            max_labeled_demos: 16,
            max_rounds: 1,
            max_errors: 5,
        }
    }
}

pub struct InferRules {
    config: InferRulesConfig,
}

impl InferRules {
    pub fn new(config: InferRulesConfig) -> Self {
        Self { config }
    }

    /// Compile: bootstrap demos, then iterate candidates with rule-augmented instructions.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
        teacher: Option<&dyn Module>,
        valset: Option<&[Example]>,
    ) -> Result<Box<dyn Module>> {
        // Split trainset if no valset provided
        let (actual_trainset, actual_valset): (Vec<Example>, Vec<Example>) = match valset {
            Some(vs) => (trainset.to_vec(), vs.to_vec()),
            None => {
                let train_size = trainset.len() / 2;
                (
                    trainset[..train_size].to_vec(),
                    trainset[train_size..].to_vec(),
                )
            }
        };

        // Phase 1: Bootstrap demos on trainset
        let bootstrap_config = BootstrapFewShotConfig {
            metric: self.config.metric.clone(),
            metric_threshold: None,
            max_bootstrapped_demos: self.config.max_bootstrapped_demos,
            max_labeled_demos: self.config.max_labeled_demos,
            max_rounds: self.config.max_rounds,
            max_errors: self.config.max_errors,
        };
        let bootstrap = BootstrapFewShot::new(bootstrap_config);
        let bootstrapped = bootstrap
            .compile(student, &actual_trainset, teacher)
            .await?;

        // Save original program state
        let original_program = bootstrapped;
        let original_instructions: Vec<String> = original_program
            .named_predictors()
            .iter()
            .map(|(_, p)| p.signature.instructions().to_string())
            .collect();

        let rules_program = RulesInductionProgram::new(self.config.num_rules);

        let mut best_score = f64::NEG_INFINITY;
        let mut best_program: Option<Box<dyn Module>> = None;

        for _candidate_idx in 0..self.config.num_candidates {
            let mut candidate_program = original_program.deep_copy();
            let candidate_predictors = candidate_program.named_predictors_mut();

            // Reset instructions to original
            for (i, (_, pred)) in candidate_predictors.into_iter().enumerate() {
                if i < original_instructions.len() {
                    pred.signature = pred.signature.with_instructions(&original_instructions[i]);
                }
            }

            // Induce rules for each predictor and update instructions
            let predictor_count = original_program.named_predictors().len();
            for i in 0..predictor_count {
                let preds = candidate_program.named_predictors();
                let sig = preds[i].1.signature.clone();
                let instr = original_instructions[i].clone();

                let rules = self
                    .induce_natural_language_rules(&sig, &actual_trainset, &rules_program)
                    .await?;

                let preds_mut = candidate_program.named_predictors_mut();
                if i < preds_mut.len() {
                    let pred = &mut preds_mut.into_iter().nth(i).unwrap().1;
                    // Reset instructions before appending rules
                    pred.signature = pred.signature.with_instructions(&instr);
                    let new_instr = format!(
                        "{}\n\nPlease adhere to the following rules when making your prediction:\n{}",
                        pred.signature.instructions(),
                        rules,
                    );
                    pred.signature = pred.signature.with_instructions(&new_instr);
                }
            }

            // Evaluate candidate
            let score = self
                .evaluate_program(candidate_program.as_ref(), &actual_valset)
                .await?;

            if score > best_score {
                best_score = score;
                best_program = Some(candidate_program);
            }
        }

        Ok(best_program.unwrap_or_else(|| original_program.deep_copy()))
    }

    /// Induce natural language rules from examples using the RulesInductionProgram.
    async fn induce_natural_language_rules(
        &self,
        signature: &Signature,
        trainset: &[Example],
        rules_program: &RulesInductionProgram,
    ) -> Result<String> {
        let mut demos = self.get_predictor_demos(trainset, signature);

        loop {
            let examples_text = self.format_examples(&demos, signature);
            let input = Example::new().field("examples_text", examples_text.as_str());

            match rules_program.call(&input).await {
                Ok(prediction) => {
                    let rules = prediction
                        .get_str("natural_language_rules")
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    return Ok(rules);
                }
                Err(_e) => {
                    if demos.len() > 1 {
                        demos.pop();
                    } else {
                        return Err(DspyError::OptimizationError(
                            "Failed to generate natural language rules since a single example couldn't fit in the model's context window.".to_string(),
                        ));
                    }
                }
            }
        }
    }

    /// Format training examples into text for rule induction.
    fn format_examples(&self, demos: &[Example], signature: &Signature) -> String {
        let input_fields: Vec<String> = signature
            .input_fields()
            .map(|(name, _)| name.clone())
            .collect();
        let output_fields: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| name.clone())
            .collect();

        let mut text = String::new();
        for demo in demos {
            let input_text: Vec<String> = input_fields
                .iter()
                .filter_map(|k| demo.get_str(k).map(|v| format!("{}: {}", k, v)))
                .collect();
            let output_text: Vec<String> = output_fields
                .iter()
                .filter_map(|k| demo.get_str(k).map(|v| format!("{}: {}", k, v)))
                .collect();

            text.push_str(&format!(
                "Input Fields:\n{}\n\n=========\nOutput Fields:\n{}\n\n",
                input_text.join("\n"),
                output_text.join("\n"),
            ));
        }

        text
    }

    /// Get demos from trainset filtered to predictor's signature fields.
    fn get_predictor_demos(&self, trainset: &[Example], signature: &Signature) -> Vec<Example> {
        let input_fields: Vec<String> = signature
            .input_fields()
            .map(|(name, _)| name.clone())
            .collect();
        let output_fields: Vec<String> = signature
            .output_fields()
            .map(|(name, _)| name.clone())
            .collect();

        trainset
            .iter()
            .map(|example| {
                let mut filtered = Example::new();
                for key in input_fields.iter().chain(output_fields.iter()) {
                    if let Some(val) = example.get(key) {
                        filtered = filtered.field(key, val.to_string().trim_matches('"'));
                    }
                }
                filtered
            })
            .collect()
    }

    /// Evaluate a program on a dataset using the Evaluate class.
    async fn evaluate_program(&self, program: &dyn Module, dataset: &[Example]) -> Result<f64> {
        let evaluate = Evaluate::new(
            dataset.to_vec(),
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

/// RulesInductionProgram: a Module that uses ChainOfThought to extract
/// natural language rules from formatted examples.
struct RulesInductionProgram {
    rules_induction: ChainOfThought,
}

impl RulesInductionProgram {
    fn new(num_rules: usize) -> Self {
        let sig = Signature::from_string("examples_text -> natural_language_rules").unwrap();
        let sig = sig.with_instructions(&format!(
            "Given a set of examples, extract a list of {} concise and non-redundant natural language \
             rules that provide clear guidance for performing the task. All rules should be actionable \
             for a well-specified scope of examples of this general kind of task.",
            num_rules,
        ));

        Self {
            rules_induction: ChainOfThought::new(sig),
        }
    }
}

#[async_trait]
impl Module for RulesInductionProgram {
    fn module_type_name(&self) -> &str {
        "RulesInductionProgram"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        self.rules_induction.call(args).await
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        self.rules_induction.named_predictors()
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        self.rules_induction.named_predictors_mut()
    }

    fn set_lm(&mut self, lm: Arc<dyn dspy_core::LM>) {
        self.rules_induction.set_lm(lm);
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(RulesInductionProgram {
            rules_induction: ChainOfThought::new(self.rules_induction.predict().signature.clone()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_core::{LMConfig, LMResponse, Message, LM};
    use std::sync::Mutex;

    /// Mock LM that returns fixed answers or rules based on the system message.
    struct MockLM {
        answer: String,
        rules: Option<String>,
        config: LMConfig,
        call_count: Mutex<usize>,
    }

    impl MockLM {
        fn new(answer: &str, rules: Option<&str>) -> Self {
            Self {
                answer: answer.to_string(),
                rules: rules.map(|r| r.to_string()),
                config: LMConfig::new("mock"),
                call_count: Mutex::new(0),
            }
        }
    }

    #[async_trait]
    impl LM for MockLM {
        async fn call(&self, messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            *self.call_count.lock().unwrap() += 1;

            let system_msg = messages.iter().find(|m| m.role == "system");
            let is_rule_induction = system_msg
                .map(|m| {
                    m.content.contains("natural language rules")
                        || m.content.contains("natural_language_rules")
                })
                .unwrap_or(false);

            let text = if is_rule_induction {
                if let Some(rules) = &self.rules {
                    format!(
                        "[[ ## reasoning ## ]]\nAnalyzing examples.\n\n[[ ## natural_language_rules ## ]]\n{}\n\n[[ ## completed ## ]]",
                        rules,
                    )
                } else {
                    format!(
                        "[[ ## answer ## ]]\n{}\n\n[[ ## completed ## ]]",
                        self.answer,
                    )
                }
            } else {
                format!(
                    "[[ ## answer ## ]]\n{}\n\n[[ ## completed ## ]]",
                    self.answer,
                )
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

    struct TestModule {
        predict: Predict,
    }

    impl TestModule {
        fn new(lm: Arc<dyn LM>) -> Self {
            let mut predict = Predict::new(Signature::from_string("question -> answer").unwrap());
            predict.set_lm(lm);
            Self { predict }
        }
    }

    #[async_trait]
    impl Module for TestModule {
        async fn forward(&self, args: &Example) -> Result<Prediction> {
            self.predict.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }
        fn deep_copy(&self) -> Box<dyn Module> {
            Box::new(TestModule {
                predict: self.predict.clone(),
            })
        }
    }

    fn make_trainset(n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                Example::new()
                    .field("question", format!("Q{i}"))
                    .field("answer", format!("A{i}"))
                    .with_inputs(&["question"])
            })
            .collect()
    }

    #[tokio::test]
    async fn test_infer_rules_basic() {
        dspy_core::reset_settings();
        let lm = Arc::new(MockLM::new(
            "A0",
            Some("Rule 1: Be concise\nRule 2: Be accurate"),
        ));
        dspy_core::configure(dspy_core::Settings {
            lm: Some(lm.clone()),
            ..Default::default()
        });

        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let config = InferRulesConfig {
            metric,
            num_candidates: 2,
            num_rules: 3,
            max_bootstrapped_demos: 1,
            max_labeled_demos: 1,
            ..InferRulesConfig::new(Arc::new(|_, _| 0.0))
        };
        let optimizer = InferRules::new(config);
        let trainset = make_trainset(4);
        let compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }

    #[tokio::test]
    async fn test_infer_rules_with_valset() {
        dspy_core::reset_settings();
        let lm = Arc::new(MockLM::new("A0", Some("Rule 1: Answer correctly")));
        dspy_core::configure(dspy_core::Settings {
            lm: Some(lm.clone()),
            ..Default::default()
        });

        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|_, _| 1.0);

        let config = InferRulesConfig {
            metric,
            num_candidates: 1,
            num_rules: 2,
            max_bootstrapped_demos: 1,
            max_labeled_demos: 0,
            ..InferRulesConfig::new(Arc::new(|_, _| 0.0))
        };

        let optimizer = InferRules::new(config);
        let trainset = make_trainset(3);
        let valset = make_trainset(2);
        let compiled = optimizer
            .compile(&student, &trainset, None, Some(&valset))
            .await
            .unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }

    #[tokio::test]
    async fn test_infer_rules_instructions_augmented() {
        dspy_core::reset_settings();
        let rules = "1. Always explain your reasoning\n2. Be specific";
        let lm = Arc::new(MockLM::new("test", Some(rules)));
        dspy_core::configure(dspy_core::Settings {
            lm: Some(lm.clone()),
            ..Default::default()
        });

        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|_, _| 1.0);

        let config = InferRulesConfig {
            metric,
            num_candidates: 1,
            num_rules: 2,
            max_bootstrapped_demos: 0,
            max_labeled_demos: 0,
            ..InferRulesConfig::new(Arc::new(|_, _| 0.0))
        };

        let optimizer = InferRules::new(config);
        let trainset = make_trainset(4);
        let compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();

        let preds = compiled.named_predictors();
        let instructions = preds[0].1.signature.instructions().to_string();
        assert!(instructions.contains("Please adhere to the following rules"));
        assert!(instructions.contains("Always explain your reasoning"));
    }

    #[tokio::test]
    async fn test_infer_rules_splits_trainset() {
        dspy_core::reset_settings();
        let lm = Arc::new(MockLM::new("test", Some("Rule 1: test")));
        dspy_core::configure(dspy_core::Settings {
            lm: Some(lm.clone()),
            ..Default::default()
        });

        let student = TestModule::new(lm);
        let metric: Metric = Arc::new(|_, _| 1.0);

        let config = InferRulesConfig {
            metric,
            num_candidates: 1,
            num_rules: 1,
            max_bootstrapped_demos: 0,
            max_labeled_demos: 0,
            ..InferRulesConfig::new(Arc::new(|_, _| 0.0))
        };

        let optimizer = InferRules::new(config);
        let trainset = make_trainset(6);
        let compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();
        assert!(!compiled.named_predictors().is_empty());
    }

    #[tokio::test]
    async fn test_infer_rules_does_not_modify_original() {
        dspy_core::reset_settings();
        let lm = Arc::new(MockLM::new("test", Some("Rule 1: test")));
        dspy_core::configure(dspy_core::Settings {
            lm: Some(lm.clone()),
            ..Default::default()
        });

        let student = TestModule::new(lm);
        let original_instructions = student.named_predictors()[0]
            .1
            .signature
            .instructions()
            .to_string();

        let metric: Metric = Arc::new(|_, _| 1.0);
        let config = InferRulesConfig {
            metric,
            num_candidates: 1,
            num_rules: 1,
            max_bootstrapped_demos: 0,
            max_labeled_demos: 0,
            ..InferRulesConfig::new(Arc::new(|_, _| 0.0))
        };

        let optimizer = InferRules::new(config);
        let trainset = make_trainset(4);
        let _compiled = optimizer
            .compile(&student, &trainset, None, None)
            .await
            .unwrap();

        // Original should be unmodified
        assert_eq!(
            student.named_predictors()[0].1.signature.instructions(),
            original_instructions,
        );
    }
}
