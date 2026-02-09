//! COPRO — Collaborative Prompt Optimization.
//! Iteratively refines instructions for each predictor using an LLM to propose improvements.
//! Python equivalent: dspy/teleprompt/copro_optimizer.py

use dspy_core::{
    Evaluate, EvaluateConfig, Example, LMConfig, Message, Metric, Module, Signature, LM,
};
use std::collections::HashMap;
use std::sync::Arc;

pub struct COPROConfig {
    pub metric: Metric,
    pub breadth: usize,
    pub depth: usize,
    pub init_temperature: f64,
    pub prompt_model: Option<Arc<dyn LM>>,
    pub max_errors: usize,
}

impl COPROConfig {
    pub fn new(metric: Metric) -> Self {
        Self {
            metric,
            breadth: 10,
            depth: 3,
            init_temperature: 1.4,
            prompt_model: None,
            max_errors: 5,
        }
    }
}

pub struct COPRO {
    config: COPROConfig,
}

impl COPRO {
    pub fn new(config: COPROConfig) -> Self {
        if config.breadth <= 1 {
            panic!("COPRO breadth must be greater than 1");
        }
        Self { config }
    }

    /// Compile: for each predictor, iteratively optimize instructions via LLM-proposed candidates.
    pub async fn compile(
        &self,
        student: &dyn Module,
        trainset: &[Example],
    ) -> dspy_core::Result<Box<dyn Module>> {
        let mut module = student.deep_copy();
        let mut module_clone = module.deep_copy();

        // Track evaluated candidates per predictor name
        let mut evaluated_candidates: HashMap<String, Vec<CandidateScore>> = HashMap::new();
        for (name, _) in module.named_predictors() {
            evaluated_candidates.insert(name.to_string(), Vec::new());
        }

        // For each depth iteration
        for d in 0..self.config.depth {
            // Go through predictors
            let predictor_names: Vec<String> = module
                .named_predictors()
                .iter()
                .map(|(name, _)| name.to_string())
                .collect();

            for pred_name in &predictor_names {
                // Get current instruction for this predictor
                let current_instruction = {
                    let preds = module_clone.named_predictors();
                    preds
                        .iter()
                        .find(|(name, _)| *name == pred_name.as_str())
                        .map(|(_, pred)| pred.signature.instructions().to_string())
                        .unwrap_or_default()
                };

                // Get signature fields for context
                let field_summary = {
                    let preds = module_clone.named_predictors();
                    preds
                        .iter()
                        .find(|(name, _)| *name == pred_name.as_str())
                        .map(|(_, pred)| self.format_field_summary(&pred.signature))
                        .unwrap_or_default()
                };

                // Generate candidate instructions
                let candidates = if d == 0 {
                    // First round: generate from basic instruction
                    self.generate_initial_candidates(&current_instruction, &field_summary, trainset)
                        .await?
                } else {
                    // Subsequent rounds: generate given previous attempts + scores
                    let prev_candidates = evaluated_candidates
                        .get(pred_name)
                        .cloned()
                        .unwrap_or_default();
                    self.generate_refined_candidates(&prev_candidates, &field_summary)
                        .await?
                };

                // Evaluate each candidate
                for candidate_instruction in &candidates {
                    // Set instruction on clone
                    for (name, pred) in module_clone.named_predictors_mut() {
                        if name == pred_name.as_str() {
                            pred.signature =
                                pred.signature.with_instructions(candidate_instruction);
                        }
                    }

                    // Evaluate
                    let evaluator = Evaluate::new(
                        trainset.to_vec(),
                        self.config.metric.clone(),
                        EvaluateConfig {
                            max_errors: self.config.max_errors,
                            ..Default::default()
                        },
                    );
                    let result = evaluator.run(module_clone.as_ref()).await?;
                    let score = result.score;

                    // Record candidate
                    if let Some(candidates) = evaluated_candidates.get_mut(pred_name) {
                        candidates.push(CandidateScore {
                            instruction: candidate_instruction.clone(),
                            score,
                        });
                    }
                }

                // Set predictor to best-performing instruction for next round
                if let Some(candidates) = evaluated_candidates.get(pred_name) {
                    if let Some(best) = candidates.iter().max_by(|a, b| {
                        a.score
                            .partial_cmp(&b.score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        for (name, pred) in module_clone.named_predictors_mut() {
                            if name == pred_name.as_str() {
                                pred.signature =
                                    pred.signature.with_instructions(&best.instruction);
                            }
                        }
                    }
                }
            }
        }

        // Apply best instructions to the final module
        for (name, pred) in module.named_predictors_mut() {
            if let Some(candidates) = evaluated_candidates.get(name) {
                if let Some(best) = candidates.iter().max_by(|a, b| {
                    a.score
                        .partial_cmp(&b.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    pred.signature = pred.signature.with_instructions(&best.instruction);
                }
            }
        }

        Ok(module)
    }

    /// Generate initial instruction candidates.
    async fn generate_initial_candidates(
        &self,
        basic_instruction: &str,
        field_summary: &str,
        trainset: &[Example],
    ) -> dspy_core::Result<Vec<String>> {
        if let Some(ref lm) = self.config.prompt_model {
            self.generate_with_lm(lm.as_ref(), basic_instruction, field_summary, trainset)
                .await
        } else {
            Ok(self.generate_heuristic_candidates(basic_instruction, field_summary))
        }
    }

    /// Generate refined candidates based on previous attempts + scores.
    async fn generate_refined_candidates(
        &self,
        prev_candidates: &[CandidateScore],
        field_summary: &str,
    ) -> dspy_core::Result<Vec<String>> {
        if let Some(ref lm) = self.config.prompt_model {
            self.generate_refined_with_lm(lm.as_ref(), prev_candidates, field_summary)
                .await
        } else {
            // Without LM, just return the best previous instruction with minor variations
            let best = prev_candidates.iter().max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let base = best.map(|c| c.instruction.as_str()).unwrap_or("");
            Ok(self.generate_heuristic_candidates(base, field_summary))
        }
    }

    /// Use LM to generate instruction candidates.
    async fn generate_with_lm(
        &self,
        lm: &dyn LM,
        basic_instruction: &str,
        field_summary: &str,
        trainset: &[Example],
    ) -> dspy_core::Result<Vec<String>> {
        let example_summary: String = trainset
            .iter()
            .take(3)
            .map(|ex| {
                ex.keys()
                    .map(|k| format!("{k}: {:?}", ex.get(k).unwrap()))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are an instruction optimizer for large language models. \
            I will give you a signature of fields and the current instruction. \
            Generate an improved instruction.\n\n\
            Task fields:\n{field_summary}\n\n\
            Example data:\n{example_summary}\n\n\
            Current instruction: \"{basic_instruction}\"\n\n\
            Generate a single improved instruction. Output only the instruction text."
        );

        let mut candidates = Vec::new();
        let config = LMConfig {
            model: lm.model().to_string(),
            temperature: Some(self.config.init_temperature),
            max_tokens: None,
            top_p: None,
            n: None,
        };

        for _ in 0..self.config.breadth {
            match lm.call(&[Message::user(&prompt)], &config).await {
                Ok(responses) => {
                    for resp in responses {
                        let text = resp.text.trim().trim_matches('"').to_string();
                        if !text.is_empty() {
                            candidates.push(text);
                        }
                    }
                }
                Err(_) => continue,
            }
        }

        // Always include the original instruction as a candidate
        candidates.push(basic_instruction.to_string());
        Ok(candidates)
    }

    /// Use LM to generate refined candidates from previous attempts.
    async fn generate_refined_with_lm(
        &self,
        lm: &dyn LM,
        prev_candidates: &[CandidateScore],
        field_summary: &str,
    ) -> dspy_core::Result<Vec<String>> {
        // Build attempts summary, sorted by score ascending
        let mut sorted = prev_candidates.to_vec();
        sorted.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let attempts: String = sorted
            .iter()
            .enumerate()
            .map(|(i, c)| {
                format!(
                    "Instruction #{}: {}\nScore #{}: {}",
                    i + 1,
                    c.instruction,
                    i + 1,
                    c.score
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        let prompt = format!(
            "You are an instruction optimizer for large language models. \
            Here are previous attempts with scores (higher is better):\n\n\
            {attempts}\n\n\
            Task fields:\n{field_summary}\n\n\
            Generate a new, improved instruction. Output only the instruction text."
        );

        let config = LMConfig {
            model: lm.model().to_string(),
            temperature: Some(self.config.init_temperature),
            max_tokens: None,
            top_p: None,
            n: None,
        };

        let mut candidates = Vec::new();
        for _ in 0..self.config.breadth {
            match lm.call(&[Message::user(&prompt)], &config).await {
                Ok(responses) => {
                    for resp in responses {
                        let text = resp.text.trim().trim_matches('"').to_string();
                        if !text.is_empty() {
                            candidates.push(text);
                        }
                    }
                }
                Err(_) => continue,
            }
        }
        Ok(candidates)
    }

    /// Fallback: generate heuristic instruction candidates without an LM.
    fn generate_heuristic_candidates(&self, current: &str, field_summary: &str) -> Vec<String> {
        let mut candidates = Vec::new();

        // Parse input/output field names from the summary
        let fields: Vec<&str> = field_summary.lines().collect();
        let inputs: Vec<&str> = fields
            .iter()
            .filter(|l| l.contains("input:"))
            .map(|l| l.trim())
            .collect();
        let outputs: Vec<&str> = fields
            .iter()
            .filter(|l| l.contains("output:"))
            .map(|l| l.trim())
            .collect();

        let input_names = if inputs.is_empty() {
            "the input".to_string()
        } else {
            inputs.join(", ")
        };
        let output_names = if outputs.is_empty() {
            "the output".to_string()
        } else {
            outputs.join(", ")
        };

        candidates.push(format!("Given {input_names}, produce {output_names}."));
        candidates.push(format!(
            "Analyze the provided input carefully and generate {output_names}."
        ));
        candidates.push(format!(
            "Process the input and provide accurate {output_names}. Think carefully."
        ));
        if !current.is_empty() {
            candidates.push(format!("{current} Be more precise and detailed."));
        }
        candidates.push(current.to_string());

        candidates.truncate(self.config.breadth);
        candidates
    }

    fn format_field_summary(&self, signature: &Signature) -> String {
        let mut lines = Vec::new();
        for (name, field) in signature.fields() {
            let kind = match field.field_type {
                dspy_core::FieldType::Input => "input",
                dspy_core::FieldType::Output => "output",
            };
            let desc = field
                .description
                .as_deref()
                .map(|d| format!(" ({d})"))
                .unwrap_or_default();
            lines.push(format!("{kind}: {name}{desc}"));
        }
        lines.join("\n")
    }
}

#[derive(Debug, Clone)]
struct CandidateScore {
    instruction: String,
    score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{Example, LMResponse, Predict, Prediction, Signature};

    struct FixedLM {
        answer: String,
        config: LMConfig,
    }

    impl FixedLM {
        fn new(answer: &str) -> Self {
            Self {
                answer: answer.to_string(),
                config: LMConfig::new("fixed"),
            }
        }
    }

    #[async_trait]
    impl LM for FixedLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse::new(format!("[[ ## answer ## ]]\n{}", self.answer), None)])
        }
        fn model(&self) -> &str {
            "fixed"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    // A simple LM that returns instruction candidates when called as prompt model
    struct InstructionLM {
        config: LMConfig,
    }

    impl InstructionLM {
        fn new() -> Self {
            Self {
                config: LMConfig::new("instruction-gen"),
            }
        }
    }

    #[async_trait]
    impl LM for InstructionLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse::new("Answer the question carefully and precisely.", None)])
        }
        fn model(&self) -> &str {
            "instruction-gen"
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
    impl dspy_core::Module for TestModule {
        async fn forward(&self, args: &Example) -> dspy_core::Result<Prediction> {
            self.predict.forward(args).await
        }
        fn named_predictors(&self) -> Vec<(&str, &Predict)> {
            vec![("predict", &self.predict)]
        }
        fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
            vec![("predict", &mut self.predict)]
        }
        fn deep_copy(&self) -> Box<dyn dspy_core::Module> {
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
    async fn test_copro_heuristic_candidates() {
        dspy_core::reset_settings();
        let lm = Arc::new(FixedLM::new("A0"));
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

        let trainset = make_trainset(3);
        let optimizer = COPRO::new(COPROConfig {
            metric,
            breadth: 3,
            depth: 1,
            init_temperature: 1.0,
            prompt_model: None, // Heuristic mode
            max_errors: 5,
        });

        let compiled = optimizer.compile(&student, &trainset).await.unwrap();
        // Should have an instruction (possibly optimized)
        let preds = compiled.named_predictors();
        assert_eq!(preds.len(), 1);
    }

    #[tokio::test]
    async fn test_copro_with_prompt_model() {
        dspy_core::reset_settings();
        let task_lm = Arc::new(FixedLM::new("A0"));
        let student = TestModule::new(task_lm);

        let prompt_lm: Arc<dyn LM> = Arc::new(InstructionLM::new());

        let metric: Metric = Arc::new(|example, prediction| {
            let expected = example.get_str("answer").unwrap_or("");
            let got = prediction.get_str("answer").unwrap_or("");
            if expected == got {
                1.0
            } else {
                0.0
            }
        });

        let trainset = make_trainset(3);
        let optimizer = COPRO::new(COPROConfig {
            metric,
            breadth: 2,
            depth: 2,
            init_temperature: 1.0,
            prompt_model: Some(prompt_lm),
            max_errors: 5,
        });

        let compiled = optimizer.compile(&student, &trainset).await.unwrap();
        let preds = compiled.named_predictors();
        assert_eq!(preds.len(), 1);
        // Instructions should be set
        assert!(!preds[0].1.signature.instructions().is_empty());
    }

    #[tokio::test]
    async fn test_copro_does_not_modify_original() {
        dspy_core::reset_settings();
        let lm = Arc::new(FixedLM::new("test"));
        let student = TestModule::new(lm);

        let metric: Metric = Arc::new(|_, _| 1.0);
        let trainset = make_trainset(2);

        let original_instructions = student.named_predictors()[0]
            .1
            .signature
            .instructions()
            .to_string();

        let optimizer = COPRO::new(COPROConfig {
            metric,
            breadth: 2,
            depth: 1,
            init_temperature: 1.0,
            prompt_model: None,
            max_errors: 5,
        });

        let _compiled = optimizer.compile(&student, &trainset).await.unwrap();
        // Original should be unmodified
        assert_eq!(
            student.named_predictors()[0].1.signature.instructions(),
            original_instructions
        );
    }
}
