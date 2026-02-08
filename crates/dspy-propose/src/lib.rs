//! Proposer system — LLM-driven instruction generation for MIPROv2.
//!
//! Implements GroundedProposer: generates candidate instructions for each predictor
//! in a program by using an LLM to analyze the program structure, data, and demo
//! examples, then propose improved instructions.

use dspy_core::{Example, Module, Predict, Signature, SignatureBuilder, LM};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Arc;

/// Tips for instruction generation — same as Python DSPy
const TIPS: &[(&str, &str)] = &[
    ("none", ""),
    ("creative", "Don't be afraid to be creative when creating the new instruction!"),
    ("simple", "Keep the instruction clear and concise."),
    ("description", "Make sure your instruction is very informative and descriptive."),
    (
        "high_stakes",
        "The instruction should include a high stakes scenario in which the LM must solve the task!",
    ),
    (
        "persona",
        "Include a persona that is relevant to the task in the instruction (ie. \"You are a ...\")",
    ),
];

/// Configuration for the GroundedProposer
pub struct ProposerConfig {
    pub program_aware: bool,
    pub use_dataset_summary: bool,
    pub use_task_demos: bool,
    pub num_demos_in_context: usize,
    pub use_instruct_history: bool,
    pub use_tip: bool,
    pub set_tip_randomly: bool,
    pub set_history_randomly: bool,
    pub view_data_batch_size: usize,
    pub init_temperature: f64,
}

impl Default for ProposerConfig {
    fn default() -> Self {
        Self {
            program_aware: true,
            use_dataset_summary: true,
            use_task_demos: true,
            num_demos_in_context: 3,
            use_instruct_history: false,
            use_tip: true,
            set_tip_randomly: true,
            set_history_randomly: false,
            view_data_batch_size: 10,
            init_temperature: 1.0,
        }
    }
}

/// GroundedProposer — generates candidate instructions using LLM meta-prompting
pub struct GroundedProposer {
    config: ProposerConfig,
    prompt_model: Arc<dyn LM>,
    rng: StdRng,
    /// Cached program description (from LLM)
    program_description: Option<String>,
    /// Cached data summary
    data_summary: Option<String>,
}

impl GroundedProposer {
    pub fn new(prompt_model: Arc<dyn LM>, config: ProposerConfig, seed: u64) -> Self {
        Self {
            config,
            prompt_model,
            rng: StdRng::seed_from_u64(seed),
            program_description: None,
            data_summary: None,
        }
    }

    /// Create a dataset summary from trainset examples
    pub fn create_data_summary(&self, trainset: &[Example]) -> String {
        let batch_size = self.config.view_data_batch_size.min(trainset.len());
        if batch_size == 0 {
            return "No training data available.".to_string();
        }

        let mut lines = vec!["Dataset summary:".to_string()];
        lines.push(format!("Total examples: {}", trainset.len()));

        // Show field names from first example
        if let Some(first) = trainset.first() {
            let field_names: Vec<String> = first.keys().cloned().collect();
            lines.push(format!("Fields: {}", field_names.join(", ")));
        }

        // Show a few example summaries
        for (i, example) in trainset.iter().take(batch_size).enumerate() {
            let mut parts = Vec::new();
            for key in example.keys() {
                if let Some(val) = example.get_str(&key) {
                    let truncated = if val.len() > 100 {
                        format!("{}...", &val[..100])
                    } else {
                        val.to_string()
                    };
                    parts.push(format!("{key}: {truncated}"));
                }
            }
            lines.push(format!("Example {}: {}", i + 1, parts.join(" | ")));
        }

        lines.join("\n")
    }

    /// Describe the program's predictors for context
    fn describe_program_predictors(&self, program: &dyn Module) -> String {
        let predictors = program.named_predictors();
        if predictors.is_empty() {
            return "Program has no predictors.".to_string();
        }

        let mut lines = Vec::new();
        for (name, pred) in &predictors {
            let sig = &pred.signature;
            lines.push(format!("Module '{}': {}", name, sig.to_shorthand()));
            if !sig.instructions().is_empty() {
                lines.push(format!("  Current instruction: {}", sig.instructions()));
            }
        }
        lines.join("\n")
    }

    /// Build a task demos string from demo candidates for a specific predictor
    fn build_task_demos_string(
        &self,
        demo_candidates: &[Vec<Vec<Example>>],
        pred_i: usize,
        demo_set_i: usize,
        program: &dyn Module,
    ) -> String {
        let predictors = program.named_predictors();
        if pred_i >= demo_candidates.len() || pred_i >= predictors.len() {
            return "No task demos provided.".to_string();
        }

        let pred_demos = &demo_candidates[pred_i];
        if demo_set_i >= pred_demos.len() || pred_demos[demo_set_i].is_empty() {
            return "No task demos provided.".to_string();
        }

        let sig = &predictors[pred_i].1.signature;
        let demos = &pred_demos[demo_set_i];
        let num = demos.len().min(self.config.num_demos_in_context);

        let mut example_strings = Vec::new();
        for demo in demos.iter().take(num) {
            let mut parts = Vec::new();
            for (name, _field) in sig.fields() {
                if let Some(val) = demo.get_str(name) {
                    parts.push(format!("{name}: {val}"));
                }
            }
            if !parts.is_empty() {
                example_strings.push(parts.join("\n"));
            }
        }

        if example_strings.is_empty() {
            "No task demos provided.".to_string()
        } else {
            example_strings.join("\n---\n")
        }
    }

    /// Generate instruction candidates for all predictors in the program
    pub async fn propose_instructions_for_program(
        &mut self,
        program: &dyn Module,
        trainset: &[Example],
        demo_candidates: Option<&[Vec<Vec<Example>>]>,
        n_candidates: usize,
    ) -> HashMap<usize, Vec<String>> {
        let mut proposed_instructions: HashMap<usize, Vec<String>> = HashMap::new();

        // Prepare context
        let data_summary = if self.config.use_dataset_summary {
            self.create_data_summary(trainset)
        } else {
            String::new()
        };
        self.data_summary = Some(data_summary.clone());

        let program_desc = if self.config.program_aware {
            self.describe_program_predictors(program)
        } else {
            String::new()
        };
        self.program_description = Some(program_desc.clone());

        let predictors = program.named_predictors();
        let use_task_demos = self.config.use_task_demos && demo_candidates.is_some();

        let num_demo_sets = if let Some(dc) = demo_candidates {
            if !dc.is_empty() && !dc[0].is_empty() {
                dc[0].len()
            } else {
                n_candidates
            }
        } else {
            n_candidates
        };

        for (pred_i, (_name, pred)) in predictors.iter().enumerate() {
            let mut instructions = Vec::new();
            let basic_instruction = pred.signature.instructions().to_string();

            let n_to_generate = n_candidates.min(num_demo_sets);

            for demo_set_i in 0..n_to_generate {
                let task_demos = if use_task_demos {
                    if let Some(dc) = demo_candidates {
                        self.build_task_demos_string(dc, pred_i, demo_set_i, program)
                    } else {
                        "No task demos provided.".to_string()
                    }
                } else {
                    "No task demos provided.".to_string()
                };

                // Select a tip
                let tip = if self.config.set_tip_randomly {
                    let idx = self.rng.gen_range(0..TIPS.len());
                    TIPS[idx].1.to_string()
                } else if self.config.use_tip {
                    TIPS[1].1.to_string() // "creative" by default
                } else {
                    String::new()
                };

                // Generate instruction via LLM
                let instruction = self
                    .generate_instruction(
                        &basic_instruction,
                        &task_demos,
                        &data_summary,
                        &program_desc,
                        &tip,
                        pred_i,
                        &pred.signature,
                    )
                    .await;

                instructions.push(instruction);
            }

            proposed_instructions.insert(pred_i, instructions);
        }

        proposed_instructions
    }

    /// Generate a single instruction using the prompt model
    async fn generate_instruction(
        &self,
        basic_instruction: &str,
        task_demos: &str,
        data_summary: &str,
        program_description: &str,
        tip: &str,
        pred_i: usize,
        pred_signature: &Signature,
    ) -> String {
        // Build the generation signature
        let sig = build_instruction_generation_signature(
            self.config.use_dataset_summary,
            self.config.program_aware,
            self.config.use_tip,
        );

        let mut generator = Predict::new(sig);
        generator.set_lm(self.prompt_model.clone());

        // Build input example
        let mut input = Example::new();
        input = input.field("basic_instruction", basic_instruction);
        input = input.field("task_demos", task_demos);

        if self.config.use_dataset_summary {
            input = input.field("dataset_description", data_summary);
        }
        if self.config.program_aware {
            input = input.field("program_code", program_description);
            let prog_desc = format!("Program with {} predictors", pred_i + 1);
            input = input.field("program_description", prog_desc.as_str());
            let shorthand = pred_signature.to_shorthand();
            input = input.field("module", shorthand.as_str());
            let mod_desc = format!(
                "Predictor {} with signature: {}",
                pred_i,
                pred_signature.to_shorthand()
            );
            input = input.field("module_description", mod_desc.as_str());
        }
        if self.config.use_tip && !tip.is_empty() {
            input = input.field("tip", tip);
        }

        // Mark all fields as inputs
        let input_keys: Vec<String> = input.keys().cloned().collect();
        let key_refs: Vec<&str> = input_keys.iter().map(|s| s.as_str()).collect();
        input = input.with_inputs(&key_refs);

        match generator.call(&input).await {
            Ok(prediction) => prediction
                .get_str("proposed_instruction")
                .unwrap_or(basic_instruction)
                .to_string(),
            Err(_) => {
                // Fallback: generate a heuristic instruction
                generate_heuristic_instruction(basic_instruction, pred_signature)
            }
        }
    }
}

/// Build the meta-signature for instruction generation
fn build_instruction_generation_signature(
    use_dataset_summary: bool,
    program_aware: bool,
    use_tip: bool,
) -> Signature {
    let mut builder = SignatureBuilder::new().instructions(
        "Use the information below to learn about a task that we are trying to solve \
             using calls to an LM, then generate a new instruction that will be used to prompt \
             a Language Model to better solve the task.",
    );

    if use_dataset_summary {
        builder = builder.input_with_desc("dataset_description", "A description of the dataset");
    }
    if program_aware {
        builder = builder
            .input_with_desc(
                "program_code",
                "Language model program designed to solve a particular task",
            )
            .input_with_desc(
                "program_description",
                "Summary of the task the program is designed to solve",
            )
            .input_with_desc("module", "The module to create an instruction for")
            .input_with_desc("module_description", "Description of the module");
    }

    builder = builder.input_with_desc("task_demos", "Example inputs/outputs of our module");
    builder = builder.input_with_desc("basic_instruction", "Basic instruction");

    if use_tip {
        builder = builder.input_with_desc("tip", "A suggestion for generating the new instruction");
    }

    builder = builder.output_with_desc(
        "proposed_instruction",
        "Propose an instruction that will be used to prompt a Language Model to perform this task",
    );

    builder.build()
}

/// Generate a heuristic instruction when no LLM is available
fn generate_heuristic_instruction(basic_instruction: &str, signature: &Signature) -> String {
    let input_names: Vec<String> = signature
        .input_fields()
        .map(|(name, _)| name.clone())
        .collect();
    let output_names: Vec<String> = signature
        .output_fields()
        .map(|(name, _)| name.clone())
        .collect();

    if !basic_instruction.is_empty() {
        format!(
            "Given the fields {}, {}. Produce the output fields {}.",
            input_names.join(", "),
            basic_instruction,
            output_names.join(", ")
        )
    } else {
        format!(
            "Given the fields {}, produce the output fields {}.",
            input_names.join(", "),
            output_names.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use dspy_core::{LMConfig, LMResponse, Message, Prediction};

    /// Mock LM that returns a fixed proposed instruction
    struct MockProposerLM {
        response: String,
        config: LMConfig,
    }

    impl MockProposerLM {
        fn new(response: &str) -> Self {
            Self {
                response: response.to_string(),
                config: LMConfig::new("mock-proposer"),
            }
        }
    }

    #[async_trait]
    impl LM for MockProposerLM {
        async fn call(
            &self,
            _messages: &[Message],
            _config: &LMConfig,
        ) -> dspy_core::Result<Vec<LMResponse>> {
            Ok(vec![LMResponse {
                text: format!("[[ ## proposed_instruction ## ]]\n{}", self.response),
                usage: None,
            }])
        }
        fn model(&self) -> &str {
            "mock-proposer"
        }
        fn config(&self) -> &LMConfig {
            &self.config
        }
        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({})
        }
    }

    /// Simple test module with one predictor
    struct SimpleModule {
        predict: Predict,
    }

    impl SimpleModule {
        fn new(sig: Signature) -> Self {
            Self {
                predict: Predict::new(sig),
            }
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

    #[tokio::test]
    async fn test_propose_basic() {
        let sig = Signature::from_string("question -> answer")
            .unwrap()
            .with_instructions("Answer the question");
        let program = SimpleModule::new(sig);
        let prompt_model: Arc<dyn LM> = Arc::new(MockProposerLM::new(
            "You are an expert. Given the question, provide a detailed and accurate answer.",
        ));

        let mut proposer = GroundedProposer::new(
            prompt_model,
            ProposerConfig {
                program_aware: false,
                use_dataset_summary: false,
                use_tip: false,
                ..Default::default()
            },
            42,
        );

        let trainset = vec![Example::new()
            .field("question", "What is 2+2?")
            .field("answer", "4")
            .with_inputs(&["question"])];

        let result = proposer
            .propose_instructions_for_program(&program, &trainset, None, 2)
            .await;

        assert!(result.contains_key(&0));
        let instructions = &result[&0];
        assert_eq!(instructions.len(), 2);
        assert!(instructions[0].contains("expert"));
    }

    #[tokio::test]
    async fn test_propose_with_demos() {
        let sig = Signature::from_string("question -> answer").unwrap();
        let program = SimpleModule::new(sig);
        let prompt_model: Arc<dyn LM> = Arc::new(MockProposerLM::new("Improved instruction"));

        let mut proposer = GroundedProposer::new(
            prompt_model,
            ProposerConfig {
                program_aware: false,
                use_dataset_summary: true,
                use_task_demos: true,
                use_tip: true,
                set_tip_randomly: true,
                ..Default::default()
            },
            42,
        );

        let trainset = vec![Example::new()
            .field("question", "Q1")
            .field("answer", "A1")
            .with_inputs(&["question"])];

        // Demo candidates: 1 predictor, 3 demo sets, 1 demo each
        let demo_candidates = vec![vec![
            vec![Example::new()
                .field("question", "D1")
                .field("answer", "DA1")],
            vec![Example::new()
                .field("question", "D2")
                .field("answer", "DA2")],
            vec![Example::new()
                .field("question", "D3")
                .field("answer", "DA3")],
        ]];

        let result = proposer
            .propose_instructions_for_program(&program, &trainset, Some(&demo_candidates), 3)
            .await;

        assert_eq!(result[&0].len(), 3);
    }

    #[test]
    fn test_heuristic_instruction() {
        let sig = Signature::from_string("question, context -> answer").unwrap();
        let instruction = generate_heuristic_instruction("Answer accurately", &sig);
        assert!(instruction.contains("question"));
        assert!(instruction.contains("context"));
        assert!(instruction.contains("answer"));
        assert!(instruction.contains("Answer accurately"));
    }

    #[test]
    fn test_build_instruction_generation_signature() {
        let sig = build_instruction_generation_signature(true, true, true);
        assert!(sig.input_fields().count() >= 6);
        assert_eq!(sig.output_fields().count(), 1);
        assert!(sig.fields().contains_key("proposed_instruction"));
    }

    #[test]
    fn test_data_summary_generation() {
        let proposer = GroundedProposer::new(
            Arc::new(MockProposerLM::new("test")),
            ProposerConfig::default(),
            42,
        );

        let trainset = vec![
            Example::new()
                .field("question", "What is AI?")
                .field("answer", "Artificial Intelligence"),
            Example::new()
                .field("question", "What is ML?")
                .field("answer", "Machine Learning"),
        ];

        let summary = proposer.create_data_summary(&trainset);
        assert!(summary.contains("Total examples: 2"));
        assert!(summary.contains("What is AI?"));
    }

    #[test]
    fn test_data_summary_empty() {
        let proposer = GroundedProposer::new(
            Arc::new(MockProposerLM::new("test")),
            ProposerConfig::default(),
            42,
        );

        let summary = proposer.create_data_summary(&[]);
        assert!(summary.contains("No training data"));
    }

    #[test]
    fn test_tips_have_entries() {
        assert!(TIPS.len() >= 5);
        assert!(TIPS.iter().any(|(k, _)| *k == "creative"));
        assert!(TIPS.iter().any(|(k, _)| *k == "persona"));
    }
}
