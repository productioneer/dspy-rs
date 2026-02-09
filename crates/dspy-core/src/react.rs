//! ReAct — Reasoning and Acting agent module.
//! Python equivalent: dspy/predict/react.py
//!
//! Iteratively reasons about the situation, selects tools, and gathers
//! information until the task is complete.

use crate::chain_of_thought::ChainOfThought;
use crate::error::{DspyError, Result};
use crate::example::Example;
use crate::lm::LM;
use crate::module_trait::Module;
use crate::predict::Predict;
use crate::prediction::Prediction;
use crate::signature::{input_field, output_field, Signature};
use crate::tool::Tool;
use crate::value::Value;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration options for ReAct.
pub struct ReActOptions {
    /// Maximum iterations before forcing extraction (default: 20).
    pub max_iters: usize,
}

impl Default for ReActOptions {
    fn default() -> Self {
        Self { max_iters: 20 }
    }
}

pub struct ReAct {
    tools: HashMap<String, Tool>,
    max_iters: usize,
    react_predict: Predict,
    extract_predict: ChainOfThought,
    original_signature: Signature,
}

impl ReAct {
    pub fn new(signature: Signature, tools: Vec<Tool>, options: Option<ReActOptions>) -> Self {
        let opts = options.unwrap_or_default();
        let original_signature = signature.clone();

        // Build tools map
        let mut tools_map = HashMap::new();
        for tool in tools {
            tools_map.insert(tool.name.clone(), tool);
        }

        // Add the "finish" tool
        let output_names: Vec<String> = signature
            .output_fields()
            .map(|(k, _)| format!("`{}`", k))
            .collect();
        let output_names_str = output_names.join(", ");

        tools_map.insert(
            "finish".to_string(),
            Tool::new(
                "finish",
                format!(
                    "Marks the task as complete. Signals that all information for producing {} is now available.",
                    output_names_str
                ),
                HashMap::new(),
                |_| async move { Ok(serde_json::json!("Completed.")) },
            ),
        );

        // Build instructions
        let input_names: Vec<String> = signature
            .input_fields()
            .map(|(k, _)| format!("`{}`", k))
            .collect();
        let input_names_str = input_names.join(", ");

        let mut instr_parts: Vec<String> = Vec::new();

        if !signature.instructions().is_empty() {
            instr_parts.push(format!("{}\n", signature.instructions()));
        }

        instr_parts.push(format!(
            "You are an Agent. In each episode, you will be given the fields {} as input. And you can see your past trajectory so far.",
            input_names_str
        ));
        instr_parts.push(format!(
            "Your goal is to use one or more of the supplied tools to collect any necessary information for producing {}.\n",
            output_names_str
        ));
        instr_parts.push(
            "To do this, you will interleave next_thought, next_tool_name, and next_tool_args in each turn, and also when finishing the task.".to_string(),
        );
        instr_parts.push(
            "After each tool call, you receive a resulting observation, which gets appended to your trajectory.\n".to_string(),
        );
        instr_parts.push(
            "When writing next_thought, you may reason about the current situation and plan for future steps.".to_string(),
        );
        instr_parts.push(
            "When selecting the next_tool_name and its next_tool_args, the tool must be one of:\n"
                .to_string(),
        );

        let mut tool_idx = 1;
        for tool in tools_map.values() {
            instr_parts.push(format!("({}) {}", tool_idx, tool));
            tool_idx += 1;
        }
        instr_parts.push(
            "When providing `next_tool_args`, the value inside the field must be in JSON format"
                .to_string(),
        );

        let instructions = instr_parts.join("\n");

        // Build react signature: inputs + trajectory -> next_thought, next_tool_name, next_tool_args
        let mut react_fields = Vec::new();
        for (name, _) in signature.input_fields() {
            react_fields.push(input_field(name));
        }
        react_fields.push(input_field("trajectory"));
        react_fields.push(output_field("next_thought"));
        react_fields.push(output_field("next_tool_name"));
        react_fields.push(output_field("next_tool_args"));

        let react_sig = Signature::new(react_fields, &instructions);

        // Build extract signature: inputs + trajectory -> outputs
        let mut extract_fields = Vec::new();
        for (name, _) in signature.input_fields() {
            extract_fields.push(input_field(name));
        }
        extract_fields.push(input_field("trajectory"));
        for (name, _) in signature.output_fields() {
            extract_fields.push(output_field(name));
        }

        let extract_sig = Signature::new(extract_fields, signature.instructions());

        let react_predict = Predict::new(react_sig);
        let extract_predict = ChainOfThought::new(extract_sig);

        Self {
            tools: tools_map,
            max_iters: opts.max_iters,
            react_predict,
            extract_predict,
            original_signature,
        }
    }

    /// Convenience constructor from string signature.
    pub fn from_string(
        spec: &str,
        tools: Vec<Tool>,
        options: Option<ReActOptions>,
    ) -> Result<Self> {
        let sig = Signature::from_string(spec)?;
        Ok(Self::new(sig, tools, options))
    }

    /// Access the underlying react Predict module.
    pub fn react_predict(&self) -> &Predict {
        &self.react_predict
    }

    pub fn react_predict_mut(&mut self) -> &mut Predict {
        &mut self.react_predict
    }

    /// Access the underlying extract ChainOfThought module.
    pub fn extract_predict(&self) -> &ChainOfThought {
        &self.extract_predict
    }

    pub fn extract_predict_mut(&mut self) -> &mut ChainOfThought {
        &mut self.extract_predict
    }

    /// Format trajectory entries into a string.
    fn format_trajectory(trajectory: &HashMap<String, String>) -> String {
        // Sort entries so they appear in order (thought_0, tool_name_0, ..., thought_1, ...)
        let mut entries: Vec<(&String, &String)> = trajectory.iter().collect();
        entries.sort_by_key(|(k, _)| {
            // Extract the suffix number for ordering
            let parts: Vec<&str> = k.rsplitn(2, '_').collect();
            let num: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let prefix = parts.last().unwrap_or(&"");
            let order = match *prefix {
                "thought" => 0,
                "tool_name" => 1,
                "tool_args" => 2,
                "observation" => 3,
                _ => 4,
            };
            (num, order)
        });

        entries
            .iter()
            .map(|(k, v)| format!("[{}] {}", k, v))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Truncate the oldest tool call (4 entries) from the trajectory.
    pub fn truncate_trajectory(
        &self,
        trajectory: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>> {
        if trajectory.len() < 4 {
            return Err(DspyError::Other(
                "Trajectory cannot be truncated — only has one tool call".to_string(),
            ));
        }

        // Sort keys to find the oldest entries
        let mut keys: Vec<&String> = trajectory.keys().collect();
        keys.sort_by_key(|k| {
            let parts: Vec<&str> = k.rsplitn(2, '_').collect();
            let num: usize = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let prefix = parts.last().unwrap_or(&"");
            let order = match *prefix {
                "thought" => 0,
                "tool_name" => 1,
                "tool_args" => 2,
                "observation" => 3,
                _ => 4,
            };
            (num, order)
        });

        // Remove the first 4 (oldest tool call)
        let to_remove: Vec<String> = keys.iter().take(4).map(|k| (*k).clone()).collect();
        let mut truncated = trajectory.clone();
        for key in &to_remove {
            truncated.remove(key);
        }
        Ok(truncated)
    }
}

#[async_trait]
impl Module for ReAct {
    fn module_type_name(&self) -> &str {
        "ReAct"
    }

    async fn forward(&self, args: &Example) -> Result<Prediction> {
        let mut trajectory: HashMap<String, String> = HashMap::new();

        // Get max_iters from args or use default
        let max_iters = args
            .get("max_iters")
            .and_then(|v| v.as_i64())
            .map(|v| v as usize)
            .unwrap_or(self.max_iters);

        // Build input args (without max_iters)
        let mut input_args = args.clone();
        input_args.remove("max_iters");

        for idx in 0..max_iters {
            // Build forward args with trajectory
            let mut forward_args = input_args.clone();
            forward_args.set("trajectory", Self::format_trajectory(&trajectory));

            let pred = match self.react_predict.call(&forward_args).await {
                Ok(p) => p,
                Err(e) => {
                    eprintln!(
                        "Ending trajectory: Agent failed to select a valid tool: {}",
                        e
                    );
                    break;
                }
            };

            let thought = pred.get_str("next_thought").unwrap_or("").to_string();
            let tool_name = pred
                .get_str("next_tool_name")
                .unwrap_or("finish")
                .to_string();

            // Parse tool args
            let raw_args = pred.get_str("next_tool_args").unwrap_or("{}");
            let tool_args: HashMap<String, serde_json::Value> =
                serde_json::from_str(raw_args).unwrap_or_default();

            trajectory.insert(format!("thought_{}", idx), thought);
            trajectory.insert(format!("tool_name_{}", idx), tool_name.clone());
            trajectory.insert(
                format!("tool_args_{}", idx),
                serde_json::to_string(&tool_args).unwrap_or_else(|_| "{}".to_string()),
            );

            // Execute tool
            if let Some(tool) = self.tools.get(&tool_name) {
                match tool.call(tool_args).await {
                    Ok(result) => {
                        trajectory.insert(format!("observation_{}", idx), result.to_string());
                    }
                    Err(e) => {
                        trajectory.insert(
                            format!("observation_{}", idx),
                            format!("Execution error in {}: {}", tool_name, e),
                        );
                    }
                }
            } else {
                trajectory.insert(
                    format!("observation_{}", idx),
                    format!("Tool '{}' not found", tool_name),
                );
            }

            if tool_name == "finish" {
                break;
            }
        }

        // Extract final outputs
        let mut extract_args = input_args.clone();
        extract_args.set("trajectory", Self::format_trajectory(&trajectory));

        let extract: Prediction = self.extract_predict.call(&extract_args).await?;

        // Build result with trajectory included
        let mut result_data: HashMap<String, Value> = HashMap::new();
        for (key, _) in self.original_signature.output_fields() {
            if let Some(val) = extract.get(key) {
                result_data.insert(key.clone(), val.clone());
            }
        }

        // Add trajectory as a serialized JSON string
        let trajectory_json =
            serde_json::to_string(&trajectory).unwrap_or_else(|_| "{}".to_string());
        result_data.insert("trajectory".to_string(), Value::String(trajectory_json));

        // Include reasoning if present
        if let Some(reasoning) = extract.get("reasoning") {
            result_data.insert("reasoning".to_string(), reasoning.clone());
        }

        Ok(Prediction::new(result_data))
    }

    fn named_predictors(&self) -> Vec<(&str, &Predict)> {
        let mut preds = vec![("react_predict", &self.react_predict)];
        preds.push(("extract_predict", self.extract_predict.predict()));
        preds
    }

    fn named_predictors_mut(&mut self) -> Vec<(&str, &mut Predict)> {
        vec![
            ("react_predict", &mut self.react_predict),
            ("extract_predict", self.extract_predict.predict_mut()),
        ]
    }

    fn set_lm(&mut self, lm: Arc<dyn LM>) {
        self.react_predict.set_lm(lm.clone());
        self.extract_predict.predict_mut().set_lm(lm);
    }

    fn deep_copy(&self) -> Box<dyn Module> {
        Box::new(Self {
            tools: HashMap::new(), // Tools contain closures, can't clone
            max_iters: self.max_iters,
            react_predict: self.react_predict.clone(),
            extract_predict: ChainOfThought::new(self.extract_predict.predict().signature.clone()),
            original_signature: self.original_signature.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::{LMConfig, LMResponse, Message};
    use crate::settings;
    use crate::tool::ToolArg;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MockReActLM {
        responses: Vec<String>,
        call_idx: AtomicUsize,
        config: LMConfig,
    }

    impl MockReActLM {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: responses.iter().map(|s| s.to_string()).collect(),
                call_idx: AtomicUsize::new(0),
                config: LMConfig::new("mock-react"),
            }
        }
    }

    #[async_trait]
    impl LM for MockReActLM {
        async fn call(&self, _messages: &[Message], _config: &LMConfig) -> Result<Vec<LMResponse>> {
            let idx = self.call_idx.fetch_add(1, Ordering::SeqCst);
            let response = &self.responses[idx % self.responses.len()];
            Ok(vec![LMResponse::new(response.clone(), None)])
        }

        fn model(&self) -> &str {
            "mock-react"
        }

        fn config(&self) -> &LMConfig {
            &self.config
        }

        fn dump_state(&self) -> serde_json::Value {
            serde_json::json!({"model": "mock-react"})
        }
    }

    #[test]
    fn test_creates_react_module_with_tools() {
        let mut search_args = HashMap::new();
        search_args.insert(
            "query".to_string(),
            ToolArg {
                arg_type: "string".to_string(),
                description: Some("Search query".to_string()),
                default: None,
            },
        );

        let search_tool = Tool::new("search", "Search the web", search_args, |_| async move {
            Ok(serde_json::json!("search results"))
        });

        let react = ReAct::from_string("question -> answer", vec![search_tool], None).unwrap();
        assert!(!react.tools.is_empty());
        assert!(react.tools.contains_key("finish"));
        assert!(react.tools.contains_key("search"));
    }

    #[tokio::test]
    async fn test_runs_tool_loop_and_extracts_answer() {
        settings::reset_settings();

        let mut search_args = HashMap::new();
        search_args.insert(
            "query".to_string(),
            ToolArg {
                arg_type: "string".to_string(),
                description: None,
                default: None,
            },
        );

        let search_tool = Tool::new(
            "search",
            "Search for information",
            search_args,
            |_| async move { Ok(serde_json::json!("Paris is the capital of France")) },
        );

        let mut react = ReAct::from_string(
            "question -> answer",
            vec![search_tool],
            Some(ReActOptions { max_iters: 5 }),
        )
        .unwrap();

        let mock_lm: Arc<dyn LM> = Arc::new(MockReActLM::new(vec![
            // First iteration: call search
            "[[ ## next_thought ## ]]\nI need to search for the capital of France.\n\n[[ ## next_tool_name ## ]]\nsearch\n\n[[ ## next_tool_args ## ]]\n{\"query\": \"capital of France\"}\n\n[[ ## completed ## ]]",
            // Second iteration: finish
            "[[ ## next_thought ## ]]\nI found the answer from the search results.\n\n[[ ## next_tool_name ## ]]\nfinish\n\n[[ ## next_tool_args ## ]]\n{}\n\n[[ ## completed ## ]]",
            // Extraction
            "[[ ## reasoning ## ]]\nBased on the search results, Paris is the capital of France.\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
        ]));

        react.set_lm(mock_lm);

        let inputs = Example::new().field("question", "What is the capital of France?");
        let result = react.forward(&inputs).await.unwrap();

        assert_eq!(result.get_str("answer"), Some("Paris"));
        assert!(result.get("trajectory").is_some());
    }

    #[tokio::test]
    async fn test_handles_tool_execution_errors() {
        settings::reset_settings();

        let mut fetch_args = HashMap::new();
        fetch_args.insert(
            "url".to_string(),
            ToolArg {
                arg_type: "string".to_string(),
                description: None,
                default: None,
            },
        );

        let fail_tool = Tool::new("fetch", "Fetch URL", fetch_args, |_| async move {
            Err(DspyError::Other("Network error".to_string()))
        });

        let mut react = ReAct::from_string(
            "question -> answer",
            vec![fail_tool],
            Some(ReActOptions { max_iters: 2 }),
        )
        .unwrap();

        let mock_lm: Arc<dyn LM> = Arc::new(MockReActLM::new(vec![
            // Try the failing tool
            "[[ ## next_thought ## ]]\nLet me try fetching.\n\n[[ ## next_tool_name ## ]]\nfetch\n\n[[ ## next_tool_args ## ]]\n{\"url\": \"http://example.com\"}\n\n[[ ## completed ## ]]",
            // Finish
            "[[ ## next_thought ## ]]\nTool failed, finishing.\n\n[[ ## next_tool_name ## ]]\nfinish\n\n[[ ## next_tool_args ## ]]\n{}\n\n[[ ## completed ## ]]",
            // Extract
            "[[ ## answer ## ]]\nCould not determine\n\n[[ ## completed ## ]]",
        ]));

        react.set_lm(mock_lm);

        let inputs = Example::new().field("question", "test");
        let result = react.forward(&inputs).await.unwrap();

        assert!(result.get_str("answer").is_some());
        // Trajectory should contain the error
        let traj_str = result.get_str("trajectory").unwrap();
        assert!(traj_str.contains("Execution error"));
    }

    #[test]
    fn test_truncate_trajectory_removes_oldest() {
        let react = ReAct::from_string("q -> a", vec![], None).unwrap();

        let mut trajectory = HashMap::new();
        trajectory.insert("thought_0".to_string(), "first".to_string());
        trajectory.insert("tool_name_0".to_string(), "search".to_string());
        trajectory.insert("tool_args_0".to_string(), "{}".to_string());
        trajectory.insert("observation_0".to_string(), "result1".to_string());
        trajectory.insert("thought_1".to_string(), "second".to_string());
        trajectory.insert("tool_name_1".to_string(), "finish".to_string());
        trajectory.insert("tool_args_1".to_string(), "{}".to_string());
        trajectory.insert("observation_1".to_string(), "done".to_string());

        let truncated = react.truncate_trajectory(&trajectory).unwrap();
        assert_eq!(truncated.len(), 4);
        assert!(!truncated.contains_key("thought_0"));
        assert!(truncated.contains_key("thought_1"));
    }

    #[test]
    fn test_truncate_trajectory_throws_on_single_entry() {
        let react = ReAct::from_string("q -> a", vec![], None).unwrap();

        let mut trajectory = HashMap::new();
        trajectory.insert("thought_0".to_string(), "only".to_string());
        trajectory.insert("tool_name_0".to_string(), "finish".to_string());
        trajectory.insert("tool_args_0".to_string(), "{}".to_string());

        let result = react.truncate_trajectory(&trajectory);
        assert!(result.is_err());
    }
}
