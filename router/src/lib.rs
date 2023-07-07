/// Text Generation Inference Webserver
mod health;
mod batcher;
pub mod server;
pub mod grpc_server;
mod validation;
mod decoder;
mod pb;
mod queue;
mod batch_types;

use batcher::Batcher;
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use validation::Validation;

#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct GenerateParameters {
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default = "default_typical_p")]
    pub typical_p: f32,
    #[serde(default = "default_max_new_tokens")]
    pub max_new_tokens: u32,
    // Indicates whether max_new_tokens is a hard
    // limit rather than user-specified limit
    pub max_is_token_limit: bool,
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,

    pub length_penalty: Option<(u32, f32)>,
    
    pub min_new_tokens: u32,
    #[serde(skip)]
    pub deadline: Option<Instant>,

    pub truncate_input_tokens: usize,

    #[serde(default)]
    pub include_input_text: bool,
    #[serde(default)]
    pub include_input_tokens: bool,
    #[serde(default)]
    pub include_gen_tokens: bool,
    #[serde(default)]
    pub include_logprobs: bool,
    #[serde(default)]
    pub include_ranks: bool,
    #[serde(default)]
    pub include_top_n: u32,

    #[serde(default)]
    pub seed: Option<u64>,

    #[serde(default)]
    pub stop_seqs: Vec<String>,
}

fn default_temperature() -> f32 {
    0.0 // => greedy
}

fn default_top_k() -> i32 {
    0
}

fn default_top_p() -> f32 {
    1.0
}

fn default_typical_p() -> f32 {
    0.0
}

fn default_repetition_penalty() -> f32 {
    1.0
}

fn default_max_new_tokens() -> u32 {
    20
}

fn default_parameters() -> GenerateParameters {
    GenerateParameters {
        temperature: default_temperature(),
        top_k: default_top_k(),
        top_p: default_top_p(),
        repetition_penalty: default_repetition_penalty(),
        max_new_tokens: default_max_new_tokens(),

        ..Default::default()
    }
}

#[derive(Clone, Debug, Deserialize, Default)]
pub(crate) struct GenerateRequest {
    pub prefix_id: Option<String>,
    pub inputs: String,
    #[serde(default = "default_parameters")]
    pub parameters: GenerateParameters,
}

#[derive(Serialize)]
pub(crate) struct Details {
    pub finish_reason: String,
    pub generated_tokens: u32,
    pub tokens: Vec<(u32, String, f32)>,
}

#[derive(Serialize)]
pub(crate) struct GeneratedText {
    pub generated_text: String,
}

#[derive(Serialize)]
pub(crate) struct ErrorResponse {
    pub error: String,
}
