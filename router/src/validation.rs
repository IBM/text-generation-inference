/// Payload validation logic
use std::collections::hash_map::RandomState;
use std::time::Duration;

use axum::{http::StatusCode, Json};
use futures::{future::try_join_all, TryFutureExt};
use moka::future::Cache;
use rand::Rng;
use text_generation_client::{ClientError, ShardedClient};
use thiserror::Error;
use tokio::time::Instant;

use crate::{tokenizer::AsyncTokenizer, ErrorResponse, GenerateParameters, GenerateRequest};
use crate::metrics::{increment_counter, observe_histogram};

const MAX_STOP_SEQS: usize = 6;
const MAX_STOP_SEQ_LENGTH: usize = 240;

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    max_sequence_length: usize,
    max_max_new_tokens: usize,
    tokenizer: AsyncTokenizer,
    client: ShardedClient,
    prefix_cache: Cache<String, usize, RandomState>,
}

pub struct RequestSize {
    pub(crate) input_length: usize,
    pub(crate) prefix_length: usize,
}

impl Validation {
    pub(crate) fn new(
        tokenizer: AsyncTokenizer,
        client: ShardedClient,
        max_sequence_length: usize,
        max_max_new_tokens: usize,
    ) -> Self {
        // Rust-level cache just stores length of prefixes
        let prefix_cache = Cache::builder()
            .max_capacity(256)
            .time_to_live(Duration::from_secs(60 * 60))
            .build();

        Self {
            max_sequence_length,
            max_max_new_tokens,
            tokenizer,
            client,
            prefix_cache,
        }
    }

    /// Validate a payload and get the number of tokens in the input
    pub(crate) async fn validate(
        &self,
        prefix_id: Option<String>,
        params: GenerateParameters,
        inputs: Vec<String>,
    ) -> Result<Vec<(RequestSize, GenerateRequest)>, ValidationError> {
        let min_new_tokens = params.min_new_tokens as usize;
        let max_new_tokens = params.max_new_tokens as usize;

        if params.temperature != 0.0 && params.temperature < 0.05 {
            return Err(ValidationError::Temperature);
        }
        if params.top_p <= 0.0 || params.top_p > 1.0 {
            return Err(ValidationError::TopP);
        }
        if params.typical_p > 1.0 {
            return Err(ValidationError::TypicalP);
        }
        if params.top_k < 0 {
            return Err(ValidationError::TopK);
        }
        if max_new_tokens > self.max_max_new_tokens {
            return Err(ValidationError::MaxNewTokens(self.max_max_new_tokens));
        }
        if min_new_tokens > max_new_tokens {
            return Err(ValidationError::MinNewTokens);
        }
        if params.repetition_penalty <= 0.0 {
            return Err(ValidationError::RepetitionPenalty);
        }
        if let Some((_, decay_factor)) = params.length_penalty {
            if !(1.0..=10.0).contains(&decay_factor) {
                return Err(ValidationError::LengthPenalty);
            }
        }
        if params.stop_seqs.len() > MAX_STOP_SEQS {
            return Err(ValidationError::StopSequences(
                MAX_STOP_SEQS,
                MAX_STOP_SEQ_LENGTH,
            ));
        }
        if (params.include_logprobs || params.include_ranks || params.include_top_n != 0)
            && !(params.include_input_tokens || params.include_gen_tokens)
        {
            return Err(ValidationError::TokenDetail);
        }

        if params
            .stop_seqs
            .iter()
            .any(|s| s.is_empty() || s.len() > MAX_STOP_SEQ_LENGTH)
        {
            return Err(ValidationError::StopSequences(
                MAX_STOP_SEQS,
                MAX_STOP_SEQ_LENGTH,
            ));
        }

        let prefix_length = if let Some(prefix_id) = &prefix_id {
            self.prefix_cache
                .try_get_with_by_ref(prefix_id, prompt_prefix_lookup(&self.client, prefix_id))
                .map_err(|e| ValidationError::PromptPrefix(prefix_id.clone(), e.to_string()))
                .await?
        } else {
            0
        };

        // Get the number of tokens in the inputs
        let futures = inputs.into_iter().map(|input| {
            self.tokenizer
                .tokenize(input, false)
                .map_ok(|(input, input_length, _)| {
                    observe_histogram("tgi_request_raw_input_length", input_length as f64);
                    (input, input_length)
                })
        });

        match try_join_all(futures).await {
            Ok(results) => {
                results
                    .into_iter()
                    .map(|(input, mut input_length)| {
                        let mut parameters = params.clone();
                        if parameters.truncate_input_tokens > 0
                            && parameters.truncate_input_tokens < input_length
                        {
                            input_length = params.truncate_input_tokens;
                        } else {
                            // Indicates no truncation is necessary
                            parameters.truncate_input_tokens = 0;
                        }
                        // Add prefix length to obtain effective token length
                        let effective_input_length = input_length + prefix_length;
                        if effective_input_length >= self.max_sequence_length {
                            // This covers the disallowed boundary case where input length == max seq length
                            Err(ValidationError::InputLength2(
                                input_length,
                                prefix_length,
                                self.max_sequence_length,
                            ))
                        } else if effective_input_length + min_new_tokens > self.max_sequence_length
                        {
                            // Input + min new tokens can't exceed global max token limit
                            Err(ValidationError::InputLength(
                                input_length,
                                prefix_length,
                                min_new_tokens,
                                self.max_sequence_length,
                            ))
                        } else {
                            // If sampling mode and seed is None, assign a random one
                            if parameters.temperature != 0.0 && parameters.seed.is_none() {
                                // We generate a 32bit seed so the values aren't too many digits,
                                // since this will be returned in the API response
                                let seed = {
                                    // Since we are async, ensure the ThreadRng goes out of scope
                                    let mut rng = rand::thread_rng();
                                    rng.gen::<u32>()
                                };
                                parameters.seed = Some(seed as u64);
                            }

                            if effective_input_length + max_new_tokens > self.max_sequence_length {
                                // If max tokens exceeds global limit, reduce it and flag so that the
                                // appropriate stop reason is returned
                                parameters.max_new_tokens =
                                    (self.max_sequence_length - effective_input_length) as u32;
                                parameters.max_is_token_limit = true;
                            }

                            Ok((
                                RequestSize {
                                    input_length,
                                    prefix_length,
                                },
                                GenerateRequest {
                                    prefix_id: prefix_id.clone(),
                                    inputs: input,
                                    parameters,
                                },
                            ))
                        }
                    })
                    .collect::<Result<Vec<(RequestSize, GenerateRequest)>, ValidationError>>()
                    .map(|results| {
                        // Only record these for successful validation
                        for (request_size, _) in &results {
                            observe_histogram(
                                "tgi_request_input_length",
                                request_size.input_length as f64
                            );
                            observe_histogram(
                                "tgi_request_max_new_tokens",
                                max_new_tokens as f64
                            );
                        }
                        results
                    })
            }
            Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
        }
    }
}

async fn prompt_prefix_lookup(
    client: &ShardedClient,
    prefix_id: &str,
) -> Result<usize, ClientError> {
    let start_time = Instant::now();
    let result = client.clone().prefix_lookup(prefix_id).await;
    if result.is_ok() {
        observe_histogram(
            "tgi_prompt_load_duration",
            start_time.elapsed().as_secs_f64()
        );
    } else {
        increment_counter("tgi_prompt_load_failure", 1);
    }
    result
}

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("temperature must be >= 0.05")]
    Temperature,
    #[error("top_p must be > 0.0 and <= 1.0")]
    TopP,
    #[error("top_k must be strictly positive")]
    TopK,
    #[error("typical_p must be <= 1.0")]
    TypicalP,
    #[error("repetition_penalty must be > 0.0")]
    RepetitionPenalty,
    #[error("length_penalty must be >= 1.0 and <= 10.0")]
    LengthPenalty,
    #[error("max_new_tokens must be <= {0}")]
    MaxNewTokens(usize),
    #[error("min_new_tokens must be <= max_new_tokens")]
    MinNewTokens,
    #[error(
        "input tokens ({0}) plus prefix length ({1}) plus min_new_tokens ({2}) must be <= {3}"
    )]
    InputLength(usize, usize, usize, usize),
    #[error("input tokens ({0}) plus prefix length ({1}) must be < {2}")]
    InputLength2(usize, usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("can specify at most {0} non-empty stop sequences, each not more than {1} UTF8 bytes")]
    StopSequences(usize, usize),
    #[error("must request input and/or generated tokens to request extra token detail")]
    TokenDetail,
    #[error("can't retrieve prompt prefix with id '{0}': {1}")]
    PromptPrefix(String, String),
    #[error("sampling parameters aren't applicable in greedy decoding mode")]
    SampleParametersGreedy,
}

impl From<ValidationError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: ValidationError) -> Self {
        (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(ErrorResponse {
                error: err.to_string(),
            }),
        )
    }
}
