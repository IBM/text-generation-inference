/// Payload validation logic
use std::collections::hash_map::RandomState;
use std::time::Duration;
use crate::{ErrorResponse, GenerateParameters, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use moka::sync::Cache;
use rand::Rng;
use rand::rngs::ThreadRng;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use tokio::time::Instant;
use text_generation_client::{ClientError, ShardedClient};

const MAX_STOP_SEQS: usize = 6;
const MAX_STOP_SEQ_TOKENS: usize = 40;

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Channel to communicate with the background validation task
    sender: mpsc::UnboundedSender<ValidationRequest>,
}

impl Validation {
    pub(crate) fn new(
        workers: usize,
        tokenizer: Tokenizer,
        client: ShardedClient,
        max_sequence_length: usize,
        max_new_tokens: usize,
    ) -> Self {
        // Create channel
        let (
            validation_sender, validation_receiver
        ) = mpsc::unbounded_channel();

        // Launch background validation task
        tokio::spawn(validation_task(
            workers,
            tokenizer,
            client,
            max_sequence_length,
            max_new_tokens,
            validation_receiver,
        ));

        Self {
            sender: validation_sender,
        }
    }

    /// Validate a payload and get the number of tokens in the input
    pub(crate) async fn validate(
        &self,
        prefix_id: Option<String>,
        parameters: GenerateParameters,
        inputs: Vec<String>,
    ) -> Result<Vec<(usize, GenerateRequest)>, ValidationError> {
        // Create response channel
        let (sender, receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender.send((prefix_id, parameters, inputs, sender)).unwrap();
        // Await on response channel
        // Unwrap is safe here
        receiver.await.unwrap()
    }
}

/// Validation task
/// Load balance the validation requests between multiple validation workers
async fn validation_task(
    workers: usize,
    tokenizer: Tokenizer,
    client: ShardedClient,
    max_sequence_length: usize,
    max_new_tokens: usize,
    mut receiver: mpsc::UnboundedReceiver<ValidationRequest>,
) {
    let mut workers_senders = Vec::with_capacity(workers);

    // Rust-level cache just stores length of prefixes
    let prefix_cache = Cache::builder()
        .max_capacity(256)
        .time_to_live(Duration::from_secs(60 * 60))
        .build();

    // Create workers
    for _ in 0..workers {
        let tokenizer_clone: Tokenizer = tokenizer.clone().into();
        // Create channel to communicate with worker
        let (worker_sender, worker_receiver) = mpsc::channel(workers);
        workers_senders.push(worker_sender);

        let client = client.clone();
        let prefix_cache = prefix_cache.clone();
        // Spawn worker
        tokio::task::spawn_blocking(move || validation_worker(
            tokenizer_clone,
            prefix_cache,
            client,
            max_sequence_length,
            max_new_tokens,
            worker_receiver,
        ));
    }

    loop {
        // Load balance requests between workers
        for sender in workers_senders.iter() {
            if let Some(validation_request) = receiver.recv().await {
                sender.send(validation_request).await.unwrap();
            } else {
                return;
            }
        }
    }
}

/// Check the parameters inside the payload and get the number of tokens inside the input using
/// the tokenizer
fn validation_worker(
    tokenizer: Tokenizer,
    mut prefix_cache: Cache<String, usize, RandomState>,
    mut client: ShardedClient,
    max_sequence_length: usize,
    max_max_new_tokens: usize,
    mut receiver: mpsc::Receiver<ValidationRequest>,
) {
    // Seed rng
    let mut rng = rand::thread_rng();

    // Loop over requests
    while let Some(
        (prefix_id, parameters, inputs, response_tx)
    ) = receiver.blocking_recv() {
        let result = validate(
            prefix_id,
            parameters,
            inputs,
            &tokenizer,
            &mut prefix_cache,
            &mut client,
            max_sequence_length,
            max_max_new_tokens,
            &mut rng,
        );
        response_tx.send(result).unwrap_or_default()
    }
}

fn prompt_prefix_lookup(
    client: &mut ShardedClient, prefix_id: &String,
) -> Result<usize, ClientError> {
    let start_time = Instant::now();
    let result = client.prefix_lookup(prefix_id);
    if result.is_ok() {
        metrics::histogram!("tgi_prompt_load_duration", start_time.elapsed().as_secs_f64());
    } else {
        metrics::increment_counter!("tgi_prompt_load_failure");
    }
    result
}

fn validate(
    prefix_id: Option<String>,
    params: GenerateParameters,
    inputs: Vec<String>,
    tokenizer: &Tokenizer,
    prefix_cache: &mut Cache<String, usize, RandomState>,
    client: &mut ShardedClient,
    max_sequence_length: usize,
    max_max_new_tokens: usize,
    rng: &mut ThreadRng,
) -> Result<Vec<(usize, GenerateRequest)>, ValidationError> {
    let min_new_tokens = params.min_new_tokens as usize;
    let max_new_tokens = params.max_new_tokens as usize;

    if params.temperature != 0.0 && params.temperature < 0.05 {
        return Err(ValidationError::Temperature);
    }
    if params.top_p <= 0.0 || params.top_p > 1.0 {
        return Err(ValidationError::TopP);
    }
    if params.typical_p >= 1.0 {
        return Err(ValidationError::TypicalP);
    }
    if params.top_k < 0 {
        return Err(ValidationError::TopK);
    }
    if max_new_tokens > max_max_new_tokens {
        return Err(ValidationError::MaxNewTokens(max_max_new_tokens));
    }
    if min_new_tokens > max_new_tokens {
        return Err(ValidationError::MinNewTokens);
    }
    if params.repetition_penalty <= 0.0 {
        return Err(ValidationError::RepetitionPenalty);
    }
    if let Some((_, decay_factor)) = params.length_penalty {
        if decay_factor < 1.0 || decay_factor > 10.0 {
            return Err(ValidationError::LengthPenalty);
        }
    }
    if params.stop_seqs.len() > MAX_STOP_SEQS {
        return Err(ValidationError::StopSequences);
    }
    if (params.include_logprobs || params.include_ranks || params.include_top_n != 0) &&
        !(params.include_input_tokens || params.include_gen_tokens) {
        return Err(ValidationError::TokenDetail);
    }

    params.stop_seqs.iter()
        .map(|s| if s.is_empty() {
            Err(ValidationError::StopSequences) // Stop sequence can't be empty string
        } else {
            match tokenizer.encode(&s[..], false) {
                Ok(enc) if enc.len() <= MAX_STOP_SEQ_TOKENS => Ok(()),
                Ok(_) => Err(ValidationError::StopSequences),
                Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
            }
        }).find(|r| r.is_err()).unwrap_or(Ok(()))?;

    let prefix_length = if let Some(prefix_id) = &prefix_id {
        prefix_cache.try_get_with_by_ref(
            prefix_id, || prompt_prefix_lookup(client, prefix_id),
        ).map_err(|e| ValidationError::PromptPrefix(
            prefix_id.clone(),
            e.to_string(),
        ))?
    } else {
        0
    };

    // Get the number of tokens in the inputs
    match inputs.iter().map(
        |input| tokenizer.encode(input.clone(), true).map(|enc| {
            let input_length = enc.len();
            metrics::histogram!("tgi_request_raw_input_length", input_length as f64);
            input_length
        })
    ).collect::<Result<Vec<usize>, tokenizers::Error>>() {
        Ok(input_lengths) => {
            input_lengths.into_iter().zip(inputs).map(|(mut input_length, input)| {
                let mut parameters = params.clone();
                if parameters.truncate_input_tokens > 0 && parameters.truncate_input_tokens < input_length {
                    input_length = params.truncate_input_tokens;
                } else {
                    // Indicates no truncation is necessary
                    parameters.truncate_input_tokens = 0;
                }
                // Add prefix length to obtain effective token length
                let effective_input_length = input_length + prefix_length;
                if effective_input_length >= max_sequence_length {
                    // This covers the disallowed boundary case where input length == max seq length
                    Err(ValidationError::InputLength2(
                        input_length,
                        prefix_length,
                        max_sequence_length,
                    ))
                } else if effective_input_length + min_new_tokens > max_sequence_length {
                    // Input + min new tokens can't exceed global max token limit
                    Err(ValidationError::InputLength(
                        input_length,
                        prefix_length,
                        min_new_tokens,
                        max_sequence_length,
                    ))
                } else {
                    // If sampling mode and seed is None, assign a random one
                    if parameters.temperature != 0.0 && parameters.seed.is_none() {
                        // We generate a 32bit seed so the values aren't too many digits,
                        // since this will be returned in the API response
                        parameters.seed = Some(rng.gen::<u32>() as u64);
                    }

                    if effective_input_length + max_new_tokens > max_sequence_length {
                        // If max tokens exceeds global limit, reduce it and flag so that the
                        // appropriate stop reason is returned
                        parameters.max_new_tokens = (max_sequence_length - effective_input_length) as u32;
                        parameters.max_is_token_limit = true;
                    }

                    Ok((
                        input_length,
                        GenerateRequest {
                            prefix_id: prefix_id.clone(),
                            inputs: input,
                            parameters,
                        }
                    ))
                }
            }).collect::<Result<Vec<(usize, GenerateRequest)>, ValidationError>>().map(|results| {
                // Only record these for successful validation
                for (input_length, _) in &results {
                    metrics::histogram!("tgi_request_input_length", *input_length as f64);
                    metrics::histogram!("tgi_request_max_new_tokens", max_new_tokens as f64);
                }
                results
            })
        },
        Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
    }
}

type ValidationRequest = (
    Option<String>,
    GenerateParameters,
    Vec<String>,
    oneshot::Sender<Result<Vec<(usize, GenerateRequest)>, ValidationError>>,
);

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("temperature must be >= 0.05")]
    Temperature,
    #[error("top_p must be > 0.0 and <= 1.0")]
    TopP,
    #[error("top_k must be strictly positive")]
    TopK,
    #[error("typical_p must be < 1.0")]
    TypicalP,
    #[error("repetition_penalty must be > 0.0")]
    RepetitionPenalty,
    #[error("length_penalty must be >= 1.0 and <= 10.0")]
    LengthPenalty,
    #[error("max_new_tokens must be <= {0}")]
    MaxNewTokens(usize),
    #[error("min_new_tokens must be <= max_new_tokens")]
    MinNewTokens,
    #[error("input tokens ({0}) plus prefix length ({1}) plus min_new_tokens ({2}) must be <= {3}")]
    InputLength(usize, usize, usize, usize),
    #[error("input tokens ({0}) plus prefix length ({1}) must be < {2}")]
    InputLength2(usize, usize, usize),
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("can specify at most 6 non-empty stop sequences, each not more than 40 tokens")]
    StopSequences,
    #[error("must request input and/or generated tokens to request extra token detail")]
    TokenDetail,
    #[error("can't retrieve prompt prefix with id '{0}': {1}")]
    PromptPrefix(String, String),
    #[error("sampling parameters aren't applicable in greedy decoding mode")]
    SampleParametersGreedy
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
