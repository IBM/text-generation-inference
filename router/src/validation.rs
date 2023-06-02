/// Payload validation logic
use std::collections::hash_map::RandomState;
use std::time::Duration;
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use moka::sync::Cache;
use rand::Rng;
use rand::rngs::ThreadRng;
use thiserror::Error;
use tokenizers::tokenizer::Tokenizer;
use tokio::sync::{mpsc, oneshot};
use text_generation_client::ShardedClient;

const MAX_STOP_SEQS: usize = 6;
const MAX_STOP_SEQ_TOKENS: usize = 40;

/// Validation
#[derive(Debug, Clone)]
pub struct Validation {
    /// Channel to communicate with the background validation task
    sender: mpsc::Sender<ValidationRequest>,
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
        let (validation_sender, validation_receiver) = mpsc::channel(128);

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
        request: GenerateRequest,
    ) -> Result<(usize, GenerateRequest), ValidationError> {
        // Create response channel
        let (sender, receiver) = oneshot::channel();
        // Send request to the background validation task
        // Unwrap is safe here
        self.sender.send((request, sender)).await.unwrap();
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
    mut receiver: mpsc::Receiver<ValidationRequest>,
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
    while let Some((request, response_tx)) = receiver.blocking_recv() {
        let result = validate(
            request, &tokenizer,
            &mut prefix_cache,
            &mut client,
            max_sequence_length,
            max_max_new_tokens,
            &mut rng,
        );
        response_tx.send(result).unwrap_or_default()
    }
}

fn validate(
    mut request: GenerateRequest,
    tokenizer: &Tokenizer,
    prefix_cache: &mut Cache<String, usize, RandomState>,
    client: &mut ShardedClient,
    max_sequence_length: usize,
    max_max_new_tokens: usize,
    rng: &mut ThreadRng,
) -> Result<(usize, GenerateRequest), ValidationError> {
    let params = &mut request.parameters;
    let min_new_tokens = params.min_new_tokens as usize;
    let max_new_tokens = params.max_new_tokens as usize;

    if params.temperature < 0.0 {
        return Err(ValidationError::Temperature);
    }
    if params.top_p <= 0.0 || params.top_p > 1.0 {
        return Err(ValidationError::TopP);
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

    // If sampling mode and seed is None, assign a random one
    if params.temperature != 0.0 && params.seed.is_none() {
        // We generate a 32bit seed so the values aren't too many digits,
        // since this will be returned in the API response
        params.seed = Some(rng.gen::<u32>() as u64);
    }

    let prefix_length = if let Some(prefix_id) = &request.prefix_id {
        prefix_cache.try_get_with_by_ref(
            prefix_id, || client.prefix_lookup(prefix_id),
        ).map_err(|e| ValidationError::PromptPrefix(
            prefix_id.clone(),
            e.to_string(),
        ))?
    } else {
        0
    };

    // Get the number of tokens in the input
    match tokenizer.encode(request.inputs.clone(), true) {
        Ok(inputs) => {
            let mut input_length = inputs.len();
            if params.truncate_input_tokens > 0 && params.truncate_input_tokens < input_length {
                input_length = params.truncate_input_tokens;
            } else {
                // Indicates no truncation is necessary
                params.truncate_input_tokens = 0;
            }
            // Add prefix length to obtain effective token length
            let effective_input_length = input_length + prefix_length;
            if effective_input_length + min_new_tokens > max_sequence_length {
                // Input + min new tokens can't exceed global max token limit
                Err(ValidationError::InputLength(
                    input_length,
                    prefix_length,
                    min_new_tokens,
                    max_sequence_length,
                ))
            } else {
                if effective_input_length + max_new_tokens > max_sequence_length {
                    // If max tokens exceeds global limit, reduce it and flag so that the
                    // appropriate stop reason is returned
                    request.parameters.max_new_tokens = (max_sequence_length - effective_input_length) as u32;
                    request.parameters.max_is_token_limit = true;
                }
                Ok((input_length, request))
            }
        },
        Err(err) => Err(ValidationError::Tokenizer(err.to_string())),
    }
}

type ValidationRequest = (
    GenerateRequest,
    oneshot::Sender<Result<(usize, GenerateRequest), ValidationError>>,
);

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("temperature must be >= 0.0")]
    Temperature,
    #[error("top_p must be > 0.0 and <= 1.0")]
    TopP,
    #[error("top_k must be strictly positive")]
    TopK,
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
    #[error("tokenizer error {0}")]
    Tokenizer(String),
    #[error("can specify at most 6 non-empty stop sequences, each not more than 40 tokens")]
    StopSequences,
    #[error("must request input and/or generated tokens to request extra token detail")]
    TokenDetail,
    #[error("can't retrieve prompt prefix with id '{0}': {1}")]
    PromptPrefix(String, String),
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
