use std::marker::PhantomData;
use crate::{
    Batcher, Details, ErrorResponse, GenerateRequest, GeneratedText, Validation,
};
use axum::extract::Extension;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;
use metrics_exporter_prometheus::{Matcher, PrometheusBuilder, PrometheusHandle};
use text_generation_client::ShardedClient;
use tokenizers::Tokenizer;
use tokio::signal;
use tokio::sync::{Notify, Semaphore};
use tokio::time::{Instant, sleep, timeout};
use tracing::{instrument, warn};
use crate::batch_types::{BatchType, FlashBatch, PaddedBatch};
use crate::decoder::Decoder;
use crate::grpc_server::start_grpc_server;
use crate::health::Health;
use crate::queue::BatchingConfig;

// Server shared state
#[derive(Clone)]
pub(crate) struct ServerState {
    pub(crate) validation: Validation,
    pub(crate) batcher: Batcher,
    pub(crate) limit_concurrent_requests: Arc<Semaphore>,
    // metadata exposed by the ModelInfo endpoint
    pub(crate) max_sequence_length: usize,
    pub(crate) max_new_tokens: usize,
    pub(crate) seq2seq: bool,
}

/// Health check method
#[instrument(skip(health))]
async fn health(mut health: Extension<Health>) -> Result<(), (StatusCode, Json<ErrorResponse>)> {
    match timeout(Duration::from_secs(5), health.check()).await {
        Ok(true) => Ok(()),
        Ok(false) => Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "unhealthy".to_string(),
            }),
        )),
        Err(_) => {
            tracing::error!("Healthcheck request timed-out");
            Err((
                StatusCode::REQUEST_TIMEOUT,
                Json(ErrorResponse {
                    error: "Healthcheck timed-out".to_string(),
                }),
            ))
        }
    }
}

/// Generate method
#[instrument(
    skip(state),
    fields(
        total_time,
        validation_time,
        queue_time,
        inference_time,
        time_per_token,
        seed,
    )
)]
async fn generate(
    state: Extension<ServerState>,
    req: Json<GenerateRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let start_time = Instant::now();
    // Limit concurrent requests by acquiring a permit from the semaphore
    let _permit = state.limit_concurrent_requests.try_acquire().map_err(|_| {
        tracing::error!("Model is overloaded");
        (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorResponse {
                error: "Model is overloaded".to_string(),
            }),
        )
    })?;

    // Validate request
    //let details = req.0.parameters.details;
    let GenerateRequest {inputs, prefix_id, parameters} = req.0;
    let (input_length, validated_request) =
        state.validation.validate(
            prefix_id, parameters, vec![inputs]
        ).await.map_err(|err| {
            tracing::error!("{err}");
            err
        })?.pop().unwrap();

    // Inference
    let response = state
        .batcher
        .infer(input_length, validated_request)
        .await
        .map_err(|err| {
            tracing::error!("{err}");
            err
        })?;

    // Token details
    // let details = match details {
    //     true => {
    //         let tokens = response
    //             .token_ids
    //             .into_iter()
    //             .zip(response.tokens.into_iter())
    //             .zip(response.logprobs.into_iter())
    //             .map(|((id, text), logprob)| (id, text, logprob))
    //             .collect();
    //         Some(Details {
    //             finish_reason: response.finish_reason,
    //             generated_tokens: response.generated_tokens,
    //             tokens,
    //         })
    //     }
    //     false => None,
    // };

    // Timings
    let total_time = start_time.elapsed();
    let times = response.times.unwrap();
    let validation_time = times.queued - start_time;
    let queue_time = times.start - times.queued;
    let inference_time = times.end - times.start;
    let time_per_token = inference_time / response.gen_token_count;

    // Headers
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-total-time",
        total_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-validation-time",
        validation_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-queue-time",
        queue_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-inference-time",
        inference_time.as_millis().to_string().parse().unwrap(),
    );
    headers.insert(
        "x-time-per-token",
        time_per_token.as_millis().to_string().parse().unwrap(),
    );

    // Tracing metadata
    tracing::Span::current().record("total_time", format!("{total_time:?}"));
    tracing::Span::current().record("validation_time", format!("{validation_time:?}"));
    tracing::Span::current().record("queue_time", format!("{queue_time:?}"));
    tracing::Span::current().record("inference_time", format!("{inference_time:?}"));
    tracing::Span::current().record("time_per_token", format!("{time_per_token:?}"));
    tracing::info!("Output: {}", response.output_text);

    // Send response
    let response = vec![GeneratedText {
        generated_text: response.output_text,
        // details,
    }];
    Ok((headers, Json(response)))
}


struct BatchConfigValidator<B: BatchType> {
    batch_type: PhantomData<B>
}

impl<B: BatchType> BatchConfigValidator<B> {
    fn validate_batch_config(
        &self,
        max_sequence_length: usize,
        max_batch_size: usize,
        max_batch_weight: Option<usize>,
        max_prefill_weight: Option<usize>,
    ) -> (usize, usize) {
        let single_request_stats = <B>::update_stats(
            &B::Stats::default(), max_sequence_length, 0
        );
        let single_request_weight = <B>::batch_weight(
            &single_request_stats, 1
        );
        let weight_upper_bound = single_request_weight * max_batch_size;

        let max_prefill_weight = max_prefill_weight.unwrap_or(
            <B>::default_max_prefill_weight()
        );

        // 0 means no max
        if max_prefill_weight > 0 {
            let single_request_prefill_weight = <B>::prefill_weight(
                &single_request_stats, 1
            );
            if max_prefill_weight < single_request_prefill_weight {
                panic!(
                    "max_prefill_weight ({}) not large enough for max_sequence_length ({}",
                    max_prefill_weight, max_sequence_length
                )
            }
        }

        let max_batch_weight = if let Some(mut max_batch_weight) = max_batch_weight {
            if max_batch_weight < single_request_weight {
                panic!(
                    "max_batch_weight ({}) not large enough for max_sequence_length ({})",
                    max_batch_weight, max_sequence_length
                )
            }
            if max_batch_weight > weight_upper_bound {
                warn!(
                    "Reducing specified max_batch_weight ({}) to ({}) which is an \
                    upper bound based on max_sequence_length ({}) and max_batch_size ({})",
                    max_batch_weight, weight_upper_bound, max_sequence_length, max_batch_size
                );
                max_batch_weight = weight_upper_bound
            }
            max_batch_weight
        } else {
            weight_upper_bound
        };

        (max_batch_weight, max_prefill_weight)
    }
}

pub struct ServerRunArgs {
    pub max_concurrent_requests: usize,
    pub max_sequence_length: usize,
    pub max_new_tokens: usize,
    pub max_batch_size: usize,
    pub max_batch_weight: Option<usize>,
    pub max_prefill_weight: Option<usize>,
    pub max_waiting_tokens: usize,
    pub client: ShardedClient,
    pub tokenizer: Tokenizer,
    pub validation_workers: usize,
    pub addr: SocketAddr,
    pub grpc_addr: SocketAddr,
    pub tls_key_pair: Option<(String, String)>,
    pub tls_client_ca_cert: Option<String>,
    pub output_special_tokens: bool,
}

async fn metrics(prom_handle: Extension<PrometheusHandle>) -> String {
    prom_handle.render()
}

/// Serving method
#[allow(clippy::too_many_arguments)]
pub async fn run(mut args: ServerRunArgs) {
    // Query shard for model info
    let (seq2seq, eos_token_id, use_padding) = args.client.model_info().await
        .expect("Error contacting model shard");
    tracing::info!("Shard model info: is_seq2seq = {seq2seq}, eos_token_id = {eos_token_id}, \
        use_padding = {use_padding}");

    if use_padding {
        do_run(args, seq2seq, eos_token_id, PaddedBatch{}).await
    } else {
        do_run(args, seq2seq, eos_token_id, FlashBatch{}).await
    }
}


/// Serving method
#[allow(clippy::too_many_arguments)]
async fn do_run<B: BatchType>(
    args: ServerRunArgs, seq2seq: bool, eos_token_id: u32, batch_type: B
) {
    let batch_config_validator = BatchConfigValidator::<B>{batch_type: PhantomData};

    // If max batch weight is not set, infer from max batch size and max seq length
    let (max_batch_weight, max_prefill_weight) = batch_config_validator
        .validate_batch_config(
            args.max_sequence_length,
            args.max_batch_size,
            args.max_batch_weight,
            args.max_prefill_weight,
        );

    // Create state
    let decoder = Decoder::new(
        args.tokenizer.clone(), seq2seq, eos_token_id, !args.output_special_tokens,
    );
    let generation_health = Arc::new(AtomicBool::new(false));
    let health_ext = Health::new(
        args.client.clone(), generation_health.clone(), &args.tokenizer,
    );
    let batcher = Batcher::new(
        args.client.clone(),
        BatchingConfig {
            size_limit: args.max_batch_size,
            weight_limit: max_batch_weight,
            prefill_weight_limit: max_prefill_weight,
        },
        args.max_waiting_tokens,
        args.max_concurrent_requests,
        decoder,
        generation_health,
        batch_type,
    );
    let validation = Validation::new(
        args.validation_workers,
        args.tokenizer.clone(),
        args.client,
        args.max_sequence_length,
        args.max_new_tokens,
    );
    let shared_state = ServerState {
        validation,
        batcher,
        limit_concurrent_requests: Arc::new(Semaphore::new(args.max_concurrent_requests)),
        max_sequence_length: args.max_sequence_length,
        max_new_tokens: args.max_new_tokens,
        seq2seq,
    };


    // Duration buckets
    let duration_matcher = Matcher::Suffix(String::from("duration"));
    let n_duration_buckets = 35;
    let mut duration_buckets = Vec::with_capacity(n_duration_buckets);
    // Minimum duration in seconds
    let mut value = 0.0001;
    for _ in 0..n_duration_buckets {
        // geometric sequence
        value *= 1.5;
        duration_buckets.push(value);
    }
    // Input Length buckets
    let input_length_matcher = Matcher::Full(String::from("tgi_request_input_length"));
    let max_sequence_length_buckets: Vec<f64> = (0..64)
        .map(|x| (args.max_sequence_length as f64 / 64.0) * (x + 1) as f64)
        .collect();
    // Generated tokens buckets
    let generated_tokens_matcher = Matcher::Full(String::from("tgi_request_generated_tokens"));
    let max_new_tokens_buckets: Vec<f64> = (0..64)
        .map(|x| (args.max_new_tokens as f64 / 64.0) * (x + 1) as f64)
        .collect();
    // Max new tokens buckets
    let max_new_tokens_matcher = Matcher::Full(String::from("tgi_request_max_new_tokens"));
    // Total tokens buckets
    let total_tokens_matcher = Matcher::Full(String::from("tgi_request_total_tokens"));
    // Batch size buckets
    let batch_size_matcher = Matcher::Full(String::from("tgi_batch_next_size"));
    let batch_inference_size_matcher = Matcher::Full(String::from("tgi_batch_inference_batch_size"));
    let batch_size_buckets: Vec<f64> = (0..args.max_batch_size).map(|x| (x + 1) as f64).collect();

    // Prometheus handler
    let builder = PrometheusBuilder::new()
        .set_buckets_for_metric(duration_matcher, &duration_buckets)
        .unwrap()
        .set_buckets_for_metric(input_length_matcher, &max_sequence_length_buckets)
        .unwrap()
        .set_buckets_for_metric(generated_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(max_new_tokens_matcher, &max_new_tokens_buckets)
        .unwrap()
        .set_buckets_for_metric(total_tokens_matcher, &max_sequence_length_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_size_matcher, &batch_size_buckets)
        .unwrap()
        .set_buckets_for_metric(batch_inference_size_matcher, &batch_size_buckets)
        .unwrap();
    let prom_handle = builder
        .install_recorder()
        .expect("failed to install metrics recorder");

    // Create router
    let app = Router::new()
        // Disabling HTTP generate endpoint for now
        //.route("/generate", post(generate))
        //.layer(Extension(shared_state.clone()))
        .route("/health", get(health))
        .layer(Extension(health_ext))
        .route("/metrics", get(metrics))
        .layer(Extension(prom_handle));

    let notify = Arc::new(Notify::new());
    let notify_clone = notify.clone();

    // Create gRPC server
    let grpc_task = start_grpc_server(
        args.grpc_addr, args.tls_key_pair, args.tls_client_ca_cert,
        shared_state, args.tokenizer, async move {
            notify_clone.notified().await
        },
    ).await;

    // Wait two seconds to ensure gRPC server does not immediately
    // fail before starting
    sleep(Duration::from_secs(2)).await;
    if grpc_task.is_finished() {
        notify.notify_one();
        grpc_task.await.expect("gRPC server startup failed");
        panic!(); // should not reach here
    }

    // Run server
    let server = axum::Server::bind(&args.addr)
        .serve(app.into_make_service())
        // Wait until all requests are finished to shut down
        .with_graceful_shutdown(shutdown_signal());

    tracing::info!("HTTP server started on port {}", args.addr.port());

    server.await.unwrap();
    tracing::info!("HTTP server shutdown complete");
    // Trigger gRPC server shutdown
    notify.notify_one();
    grpc_task.await.unwrap();
}

/// Shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("signal received, starting graceful shutdown");
}
