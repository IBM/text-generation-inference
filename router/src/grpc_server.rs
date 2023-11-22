use std::borrow::Cow;
use std::future::Future;
use std::net::SocketAddr;
use std::ops::Add;
use futures::future::try_join_all;
use futures::TryFutureExt;
use tokio::fs::read;
use tokio::sync::OwnedSemaphorePermit;
use tokio::task::JoinHandle;
use tokio::time::{Instant, Duration};
use tonic::{Request, Response, Status};
use tonic::transport::{Certificate, Identity, Server, ServerTlsConfig};
use tracing::{info_span, instrument, Span};
use crate::{default_parameters, GenerateParameters, GenerateRequest};
use crate::batcher::{InferError, InferResponse, ResponseStream, Times};
use crate::pb::fmaas::{
    BatchedGenerationRequest, BatchedGenerationResponse, GenerationResponse,
    SingleGenerationRequest, BatchedTokenizeRequest, BatchedTokenizeResponse,
    TokenizeResponse, Parameters, DecodingMethod, StopReason, ModelInfoRequest, ModelInfoResponse
};
use crate::pb::fmaas::StopReason::{Error, Cancelled, TokenLimit};

use crate::pb::fmaas::generation_service_server::{GenerationService, GenerationServiceServer};
use crate::server::ServerState;
use unicode_truncate::UnicodeTruncateStr;
use crate::pb::fmaas::model_info_response::ModelKind;
use crate::tokenizer::AsyncTokenizer;
use crate::validation::ValidationError;

/// Whether to fail if sampling parameters are provided in greedy-mode requests
/// or to silently ignore them.
const STRICT_PARAMETER_VALIDATION: bool = false;

pub(crate) async fn start_grpc_server<F: Future<Output = ()> + Send +'static> (
    grpc_addr: SocketAddr,
    tls_key_pair: Option<(String, String)>,
    tls_client_ca_cert: Option<String>,
    shared_state: ServerState,
    tokenizer: AsyncTokenizer,
    signal: F,
) -> JoinHandle<()> {

    let mut builder = Server::builder();

    // Configure TLS if requested
    if let Some((cert_path, key_path)) = tls_key_pair {
        let mut tls_config = ServerTlsConfig::new();
        let cert_pem = load_pem(cert_path, "cert").await;
        let key_pem = load_pem(key_path, "key").await;
        if let Some(ca_cert_path) = tls_client_ca_cert {
            let ca_cert_pem = load_pem(ca_cert_path, "client ca cert").await;
            tls_config = tls_config.client_ca_root(Certificate::from_pem(ca_cert_pem));
        }
        tls_config = tls_config.identity(Identity::from_pem(cert_pem, key_pem));
        builder = builder.tls_config(tls_config).expect("tls configuration error");
    }

    // Build and start server
    let grpc_service = GenerationServicer {
        state: shared_state,
        tokenizer,
        input_counter: metrics::register_counter!("tgi_request_input_count"),
        tokenize_input_counter: metrics::register_counter!("tgi_tokenize_request_input_count"),
    };
    let grpc_server = builder
        .add_service(GenerationServiceServer::new(grpc_service))
        .serve_with_shutdown(grpc_addr, signal);

    // Await in spawned task
    tokio::spawn(async move {
        tracing::info!("gRPC server started on port {}", grpc_addr.port());
        grpc_server.await.expect("gRPC server failed");
        tracing::info!("gRPC server shutdown complete");
    })
}

async fn load_pem(path: String, name: &str) -> Vec<u8> {
    read(&path).await.expect(&*format!("couldn't load {name} from {path}"))
}

//  #[derive(Debug, Default)]
pub struct GenerationServicer {
    state: ServerState,
    tokenizer: AsyncTokenizer,
    input_counter: metrics::Counter,
    tokenize_input_counter: metrics::Counter,
}

#[tonic::async_trait]
impl GenerationService for GenerationServicer {
    #[instrument(
        skip_all,
        fields(
            input=?request.get_ref().requests.iter().map(|r| truncate(&r.text, 32)).collect::<Vec<Cow<'_,str>>>(),
            prefix_id=?request.get_ref().prefix_id,
            correlation_id=?request.metadata().get("x-correlation-id").map(|mv| mv.to_str().unwrap_or("<non-ascii>")).unwrap_or("<none>"),
            input_bytes=?request.get_ref().requests.iter().map(|r| r.text.len()).collect::<Vec<usize>>(),
            params=?request.get_ref().params,
        )
    )]
    async fn generate(&self, request: Request<BatchedGenerationRequest>)
        -> Result<Response<BatchedGenerationResponse>, Status> {
        let start_time = Instant::now();
        let br = request.into_inner();
        let batch_size = br.requests.len();
        let kind = if batch_size == 1 { "single" } else { "batch" };
        metrics::increment_counter!("tgi_request_count", "kind" => kind);
        if batch_size == 0 {
            return Ok(Response::new(BatchedGenerationResponse{ responses: vec![] }));
        }
        self.input_counter.increment(batch_size as u64);
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self.state.limit_concurrent_requests
            .try_acquire_many(batch_size as u32)
            .map_err(|_| {
                metrics::increment_counter!("tgi_request_failure", "err" => "conc_limit");
                tracing::error!("Model is overloaded");
                Status::resource_exhausted("Model is overloaded")
            })?;

        let valids = self.validate(
            br.prefix_id,
            br.params,
            br.requests.into_iter().map(move |r| r.text).collect(),
            start_time,
        ).await?;

        if batch_size == 1 {
            // Single request case
            let (input_length, request) = valids.into_iter().next().unwrap();
            self.state.batcher.infer(input_length, request)
                .map_ok(|response| {
                    log_response(
                        &response.times, input_length, response.gen_token_count, response.reason,
                        &response.output_text, start_time, "single", "Request", response.request_id
                    );
                    vec![response.into()]
                }).await
        } else {
            // Batch size > 1
            let input_tokens = valids.iter().map(|r| r.0).collect::<Vec<usize>>();
            match self.state.batcher.infer_batch(valids).await {
                Ok(response_chans) => {
                    try_join_all(response_chans.into_iter().zip(input_tokens).enumerate()
                        .map(|(i, (f, in_len))| f.map_ok(move |r| {
                            log_response(
                                &r.times, in_len, r.gen_token_count, r.reason,&r.output_text, start_time,
                                "batch", &format!("Sub-request {} from batch of {}", i + 1, batch_size), r.request_id
                            );
                            r.into()
                        }))
                    ).await
                },
                Err(err) => Err(err),
            }
        }.map_err(|err| match err {
            InferError::RequestQueueFull() => {
                metrics::increment_counter!("tgi_request_failure", "err" => "queue_full");
                Status::resource_exhausted(err.to_string())
            },
            _ => {
                metrics::increment_counter!("tgi_request_failure", "err" => "generate");
                tracing::error!("{err}");
                Status::from_error(Box::new(err))
            },
        }).map(
            |responses| Response::new(BatchedGenerationResponse{ responses })
        )
    }

    type GenerateStreamStream = ResponseStream<Result<GenerationResponse, Status>, StreamContext>;

    #[instrument(
        skip_all,
        fields(
            input=?truncate(&request.get_ref().request.as_ref().map(|r| &*r.text).unwrap_or(""), 32),
            prefix_id=?request.get_ref().prefix_id,
            correlation_id=?request.metadata().get("x-correlation-id").map(|mv| mv.to_str().unwrap_or("<non-ascii>")).unwrap_or("<none>"),
            input_bytes=?request.get_ref().request.as_ref().map(|r| r.text.len()).unwrap_or(0),
            params=?request.get_ref().params,
        )
    )]
    async fn generate_stream(
        &self, request: Request<SingleGenerationRequest>
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let start_time = Instant::now();
        metrics::increment_counter!("tgi_request_count", "kind" => "stream");
        self.input_counter.increment(1);
        let permit = self.state.limit_concurrent_requests.clone()
            .try_acquire_owned().map_err(|_| {
                metrics::increment_counter!("tgi_request_failure", "err" => "conc_limit");
                tracing::error!("Model is overloaded");
                Status::resource_exhausted("Model is overloaded")
        })?;
        let sr = request.into_inner();
        let req = sr.request.ok_or_else(
            || Status::invalid_argument("missing request")
        )?;

        // Validate request
        let (input_length, validated_request) = self
            .validate(sr.prefix_id, sr.params, vec![req.text], start_time)
            .await?
            .pop().unwrap();

        let stream = self.state.batcher
            .infer_stream(input_length, validated_request, |r| match r {
                Ok(resp) => Ok(resp.into()),
                Err(err) => Err(Status::from_error(Box::new(err))),
            }, |ctx, count, reason, request_id, times, out, err| {
                let _enter = ctx.span.enter();
                if let Some(e) = err {
                    metrics::increment_counter!("tgi_request_failure", "err" => "generate");
                    tracing::error!("Streaming response failed after {count} tokens, \
                        output so far: '{:?}': {e}", truncate(&out, 32));
                } else {
                    log_response(
                        &times, ctx.input_token_count, count,
                        reason,&out, ctx.start_time,
                        "stream", "Streaming response", request_id
                    );
                }
            }, StreamContext {
                span: Span::current(),
                input_token_count: input_length,
                start_time,
                _permit: permit,
            })
            .await
            .map_err(|err| match err {
                InferError::RequestQueueFull() => {
                    metrics::increment_counter!("tgi_request_failure", "err" => "queue_full");
                    Status::resource_exhausted(err.to_string())
                },
                _ => {
                    metrics::increment_counter!("tgi_request_failure", "err" => "unknown");
                    tracing::error!("{err}");
                    Status::from_error(Box::new(err))
                },
            })?;

        // Inference
        Ok(Response::new(stream))
    }

    async fn tokenize(
        &self, request: Request<BatchedTokenizeRequest>
    ) -> Result<Response<BatchedTokenizeResponse>, Status> {
        let br = request.into_inner();
        metrics::increment_counter!("tgi_tokenize_request_count");
        let start_time = Instant::now();
        self.tokenize_input_counter.increment(br.requests.len() as u64);

        let responses = try_join_all(br.requests.into_iter().map(|tr|
            self.tokenizer.tokenize(tr.text, br.return_tokens).map_ok(
                |(_, token_count, encoding)| TokenizeResponse {
                    token_count: token_count as u32,
                    tokens: encoding.map_or_else(
                        Vec::new, |e| e.get_tokens().to_vec()
                    ),
                }
            ))).map_err(Status::from_error).await?;

        let token_total: u32 = responses.iter().map(|tr| tr.token_count).sum();
        metrics::histogram!("tgi_tokenize_request_tokens", token_total as f64);
        metrics::histogram!("tgi_tokenize_request_duration", start_time.elapsed().as_secs_f64());

        Ok(Response::new(BatchedTokenizeResponse { responses }))
    }

    async fn model_info(
        &self, _request: Request<ModelInfoRequest>
    ) -> Result<Response<ModelInfoResponse>, Status> {
        Ok(Response::new(ModelInfoResponse {
            model_kind: i32::from(if self.state.seq2seq {
                ModelKind::EncoderDecoder
            } else {
                ModelKind::DecoderOnly
            }),
            max_sequence_length: self.state.max_sequence_length as u32,
            max_new_tokens: self.state.max_new_tokens as u32,
        }))
    }
}

pub struct StreamContext {
    span: Span,
    input_token_count: usize,
    start_time: Instant,
    _permit: OwnedSemaphorePermit, // dropped (released) when the stream is dropped
}

impl GenerationServicer {
    pub(crate) async fn validate(
        &self,
        prefix_id: Option<String>,
        parameters: Option<Parameters>,
        inputs: Vec<String>,
        start_time: Instant,
    ) -> Result<Vec<(usize, GenerateRequest)>, Status> {
        match convert_params(parameters, self.state.default_include_stop_seqs) {
            Ok(params) => self.state.validation.validate(
                prefix_id, params, inputs
            ).await,
            Err(err) => Err(err),
        }.map_err(|err| {
            metrics::increment_counter!("tgi_request_failure", "err" => "validation");
            tracing::error!("{err}");
            Status::invalid_argument(err.to_string())
        }).map(|requests| {
            metrics::histogram!("tgi_request_validation_duration", start_time.elapsed().as_secs_f64());
            requests
        })
    }
}

fn log_response(
    times: &Option<Times>,
    input_tokens: usize,
    generated_tokens: u32,
    reason: StopReason,
    output: &String,
    start_time: Instant,
    kind: &'static str,
    kind_log: &str,
    request_id: Option<u64>,
) {
    let span;
    let _enter;
    // Timings
    let total_time = Instant::now() - start_time;
    if let Some(times) = times.as_ref() {
        let validation_time = times.queued - start_time;
        let queue_time = times.start - times.queued;
        let inference_time = times.end - times.start;
        let time_per_token = inference_time.checked_div(generated_tokens)
            .unwrap_or_else(|| Duration::new(0, 0));

        // Tracing metadata
        span = info_span!(
            "",
            validation_time = ?validation_time,
            queue_time = ?queue_time,
            inference_time = ?inference_time,
            time_per_token = ?time_per_token,
            total_time = ?total_time,
            input_toks = input_tokens,
            request_id = request_id,
        );
        _enter = span.enter();

        metrics::histogram!("tgi_request_inference_duration", inference_time.as_secs_f64());
        metrics::histogram!("tgi_request_mean_time_per_token_duration", time_per_token.as_secs_f64());
    }

    // Metrics
    match reason {
        Error => metrics::increment_counter!("tgi_request_failure", "err" => "generate"),
        Cancelled => (), // recorded where cancellation is detected
        _ => {
            metrics::increment_counter!(
                "tgi_request_success", "stop_reason" => reason.as_str_name(), "kind" => kind
            );
            metrics::histogram!("tgi_request_duration", total_time.as_secs_f64());
            metrics::histogram!("tgi_request_generated_tokens", generated_tokens as f64);
            metrics::histogram!(
                "tgi_request_total_tokens", (generated_tokens as usize + input_tokens) as f64
            );
        }
    }

    let len = output.len();
    let output = truncate(output, 32);
    match reason {
        Error => tracing::error!(
            "{kind_log} generated {generated_tokens} tokens before {reason:?}, output {len} bytes: {output:?}",
        ),
        Cancelled | TokenLimit => tracing::warn!(
            "{kind_log} generated {generated_tokens} tokens before {reason:?}, output {len} bytes: {output:?}",
        ),
        _ => tracing::info!(
            "{kind_log} generated {generated_tokens} tokens before {reason:?}, output {len} bytes: {output:?}",
        ),
    };
}

fn truncate(string: &str, len: usize) -> Cow<str> {
    let orig_len = string.len();
    let (string, tlen) = string.unicode_truncate(len);
    if tlen == orig_len {
        string.into()
    } else {
       [string, "..."].concat().into()
    }
}

fn convert_params(
    params: Option<Parameters>, default_include_stop_seqs: bool
) -> Result<GenerateParameters, ValidationError> {
    match params {
        Some(p) => {
            let mut gp = default_parameters();
            // Input token truncation
            gp.truncate_input_tokens = p.truncate_input_tokens as usize;
            // Response Options
            if let Some(r) = p.response {
                gp.include_input_text = r.input_text;
                gp.include_input_tokens = r.input_tokens;
                gp.include_gen_tokens = r.generated_tokens;
                gp.include_logprobs = r.token_logprobs;
                gp.include_ranks = r.token_ranks;
                gp.include_top_n = r.top_n_tokens;
            }
            // Decoding Parameters
            if let Some(d) = p.decoding {
                if d.repetition_penalty != 0.0 {
                    gp.repetition_penalty = d.repetition_penalty;
                    gp.length_penalty = d.length_penalty
                        .map(|lp| (lp.start_index, lp.decay_factor));
                }
            }
            // Stopping Criteria
            if let Some(s) = p.stopping {
                if s.max_new_tokens != 0 { gp.max_new_tokens = s.max_new_tokens }
                gp.min_new_tokens = s.min_new_tokens;
                gp.stop_seqs = s.stop_sequences;
                gp.include_stop_seq = s.include_stop_sequence
                    .unwrap_or(default_include_stop_seqs);
                if s.time_limit_millis > 0 {
                    gp.deadline = Some(Instant::now()
                        .add(Duration::from_millis(s.time_limit_millis as u64)));
                }
            }
            // Sampling Parameters
            if p.method == DecodingMethod::Sample as i32 {
                if let Some(s) = p.sampling {
                    gp.temperature = s.temperature;
                    gp.top_k = s.top_k as i32;
                    if s.top_p != 0.0 { gp.top_p = s.top_p }
                    if s.typical_p != 0.0 { gp.typical_p = s.typical_p }
                    gp.seed = s.seed;
                }
                if gp.temperature == 0.0 {
                    gp.temperature = 1.0; // sampling and temp 0 => disabled i.e. temp 1
                }
            } else if STRICT_PARAMETER_VALIDATION {
                if let Some(s) = p.sampling {
                    if s.temperature != 0.0 || s.top_p != 0.0 || s.typical_p != 0.0
                        || s.top_k != 0 || s.seed.is_some() {
                        return Err(ValidationError::SampleParametersGreedy)
                    }
                }
            }
            // else temperature = 0.0 => greedy
            Ok(gp)
        },
        None => Ok(default_parameters()),
    }
}

impl From<InferResponse> for GenerationResponse {
    fn from(resp: InferResponse) -> Self {
        Self{
            input_token_count: resp.in_token_count,
            text: resp.output_text,
            generated_token_count: resp.gen_token_count,
            stop_reason: resp.reason as i32,
            stop_sequence: resp.stop_sequence.unwrap_or_default(),
            tokens: resp.tokens.to_final_vec(),
            input_tokens: resp.in_tokens.to_final_vec(),
            seed: resp.seed,
        }
    }
}