use std::borrow::Cow;
use std::future::Future;
use std::net::SocketAddr;
use std::ops::Add;
use futures::future::try_join_all;
use tokenizers::tokenizer::Tokenizer;
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


pub(crate) async fn start_grpc_server<F: Future<Output = ()> + Send +'static> (
    grpc_addr: SocketAddr,
    tls_key_pair: Option<(String, String)>,
    tls_client_ca_cert: Option<String>,
    shared_state: ServerState,
    tokenizer: Tokenizer,
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
    let grpc_service = GenerationServicer { state: shared_state, tokenizer };
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
    tokenizer: Tokenizer,
}

#[tonic::async_trait]
impl GenerationService for GenerationServicer {
    #[instrument(
        skip_all,
        fields(
            input=?request.get_ref().requests.iter().map(|r| truncate(&r.text, 32)).collect::<Vec<Cow<'_,str>>>(),
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
        if batch_size == 0 {
            return Ok(Response::new(BatchedGenerationResponse{ responses: vec![] }));
        }
        // Limit concurrent requests by acquiring a permit from the semaphore
        let _permit = self.state.limit_concurrent_requests
            .try_acquire_many(batch_size as u32)
            .map_err(|_| {
                tracing::error!("Model is overloaded");
                Status::resource_exhausted("Model is overloaded")
            })?;

        let params = convert_params(br.params);

        let responses = if batch_size == 1 {
            // Simpler single request case
            let response = self.generate_single(GenerateRequest {
                prefix_id: br.prefix_id,
                inputs: br.requests.into_iter().next().unwrap().text,
                parameters: params,
            }, start_time).await?;
            vec![response]
        } else {
            // Batch size > 1

            //TODO duplicating parameter validation for now, can change to once plus batch tokenization

            let valids = br.requests.into_iter().map(move |r|
                self.state.validation.validate(GenerateRequest {
                    prefix_id: br.prefix_id.clone(),
                    inputs: r.text,
                    parameters: params.clone(),
                })
            );

            let results = try_join_all(valids).await.map_err(|err| {
                tracing::error!("{err}");
                Status::invalid_argument(err.to_string())
            })?;

            let in_toks = results.iter().map(|r| r.0).collect::<Vec<usize>>();

            let response_chans = self.state.batcher
                .infer_batch(results).await
                .map_err(|err| match err {
                    InferError::RequestQueueFull() => Status::resource_exhausted(err.to_string()),
                    _ => Status::from_error(Box::new(err)),
                })?;

            try_join_all(response_chans.into_iter().zip(in_toks).enumerate()
                .map(|(i, (f, in_len))| f.map_ok(move |r| {
                    log_response(
                        &r.times, in_len, r.gen_token_count, r.reason,&r.output_text, start_time,
                        &format!("Sub-request {} from batch of {}", i + 1, batch_size)
                    );
                    r.into()
                }))
            )
            .await
            .map_err(|err| {
                tracing::error!("{err}");
                Status::from_error(Box::new(err))
            })?
        };

        Ok(Response::new(BatchedGenerationResponse{ responses }))
    }

    type GenerateStreamStream = ResponseStream<Result<GenerationResponse, Status>, StreamContext>;

    #[instrument(
        skip_all,
        fields(
            input=?truncate(&request.get_ref().request.as_ref().map(|r| &*r.text).unwrap_or(""), 32),
            correlation_id=?request.metadata().get("x-correlation-id").map(|mv| mv.to_str().unwrap_or("<non-ascii>")).unwrap_or("<none>"),
            input_bytes=?request.get_ref().request.as_ref().map(|r| r.text.len()).unwrap_or(0),
            params=?request.get_ref().params,
        )
    )]
    async fn generate_stream(
        &self, request: Request<SingleGenerationRequest>
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        let start_time = Instant::now();
        let permit = self.state.limit_concurrent_requests.clone()
            .try_acquire_owned().map_err(|_| {
            tracing::error!("Model is overloaded");
            Status::resource_exhausted("Model is overloaded")
        })?;
        let sr = request.into_inner();
        let req = sr.request.ok_or(Status::invalid_argument("missing request"))?;

        // Validate request
        let (input_length, validated_request) = self.state.validation
            .validate(GenerateRequest {
                prefix_id: sr.prefix_id,
                inputs: req.text,
                parameters: convert_params(sr.params),
            })
            .await
            .map_err(|err| {
                tracing::error!("{err}");
                Status::invalid_argument(err.to_string())
            })?;

        let stream = self.state.batcher
            .infer_stream(input_length, validated_request, |r| match r {
                Ok(resp) => Ok(resp.into()),
                Err(err) => Err(Status::from_error(Box::new(err))),
            }, |ctx, count, reason, times, out, err| {
                let _enter = ctx.span.enter();
                if let Some(e) = err {
                    tracing::error!("Streaming response failed after {count} tokens, \
                        output so far: '{out}': {e}");
                } else {
                    log_response(
                        &times, ctx.input_token_count, count,
                        reason,&out, ctx.start_time, "Streaming response"
                    );
                }
            }, StreamContext {
                span: Span::current(),
                input_token_count: input_length,
                start_time,
                _permit: permit,
            }).await
            //TODO move this somewhere common .. use From trait
            .map_err(|err| match err {
                InferError::RequestQueueFull() => Status::resource_exhausted(err.to_string()),
                _ => Status::from_error(Box::new(err)),
            })?;

        // Inference
        Ok(Response::new(stream))
    }

    async fn tokenize(
        &self, request: Request<BatchedTokenizeRequest>
    ) -> Result<Response<BatchedTokenizeResponse>, Status> {
        let br = request.into_inner();

        let responses = self.tokenizer.encode_batch(
            br.requests.into_iter().map(|tr| tr.text).collect(), true
        )
            .map_err(Status::from_error)?
            .into_iter().map(|e| TokenizeResponse {
                token_count: e.len() as u32,
                tokens: if br.return_tokens { e.get_tokens().to_vec() } else { vec![] },
        }).collect();

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
    // Simpler single request case
    async fn generate_single(&self, request: GenerateRequest, start_time: Instant)
        -> Result<GenerationResponse, Status> {
        let (input_length, valid_request) = self.state.validation
            .validate(request)
            .await
            .map_err(|err| {
                tracing::error!("{err}");
                Status::invalid_argument(err.to_string())
            })?;

        let response = self.state.batcher
            .infer(input_length, valid_request)
            .await
            .map_err(|err| match err {
                InferError::RequestQueueFull() => Status::resource_exhausted(err.to_string()),
                _ => {
                    tracing::error!("{err}");
                    Status::from_error(Box::new(err))
                },
            })?;
        log_response(
            &response.times, input_length, response.gen_token_count, response.reason,
            &response.output_text, start_time, "Request"
        );
        Ok(response.into())
    }
}

fn log_response(
    times: &Option<Times>, in_toks: usize, toks: u32, reason: StopReason,
    output: &String, start_time: Instant, kind: &str,
) {
    let span;
    let _enter;
    // Timings
    if let Some(times) = times.as_ref() {
        let validation_time = times.queued - start_time;
        let queue_time = times.start - times.queued;
        let inference_time = times.end - times.start;
        let time_per_token = inference_time.checked_div(toks)
            .unwrap_or_else(|| Duration::new(0, 0));
        let total_time = Instant::now() - start_time;

        // Tracing metadata
        span = info_span!("",
        validation_time = ?validation_time,
        queue_time = ?queue_time,
        inference_time = ?inference_time,
        time_per_token = ?time_per_token,
        total_time = ?total_time,
        input_toks = in_toks);
        _enter = span.enter();
    }

    let len = output.len();
    let output = truncate(output, 32);
    match reason {
        Error => tracing::error!(
            "{kind} generated {toks} tokens before {reason:?}, output {len} bytes: {output:?}",
        ),
        Cancelled | TokenLimit => tracing::warn!(
            "{kind} generated {toks} tokens before {reason:?}, output {len} bytes: {output:?}",
        ),
        _ => tracing::info!(
            "{kind} generated {toks} tokens before {reason:?}, output {len} bytes: {output:?}",
        )
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

fn convert_params(params: Option<Parameters>) -> GenerateParameters {
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
                    gp.seed = s.seed;
                }
                if gp.temperature == 0.0 {
                    gp.temperature = 1.0; // sampling and temp 0 => disabled i.e. temp 1
                }
            };
            // else temperature = 0.0 => greedy
            gp
        },
        None => default_parameters(),
    }
}

impl From<InferResponse> for GenerationResponse {
    fn from(resp: InferResponse) -> Self {
        Self{
            input_token_count: resp.in_token_count,
            text: resp.output_text,
            generated_token_count: resp.gen_token_count,
            stop_reason: resp.reason as i32,
            tokens: resp.tokens.to_final_vec(),
            input_tokens: resp.in_tokens.to_final_vec(),
            seed: resp.seed,
        }
    }
}