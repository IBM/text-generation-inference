use std::cmp::max;
/// Batching and inference logic
use crate::queue::{BatchingConfig, Entry, Queue};
use crate::{ErrorResponse, GenerateRequest};
use axum::http::StatusCode;
use axum::Json;
use std::future::Future;
use std::mem::take;
use std::ops::Add;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;
use futures::{FutureExt, pin_mut, TryFutureExt};
use futures::future::Map;
use nohash_hasher::IntMap;
use text_generation_client::{
    ClientError, Token, ShardedClient, CachedBatch, RequestsStatus,
    InputTokens, GenerateError, Batch, GenerateTokenResponse
};
use thiserror::Error;
use tokio::select;

use tokio::sync::oneshot;
use tokio::sync::mpsc::{channel, Sender, unbounded_channel, UnboundedReceiver};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::oneshot::error::RecvError;
use tokio::sync::oneshot::Receiver;
use tokio::time::Instant;
use tokio_stream::Stream;
use tracing::{debug, info, instrument, warn, enabled, Level, error};
use crate::batch_types::BatchType;
use crate::batcher::InferError::{GenerationError, RequestQueueFull};
use crate::batcher::TokenInfos::{WithIds, WithStrings};
use crate::decoder::{Decoder, IncrementalDecoder, IncrementalDecoderWrapper};
use crate::pb::fmaas::{StopReason, TokenInfo};
use crate::pb::fmaas::StopReason::{
    Cancelled, EosToken, Error, MaxTokens, NotFinished, StopSequence, TimeLimit, TokenLimit
};
use crate::pb::fmaas::token_info::TopToken;

/// Batcher
#[derive(Clone)]
pub(crate) struct Batcher {
    /// Request queue
    sender: Sender<Vec<Entry>>,
    /// Tokenizer
    decoder: Arc<Decoder>,
}

impl Batcher {
    pub(crate) fn new<B: BatchType>(
        client: ShardedClient,
        config: BatchingConfig,
        max_waiting_tokens: usize,
        queue_size: usize,
        decoder: Decoder,
        generation_health: Arc<AtomicBool>,
        batch_type: B,
    ) -> Self {
        // Set up queue
        let (sender, receiver) = channel(queue_size);
        let decoder = Arc::new(decoder);

        // Spawn batching background task that contains all the inference logic
        tokio::spawn(std::panic::AssertUnwindSafe(batching_task(
            client,
            max_waiting_tokens,
            Queue::new(config, batch_type, receiver),
            decoder.clone(),
            generation_health,
        )).catch_unwind().map_err(|panic| {
            error!("Batching task panicked: {panic:?}");
            std::process::exit(1);
        }));

        Self { sender, decoder }
    }

    // Returns input if queue is full
    fn enqueue_request(&self, entries: Vec<Entry>) -> Result<(), InferError> {
        self.sender.try_send(entries).map_err(|se| match se {
            TrySendError::Full(ents) => {
                warn!(
                    "Unexpected: Rejecting request of {} input(s) due to full request queue",
                    ents.len()
                );
                RequestQueueFull()
            },
            TrySendError::Closed(_) => panic!("Queue closed"),
        })
    }

    /// Add a new request to the queue and return a future that will generate the text
    pub(crate) async fn infer(
        &self,
        input_length: usize,
        request: GenerateRequest,
    ) -> Result<InferResponse, InferError> {
        // One shot channel to communicate with the background batching task
        let (response_tx, response_rx) = oneshot::channel();

        // Try to add the request to the queue
        self.enqueue_request(vec![
            Entry::new(request, input_length, Some(response_tx), None),
        ])?;

        // Await on the response from the background task
        // We can safely unwrap as the background task will never drop the sender
        match response_rx.await.unwrap() {
            Ok(ir) => ir.ensure_decoded(&self.decoder),
            Err(err) => Err(GenerationError(err.to_string())),
        }
    }

    // Add a batch of new requests to the queue and return an vec of futures that will generate the text
    pub(crate) async fn infer_batch(
        &self,
        requests: Vec<(usize, GenerateRequest)>,
    ) -> Result<Vec<Map<Receiver<Result<InferResponse, ClientError>>,
        impl FnOnce(Result<Result<InferResponse, ClientError>, RecvError>) -> Result<InferResponse, InferError> + '_>>, InferError> {

        let mut response_chans= vec![];

        let entries: Vec<Entry> = requests.into_iter()
            .map(|(input_length, request)| {
                // One shot channel to communicate with the background batching task
                let (response_tx, response_rx) = oneshot::channel();
                response_chans.push(response_rx
                    .map(move |r: Result<Result<InferResponse, ClientError>, RecvError>| match r.unwrap() {
                        Ok(ir) => ir.ensure_decoded(&self.decoder),
                        Err(err) => Err(GenerationError(err.to_string())),
                    })
                );

                Entry::new(request, input_length, Some(response_tx), None)
            }).collect();

        // Try to add the request to the queue
        self.enqueue_request(entries)?;

        Ok(response_chans)
    }

    /// Add a new request to the queue and return a stream that will generate the text
    pub(crate) async fn infer_stream<T, C>(
        &self,
        input_length: usize,
        request: GenerateRequest,
        result_map: fn (Result<InferResponse, InferError>) -> T,
        on_drop: fn (&C, u32, StopReason, Option<u64>, Option<Times>, String, Option<InferError>),
        on_drop_context: C,
    ) -> Result<ResponseStream<T, C>, InferError> {
        // Channel to communicate with the background batching task
        let (response_tx, response_rx) = unbounded_channel();

        // Send first response with input token count (and text if requested), and random seed used
        response_tx.send(Ok(InferResponse{
            in_token_count: input_length as u32,
            output_text: request.parameters.include_input_text
                .then(|| request.inputs.clone())
                .unwrap_or_default(),
            seed: request.parameters.seed.unwrap_or_default(),
            ..Default::default()
        })).unwrap_or_default();

        let has_stop_seq = !request.parameters.stop_seqs.is_empty();
        let include_token_info = request.parameters.include_gen_tokens;

        // Try to add the request to the queue
        self.enqueue_request(vec![
            Entry::new(request, input_length, None, Some(response_tx)),
        ])?;

        Ok(ResponseStream {
            inner: response_rx,
            map_func: result_map,
            decoder: Some(self.decoder.clone()),
            include_token_info,
            on_drop,
            on_drop_context: Arc::new(on_drop_context),
            token_count: 0,
            output: if has_stop_seq {
                // If stop sequences are requested, incremental decoding is already done in
                // the batching loop
                Accumulator::String(String::new())
            } else {
                Accumulator::Decoder(IncrementalDecoderWrapper::for_decoder(
                    &self.decoder, self.decoder.seq2seq, 0,
                ))
            },
            times: None,
            request_id: None,
            stop_reason: NotFinished,
            err: None,
        })
    }
}

enum Accumulator {
    String(String),
    Decoder(IncrementalDecoderWrapper)
}

impl Accumulator {
    fn into_string(self) -> String {
        match self {
            Self::String(string) => string,
            Self::Decoder(idw) => idw.into_string(),
        }
    }
}

impl Default for Accumulator {
    fn default() -> Self {
        Self::String(String::new())
    }
}

/// State associated with the ongoing response stream
pub struct ResponseStream<T, C> {
    inner: UnboundedReceiver<Result<InferResponse, ClientError>>,
    map_func: fn (Result<InferResponse, InferError>) -> T,
    // This is only an option to avoid Arc clones when used in poll_next
    decoder: Option<Arc<Decoder>>,
    include_token_info: bool,
    on_drop: fn (&C, u32, StopReason, Option<u64>, Option<Times>, String, Option<InferError>),
    on_drop_context: Arc<C>,
    token_count: u32,
    output: Accumulator,
    times: Option<Times>,
    request_id: Option<u64>,
    stop_reason: StopReason,
    err: Option<InferError>,
}

impl<T, C> Drop for ResponseStream<T, C> {
    fn drop(&mut self) {
        if self.stop_reason == NotFinished {
            self.stop_reason = match self.err {
                Some(_) => Error,
                None => Cancelled,
            }
        }
        (self.on_drop)(
            &self.on_drop_context, self.token_count, self.stop_reason, self.request_id,
            take(&mut self.times),
            take(&mut self.output).into_string(),
            take(&mut self.err)
        );
    }
}

impl<T, C> Stream for ResponseStream<T, C> {
    type Item = T;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let next = self.inner.poll_recv(cx)
                .map_err(|err| GenerationError(err.to_string()))
                .map(|o| match o {
                    Some(mut res) => {
                        let mut decode_err = None;
                        match &mut res {
                            Ok(ir) => {
                                self.token_count = ir.gen_token_count;
                                self.stop_reason = ir.reason;
                                if ir.times.is_some() {
                                    self.times = take(&mut ir.times);
                                }
                                if let Some(rid) = ir.request_id {
                                    self.request_id = Some(rid);
                                }
                                let token = match &ir.tokens {
                                    WithIds(toks) if !toks.is_empty() => Some(&toks[0]),
                                    _ => None
                                };
                                // Detatch and reattach the decoder to appease borrow checker
                                // while avoiding having to clone Arcs
                                let decoder = take(&mut self.decoder);
                                match &mut self.output {
                                    Accumulator::String(str) => {
                                        str.push_str(&*ir.output_text);
                                    },
                                    Accumulator::Decoder(id) => {
                                        if let Some(tok) = token {
                                            match id.next(
                                                tok.token_id,
                                                decoder.as_ref().unwrap(),
                                            ) {
                                                Ok((text, _)) => ir.output_text = text,
                                                Err(err) => decode_err = Some(err),
                                            }
                                        }
                                        // Add remainder if this is the last one
                                        if decode_err.is_none() && ir.reason != NotFinished {
                                            match id.flush(None, decoder.as_ref().unwrap()) {
                                                Ok(text) => ir.output_text += &text,
                                                Err(err) => decode_err = Some(err),
                                            }
                                        }
                                    }
                                }
                                self.decoder = decoder;
                                if !self.include_token_info {
                                    ir.tokens.clear();
                                }
                                ir.decode_token_infos(&self.decoder.as_ref().unwrap());
                                if ir.tokens.is_empty() && ir.output_text.is_empty()
                                    && ir.reason == NotFinished && ir.gen_token_count != 0 {
                                    // Don't include response if it's empty, unless it's the first
                                    return None
                                }
                            },
                            Err(err) => {
                                self.err = Some(err.clone());
                            }
                        }
                        if let Some(err) = decode_err {
                            self.err = Some(err.clone());
                            res = Err(err);
                        }
                        Some(Some((self.map_func)(res)))
                    },
                    None => Some(None),
                });
            if let Poll::Ready(None) = next {
                // Skip if output is empty (for example was a special token)
                continue
            }
            return next.map(Option::unwrap);
        }
    }
}

/// Batching logic
/// Will be launched in a background Tokio task
///
/// Batches requests and sends them to the inference server
// #[instrument(skip(client, receiver, shared))]
async fn batching_task<B: BatchType>(
    mut client: ShardedClient,
    max_waiting_tokens: usize,
    mut queue: Queue<B>,
    decoder: Arc<Decoder>,
    generation_health: Arc<AtomicBool>,
) {
    let mut processor = TokenProcessor {
        entries: IntMap::default(),
        decoder: &decoder,
        generation_health,
    };

    // Get the next batch from the queue
    while let Some(batch) = queue.next_batch(processor.entries()).await {
        if enabled!(Level::DEBUG) {
            debug!["Pulled batch of {} request(s) from queue: {:?}", batch.requests.len(),
                batch.requests.iter().map(|r| r.id).collect::<Vec<u64>>()];
        }
        log_new_batch(batch.id, processor.entries());

        let (mut cached_batch, _) = processor.prefill(
            &mut client, batch, vec![], None, &mut queue,
        ).await;
        let mut waiting_tokens = 1;
        let mut batch_max_remaining_tokens = None;
        let mut next_prefill_after = None;

        // We loop until we do not receive any cached batch from the inference server (== until
        // all requests have met their stopping criteria)
        while let Some(batch) = cached_batch {
            let batch_size = processor.entries().len();
            let batch_id = batch.batch_id;
            let mut batches = vec![batch];

            // Recompute or decrement batch_remaining_tokens as appropriate
            batch_max_remaining_tokens = Some(batch_max_remaining_tokens.map_or_else(
                || processor.max_remaining_tokens(), |t| t - 1
            ));

            let batch_tokens = <B>::count_tokens(
                processor.entries().iter().map(
                    |(_, e)| e.input_length + e.generated_tokens as usize
                ),
                batch_size,
            );

            metrics::gauge!("tgi_batch_current_size", batch_size as f64);
            metrics::gauge!("tgi_batch_input_tokens", batch_tokens as f64);
            metrics::gauge!("tgi_batch_max_remaining_tokens", batch_max_remaining_tokens.unwrap() as f64);

            // Don't interfere with current batch if it's about to complete
            if batch_max_remaining_tokens.unwrap() >= 2 &&
                next_prefill_after.map_or(true, |t| Instant::now() > t) {
                // Determine min num of requests for add-on batch based on current batch size and
                // tokens since last prefill
                let min_size = if batch_size <= 1 || waiting_tokens >= max_waiting_tokens {
                    1
                } else {
                    max(1, (batch_size * (max_waiting_tokens - waiting_tokens)) / max_waiting_tokens)
                };

                // Try to get a new batch
                if let Some(new_batch) = queue.try_next_batch(processor.entries(), min_size) {
                    info!(
                        "DEBUG: Pulled batch of {} extra request(s) from queue: {:?}",
                        new_batch.requests.len(),
                        new_batch.requests.iter().map(|r| r.id).collect::<Vec<u64>>()
                    );

                    // Determine whether existing batch needs pruning
                    let to_prune = match &batches[0].status {
                        Some(rs) if rs.completed_ids.is_empty() => vec![],
                        _ => batches.clone(),
                    };

                    // Generate one token for this new batch to have the attention past in cache
                    let first_new_id = new_batch.requests.first()
                        .expect("Batch can't be empty here").id;
                    let (new_cached_batch, prefill_time) = processor.prefill(
                        &mut client, new_batch, to_prune, Some(first_new_id), &mut queue
                    ).await;

                    // Hack for now - update existing batch based on pruning that would have been done
                    match batches[0].status.as_mut() {
                        Some(rs) => rs.completed_ids.clear(),
                        None => batches.clear(),
                    };

                    // Reset waiting counter and batch_remaining_tokens
                    waiting_tokens = 1;
                    batch_max_remaining_tokens = None;
                    // Ensure we wait at least half as long as the last prefill took
                    // before we do another prefill (unless the entire batch completes by then)
                    next_prefill_after = Some(Instant::now().add(prefill_time / 2));
                    // Extend current batch with the new batch
                    if let Some(new_batch) = new_cached_batch {
                        let new_batch_id = new_batch.batch_id;
                        batches.push(new_batch);
                        let new_batch_size = processor.entries().len();
                        let added_batch_size = new_batch_size - batch_size;
                        let combined_batch_id;
                        if batch_size > 0 {
                            combined_batch_id = batch_id;
                            if added_batch_size > 0 {
                                info!("Extending batch #{} of {} with additional batch #{} of {}",
                                batch_id, batch_size, new_batch_id, added_batch_size);
                                metrics::increment_counter!("tgi_batch_concatenation_count");
                            }
                        } else {
                            combined_batch_id = new_batch_id;
                            if new_batch_size > 0 {
                                info!("Replacing completed batch #{} with new batch #{} of {}",
                                batch_id, new_batch_id, new_batch_size);
                            }
                        }
                        if added_batch_size > 0 {
                            log_new_batch(combined_batch_id, processor.entries());
                        }
                    } else if batches.is_empty() {
                        // All batches completed or failed, fetch a new one
                        break
                    }
                } else {
                    next_prefill_after = None;
                }
            }

            (cached_batch, _) = processor.next_token(&mut client, batches, &mut queue).await;
            waiting_tokens += 1;
            // Reset batch_remaining_tokens if any requests in the batch completed
            if batch_max_remaining_tokens.is_some() && some_completed(&cached_batch) {
                batch_max_remaining_tokens = None;
            }
        }

        metrics::gauge!("tgi_batch_current_size", 0.0);
        metrics::gauge!("tgi_batch_input_tokens", 0.0);
        metrics::gauge!("tgi_batch_max_remaining_tokens", 0.0);
    }

    info!("Batching loop exiting");
}


fn log_new_batch(id: u64, entries: &IntMap<u64, Entry>) {
    let bs = entries.len();
    if bs != 0 {
        //TODO improve what's printed here
        let total_toks = entries.iter().map(|(_, e)| e.input_length).sum::<usize>();
        let max_new_toks = entries.iter().map(
            |(_, e)| e.request.parameters.max_new_tokens - e.generated_tokens
        ).max().unwrap();
        info!["New or updated batch #{} of size {} ({} total toks), max new toks = {}",
                        id, bs, total_toks, max_new_toks];
    }
}

fn some_completed(batch: &Option<CachedBatch>) -> bool {
    batch.as_ref().map_or(
        true,|b| b.status.as_ref().map_or(
            true, |s| !s.completed_ids.is_empty()
        )
    )
}

struct TokenProcessor<'a> {
    entries: IntMap<u64, Entry>,
    decoder: &'a Decoder,
    generation_health: Arc<AtomicBool>,
}

impl<'a> TokenProcessor<'a> {
    /// Mutably borrow the entries map
    fn entries(&mut self) -> &mut IntMap<u64, Entry> {
        &mut self.entries
    }

    /// Max number of tokens to generate before the current batch will complete
    fn max_remaining_tokens(&self) -> u32 {
        self.entries.iter().map(
            |(_, e)| e.request.parameters.max_new_tokens - e.generated_tokens
        ).sum()
    }

    async fn prefill<B: BatchType>(
        &mut self,
        client: &mut ShardedClient,
        batch: Batch,
        to_prune: Vec<CachedBatch>,
        // First request id in this batch if it doesn't comprise all current entries
        start_id: Option<u64>,
        queue: &mut Queue<B>,
    ) -> (Option<CachedBatch>, Duration) {
        let batch_size = batch.requests.len();
        let batch_tokens = batch.total_tokens;
        let start_time = Instant::now();
        metrics::histogram!("tgi_batch_next_tokens", batch_tokens as f64);
        metrics::histogram!(
            "tgi_batch_inference_batch_size", batch_size as f64, "method" => "prefill"
        );
        let (result, prefill_time) = self._wrap_future(
            client.prefill(batch, to_prune), "prefill", start_time, start_id, queue
        ).await;
        info!("Prefill took {prefill_time:?} for {batch_size} inputs, {batch_tokens} total tokens");
        (result, prefill_time)
    }

    async fn next_token<B: BatchType>(
        &mut self, client: &mut ShardedClient, batches: Vec<CachedBatch>, queue: &mut Queue<B>,
    ) -> (Option<CachedBatch>, Duration) {
        metrics::histogram!(
            "tgi_batch_inference_batch_size", self.entries.len() as f64, "method" => "next_token"
        );
        let start_time = Instant::now();
        self._wrap_future(
            client.next_token(batches), "next_token", start_time, None, queue
        ).await
    }

    /// Wrap a future inside a match statement to handle errors and send the response to the Batcher
    async fn _wrap_future<B: BatchType>(
        &mut self,
        future: impl Future<Output = Result<Option<GenerateTokenResponse>, ClientError>>,
        method: &'static str,
        start_time: Instant,
        // First request id in this batch if it doesn't comprise all current entries
        start_id: Option<u64>,
        queue: &mut Queue<B>,
    ) -> (Option<CachedBatch>, Duration) {
        metrics::increment_counter!("tgi_batch_inference_count", "method" => method);

        // We process the shared queue while waiting for the response from the python shard(s)
        let queue_servicer = queue.service_queue().fuse();
        pin_mut!(future, queue_servicer);
        let result = loop {
            select! {
                result = &mut future => break result,
                _ = &mut queue_servicer => (),
            }
        };

        let elapsed = start_time.elapsed();
        let result = match result {
            Ok(
                Some((generated_tokens, input_tokens, errors, next_batch_id, forward_duration))
            ) => {
                let pre_token_process_time = Instant::now();
                self.process_input_tokens(input_tokens);
                let completed_request_ids = self.process_next_tokens(
                    generated_tokens, errors,
                );
                // Update health
                self.generation_health.store(true, Ordering::SeqCst);
                metrics::histogram!(
                    "tgi_batch_inference_duration",
                    elapsed.as_secs_f64(),
                    "method" => method,
                    "makeup" => "single_only", // later will possibly be beam_only or mixed
                );
                metrics::histogram!(
                    "tgi_batch_inference_forward_duration",
                    forward_duration,
                    "method" => method,
                    "makeup" => "single_only", // later will possibly be beam_only or mixed
                );
                metrics::histogram!(
                    "tgi_batch_inference_tokproc_duration",
                    pre_token_process_time.elapsed().as_secs_f64(),
                    "method" => method,
                    "makeup" => "single_only", // later will possibly be beam_only or mixed
                );
                // Probably don't need this additional counter because the duration histogram
                // records a total count
                metrics::increment_counter!("tgi_batch_inference_success", "method" => method);
                Some(CachedBatch{
                    batch_id: next_batch_id,
                    status: completed_request_ids.map(|c| RequestsStatus{completed_ids: c}),
                })
            },
            // No inference was performed, only batch cleanup
            Ok(None) => None,
            // If we have an error, we discard the whole batch
            Err(err) => {
                // Update health
                self.generation_health.store(false, Ordering::SeqCst);
                let reason = match err {
                    ClientError::OutOfMemory() => "oom",
                    ClientError::Connection(_) => "connection",
                    _ => "error"
                };
                metrics::increment_counter!("tgi_batch_inference_failure", "method" => method, "reason" => reason);
                self.send_errors(err, start_id);
                None
            },
        };

        (result, elapsed)
    }

    /// Send errors to the Batcher for all `request_ids`
    fn send_errors(&mut self, error: ClientError, start_id: Option<u64>) {
        self.entries.retain(|id, entry| {
            if matches![start_id, Some(sid) if *id < sid] {
                // Keep entries that weren't in the failed request batch
                return true
            }
            // unwrap_or is valid here as we don't care if the receiver is gone.
            entry.send_final(Err(error.clone())).unwrap_or_default();
            false
        });
    }

    /// If stop reason is StopSequence, second element of returned tuple will be
    /// Some((index into stop seq array, index of first byte of stop seq in entry.output))
    fn check_stopping_criteria(
        entry: &Entry, last_token_id: u32, eos_token_id: u32, new_bytes: Option<usize>,
    ) -> (StopReason, Option<(usize, usize)>) {
        let params = &entry.request.parameters;
        match params.deadline {
            Some(deadline) if Instant::now() > deadline => (TimeLimit, None),
            _ if entry.generated_tokens < params.min_new_tokens => (NotFinished, None),
            _ if last_token_id == eos_token_id => (EosToken, None),
            _ if entry.generated_tokens >= params.max_new_tokens =>
                (if params.max_is_token_limit { TokenLimit } else { MaxTokens }, None),
            _ => if let Some(ss_match) = TokenProcessor::matches_stop_sequence(entry, new_bytes) {
                (StopSequence, Some(ss_match))
            } else {
                (NotFinished, None)
            }
        }
    }

    /// If stop sequence matches, returns tuple
    /// (index into stop seq array, index of first byte of stop seq in entry.output)
    fn matches_stop_sequence(entry: &Entry, new_bytes: Option<usize>) -> Option<(usize, usize)> {
        new_bytes.map(|new_bytes_count| {
            // We compare byte subslices to avoid utf8 boundary problem
            let output = entry.output.as_ref().unwrap().output().as_bytes();
            let next_off = (output.len() + 1) - new_bytes_count;
            entry.request.parameters.stop_seqs.iter()
                .map(|ss| (ss.as_bytes(), ss.len(), next_off.saturating_sub(ss.len()))).enumerate()
                .find_map(|(ss_idx, (ss, len, start_off))| output[start_off..]
                    .windows(len).rposition(|w| w == ss)
                    .map(|pos| (ss_idx, start_off + pos))
                )
        }).flatten()
    }

    /// Add returned input tokens to their corresponding entries
    fn process_input_tokens(&mut self, inputs: Vec<InputTokens>) {
        for input in inputs.into_iter() {
            let request_id = input.request_id;
            let e = self.entries.get_mut(&request_id)
                .expect("ID not found. This is a bug.");
            // This should be before any generated tokens are processed
            assert_eq!(e.generated_tokens, 0);

            if let Some(stream) = e.stream_tx.as_ref() {
                // In progress stream, send individual token response
                let response = InferResponse::stream_input_info(
                    input.tokens, request_id
                );
                stream.send(Ok(response)).unwrap_or_default();
            } else {
                e.input_tokens = input.tokens;
            }
        }
    }

    /// Store next token for each sequence, evaluate stopping criteria,
    /// send output back for streaming or completed requests
    fn process_next_tokens(
        &mut self, outputs: Vec<Token>, errors: Vec<GenerateError>,
    ) -> Option<Vec<u64>> {
        let mut completed_ids = vec![];
        let request_count = outputs.len();
        for output in outputs.into_iter() {
            let request_id = output.request_id;
            let next_token_id = output.token_id;

            let e = self.entries.get_mut(&request_id)
                .expect("ID not found. This is a bug.");

            let is_stream = e.stream_tx.is_some();
            let stop_seqs = &e.request.parameters.stop_seqs;

            if e.generated_tokens == 0 && !stop_seqs.is_empty() {
                let hold_back_bytes = match is_stream {
                    // No need to hold back bytes if we aren't streaming
                    false => 0,
                    // Ensure at least one token is held back so that its text can be trimmed
                    _ if e.request.parameters.include_stop_seq => 1,
                    // If stop sequences aren't to be output then we need to hold back at least
                    // the number of bytes that comprise the longest one
                    _ => stop_seqs.iter().map(|ss| ss.len()).max().unwrap(),
                };
                e.output = Some(IncrementalDecoderWrapper::for_decoder(
                    &self.decoder, self.decoder.seq2seq, hold_back_bytes,
                ));
            }

            e.generated_tokens += 1;
            let token = match is_stream {
                true => Some(output),
                false => {
                    // Only accumulate token vecs in the entry if this is a non-streaming request
                    // (otherwise they're sent immediately)
                    e.token_ids.push(next_token_id);
                    if e.request.parameters.include_gen_tokens {
                        e.tokens.push(output);
                    }
                    None
                }
            };

            let mut text = None;
            let mut bytes_added = None;
            if let Some(idecoder) = &mut e.output {
                // We only do the token decoding at this stage if stop_sequence(s) are provided,
                // otherwise it can be deferred to run in per-response tasks rather than
                // the main batching loop
                match idecoder.next(next_token_id, self.decoder) {
                    Ok((decoded, added)) => {
                        text = Some(decoded);
                        bytes_added = Some(added);
                    },
                    Err(err) => {
                        // Decoding error, abort the request
                        e.send_final(Err(ClientError::Generation(err.to_string())))
                            .unwrap_or_default();
                        self.entries.remove(&request_id).unwrap();
                        info!("DEBUG: Completed req id {request_id} with reason {Error:?}");
                        completed_ids.push(request_id);
                        continue
                    },
                }
            }

            // Evaluate stopping criteria
            let (mut stop_reason, stop_seq_match) = TokenProcessor::check_stopping_criteria(
                e, next_token_id, self.decoder.eos_token_id, bytes_added
            );

            if stop_reason != NotFinished {
                // Stop criteria met, send final response for both streaming and unary cases
                let mut e = self.entries.remove(&request_id).unwrap();

                // Handle stop sequence if we are stopping due to one
                let mut stop_sequence = None;
                let mut truncate_to = None;
                if let Some((ss_index, ss_byte_offset)) = stop_seq_match {
                    // assert stop_reason == StopSequence
                    let stop_seq = e.request.parameters.stop_seqs.get(ss_index).unwrap();
                    stop_sequence = Some(stop_seq.clone());
                    truncate_to = match e.request.parameters.include_stop_seq {
                        true => Some(ss_byte_offset + stop_seq.len()),
                        false => Some(ss_byte_offset),
                    };
                }

                // Flush the output if we are doing incremental decoding
                let mut decode_err = None;
                if let Some(t) = text.as_mut() {
                    if let Err(err) = e.output.as_mut().unwrap()
                        .flush(truncate_to, self.decoder).map(|s| t.push_str(&s)) {
                        decode_err = Some(err);
                    }
                }
                let response = match decode_err {
                    Some(err) => Err(ClientError::Generation(err.to_string())),
                    _ if is_stream => Ok(InferResponse::stream_final(
                        token.unwrap(), text, &e, request_id, stop_reason, stop_sequence
                    )),
                    _ => Ok(InferResponse::unary(
                        &mut e, request_id, self.decoder.seq2seq, stop_reason, stop_sequence
                    )),
                };
                // unwrap_or is valid here as we don't care if the receiver is gone.
                e.send_final(response).unwrap_or_default();

            } else if is_stream {
                // In progress stream, send individual token response
                let response = InferResponse::stream_inprog(
                    token.unwrap(), e.generated_tokens, text, request_id
                );
                if e.stream_tx.as_ref().unwrap().send(Ok(response)).is_err() {
                    // If receiver closed (request cancelled), cancel this entry
                    let e = self.entries.remove(&request_id).unwrap();
                    stop_reason = Cancelled;
                    metrics::increment_counter!("tgi_request_failure", "err" => "cancelled");
                    //TODO include request context in log message
                    warn!("Aborted streaming request {request_id} cancelled by client \
                        after generating {} token(s)", e.generated_tokens);
                }
            }

            // Only check non-streaming response channel every 16 tokens to avoid repeated atomic access
            else if e.generated_tokens % 16 == 0 && e.response_tx.as_ref().unwrap().is_closed() {
                // If receiver closed (request cancelled), cancel this entry
                let e = self.entries.remove(&request_id).unwrap();
                stop_reason = Cancelled;
                metrics::increment_counter!("tgi_request_failure", "err" => "cancelled");
                //TODO include request context in log message
                warn!("Aborted request {request_id} cancelled by client \
                    after generating {} token(s)", e.generated_tokens);
            }

            if stop_reason != NotFinished {
                debug!("Completed req id {request_id} with reason {stop_reason:?}");
                completed_ids.push(request_id);
            }
        }

        // Process any errors
        for error in errors.into_iter() {
            let request_id = error.request_id;

            let e = self.entries.get_mut(&request_id)
                .expect("ID not found. This is a bug.");

                // Abort the request
                // TODO maybe send Ok result with Error stop reason instead,
                // so that any tokens already generated will be included in unary case
                let message = match e.generated_tokens {
                    0 => error.message.clone(),
                    n => format!["Error after generating {} tokens: {}", n, error.message],
                };
                e.send_final(Err(ClientError::Generation(message))).unwrap_or_default();
                self.entries.remove(&request_id).unwrap();
                info!("DEBUG: Completed req id {request_id} with reason {Error:?}: {}", error.message);
                completed_ids.push(request_id);
        }

        // Return None if all requests in this batch have completed, otherwise the list of completed ids
        if completed_ids.len() == request_count { None } else { Some(completed_ids) }
    }
}

#[derive(Debug)]
pub(crate) struct Times {
    // Queue start time
    pub(crate) queued: Instant,
    // Generation start time
    pub(crate) start: Instant,
    // Generation end time
    pub(crate) end: Instant,
}

impl From<&Entry> for Times {
    fn from(entry: &Entry) -> Self {
        Self{
            queued: entry.queue_time, start: entry.batch_time.unwrap(), end: Instant::now(),
        }
    }
}

/// This enum initially contains a vec of Token structs
/// received from the shards and containing token ids.
/// It is decoded to a vec of TokenInfo structs containing
/// the token strings, which is sent in the external gRPC response.
#[derive(Debug)]
pub(crate) enum TokenInfos {
    WithIds(Vec<Token>),
    WithStrings(Vec<TokenInfo>)
}

impl Default for TokenInfos {
    fn default() -> Self {
        WithIds(vec![])
    }
}

impl TokenInfos {
    fn clear(&mut self) {
        match self {
            WithStrings(tis) => tis.clear(),
            WithIds(tis) => tis.clear(),
        }
    }
    fn is_empty(&self) -> bool {
        match self {
            WithStrings(tis) => tis.is_empty(),
            WithIds(tis) => tis.is_empty(),
        }
    }
    pub(crate) fn to_final_vec(self) -> Vec<TokenInfo> {
        match self {
            WithStrings(tis) => tis,
            _ => vec![],
        }
    }
    fn decode(&mut self, decoder: &Decoder) {
        if let WithIds(toks) = &self {
            *self = WithStrings(toks.iter()
                .map(|t| TokenInfos::decode_token_info(t, decoder))
                .collect());
        }
    }
    fn decode_token_info(with_ids: &Token, decoder: &Decoder) -> TokenInfo {
        TokenInfo{
            text: decoder.id_to_token(with_ids.token_id),
            logprob: with_ids.logprob,
            rank: with_ids.rank,
            top_tokens: with_ids.top_tokens.iter().map(|tt| TopToken{
                text: decoder.id_to_token(tt.token_id),
                logprob: tt.logprob,
            }).collect(),
        }
    }
}


#[derive(Debug, Default)]
pub(crate) struct InferResponse {
    pub(crate) output_text: String,
    /// whether or not the token ids have been decoded yet
    pub(crate) is_decoded: bool,
    /// Total generated tokens so far
    pub(crate) gen_token_count: u32,
    // Set/used only for unary responses
    pub(crate) token_ids: Vec<u32>,
    // This will be max length 1 in streaming case
    // Only set in unary case if extra token info is requested
    pub(crate) tokens: TokenInfos,
    pub(crate) in_tokens: TokenInfos,
    pub(crate) reason: StopReason,
    // Stop sequence encountered iff reason == StopSequence
    pub(crate) stop_sequence: Option<String>,
    pub(crate) in_token_count: u32,
    pub(crate) times: Option<Times>,
    pub(crate) request_id: Option<u64>,
    /// Random seed used, only applicable to sampling
    pub(crate) seed: u64,
}

impl InferResponse {
    /// A dedicated message is sent with the input token info, if requested
    fn stream_input_info(in_tokens: Vec<Token>, request_id: u64) -> Self {
        Self {
            in_token_count: in_tokens.len() as u32,
            in_tokens: WithIds(in_tokens),
            is_decoded: true,
            request_id: Some(request_id),
            ..Default::default()
        }
    }
    /// Response message for in-progress stream
    fn stream_inprog(token: Token, count: u32, text: Option<String>, request_id: u64) -> Self {
        Self {
            is_decoded: text.is_some(),
            output_text: text.unwrap_or_default(),
            gen_token_count: count,
            tokens: WithIds(vec![token]),
            request_id: Some(request_id),
            ..Default::default()
        }
    }
    /// Final stream response message
    fn stream_final(
        token: Token,
        text: Option<String>,
        entry: &Entry,
        request_id: u64,
        stop_reason: StopReason,
        stop_sequence: Option<String>,
    ) -> Self {
        Self {
            is_decoded: text.is_some(),
            output_text: text.unwrap_or_default(),
            gen_token_count: entry.generated_tokens,
            tokens: WithIds(vec![token]),
            reason: stop_reason,
            stop_sequence,
            times: Some(entry.into()),
            request_id: Some(request_id),
            seed: entry.request.parameters.seed.unwrap_or_default(),
            ..Default::default()
        }
    }
    /// Unary response message
    fn unary(
        entry: &mut Entry,
        request_id: u64,
        seq2seq: bool,
        stop_reason: StopReason,
        stop_sequence: Option<String>,
    ) -> Self {
        let mut text = String::new();
        if entry.request.parameters.include_input_text {
            text += &*entry.request.inputs;
            if seq2seq {
                text += "\n\n";
            }
        }
        let is_decoded;
        if let Some(out_decoder) = take(&mut entry.output) {
            is_decoded = true;
            if text.is_empty() {
                text = out_decoder.into_string();
            } else {
                text.push_str(out_decoder.output())
            }
        } else {
            is_decoded = false;
        }
        Self {
            output_text: text,
            is_decoded,
            gen_token_count: entry.generated_tokens,
            token_ids: take(&mut entry.token_ids),
            tokens: WithIds(take(&mut entry.tokens)),
            in_tokens: WithIds(take(&mut entry.input_tokens)),
            reason: stop_reason,
            stop_sequence,
            times: Some((&*entry).into()),
            request_id: Some(request_id),
            in_token_count: entry.input_length as u32,
            seed: entry.request.parameters.seed.unwrap_or_default(),
        }
    }
    /// If time limit is expired before generation starts
    pub(crate) fn early_timeout(entry: &Entry) -> Self {
        Self {
            reason: TimeLimit,
            is_decoded: true,
            // We only include input token count in the unary case, since it will have
            // already been sent in the streaming case
            in_token_count: if entry.response_tx.is_some() { entry.input_length as u32 } else { 0 },
            times: Some((&*entry).into()),
            ..Default::default()
        }
    }

    pub(crate) fn decode_output_text(&mut self, decoder: &Decoder) -> Result<(), InferError> {
        if !self.is_decoded {
            self.output_text += &*decoder.decode(
                take(&mut self.token_ids), true, true,
            )?;
            self.is_decoded = true;
        }
        Ok(())
    }

    pub(crate) fn decode_token_infos(&mut self, decoder: &Decoder) {
        self.tokens.decode(decoder);
        self.in_tokens.decode(decoder);
    }

    pub(crate) fn ensure_decoded(
        mut self, decoder: &Decoder
    ) -> Result<InferResponse, InferError> {
        self.decode_token_infos(decoder);
        self.decode_output_text(decoder).map(|_| self)
    }
}

#[derive(Debug, Error, Clone)]
pub enum InferError {
    #[error("Request failed during generation: {0}")]
    GenerationError(String),
    #[error("Request failed during detokenization: {0}")]
    DetokenizationError(String),
    #[error("Server too busy")]
    RequestQueueFull(),
}

/// Convert to Axum supported format
impl From<InferError> for (StatusCode, Json<ErrorResponse>) {
    fn from(err: InferError) -> Self {
        match err {
            _ => (
                StatusCode::FAILED_DEPENDENCY,
                Json(ErrorResponse {
                    error: err.to_string(),
                }),
            ),
        }
    }
}