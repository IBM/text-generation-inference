use crate::{GenerateParameters, GenerateRequest};
use std::collections::{BTreeSet, VecDeque};
use std::mem::take;
use std::ops::Add;
use std::time::Duration;
use nohash_hasher::IntMap;
use tokio::sync::mpsc::{Receiver, UnboundedSender};
use tokio::sync::mpsc::error::TryRecvError::{Disconnected, Empty};
use text_generation_client::{
    Batch, ClientError, LengthPenalty, NextTokenChooserParameters, Request, RequestedDetails, Token
};
use tokio::sync::oneshot::Sender;
use tokio::time::Instant;
use tracing::info;
use crate::batch_types::BatchType;
use crate::batcher::InferResponse;
use crate::decoder::IncrementalDecoderWrapper;

// Requests that fit into the next batch can overtake others
// that don't as long as they arrive within this amount of time after
const CUTOFF_DURATION: Duration = Duration::from_secs(1);


/// Queue entry / in-progress request state
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: GenerateRequest,
    /// Response senders to communicate between the Batcher and the batching_task
    /// Exactly one of these will be non-None
    pub response_tx: Option<Sender<Result<InferResponse, ClientError>>>,
    pub stream_tx: Option<UnboundedSender<Result<InferResponse, ClientError>>>,
    /// Number of tokens in the input
    pub input_length: usize,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch (queue end time)
    pub batch_time: Option<Instant>,
    /// Generated token ids, populated only in non-streaming case
    pub token_ids: Vec<u32>,
    /// Generated tokens
    pub tokens: Vec<Token>,
    /// Input tokens
    pub input_tokens: Vec<Token>,
    /// Accumulates output, used only when stop sequences are provided
    pub output: Option<IncrementalDecoderWrapper>,
    /// Generated token count
    pub generated_tokens: u32,
}

impl Entry {
    pub(crate) fn new(
        request: GenerateRequest,
        input_length: usize,
        response_tx: Option<Sender<Result<InferResponse, ClientError>>>,
        stream_tx: Option<UnboundedSender<Result<InferResponse, ClientError>>>,
    ) -> Self {
        Self {
            request,
            response_tx,
            stream_tx,
            input_length,
            input_tokens: vec![],
            queue_time: Instant::now(),
            batch_time: None,
            token_ids: vec![],
            tokens: vec![],
            output: None,
            generated_tokens: 0,
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        if self.response_tx.is_some() {
            self.response_tx.as_ref().unwrap().is_closed()
        } else {
            self.stream_tx.as_ref().unwrap().is_closed()
        }
    }

    pub(crate) fn deadline_exceeded(&self) -> bool {
        matches![self.request.parameters.deadline, Some(d) if d < Instant::now()]
    }

    // Convenience method for sending a terminating response
    pub(crate) fn send_final(
        &mut self, result: Result<InferResponse, ClientError>
    ) -> Result<(), Result<InferResponse, ClientError>> {
        if self.response_tx.is_some() {
            let rtx = take( &mut self.response_tx );
            rtx.unwrap().send(result)
        } else {
            self.stream_tx.as_mut().unwrap().send(result).map_err(|s| s.0)
        }
    }
}


#[derive(Debug)]
pub(crate) struct BatchingConfig {
    /// Upper bound on number of requests in a batch
    pub(crate) size_limit: usize,
    /// Maximum batch "weight" at any point of time (takes sequence lengths into account)
    pub(crate) weight_limit: usize,
    /// Maximum percentage of pad tokens in prefill batches. In range [0, 1]
    pub(crate) prefill_padding_limit: f32,
}

/// Request Queue
#[derive(Debug)]
pub(crate) struct Queue<B: BatchType> {
    /// Batching config
    config: BatchingConfig,
    /// Batch type
    batch_type: B,

    receiver: Receiver<Vec<Entry>>,
    // Staging buffer, filled until max_size is reached
    buffer: VecDeque<Entry>,
    /// Id of the next entry
    next_id: u64,
    /// Id of the next batch
    next_batch_id: u64,

    // Keep track what was logged in the last call to try_next_batch
    // so as to avoid many repeating entries in the log
    last_logged: Option<(usize, usize)>,

    /// Just a constant empty map to reuse
    empty_map: IntMap<u64, Entry>,
}

impl<B: BatchType> Queue<B> {
    pub(crate) fn new(
        config: BatchingConfig, _batch_type: B, receiver: Receiver<Vec<Entry>>
    ) -> Self {
        Self {
            config,
            receiver,
            buffer: VecDeque::new(),
            next_id: 0,
            next_batch_id: 1,
            batch_type: _batch_type,
            last_logged: None,
            empty_map: IntMap::default(),
        }
    }

    /// Get the next batch, blocking until available
    /// Corresponding entries are added to the entries map
    /// Returns None only if the queue has been closed
    pub(crate) async fn next_batch(&mut self, entries: &mut IntMap<u64, Entry>) -> Option<Batch> {
        loop {
            if self.buffer.is_empty() {
                // Await on the queue while the buffer is empty
                match self.receiver.recv().await {
                    Some(ents) => self.add_to_buffer(ents),
                    // Queue closed, we must be shutting down
                    None => return None,
                }
                loop {
                    match self.receiver.try_recv() {
                        Ok(ents) => self.add_to_buffer(ents),
                        Err(Empty) => break,
                        Err(Disconnected) => return None,
                    }
                }
            }
            // We have at least one entry in the buffer
            if let Some(batch) = self.try_next_batch(entries, 1) {
                return Some(batch)
            }
        }
    }

    /// Returns a future that can be awaited to consume requests from the queue's
    /// shared channel into it's internal buffer. The future never completes.
    pub(crate) async fn service_queue(&mut self) {
        // First prune existing cancelled or expired requests
        let mut pruned = false;
        self.buffer.retain_mut(|entry| match entry {
            entry if entry.is_cancelled() => {
                metrics::increment_counter!("tgi_request_failure", "err" => "cancelled");
                pruned = true;
                false
            },
            entry if entry.deadline_exceeded() => {
                // Send timeout response
                metrics::increment_counter!("tgi_request_failure", "err" => "timeout");
                entry.batch_time = Some(Instant::now());
                entry.send_final(Ok(InferResponse::early_timeout(&entry)))
                    .unwrap_or_default();
                pruned = true;
                false
            },
            _ => true,
        });

        if pruned {
            metrics::gauge!("tgi_queue_size", self.buffer.len() as f64);
        }

        while let Some(ents) = self.receiver.recv().await {
            self.add_to_buffer(ents);
        }
    }

    fn add_to_buffer(&mut self, new_entries: Vec<Entry>) {
        self.buffer.extend(new_entries);
        metrics::gauge!("tgi_queue_size", self.buffer.len() as f64);
    }

    /// Get the next batch without blocking.
    /// Corresponding entries are added to the entries map
    pub(crate) fn try_next_batch(
        &mut self, entries: &mut IntMap<u64, Entry>, min_size: usize,
    ) -> Option<Batch> {

        let buffer_size = self.buffer.len();
        if buffer_size < min_size {
            // Not enough requests waiting to reach min_size
            self.last_logged = None;
            return None
        }

        let mut total_count = entries.len();
        if total_count + min_size > self.config.size_limit {
            // Not enough space to fit min_size within max batch size
            self.last_logged = None;
            return None
        }

        // Indices into buffer of entries chosen to add to next batch
        let mut chosen_indices = vec![];
        let mut btree = None;
        let mut time_cutoff = None;

        let now = Instant::now();
        let mut batch_stats = <B>::compute_stats(entries);
        let mut prefill_stats = <B>::compute_stats(&self.empty_map);

        // Compute the effective prefill weight limit, taking into account space already consumed
        // by the in-progress batch
        let effective_prefill_weight_limit = match self.config.weight_limit {
            prefill_limit if prefill_limit == 0 || total_count == 0 => prefill_limit,
            prefill_limit => {
                let current_batch_weight = self.batch_type.batch_initial_weight(&batch_stats, total_count);
                let pct_space_free = 1.0 - (
                    current_batch_weight as f64 / self.config.weight_limit as f64
                );
                let limit = (pct_space_free * prefill_limit as f64) as usize;
                if limit == 0 {
                    return None
                }
                limit
            },
        };
        let max_prefill_padding = self.config.prefill_padding_limit;

        // We first do a read-only pass over the queue to allow skipping over large entries
        // that don't fit in the current batch to reach smaller entries that do
        for (index, entry) in self.buffer.iter().enumerate() {
            let config = &self.config;
            if matches!(time_cutoff, Some(t) if entry.queue_time > t) {
                break
            }

            let input_len = entry.input_length;
            let output_len = entry.request.parameters.max_new_tokens as usize;
            let next_stats = <B>::update_stats(
                &batch_stats, input_len, output_len
            );

            // Avoid more granular analysis if possible
            if self.batch_type.batch_max_weight(&next_stats, total_count + 1) > config.weight_limit {
                // We aren't sure whether this next request will fit, so populate
                // a btree with the current batch of requests, the set of
                // requests already evaluated, and this one, and perform more
                // granular analysis to verify that the batch shape won't exceed
                // the limit at any point

                // Allocate btree the first time it's required
                let tree = btree.get_or_insert_with(|| {
                    let mut t = Box::new(BTreeSet::new());
                    // Populate with records corresponding to all existing and pending entries
                    let pending = chosen_indices.iter()
                        .map(|i| (&0, self.buffer.get(*i).unwrap()));
                    for (_, e) in entries.iter().chain(pending) {
                        let generated_count = e.generated_tokens as usize;
                        t.insert((
                            e.request.parameters.max_new_tokens as usize - generated_count,
                            e.input_length + generated_count,
                            t.len(),
                        ));
                    }
                    t
                });
                // Add the current entry
                tree.insert((output_len, input_len, tree.len()));

                // Perform analysis
                if self.batch_type.exceeds_weight(tree, config.weight_limit, output_len) {
                    if chosen_indices.len() + buffer_size < min_size + index + 1 {
                        // We don't have enough remaining to meet min_size
                        self.last_logged = None;
                        return None
                    }
                    // Remove our tuple from the set
                    tree.remove(&(output_len, input_len, tree.len() - 1));
                    time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                    continue
                }
                metrics::increment_counter!("tgi_granular_batch_addition");
            } else if let Some(tree) = btree.as_mut() {
                // If we initialized the btree for a prior request, keep it updated
                tree.insert((output_len, input_len, tree.len()));
            }
            // Here, we can add this request to the batch without breaching memory limit
            if time_cutoff.is_some() {
                metrics::increment_counter!("tgi_queue_jump");
            }

            // Also check whether adding this request will breach the prefill weight limit
            if effective_prefill_weight_limit > 0 || max_prefill_padding < 1.0 {
                let next_prefill_stats = <B>::update_stats(
                    &prefill_stats, input_len, 0
                );
                let batch_size = chosen_indices.len() + 1;
                let mut skip = false;
                if effective_prefill_weight_limit > 0 {
                    let prefill_weight = self.batch_type.prefill_weight(
                        &next_prefill_stats, batch_size
                    );
                    if prefill_weight > effective_prefill_weight_limit {
                        skip = true;
                        metrics::increment_counter!("tgi_prefill_weight_limit_exceeded");
                    }
                }
                if !skip && max_prefill_padding < 1.0 {
                    let percentage_padding = <B>::percent_padding(&next_prefill_stats, batch_size);
                    if percentage_padding > max_prefill_padding {
                        skip = true;
                        //TODO if we skip due to padding and added other requests from queue,
                        // we could consider doing another pass since the padding proportion may have decreased
                        metrics::increment_counter!("tgi_prefill_padding_limit_exceeded");
                    }
                }
                if skip {
                    if let Some(tree) = btree.as_mut() {
                        // Remove our tuple from the set
                        tree.remove(&(output_len, input_len, tree.len() - 1));
                    }
                    time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                    metrics::increment_counter!("tgi_prefill_weight_limit_exceeded");
                    continue
                }
                prefill_stats = next_prefill_stats;
            }

            batch_stats = next_stats;

            chosen_indices.push(index);
            total_count += 1;
            if total_count >= config.size_limit {
                break
            }
        }

        let chosen_count = chosen_indices.len();
        if chosen_count == 0 {
            // Don't repeatedly log when no requests were chosen if the current/waiting
            // request counts haven't changed
            let current_counts = Some((buffer_size, total_count));
            if self.last_logged != current_counts {
                self.last_logged = current_counts;
                info!("Chose 0 out of {buffer_size} requests from buffer, total now {total_count}");
            }
            return None
        }

        self.last_logged = None;
        info!("Chose {chosen_count} out of {buffer_size} requests from buffer, \
                total now {total_count}");

        let some_now = Some(now);
        let requests = chosen_indices.iter().enumerate().map(|(i, index)| {
            let mut entry = self.buffer.remove(index - i).expect("bug");
            // Allocate new id
            let id = self.next_id;
            self.next_id += 1;
            let request = Request {
                id,
                prefix_id: entry.request.prefix_id.clone().unwrap_or_default(),
                inputs: entry.request.inputs.clone(),
                input_length: entry.input_length as u32,
                max_output_length: entry.request.parameters.max_new_tokens,
                truncate: entry.request.parameters.truncate_input_tokens > 0,
                parameters: Some((&entry.request.parameters).into()),
                stream_response: entry.stream_tx.is_some(),
                details: (&entry.request.parameters).into(),
            };
            // Set batch_time
            entry.batch_time = some_now;
            metrics::histogram!("tgi_request_queue_duration", (now - entry.queue_time).as_secs_f64());
            // Insert into entries IntMap
            entries.insert(id, entry);
            request
        }).collect::<Vec<Request>>();

        let batch_tokens = <B>::count_tokens(
            requests.iter().map(|r| r.input_length as usize),
            chosen_count,
        );
        metrics::gauge!("tgi_queue_size", self.buffer.len() as f64);
        let batch = Batch { id: self.next_batch_id, requests, total_tokens: batch_tokens as u32 };
        // Increment batch id
        self.next_batch_id += 1;
        Some(batch)
    }
}

impl From<&GenerateParameters> for NextTokenChooserParameters {
    fn from(parameters: &GenerateParameters) -> Self {
        Self {
            temperature: parameters.temperature,
            top_k: parameters.top_k as u32,
            top_p: parameters.top_p,
            typical_p: parameters.typical_p,
            min_new_tokens: parameters.min_new_tokens,
            seed: parameters.seed,
            repetition_penalty: match parameters.repetition_penalty {
                x if x == 1.0 || x == 0.0 => None,
                theta => Some(theta),
            },
            length_penalty: parameters.length_penalty
                .map(|(start_index, decay_factor)| LengthPenalty {
                    start_index, decay_factor
                }),
        }
    }
}

impl From<&GenerateParameters> for Option<RequestedDetails> {
    fn from(parameters: &GenerateParameters) -> Self {
        Some(RequestedDetails {
            input_toks: parameters.include_input_tokens,
            logprobs: parameters.include_logprobs,
            ranks: parameters.include_ranks,
            top_n_toks: parameters.include_top_n,
        })
    }
}
