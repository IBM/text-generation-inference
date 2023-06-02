use crate::{GenerateParameters, GenerateRequest};
use std::collections::{BTreeSet, VecDeque};
use std::marker::PhantomData;
use std::mem::take;
use std::ops::Add;
use std::time::Duration;
use nohash_hasher::IntMap;
use tokio::sync::mpsc::{Receiver, UnboundedSender};
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

// Time after which a waiting request will be admitted into a batch
// immediately regardless of counts of in-progress/waiting request counts
pub(crate) const MAX_WAITING_DURATION: Duration = Duration::from_millis(1500);

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
    /// Maximum weight of individual prefill batches
    pub(crate) prefill_weight_limit: usize,
}

/// Request Queue
#[derive(Debug)]
pub(crate) struct Queue<B: BatchType> {
    /// Batching config
    config: BatchingConfig,
    /// Just for type inference
    batch_type: PhantomData<B>,

    receiver: Receiver<Vec<Entry>>,
    // Staging buffer, filled until max_size is reached
    buffer: VecDeque<Entry>,
    /// Id of the next entry
    next_id: u64,
    /// Id of the next batch
    next_batch_id: u64,
    // Remembered size of the last batch, used to determine
    // when entries have completed between calls to the
    // next_batch function
    last_seen_batch_size: usize,
    // Index in the queue up to which entries have been
    // checked to see if they can fit into the current batch.
    // Reset to zero when any existing entries complete
    checked_request_count: usize,
    /// true if it's known that the size of the requests in the buffer
    /// is to small to prefill an add-on batch
    pub(crate) buffer_contents_insufficient: bool,

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
            last_seen_batch_size: 0,
            checked_request_count: 0,
            buffer_contents_insufficient: false,
            batch_type: PhantomData,
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
                    Some(ents) => {
                        self.buffer.extend(ents);
                        self.buffer_contents_insufficient = false;
                    },
                    // Queue closed, we must be shutting down
                    None => return None,
                }
            }
            // We have at least one entry in the buffer, try to fill it further up to max_size
            if let Some(batch) = self.try_next_batch(entries) {
                return Some(batch)
            }
        }
    }

    /// Attempt to add a single batch of requests from the queue without waiting
    /// Returns false if there are none
    fn try_dequeue_some(&mut self) -> bool {
        match self.receiver.try_recv() {
            Ok(ents) => {
                self.buffer.extend(ents);
                self.buffer_contents_insufficient = false;
                true
            },
            // Err(TryRecvError::Empty) => break,
            // Err(TryRecvError::Disconnected) => break, //TODO TBD
            _ => false
        }
    }

    /// Get the next batch without blocking
    /// Corresponding entries are added to the entries map
    pub(crate) fn try_next_batch(&mut self, entries: &mut IntMap<u64, Entry>) -> Option<Batch> {
        let mut total_count = entries.len();
        if total_count >= self.config.size_limit {
            // We are already at max batch size
            return None
        }

        if total_count != self.last_seen_batch_size {
            // Reset the count of checked requests if any completed since last check
            self.checked_request_count = 0;
            self.last_seen_batch_size = total_count
        }

        // Filter timed-out or cancelled entries from the front of the queue,
        // so that next-entry waiting time is accurate
        while let Some(entry) = self.buffer.front() {
            if entry.is_cancelled() {
                self.buffer.pop_front();
            } else if entry.deadline_exceeded() {
                let mut entry = self.buffer.pop_front().unwrap();
                // Send timeout response
                entry.batch_time = Some(Instant::now());
                entry.send_final(Ok(InferResponse::early_timeout(&entry)))
                    .unwrap_or_default();
            } else {
                break
            }
            // Reset the count of checked requests if any were cancelled since last check
            self.checked_request_count = 0
        }

        // This will generally be zero, but if no requests have been completed
        // since last time, we don't need to reconsider those already checked
        let mut checked_up_to_index = self.checked_request_count;

        if !entries.is_empty() {
            // If we don't have any new requests in the buffer to check
            if self.buffer.len() <= checked_up_to_index || (
                // Or the current buffer isn't large enough to satisfy the min prefill requirement
                self.buffer_contents_insufficient && !self.next_entry_waiting_too_long()) {
                // Try to pull new requests from the queue
                // If there aren't any then there's nothing to be done
                if !self.try_dequeue_some() {
                    return None
                }
            }
        }

        // Indices into buffer of entries chosen to add to next batch
        let mut chosen_indices = vec![];
        // Indices to drop due to client cancellation
        let mut indices_to_drop = vec![];
        let mut btree = None;
        let mut time_cutoff = None;
        let mut hit_prefill_weight_limit = false;

        let now = Instant::now();
        let mut batch_stats = <B as BatchType>::compute_stats(entries);
        let mut prefill_stats = <B as BatchType>::compute_stats(&self.empty_map);
        // We first do a read-only pass over the queue to allow skipping over large entries
        // that don't fit in the current batch to reach smaller entries that do
        let mut buffer_index = checked_up_to_index;
        'outer: loop {
            'inner: for entry in self.buffer.range(buffer_index..) {
                let config = &self.config;
                if matches!(time_cutoff, Some(t) if entry.queue_time > t) {
                    break 'outer
                }
                buffer_index += 1;
                if now - entry.queue_time > Duration::from_millis(500)
                    && (entry.deadline_exceeded() || entry.is_cancelled()) {
                    // Eject cancelled or expired entry from queue
                    indices_to_drop.push(buffer_index - 1);
                    continue
                }

                // This is the index into the queue after cancelled entries
                // have been pruned
                checked_up_to_index += 1;

                let input_len = entry.input_length;
                let output_len = entry.request.parameters.max_new_tokens as usize;
                let next_stats = <B as BatchType>::update_stats(
                    &batch_stats, input_len, output_len
                );

                // Avoid more granular analysis if possible
                if <B as BatchType>::batch_weight(
                    &batch_stats, total_count + 1
                ) > config.weight_limit {
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
                    if <B as BatchType>::exceeds_weight(
                        tree, config.weight_limit, output_len,
                    ) {
                        // Remove our tuple from the set
                        tree.remove(&(output_len, input_len, tree.len() - 1));
                        time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                        continue 'inner
                    }
                } else if let Some(tree) = btree.as_mut() {
                    // If we initialized the btree for a prior request, keep it updated
                    tree.insert((output_len, input_len, tree.len()));
                }
                // Here, we can add this request to the batch without breaching memory limit

                // Also check whether adding this request will make the batch of new requests
                // too expensive latency-wise to perform in a single forward-pass.
                if config.prefill_weight_limit > 0 {
                    let next_prefill_stats = <B as BatchType>::update_stats(
                        &prefill_stats, input_len, 0
                    );
                    let prefill_weight = <B as BatchType>::prefill_weight(
                        &next_prefill_stats, chosen_indices.len() + 1
                    );
                    if prefill_weight > config.prefill_weight_limit {
                        if let Some(tree) = btree.as_mut() {
                            // Remove our tuple from the set
                            tree.remove(&(output_len, input_len, tree.len() - 1));
                            hit_prefill_weight_limit = true;
                        }
                        time_cutoff.get_or_insert_with(|| entry.queue_time.add(CUTOFF_DURATION));
                        continue 'inner
                    }
                    prefill_stats = next_prefill_stats;
                }

                batch_stats = next_stats;

                chosen_indices.push(checked_up_to_index - 1);
                total_count += 1;
                if total_count >= config.size_limit {
                    break 'outer
                }
            }

            if !self.try_dequeue_some() {
                break 'outer
            }
        }

        // Drop any cancelled requests
        if !indices_to_drop.is_empty() {
            info!(
                "Pruning {} cancelled requests from queue before building batch",
                indices_to_drop.len()
            );
            // Iterate in reverse so that we don't shift the indices
            indices_to_drop.iter().rev().for_each(|i| {
                let mut entry = self.buffer.remove(*i).unwrap();
                if entry.deadline_exceeded() && !entry.is_cancelled() {
                    // Send timeout response
                    entry.batch_time = Some(Instant::now());
                    entry.send_final(Ok(InferResponse::early_timeout(&entry)))
                        .unwrap_or_default();
                }
            });
        }

        let chosen_count = chosen_indices.len();
        let buffer_size = self.buffer.len();
        if buffer_size > 0 {
            info!("Chose {chosen_count} out of {buffer_size} requests from buffer");
        }
        if chosen_count == 0 {
            // This gets reset to zero when any requests in the existing batch are removed
            self.checked_request_count = checked_up_to_index;
            return None
        }
        self.checked_request_count = 0;

        if !hit_prefill_weight_limit && !entries.is_empty() {
            // If this is to be added to an existing batch, ensure it meets urgency or size
            // requirements to avoid too frequent prefills
            if !self.next_entry_waiting_too_long() {
                if <B as BatchType>::batch_weight(&batch_stats, total_count) <
                    self.config.weight_limit / 2 {

                    // Don't add this new batch yet because it's not large enough
                    info!("Aborting new batch addition because it is not large enough");
                    self.checked_request_count = checked_up_to_index;
                    self.buffer_contents_insufficient = true;
                    return None
                }
            }
        }

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
            // Insert into entries IntMap
            entries.insert(id, entry);
            request
        }).collect::<Vec<Request>>();

        let batch = Batch { id: self.next_batch_id, requests };
        // Increment batch id
        self.next_batch_id += 1;
        self.buffer_contents_insufficient = false;
        Some(batch)
    }

    /// Returns true if the entry at the front of the queue has been waiting for longer
    /// than MAX_WAITING_DURATION
    pub(crate) fn next_entry_waiting_too_long(&self) -> bool {
        matches!(
            self.buffer.front(), Some(e) if e.queue_time.elapsed() > MAX_WAITING_DURATION
        )
    }
}

impl From<&GenerateParameters> for NextTokenChooserParameters {
    fn from(parameters: &GenerateParameters) -> Self {
        Self {
            temperature: parameters.temperature,
            top_k: parameters.top_k as u32,
            top_p: parameters.top_p,
            min_new_tokens: parameters.min_new_tokens,
            seed: parameters.seed,
            repetition_penalty: match parameters.repetition_penalty {
                x if x == 1.0 || x == 0.0 => None,
                theta => Some(theta),
            },
            length_penalty: parameters.length_penalty
                .map(|lp| LengthPenalty {
                    start_index: lp.0,
                    decay_factor: lp.1,
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