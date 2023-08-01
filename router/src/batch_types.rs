use std::cmp::max;
use std::collections::BTreeSet;
use nohash_hasher::IntMap;
use num::integer::Roots;
use crate::queue::Entry;

pub(crate) trait BatchType: Send + Sync + Clone + 'static {
    type Stats: Default;

    /// Update batch statistics with an additional request
    fn update_stats(stats: &Self::Stats, input_length: usize, output_length: usize) -> Self::Stats;
    /// Calculate batch weight given batch statistics
    fn batch_weight(stats: &Self::Stats, batch_size: usize) -> usize;
    /// Calculate prefill batch weight given prefill batch statistics
    fn prefill_weight(prefill_stats: &Self::Stats, batch_size: usize) -> usize;
    /// Indicate whether a hypothetical batch will exceed the combined weight limit
    fn exceeds_weight(
        tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool;
    /// Provide a count of tokens for a given batch, including padding tokens if applicable
    fn count_tokens(input_lengths: impl Iterator<Item=usize>, batch_size: usize) -> usize;

    /// max_prefill_weight to use when none is specified
    fn default_max_prefill_weight() -> usize;

    /// Compute batch statistics given map of entries
    fn compute_stats(entries: &IntMap<u64, Entry>) -> Self::Stats {
        entries.iter().fold(
            Self::Stats::default(),
            |stats, (_, entry)| {
                let generated_count = entry.generated_tokens;
                Self::update_stats(
                    &stats,
                    entry.input_length + generated_count as usize,
                    (entry.request.parameters.max_new_tokens - generated_count) as usize,
                )
            }
        )
    }
}

/// Non-padded batch used in flash attention
#[derive(Clone)]
pub(crate) struct FlashBatch {}

impl BatchType for FlashBatch {
    /// Keep track of total number of tokens in the batch
    type Stats = usize;

    fn update_stats(
        total_tokens: &Self::Stats, input_length: usize, output_length: usize
    ) -> Self::Stats {
        total_tokens + input_length + output_length
    }

    fn batch_weight(total_tokens: &Self::Stats, _batch_size: usize) -> usize {
        *total_tokens
    }

    fn prefill_weight(total_tokens: &Self::Stats, _batch_size: usize) -> usize {
        *total_tokens
    }

    fn exceeds_weight(
        tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool {
        let mut in_sum = 0;
        // Work backwards from longest projected entry
        for (batch_size, (out_len, in_len, _)) in tree.iter().rev().enumerate() {
            let this_out_len = *out_len;
            in_sum += *in_len;
            // Only need to check segments with output_len > current_output_len
            // will have been checked in a prior iteration
            if this_out_len <= current_output_len {
                // Check if we breach max space for this segment
                let token_count = in_sum + (batch_size + 1) * this_out_len;
                if token_count > max_total_weight {
                    return true
                }
            }
        }
        false
    }

    fn count_tokens(input_lengths: impl Iterator<Item=usize>, _: usize) -> usize {
        input_lengths.sum()
    }

    fn default_max_prefill_weight() -> usize {
        8192
    }
}

/// Regular rectangular padded
#[derive(Clone)]
pub(crate) struct PaddedBatch {}

impl BatchType for PaddedBatch {
    /// Keep track of maximum input length, maximum output length
    type Stats = (usize, usize);

    fn update_stats(
        max_in_out_lengths: &Self::Stats, input_length: usize, output_length: usize
    ) -> Self::Stats {
        let (max_input_length, max_output_length) = max_in_out_lengths;
        (max(*max_input_length, input_length), max(*max_output_length, output_length))
    }

    fn batch_weight(max_in_out_lengths: &Self::Stats, batch_size: usize) -> usize {
        let (max_input_length, max_output_length) = max_in_out_lengths;
        let max_seq_len = max_input_length + max_output_length;
        // Memory requirement roughly proportional to batch_size * seq_len^2
        batch_size * max_seq_len.pow(2)
    }

    fn prefill_weight(max_in_out_lengths: &Self::Stats, batch_size: usize) -> usize {
        // Empirically, prefill latency is proportional to batch_size * seq_len^(3/2)
        let (max_input_length, _) = max_in_out_lengths;
        batch_size * max_input_length.pow(3).sqrt()
    }

    fn exceeds_weight(
        tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool {
        let mut max_in_len = 0;
        // Work backwards from longest projected entry
        for (batch_size, (out_len, in_len, _)) in tree.iter().rev().enumerate() {
            let this_out_len = *out_len;
            max_in_len = max(max_in_len, *in_len);
            if this_out_len <= current_output_len {
                // Check if we breach max space for this segment
                let seq_len = max_in_len + this_out_len;
                if seq_len.pow(2) * (batch_size + 1) > max_total_weight {
                    return true
                }
            }
        }
        false
    }

    fn count_tokens(input_lengths: impl Iterator<Item=usize>, batch_size: usize) -> usize {
        input_lengths.max().unwrap_or(0) * batch_size
    }

    fn default_max_prefill_weight() -> usize {
        300000
    }
}
