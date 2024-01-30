use std::cmp::max;
use std::collections::BTreeSet;
use nohash_hasher::IntMap;
use crate::queue::Entry;


pub(crate) trait BatchType: Send + Sync + Clone + 'static {
    type Stats: Default;

    /// Update batch statistics with an additional request
    fn update_stats(stats: &Self::Stats, input_length: usize, output_length: usize) -> Self::Stats;
    /// Calculate worst-case max batch weight given batch statistics
    fn batch_max_weight(&self, stats: &Self::Stats, batch_size: usize) -> usize;
    /// Calculate initial max batch weight given batch statistics (based on input lengths only)
    fn batch_initial_weight(&self, stats: &Self::Stats, batch_size: usize) -> usize;
    /// Calculate prefill batch weight given prefill batch statistics
    fn prefill_weight(&self, prefill_stats: &Self::Stats, batch_size: usize) -> usize;
    /// Percentage of batch tokens that are padding
    fn percent_padding(prefill_stats: &Self::Stats, batch_size: usize) -> f32;
    /// Indicate whether a hypothetical batch will exceed the combined weight limit
    fn exceeds_weight(
        &self, tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool;
    /// Provide a count of tokens for a given batch, including padding tokens if applicable
    fn count_tokens(input_lengths: impl Iterator<Item=usize>, batch_size: usize) -> usize;

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
pub(crate) struct FlashBatch {
    pub(crate) prefill_gradient: f64,
    pub(crate) nexttoken_gradient: f64,
}

impl BatchType for FlashBatch {
    /// Keep track of total number of input and output tokens in the batch
    type Stats = (usize, usize);

    fn update_stats(
        total_tokens: &Self::Stats, input_length: usize, output_length: usize
    ) -> Self::Stats {
        let (total_in_tokens, total_out_tokens) = total_tokens;
        (total_in_tokens + input_length, total_out_tokens + output_length)
    }

    fn batch_max_weight(&self, total_tokens: &Self::Stats, _batch_size: usize) -> usize {
        let (total_in_tokens, total_out_tokens) = total_tokens;
        ((*total_in_tokens + *total_out_tokens) as f64 * self.nexttoken_gradient) as usize
    }

    fn batch_initial_weight(&self, (total_in_tokens, _): &Self::Stats, _batch_size: usize) -> usize {
        (*total_in_tokens as f64 * self.nexttoken_gradient) as usize
    }

    fn prefill_weight(&self, (total_in_tokens, _): &Self::Stats, _batch_size: usize) -> usize {
        (*total_in_tokens as f64 * self.prefill_gradient) as usize
    }

    fn percent_padding(_: &Self::Stats, _batch_size: usize) -> f32 {
        0.0
    }

    fn exceeds_weight(
        &self, tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool {
        let mut in_sum = 0;
        // Work backwards from longest projected entry
        for (batch_size, (out_len, in_len, _)) in tree.iter().rev().enumerate() {
            let total_weight_limit = max_total_weight as f64;
            let this_out_len = *out_len;
            in_sum += *in_len;
            // Only need to check segments with output_len > current_output_len
            // will have been checked in a prior iteration
            if this_out_len <= current_output_len {
                // Check if we breach max space for this segment
                let seg_max_tokens = in_sum + (batch_size + 1) * this_out_len;
                if seg_max_tokens as f64 * self.nexttoken_gradient > total_weight_limit {
                    return true
                }
            }
        }
        false
    }

    fn count_tokens(input_lengths: impl Iterator<Item=usize>, _: usize) -> usize {
        input_lengths.sum()
    }

}

/// Regular rectangular padded
#[derive(Clone)]
pub(crate) struct PaddedBatch {
    pub(crate) prefill_linear_coef1: f64,
    pub(crate) prefill_quadratic_coef1: f64,
    pub(crate) prefill_quadratic_coef2: f64,
    pub(crate) nexttoken_gradient: f64,
}

impl BatchType for PaddedBatch {
    /// Keep track of maximum input length, maximum output length, input token count
    type Stats = (usize, usize, usize);

    fn update_stats(
        max_in_out_lengths: &Self::Stats, input_length: usize, output_length: usize
    ) -> Self::Stats {
        let (max_input_length, max_output_length, total_in_tokens) = max_in_out_lengths;
        (
            max(*max_input_length, input_length),
            max(*max_output_length, output_length),
            total_in_tokens + input_length
        )
    }

    fn batch_max_weight(&self, max_in_out_lengths: &Self::Stats, batch_size: usize) -> usize {
        let (max_input_length, max_output_length, _) = max_in_out_lengths;
        let seq_len_upper_bound = max_input_length + max_output_length;
        ((seq_len_upper_bound * batch_size) as f64 * self.nexttoken_gradient) as usize
    }

    fn batch_initial_weight(&self, (max_input_length, _, _): &Self::Stats, batch_size: usize) -> usize {
        ((*max_input_length * batch_size) as f64 * self.nexttoken_gradient) as usize
    }

    fn prefill_weight(&self, (max_input_length, _, _): &Self::Stats, batch_size: usize) -> usize {
        // Empirically, prefill latency is proportional to batch_size * seq_len^(3/2)
        let input_tokens = batch_size * max_input_length;
        let quad_input_tokens = (input_tokens * max_input_length) as f64;
        let input_tokens = input_tokens as f64;
        let linear = input_tokens * self.prefill_linear_coef1;
        let quadratic = input_tokens * self.prefill_quadratic_coef1 +
            quad_input_tokens * self.prefill_quadratic_coef2;

        f64::max(linear, quadratic) as usize
    }

    fn percent_padding((max_input_length, _, total_in_tokens): &Self::Stats, batch_size: usize) -> f32 {
        let total_toks = max_input_length * batch_size;
        match total_toks {
            0 => 0.0,
            total_toks => (total_toks - total_in_tokens) as f32 / total_toks as f32,
        }
    }

    fn exceeds_weight(
        &self, tree: &BTreeSet<(usize, usize, usize)>, max_total_weight: usize, current_output_len: usize
    ) -> bool {
        let total_weight_limit = max_total_weight as f64;
        let mut max_in_len = 0;
        // Work backwards from longest projected entry
        for (batch_size, (out_len, in_len, _)) in tree.iter().rev().enumerate() {
            let this_out_len = *out_len;
            max_in_len = max(max_in_len, *in_len);
            if this_out_len <= current_output_len {
                // Check if we breach max space for this segment
                let seg_max_tokens = (max_in_len + this_out_len) * (batch_size + 1);
                if seg_max_tokens as f64 * self.nexttoken_gradient > total_weight_limit {
                    return true
                }
            }
        }
        false
    }

    fn count_tokens(input_lengths: impl Iterator<Item=usize>, batch_size: usize) -> usize {
        input_lengths.max().unwrap_or(0) * batch_size
    }
}
