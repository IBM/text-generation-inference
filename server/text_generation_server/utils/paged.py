from typing import Optional, List
from multiprocessing import Queue
import os
import torch

from fms_extras.models.speculator import flatten_batch, apply_index_map

# number of candidates during speculation
SPECULATOR_NUM_CANDIDATES = int(os.getenv("SPECULATOR_NUM_CANDIDATES", "5"))

# number of candidates per head
SPECULATOR_THRESHES = [
    int(x) for x in os.getenv("SPECULATOR_THRESHES", "5,3,2").strip().split(',')
]

def fit_memory_scaling_model(
        model_name: str,
        revision: Optional[str],
        deployment_framework: str,
        dtype_str: Optional[str],
        quantize: Optional[str],
        max_sequence_length: int,
        batch_safety_margin: int,
        cuda_process_memory_fraction: float,
        q_out: Queue
    ):

    import os

    os.environ["PAGED_ATTENTION"] = "false"

    import torch
    from text_generation_server.models import get_model
    from text_generation_server.utils import Estimator

    cuda_available = torch.cuda.is_available()

    # Set the fraction of cuda/gpu mem available to this process, then load the model
    if cuda_available and cuda_process_memory_fraction < 1:
        torch.cuda.set_per_process_memory_fraction(cuda_process_memory_fraction)

    model = get_model(
        model_name, revision, deployment_framework, dtype_str, quantize, max_sequence_length
    )

    memory_scaling_model = Estimator.build_from_env(
        model,
        batch_safety_margin,
    ).run()

    q_out.put(memory_scaling_model)


def truncate_and_flatten(tensor: torch.Tensor, num_tokens_per_sequence: List[int]):
    tensor_list = torch.tensor_split(tensor, tensor.size(0))
    return torch.cat([tensor_i[0, -num_tokens:] for tensor_i, num_tokens in
                                  zip(tensor_list, num_tokens_per_sequence)], dim=0)

def prepare_inputs_without_speculation(
    input_ids,
    embeds,
    parent_sequence_ids,
    kv_cache_manager
    ):
    
    bsize = input_ids.shape[0]

    num_tokens_per_sequence = [1 for _ in range(bsize)]

    cache_data = kv_cache_manager.allocate_tokens(
        num_tokens_per_sequence, parent_sequence_ids
    )
    # tpa --is this needed?
    parent_sequence_ids = cache_data.sequence_ids

    # ** SPECIAL HANDLING FOR FLASH V2 **
    # fix slot mappings for flash attention v2
    cache_data.slot_mapping = truncate_and_flatten(
        cache_data.slot_mapping, num_tokens_per_sequence
    )
    position_ids = truncate_and_flatten(
        cache_data.position_ids,
        num_tokens_per_sequence,
    )
    input_ids = truncate_and_flatten(input_ids, num_tokens_per_sequence)

    context_lengths = cache_data.context_lengths  # bk
    inflate_factor = (
        cache_data.query_length
        if cache_data.unflatten_indices is None
        else cache_data.unflatten_indices.size(-1)
    )
    context_lengths = context_lengths.unsqueeze(1).expand(
        -1, inflate_factor
    )  # bk n
    context_lengths = (
        context_lengths.sub(context_lengths.sign().cumsum(1).flip([1]).sub(1))
        .int()
        .view(-1)
    )  # bkn
    block_mappings = cache_data.block_mapping.repeat_interleave(
        inflate_factor, dim=0
    )  # bkn n_blocks
    if cache_data.flatten_indices is not None:
        context_lengths = apply_index_map(
            context_lengths, cache_data.flatten_indices
        )  # n'
        block_mappings = apply_index_map(
            block_mappings, cache_data.flatten_indices
        )  # n' n_blocks
    cache_data.block_mapping = block_mappings
    cache_data.context_lengths = context_lengths

    return input_ids, position_ids, cache_data


def prepare_inputs_for_prefill(
    input_ids,
    num_tokens_per_sequence,
    kv_cache_manager,
    ):

    cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence)

    # ** SPECIAL HANDLING FOR FLASH V2 **
    # fix slot mappings for flash attention v2
    cache_data.slot_mapping = truncate_and_flatten(cache_data.slot_mapping, num_tokens_per_sequence)
    position_ids = truncate_and_flatten(cache_data.position_ids,
                                          num_tokens_per_sequence)

    cum_seq_lengths = torch.cumsum(
        torch.tensor(num_tokens_per_sequence, dtype=torch.int32, device=input_ids.device), dim=0)
    cache_data.context_lengths = torch.cat(
        tensors=(torch.zeros(1, dtype=torch.int32, device=input_ids.device), cum_seq_lengths),
        dim=0
    )

    return input_ids, position_ids, cache_data


def prepare_inputs_with_speculation(
    input_ids,
    embeds,
    parent_sequence_ids,
    kv_cache_manager,
    speculator,
    spec_ind,
    pad_token_id,
    ):

    bsize = input_ids.shape[0]

    n_adds = speculator.n_predict + 1

    if len(SPECULATOR_THRESHES) != speculator.n_predict:
        raise ValueError(
            f"Length of SPECULATOR_THRESHES ({SPECULATOR_THRESHES}) does not match SPECULATOR_N_PREDICT ({SPECULATOR_N_PREDICT})"
        )

    #hard-code some values
    top_k = SPECULATOR_NUM_CANDIDATES
    threshes = SPECULATOR_THRESHES
    flatting=True

    # create candidate sequences
    child_sequence_ids_list = []
    child_sequence_ids_flattened = []
    num_tokens_per_sequence = [n_adds for _ in range(bsize * top_k)]
    # each parent will have top_k child sequences
    for parent_sequence_id in parent_sequence_ids:
        child_sequence_ids = kv_cache_manager.add_child_sequences(parent_sequence_id, top_k)
        child_sequence_ids_list.append(child_sequence_ids)
        child_sequence_ids_flattened.extend(child_sequence_ids)

    # add n_adds tokens to each candidate
    cache_data = kv_cache_manager.allocate_tokens(num_tokens_per_sequence, child_sequence_ids_flattened)
    position_ids = cache_data.position_ids

    # Get candidate set of speculations
    adds = speculator.generate_suffixes(embeds[spec_ind, :], input_ids[spec_ind,:], threshes, top_k)  # b k h

    adds_all = adds.new_full(size=(bsize, adds.shape[1], adds.shape[2]), fill_value=pad_token_id)
    adds_all[spec_ind, :, :] = adds

    input_ids = torch.cat(
        [input_ids.unsqueeze(1).expand(bsize, top_k, 1), adds_all], dim=-1
    ).int()  # b k 1+h

    this_flatting = False
    if flatting:
        flat_inputs, unflat_indices, flat_indices = flatten_batch(input_ids) # b', b k 1+h, b'
        compression = flat_inputs.numel() / input_ids.numel()
        if compression < .75:
            this_flatting = True
            flat_inputs = flat_inputs[None,] # 1 b'
            cache_data.unflatten_indices = unflat_indices
            cache_data.flatten_indices = flat_indices
            position_ids = apply_index_map(position_ids.view(-1), flat_indices)[None,]
    input_ids = input_ids.view(-1, n_adds)  # bk 1+h

    context_lengths = cache_data.context_lengths  # bk
    inflate_factor = cache_data.query_length if cache_data.unflatten_indices is None else cache_data.unflatten_indices.size(
        -1)
    context_lengths = context_lengths.unsqueeze(1).expand(-1, inflate_factor)  # bk n
    context_lengths = context_lengths.sub(context_lengths.sign().cumsum(1).flip([1]).sub(1)).int().view(-1)  # bkn
    block_mappings = cache_data.block_mapping.repeat_interleave(inflate_factor, dim=0)  # bkn n_blocks
    if cache_data.flatten_indices is not None:
        context_lengths = apply_index_map(context_lengths, cache_data.flatten_indices)  # n'
        block_mappings = apply_index_map(block_mappings, cache_data.flatten_indices)  # n' n_blocks
    cache_data.block_mapping = block_mappings
    cache_data.context_lengths = context_lengths

    cache_data.slot_mapping = cache_data.slot_mapping.view(-1)
    position_ids = position_ids.view(-1)

    input_ids_unflat = input_ids
    if this_flatting:
        input_ids = flat_inputs

    input_ids = input_ids.view(-1)

    return input_ids, position_ids, cache_data, this_flatting, unflat_indices, input_ids_unflat, child_sequence_ids_list


def process_outputs_with_speculation(
    logits,
    embeds,
    kv_cache_manager,
    speculator,
    this_flatting,
    unflat_indices,
    input_ids_unflat,
    child_sequence_ids_list,
    ):

    bsize, top_k, n_adds = unflat_indices.shape

    logits = logits.unsqueeze(0) # 1 n' v
    embeds = embeds.unsqueeze(0) # 1 n' v
    next_vals = torch.argmax(logits, dim=-1) # 1 n'

    if this_flatting:
        unflat_indices = unflat_indices.view(-1, unflat_indices.size(2))
        next_vals = apply_index_map(next_vals[0], unflat_indices) # bk 1+h
        embeds = apply_index_map(embeds[0], unflat_indices) # bk 1+h d
        logits = apply_index_map(logits[0], unflat_indices) # bk 1+h d
    else:
        next_vals = next_vals.view(-1, n_adds)
        embeds = embeds.view(next_vals.size(0), n_adds, -1)
        logits = logits.view(next_vals.size(0), n_adds, -1)


    # Check correctness of speculator predictions
    test = input_ids_unflat.roll(-1, 1).eq(next_vals).cumprod(1)
    n_correct = (
        test.sum(1).clamp(0, n_adds - 1).view(bsize, top_k)
    )  # clamp in case pred[0]==targ[-1]
    best_guess = n_correct.argmax(1)  # b
    best_guess_unflat = (
        best_guess.unsqueeze(1).expand(bsize, n_adds).unsqueeze(1)
    )  # b 1 1+h


    # Set global values to those of best guess
    next_vals = next_vals.view(bsize, top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)  # b 1+h
    n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b

    embeds = embeds.view(bsize, top_k, *embeds.size()[1:]).gather(
        1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
    ).squeeze(1)  # b 1+h d
    logits = logits.view(bsize, top_k, *logits.size()[1:]).gather(
        1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, logits.size(2))
    ).squeeze(1)  # b 1+h d

    # free all worst candidates and keep best candidates as parents
    parent_sequence_ids = []
    for parent_index, child_sequence_ids in enumerate(child_sequence_ids_list):
        best_index = best_guess[parent_index].item()

        # free all bad candidates
        kv_cache_manager.free_sequences(child_sequence_ids[:best_index] + child_sequence_ids[best_index + 1:])

        # decrease the context length of the sequence which used to be sequence length + n_adds by the number of incorrect tokens
        # for the correct candidate
        best_sequence_id = child_sequence_ids[best_index]
        parent_sequence_ids.append(best_sequence_id)
        kv_cache_manager.remove_tokens(best_sequence_id, n_adds - n_correct[parent_index].item() - 1)

    # Toss any wrong speculator tokens
    next_vals_split = list(next_vals)
    next_vals_split = [
        next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
    ]  # [b] h'
    embeds = embeds.gather(
        1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
    )  # Grab last correct embed

    return logits, embeds, parent_sequence_ids, next_vals_split
