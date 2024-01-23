from typing import Optional
from text_generation_server.pb import generate_pb2
import numpy as np
import torch


def pt2_compile_warmup(
    model: 'Model',
    max_batch_size: int,
    max_new_tokens: int,
    max_sequence_length: int,
    max_batch_weight: Optional[int] = None,
    n_samples: int = 10,
    verbose: bool = False
):
    text = "test " * 10_000

    def __generate_prefill_request(batch_size: int, in_tokens: int, num_new_tokens: int):
        return generate_pb2.PrefillRequest(
            batch=generate_pb2.Batch(
                id=0,
                requests=[
                    generate_pb2.Request(
                        id=i, inputs=text, input_length=in_tokens, truncate=True, max_output_length=num_new_tokens
                    ) for i in range(batch_size)
                ]
            )
        )

    def __force_contiguous(x):
        # Update tensors in place so that memory can be freed incrementally
        x.data = x.data.contiguous()
        strides = list(x.stride())
        if strides != sorted(strides, reverse=True):
            x.data = x.data.contiguous(memory_format=torch.channels_last).contiguous()
        return x

    def __eval_shape(batch_size: int, sequence_length: int, num_new_tokens: int):
        if verbose:
            print(
                ">> evaluating shape (batch_size: %d, sequence_length: %d, num_new_tokens: %d)"
                % (batch_size, sequence_length, num_new_tokens)
            )

        input_length = sequence_length - num_new_tokens

        # prefill
        request = __generate_prefill_request(batch_size, input_length, num_new_tokens)

        batch, errors = model.batch_type.from_pb(
            request.batch,
            tokenizer=model.tokenizer,
            dtype=model.dtype,
            device=model.device,
            embeddings_lookup=model.word_embeddings,
            prefix_cache=model.prefix_cache,
            use_position_ids=model.use_position_ids,
        )

        model.generate_token(batch, first=True, for_concat=False)

        for i in range(num_new_tokens - 1):
            model.generate_token(batch)

    def __safe_eval_shape(batch_size: int, sequence_length: int, num_new_tokens: int):
        try:
            __eval_shape(batch_size, sequence_length, num_new_tokens)
        except torch.cuda.OutOfMemoryError as e:
            print(">> caught OOM error: ", e)

    def __max_sequence_length_for_batch_size(batch_size: int):
        return max_sequence_length if max_batch_weight is None else min(
            max_sequence_length, int(np.floor(np.sqrt(max_batch_weight/batch_size)))
        )

    def __max_new_tokens_for_sequence_length(sequence_length: int):
        return min(max_new_tokens, sequence_length - 1)

    if verbose:
        print("[Phase 1] Probing boundaries.")

    for batch_size in [1, max_batch_size]:
        max_sequence_length_for_batch_size = __max_sequence_length_for_batch_size(batch_size)
        for sequence_length in [2, 3, max_sequence_length_for_batch_size]:
            __safe_eval_shape(
                batch_size=batch_size,
                sequence_length=sequence_length,
                num_new_tokens=1,
            )
            if sequence_length > 2:
                __safe_eval_shape(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    num_new_tokens=2,
                )
            if sequence_length > 3:
                __safe_eval_shape(
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    num_new_tokens=__max_new_tokens_for_sequence_length(sequence_length),
                )

    if verbose:
        print("[Phase 2] Probing random valid tensor shapes.")

    n_compiles = model.n_kernels

    valid_shapes = []
    for batch_size in range(1, 1 + max_batch_size):
        this_max_sequence_length = __max_sequence_length_for_batch_size(batch_size)
        for sequence_length in range(1, 1 + this_max_sequence_length):
            this_max_new_tokens = __max_new_tokens_for_sequence_length(sequence_length)
            for new_tokens in range(1, 1 + this_max_new_tokens):
                valid_shapes.append((batch_size, sequence_length, new_tokens))

    rs = np.random.RandomState(seed=42)
    for i in range(n_samples):
        shape = valid_shapes[rs.randint(low=0, high=len(valid_shapes))]
        __safe_eval_shape(
            batch_size=shape[0],
            sequence_length=shape[1],
            num_new_tokens=shape[2],
        )
        if verbose:
            print(">> n_samples: %3d, n_new_compiles: %3d" % (i+1, model.n_kernels-n_compiles))
