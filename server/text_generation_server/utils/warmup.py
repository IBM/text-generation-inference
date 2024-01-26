from text_generation_server.pb import generate_pb2
import numpy as np
import torch

def pt2_compile_warmup(
    model: 'Model',
    memory_scaling_model: 'MemoryScalingModel',
    max_batch_size: int,
    max_new_tokens: int,
    max_sequence_length: int,
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

    def __eval_shape(batch_size: int, input_length: int, num_new_tokens: int):

        if verbose:
            print(
                ">> evaluating shape (batch_size: %3d, input_length: %4d, num_new_tokens: %4d)"
                % (batch_size, input_length, num_new_tokens)
            )

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

    def __safe_eval_shape(batch_size: int, input_length: int, num_new_tokens: int):
        try:
            __eval_shape(batch_size, input_length, num_new_tokens)
        except torch.cuda.OutOfMemoryError as e:
            print(">> caught OOM error: ", e)

    def __get_n_kernels():
        return model.n_kernels if hasattr(model, 'n_kernels') else 0

    if verbose:
        print("[Phase 1] Probing PREFILL boundaries")

    for batch_size in [1, max_batch_size]:
        __safe_eval_shape(
            batch_size=batch_size,
            input_length=1,
            num_new_tokens=1,
        )

        max_input_len = memory_scaling_model.max_input_len_for_prefill(batch_size, max_sequence_length-1)
        __safe_eval_shape(
            batch_size=batch_size,
            input_length=max_input_len,
            num_new_tokens=1,
        )

    if verbose:
        print("[Phase 2] Probing NEXT_TOKEN boundaries")

    for batch_size in [1, max_batch_size]:
        __safe_eval_shape(
            batch_size=batch_size,
            input_length=1,
            num_new_tokens=2,
        )

        max_input_len = memory_scaling_model.max_input_len_for_nt(batch_size, 2, max_sequence_length-2)
        __safe_eval_shape(
            batch_size=batch_size,
            input_length=max_input_len,
            num_new_tokens=2,
        )

        max_output_len = memory_scaling_model.max_output_len_for_nt(batch_size, 1, max_new_tokens)
        __safe_eval_shape(
            batch_size=batch_size,
            input_length=1,
            num_new_tokens=max_output_len,
        )

        if max_output_len == max_new_tokens:
            max_input_len = memory_scaling_model.max_input_len_for_nt(batch_size, max_new_tokens, max_sequence_length-max_new_tokens)

            __safe_eval_shape(
                batch_size=batch_size,
                input_length=max_input_len,
                num_new_tokens=max_new_tokens,
            )
        else:
            mid_output_len = max_output_len // 2

            max_input_len = memory_scaling_model.max_input_len_for_nt(batch_size, mid_output_len, max_sequence_length-mid_output_len)
            __safe_eval_shape(
                batch_size=batch_size,
                input_length=max_input_len,
                num_new_tokens=mid_output_len,
            )

    if verbose:
        print("[Phase 3] Probing random valid shapes")

    n_kernels_init = __get_n_kernels()

    rs = np.random.RandomState(seed=42)
    for i in range(n_samples):
        batch_size = rs.randint(low=1, high=(max_batch_size+1))
        max_input_len = memory_scaling_model.max_input_len_for_prefill(batch_size, max_sequence_length-1)
        input_len = rs.randint(low=1, high=(max_input_len+1))
        try:
            max_output_len = np.minimum(max_sequence_length-input_len, max_new_tokens)
            max_output_len_ = memory_scaling_model.max_output_len_for_nt(batch_size, input_len, max_output_len)
            output_len = rs.randint(low=1, high=(max_output_len_+1))
        except:
            output_len = 1

        __safe_eval_shape(
            batch_size=batch_size,
            input_length=input_len,
            num_new_tokens=output_len,
        )

        if verbose:
            n_kernels = __get_n_kernels()
            print(">> n_samples: %3d, n_new_compiles: %3d" % (i+1, n_kernels-n_kernels_init))
