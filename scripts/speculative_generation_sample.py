from text_generation_server.models import get_model
from text_generation_server.pb import generate_pb2
from typing import List
import time
import torch

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"

text = template.format(
    "Provide a list of instructions for preparing chicken soup."
)

def __generate_prefill_request(id: int, batch_size: int, num_new_tokens: List[int]):

    out = generate_pb2.PrefillRequest(
        batch=generate_pb2.Batch(
            id=id,
            requests=[
                generate_pb2.Request(
                    id=i, inputs=text, input_length=49, max_output_length=num_new_tokens[i],
                    parameters=generate_pb2.NextTokenChooserParameters(
                        temperature=0.0,
                    )
                ) for i in range(batch_size)
            ]
        )
    )
    return out

model = get_model(
    model_name="/net/storage149/mnt/md0/jmrosenk/llama_weights/hf/7B-F",
    revision=None,
    deployment_framework="tgis_native",
    dtype_str="float16",
    quantize=None,
    max_sequence_length=2048
)


num_new_tokens = [100]

request1 = __generate_prefill_request(0, 1, num_new_tokens)

batch1, errors = model.batch_type.from_pb(
    request1.batch,
    tokenizer=model.tokenizer,
    dtype=model.dtype,
    device=model.device,
    embeddings_lookup=model.word_embeddings,
    prefix_cache=model.prefix_cache,
    use_position_ids=model.use_position_ids,
)

# token info (token ids) - have tokenizer in model,
result = ""
token_info_list, _, _, _ = model.generate_token(batch1, first=True, for_concat=False)

cumulative_t = 0
total_decode_tokens = 0
token_ids_out = [token_info_list[0].token_id]
while batch1.input_lengths[0] < batch1.total_lengths[0]:
    t0 = time.time_ns()
    token_info_list, _, _, _ = model.generate_token(batch1)
    torch.cuda.synchronize(device=model.device)
    t_tok = time.time_ns()-t0
    cumulative_t += t_tok
    total_decode_tokens += len(token_info_list)
    print(f"number of tokens per step: {len(token_info_list)}")
    for token_info in token_info_list:
        token_ids_out.append(token_info.token_id)
    print("t_tok: %.2f ms" % (t_tok / len(token_info_list) / 1000.0 / 1000.0))

print(model.tokenizer.decode(token_ids_out))
print(f"avg per token: {cumulative_t / total_decode_tokens / 1000000}")