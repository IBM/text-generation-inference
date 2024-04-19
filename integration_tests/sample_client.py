import time
import grpc
from google.protobuf import json_format
from text_generation_tests.pb import generation_pb2_grpc as gpb2, generation_pb2 as pb2


def get_streaming_response_tgis(response):
    stop = False
    generated_tokens = 0
    while not stop:
        try:
            x = next(response)
            timestamp = time.time_ns()
            data = json_format.MessageToDict(x)
            # skip first response (tokenizer output only)
            if "inputTokenCount" not in data:
                n_tokens = data["generatedTokenCount"] - generated_tokens
                generated_tokens = data["generatedTokenCount"]
                yield data, n_tokens, timestamp, True, None
        except Exception as e:
            timestamp = time.time_ns()
            yield None, 0, timestamp, False, e


channel = grpc.insecure_channel("localhost:8033")
stub = gpb2.GenerationServiceStub(channel)
max_new_tokens = 100

template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:"
num_req = 0
while True:
    prompt_input = input(f"\n{num_req}) Enter a prompt:\n")

    print("-" * 40)
    print("Output:")
    prompt = template.format(prompt_input)
    sample_request = {
        "model_id": "dummy-model-name",
        "request": {"text": prompt},
        "params": {
            "method": "GREEDY",
            "stopping": {
                "max_new_tokens": max_new_tokens,
                "min_new_tokens": max_new_tokens,
            },
        },
    }
    message = json_format.ParseDict(sample_request, pb2.SingleGenerationRequest())
    output = []
    total_time = 0
    response = stub.GenerateStream(message)
    response_generator = get_streaming_response_tgis(response)
    t0 = time.time_ns()
    response = ""
    stop = False
    while not stop:
        r, n_tokens, t, ok, err = next(response_generator)

        if not ok:
            stop = True
            # check if we have reached end of stream
            if type(err) is StopIteration:
                continue
        duration = (t - t0) / 1000.0 / 1000.0
        record = {
            "response": r,
            "ok": ok,
            "error": str(err),
            "timestamp": t,
            "duration_ms": duration,
            "n_tokens": n_tokens,
        }
        total_time += duration
        response += r["text"]
        output.append(record)
        t0 = t

    # print(json.dumps(output, indent=4))
    print("-" * 40)
    print(response)
    print("-" * 40)
    print(f"Total_time : {total_time}ms")
    print(f"Time_per_token : {total_time/max_new_tokens}ms")
    print("-" * 40)
    num_req += 1
