import time
import json
from durations import Duration
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
sample_request = {
    "model_id": "dummy-model-name",
    "request": {"text": "Hello! My favorite Bible verse is "},
    "params": {
        "method": "SAMPLE",
        "sampling": {},
        "decoding": {"repetition_penalty": 2},
        "stopping": {
            "min_new_tokens": 16,
            "max_new_tokens": 20,
            "stop_sequences": ["Peter ", "Timothy ", "joseph", "Corinthians"],
        },
    },
}
message = json_format.ParseDict(sample_request, pb2.SingleGenerationRequest())
output = []
t0 = time.time_ns()
response = stub.GenerateStream(message)
response_generator = get_streaming_response_tgis(response)

stop = False
while not stop:
    r, n_tokens, t, ok, err = next(response_generator)

    if not ok:
        stop = True
        # check if we have reached end of stream
        if type(err) is StopIteration:
            continue

    record = {
        "response": r,
        "ok": ok,
        "error": str(err),
        "timestamp": t,
        "duration_ms": (t - t0) / 1000.0 / 1000.0,
        "n_tokens": n_tokens,
    }

    output.append(record)
    t0 = t

with open("results.json", "w") as f:
    json.dump(output, f)
