import json
import time

import grpc
import requests
from google.protobuf import json_format

import generation_pb2 as pb2
import generation_pb2_grpc as gpb2

port = 8033
channel = grpc.insecure_channel(f"localhost:{port}")
stub = gpb2.GenerationServiceStub(channel)

# warmup inference
for i in range (5):
    text = "hello world"
    message = json_format.ParseDict(
        {"requests": [{"text": text}]}, pb2.BatchedGenerationRequest()
    )
    response = stub.Generate(message)

# time inference
for prompt in ["The weather is", "The cat is walking on", "I would like to"]:
# for prompt in ["def hello_world():"]:
    message = json_format.ParseDict(
        {"requests": [{"text": prompt}]}, pb2.BatchedGenerationRequest()
    )
    start = time.perf_counter()
    response = stub.Generate(message)
    end = time.perf_counter()
    print(prompt, response)
    print(f"Duration: {end-start:.2f}")