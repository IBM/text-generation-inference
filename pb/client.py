import generation_pb2 as pb2
import generation_pb2_grpc as gpb2
import grpc
from google.protobuf import json_format

port = 8033
channel = grpc.insecure_channel(f"localhost:{port}")
stub = gpb2.GenerationServiceStub(channel)

# optional: parameters for inference
params = pb2.Parameters(
    method="GREEDY", stopping=pb2.StoppingCriteria(min_new_tokens=20, max_new_tokens=40)
)

prompt = "The weather is"

message = json_format.ParseDict(
    {"requests": [{"text": prompt}]}, pb2.BatchedGenerationRequest(params=params)
)
response = stub.Generate(message)
print(prompt, response)
