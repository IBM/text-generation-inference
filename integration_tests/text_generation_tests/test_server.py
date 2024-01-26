import asyncio
import glob
import os
import random
import sys
import yaml
import subprocess
import threading
import time

import grpc
import pytest
import requests
from google.protobuf import json_format

from text_generation_tests.pb import generation_pb2_grpc as gpb2, generation_pb2 as pb2
from text_generation_tests.approx import approx

INCLUDE_STREAMING = True
TESTS_TIMEOUT = 300.0  # 5 mins
TESTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))


def start_server(
    model_name: str,
    extensions: str,
    num_shard: int,
    port: int,
    master_port: int,
    timeout=20,
    model_path=None,
    include_cache_env_vars=True,
    output_special_tokens=False,
):
    # Download weights to the cache first
    print(f"Downloading files for model {model_name}...")
    cp = subprocess.run(
        ["text-generation-server", "download-weights", "--extension", extensions,  model_name],
        capture_output=True,
    )
    if cp.stderr:
        sys.stderr.write(str(cp.stderr))
    cp.check_returncode()

    # Override name with path for testing explicit path case
    model_name_or_path = model_path if model_path is not None else model_name

    args = [
        "text-generation-launcher",
        "--model-name", model_name_or_path,
        "--num-shard", str(num_shard),
        "--dtype-str", "float32",
        "--deployment-framework", "hf_accelerate",
        "--port", str(port),
        "--master-port", str(master_port),
        "--shard-uds-path",
        f"/tmp/test-{num_shard}-{port}-{master_port}",
        # Reduce these from defaults for tests, to force more batch concats
        "--max-batch-size", "8",
        "--max-waiting-tokens", "10",
        # Reduce this so we can more easily test limit behaviour
        "--max-sequence-length", "200",
        "--max-new-tokens", "169",
    ]

    if output_special_tokens:
        args.append("--output-special-tokens")

    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "full"
    env["ESTIMATE_MEMORY"] = "manual"
    env["PREFIX_STORE_PATH"] = os.path.join(TESTS_DIR, "prompt_prefixes")
    if not include_cache_env_vars:
        env.pop("TRANSFORMERS_CACHE", None)
        env.pop("HUGGING_FACE_HUB_CACHE", None)

    # Start the process
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)

    # Start a thread to continuously read and print the process's stdout and stderr
    def print_output():
        while True:
            output = process.stdout.readline()
            if output == b'' and process.poll() is not None:
                sys.stdout.flush()
                break
            if output:
                print(output.decode().rstrip())
        process.stdout.close()

    t = threading.Thread(target=print_output)
    t.start()

    # Poll the process's /health endpoint until it returns HTTP 200 OK
    start_time = time.time()
    while True:
        if process.poll() is not None:
            t.join()
            raise Exception("Server failed to start")
        try:
            response = requests.get(f'http://localhost:{port}/health')
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            pass

        if time.time() - start_time > timeout:
            process.terminate()
            raise TimeoutError('Timed out waiting for process to start')

        time.sleep(2)

    return process


@pytest.fixture
def server_fixture(request):
    model_name = request.node.get_closest_marker("model").args[0]
    shards = int(request.node.get_closest_marker("shards").args[0])
    extensions = request.node.get_closest_marker("extensions").args[0]
    ost = request.node.get_closest_marker("output_special_tokens")
    ost = ost is not None and ost.args[0]
    p = start_server(
        model_name, extensions, shards, 3000, 29502, output_special_tokens=ost
    )
    yield p
    p.terminate()
    assert p.wait(8.0) == 0


@pytest.fixture
def test_cases(request):
    filename = request.node.get_closest_marker("test_case_file").args[0]
    with open(os.path.join(TESTS_DIR, filename)) as f:
        return yaml.load(f, Loader=yaml.Loader)


def _verify_error(expected_err: grpc.RpcError, err):
    if not expected_err:
        raise err
    if "code" in expected_err:
        assert err.code().name == expected_err["code"]
    if "message" in expected_err:
        assert err.details() == expected_err["message"]


async def run_unary_test_case(stub, case):
    # Get case details
    request = case["request"]
    request_type = case.get("request_type", "generate")
    skip_check = "skip_check" in case
    expected = case.get("response")
    expected_err = case.get("error")
    # Create gRPC message from test request
    message = json_format.ParseDict(
        request,
        pb2.BatchedTokenizeRequest() if request_type == "tokenize"
        else pb2.BatchedGenerationRequest(),
    )
    print(f'================ Test: {case.get("name")}:')
    try:
        # Send request
        if request_type == "tokenize":
            response = await stub.Tokenize(message)
        else:
            response = await stub.Generate(message)
        assert not expected_err
        # Convert response back to dict
        response_dict = json_format.MessageToDict(response)
        if not skip_check:
            if response_dict != approx(expected):
                print(f'================ Test: {case.get("name")}:')
                print(yaml.dump(response_dict))
            # Check result matches expected
            assert response_dict == approx(expected)
        elif request_type == "generate":
            assert response_dict["responses"][0]["inputTokenCount"] == expected["responses"][0]["inputTokenCount"]
            assert response_dict["responses"][0].get("inputTokens") == expected["responses"][0].get("inputTokens")
    except grpc.RpcError as e:
        _verify_error(expected_err, e)


async def run_streaming_test_case(stub, case, seq2seq_model):
    # Get case details
    request = case["request"].copy()
    # Convert to SingleGeneration Request
    request["request"] = request.pop("requests")[0]
    skip_check = "skip_check" in case
    expected = None
    response = case.get("response")
    if response and "responses" in response:
        expected = case["response"]["responses"][0]
    expected_err = case.get("error")
    message = json_format.ParseDict(request, pb2.SingleGenerationRequest())
    print(f'============ Stream Test: {case.get("name")}:')
    response_stream = stub.GenerateStream(message)
    response_iter = response_stream.__aiter__()
    output = ""
    out_tokens = []
    try:
        # Verify first response which should contain input token count
        first = json_format.MessageToDict(await response_iter.__anext__())
        incl_input_text = "params" in request and "response" in request["params"] \
            and request["params"]["response"].get("inputText")
        assert first["inputTokenCount"] == expected["inputTokenCount"]
        if not skip_check:
            assert first.get("seed") == expected.get("seed")
        # assert first.get("inputTokens") == expected.get("inputTokens")
        assert "generatedTokenCount" not in first
        assert "stopReason" not in first
        if incl_input_text:
            # If inputText was specified, ensure it also contains the input text
            output = first["text"]
            assert output == request["request"]["text"]
            if seq2seq_model:
                output += "\n\n"  # Add separator between encoder input and decoder output
        else:
            assert "text" not in first

        incl_input_toks = "params" in request and "response" in request["params"] \
            and request["params"]["response"].get("inputTokens")
        if incl_input_toks:
            # If inputTokens was specified, check for a second message containing
            # just the input token information
            second = json_format.MessageToDict(await response_iter.__anext__())
            assert "generatedTokenCount" not in second
            assert "stopReason" not in second
            assert "text" not in second
            assert second.get("inputTokens") == approx(expected.get("inputTokens"))

        stop_reason = None
        last_count = 0
        seed = None
        async for response in response_iter:
            if stop_reason is not None:
                pytest.fail("stop reason before last response")
            if seed is not None:
                pytest.fail("seed before last response")
            response_dict = json_format.MessageToDict(response)
            if "text" in response_dict:
                output += response_dict["text"]
            if "tokens" in response_dict:
                out_tokens.extend(response_dict["tokens"])
            count = response_dict.get("generatedTokenCount", 0)
            stop_reason = response_dict.get("stopReason")
            seed = response_dict.get("seed")
            assert count > last_count or (count == 0 and stop_reason != "NOT_FINISHED")
            assert "inputTokenCount" not in response_dict
            assert "inputTokens" not in response_dict
            last_count = count
        if not skip_check:
            assert output == expected.get("text", "")
            assert last_count == expected["generatedTokenCount"]
            assert stop_reason == expected["stopReason"]
            assert seed == expected.get("seed")

            assert out_tokens == approx(expected.get("tokens", []))
    except grpc.RpcError as e:
        _verify_error(expected_err, e)


async def run_test_cases_async(test_cases, seq2seq_model=False, sharded=False):
    async with grpc.aio.insecure_channel('localhost:8033') as channel:
        stub = gpb2.GenerationServiceStub(channel)
        tasks = []
        random.shuffle(test_cases)
        for case in test_cases:
            name = case.get("name")
            if sharded and case.get("singleShardOnly", False):
                print(f"Skipping single-shard-only test in sharded mode: {name}")
                continue

            tasks.append(asyncio.create_task(
                run_unary_test_case(stub, case), name=f"Generate: {name}",
            ))
            # For single-input tests, try the same thing as a streaming request
            if INCLUDE_STREAMING and len(case["request"].get("requests", [])) == 1 \
                    and case.get("request_type", "generate") == "generate":
                tasks.append(asyncio.create_task(
                    run_streaming_test_case(stub, case, seq2seq_model),
                    name=f"GenerateStream: {name}",
                ))
            await asyncio.sleep(0.3)
        # Add in the multi-input seed case
        tasks.append(asyncio.create_task(_test_multi_input_seeds(stub)))
        done, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_EXCEPTION, timeout=TESTS_TIMEOUT,
        )
        unfinished = len(pending)
        for p in pending:  # empty in success case
             p.cancel()
        for p in pending:  # empty in success case
            try:
                await p
            except (asyncio.CancelledError, UnboundLocalError):
                # Unbound local error too due to pygrpc bug
                pass
        for d in done:
            d.result()  # will raise if failed
        assert unfinished == 0, f"{unfinished} tests not finished before timeout of {TESTS_TIMEOUT}s"

        # Verify metrics endpoint
        response = requests.get(f'http://localhost:{3000}/metrics')
        assert response.status_code == 200


async def _test_multi_input_seeds(stub):
    # Ensure that sending a batch of identical inputs in sampling mode results
    # in different output seeds and texts
    with open(os.path.join(TESTS_DIR, "test_cases_common.yaml")) as f:
        test_case = yaml.load(f, Loader=yaml.Loader)
        request = test_case["seed_test"]["request"]
        message = json_format.ParseDict(request, pb2.BatchedGenerationRequest())
        response = await stub.Generate(message)
        response_dict = json_format.MessageToDict(response)

        outputs = {output["text"] for output in response_dict["responses"]}
        seeds = {int(output["seed"]) for output in response_dict["responses"]}
        print(f"texts: {outputs}")
        print(f"seeds: {seeds}")

        assert len(outputs) == len(request["requests"])
        assert len(seeds) == len(request["requests"])

        for seed in seeds:
            assert 0 <= seed <= 4294967295


async def run_time_limit_test(stub, *, streaming=False, time_limit=200, min_generated_tokens=2):
    generation_request = pb2.GenerationRequest(
        text='def doit():\n'
    )
    generation_params = pb2.Parameters(
        method=pb2.GREEDY,
        stopping=pb2.StoppingCriteria(
            max_new_tokens=169,
            min_new_tokens=169,
            time_limit_millis=time_limit,
        )
    )

    start = time.time_ns()
    if streaming:
        response = pb2.GenerationResponse()
        async for resp in stub.GenerateStream(
            pb2.SingleGenerationRequest(
                request=generation_request,
                params=generation_params
            )
        ):
            response.generated_token_count = resp.generated_token_count
            response.stop_reason = resp.stop_reason
    else:
        response = await stub.Generate(
            pb2.BatchedGenerationRequest(
                requests=[generation_request],
                params=generation_params,
            )
        )
        # single req/resp in the batch
        response = response.responses[0]
    end = time.time_ns()

    assert response.stop_reason == pb2.StopReason.TIME_LIMIT
    # ensure that some tokens were actually generated
    assert min_generated_tokens <= response.generated_token_count < 100
    # generating all tokens takes a few seconds
    assert time_limit < (end-start) / (10**6)  < time_limit+300


@pytest.mark.model("gpt2")
@pytest.mark.extensions(".safetensors,.json")
@pytest.mark.shards(1)
@pytest.mark.test_case_file("test_cases_gpt2.yaml")
@pytest.mark.asyncio
async def test_gpt2(server_fixture, test_cases):
    await run_test_cases_async(test_cases)

@pytest.mark.model("bigscience/bloom-560m")
@pytest.mark.extensions(".safetensors,.json,.model")
@pytest.mark.shards(1)
@pytest.mark.test_case_file("test_cases_bloom560m.yaml")
@pytest.mark.asyncio
async def test_bloom(server_fixture, test_cases):
    await run_test_cases_async(test_cases)


@pytest.mark.model("bigscience/mt0-small")
@pytest.mark.extensions(".bin,.json,.model")
@pytest.mark.shards(1)
@pytest.mark.test_case_file("test_cases_mt0small.yaml")
@pytest.mark.asyncio
async def test_mt0(server_fixture, test_cases):
    await run_test_cases_async(test_cases, seq2seq_model=True)

# test with tiny GPTBigCode model for the merged kv cache
@pytest.mark.model("bigcode/tiny_starcoder_py")
@pytest.mark.extensions(".safetensors,.json")
@pytest.mark.shards(2)
@pytest.mark.test_case_file("test_cases_tinystarcoderpy.yaml")
@pytest.mark.asyncio
async def test_gptbigcode(server_fixture, test_cases):
    await run_test_cases_async(test_cases)

# test with Llama model which has tokenizer.add_bos_token == true
@pytest.mark.model("Maykeye/TinyLLama-v0")
@pytest.mark.extensions(".bin,.json,.model")
@pytest.mark.shards(2)
@pytest.mark.test_case_file("test_cases_tinyllama.yaml")
@pytest.mark.asyncio
async def test_llama(server_fixture, test_cases):
    await run_test_cases_async(test_cases)

# Test distributed inference - two shards
@pytest.mark.model("bigscience/bloom-560m")
@pytest.mark.extensions(".safetensors,.json,.model")
@pytest.mark.shards(2)
@pytest.mark.test_case_file("test_cases_bloom560m.yaml")
@pytest.mark.asyncio
async def test_bloom(server_fixture, test_cases):
    await run_test_cases_async(test_cases, sharded=True)


@pytest.mark.model("bigscience/mt0-small")
@pytest.mark.extensions(".bin,.json")
@pytest.mark.shards(1)
@pytest.mark.output_special_tokens(True)
@pytest.mark.test_case_file("test_cases_mt0_ost.yaml")
@pytest.mark.asyncio
async def test_mt0_output_special_tokens(server_fixture, test_cases):
    await run_test_cases_async(test_cases)


# Test that the time based stopping criteria works
@pytest.mark.model("bigcode/tiny_starcoder_py")
@pytest.mark.extensions(".safetensors,.json")
@pytest.mark.shards(1)
@pytest.mark.asyncio
async def test_time_limit_stopping(server_fixture):
    async with grpc.aio.insecure_channel('localhost:8033') as channel:
        stub = gpb2.GenerationServiceStub(channel)
        # verify server is up with metrics request
        response = requests.get(f'http://localhost:{3000}/metrics')
        assert response.status_code == 200

        # batched
        await run_time_limit_test(stub)
        # one token should always be generated
        await run_time_limit_test(stub, time_limit=1, min_generated_tokens=1)

        # streaming
        await run_time_limit_test(stub, streaming=True)
        # one token should always be generated
        await run_time_limit_test(stub, streaming=True, time_limit=1, min_generated_tokens=1)

# Test loading when an explicit local path is provided
def test_explicit_path():
    # Test with and without providing TRANSFORMERS_CACHE env var
    path = glob.glob(f'{os.environ["TRANSFORMERS_CACHE"]}/models--bigscience--mt0-small/snapshots/*')[0]
    for include_env_vars in [False, True]:
        p = start_server(
            "bigscience/mt0-small",
            ".bin,.json,.model",
            1,
            3000,
            29502,
            model_path=path,
            include_cache_env_vars=include_env_vars,
        )
        try:
            async def test_model_info() -> pb2.ModelInfoResponse:
                async with grpc.aio.insecure_channel('localhost:8033') as channel:
                    return await gpb2.GenerationServiceStub(channel).ModelInfo(pb2.ModelInfoRequest(model_id="unused"))

            result = asyncio.get_event_loop().run_until_complete(test_model_info())
            assert result.max_sequence_length == 200
            assert result.max_new_tokens == 169
            assert result.model_kind == pb2.ModelInfoResponse.ModelKind.ENCODER_DECODER
        finally:
            p.terminate()

    assert p.wait(8.0) == 0


# To avoid errors related to event loop shutdown timing
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()
