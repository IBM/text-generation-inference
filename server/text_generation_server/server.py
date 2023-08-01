import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from functools import partial

import torch.cuda
from grpc import aio, StatusCode
from grpc._cython.cygrpc import AbortError

from grpc_reflection.v1alpha import reflection
from pathlib import Path
from typing import List, Optional

from text_generation_server.cache import Cache
from text_generation_server.models import Model, get_model, Seq2SeqLM
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.pb import generate_pb2_grpc, generate_pb2
from text_generation_server.pb.generate_pb2 import ModelInfoResponse
from text_generation_server.prompt_cache import PrefixNotFound
from text_generation_server.utils import pt2_compile_warmup

COMPACT_BEFORE_PREFILL = os.getenv("COMPACT_BEFORE_PREFILL", "true") != "false"

HEALTHCHECK_BATCH_ID = (1 << 64) - 1


def log_errs(func):
    async def func_with_log(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except AbortError:
            raise
        except Exception:
            logging.exception(f"{func.__name__} failed")
            raise
    return func_with_log


class TextGenerationService(generate_pb2_grpc.TextGenerationServiceServicer):
    def __init__(self, model: Model, cache: Cache, server_urls: List[str]):
        self.cache = cache
        self.model = model
        self.server_urls = server_urls

    async def ServiceDiscovery(
        self, request: generate_pb2.ServiceDiscoveryRequest, context
    ) -> generate_pb2.ServiceDiscoveryResponse:
        return generate_pb2.ServiceDiscoveryResponse(urls=self.server_urls)

    @log_errs
    async def ClearCache(
        self, request: generate_pb2.ClearCacheRequest, context
    ) -> generate_pb2.ClearCacheResponse:
        self.cache.clear()
        return generate_pb2.ClearCacheResponse()

    @log_errs
    async def ModelInfo(self, request: generate_pb2.ModelInfoRequest, context) -> generate_pb2.ModelInfoResponse:
        return generate_pb2.ModelInfoResponse(
            model_type=ModelInfoResponse.ModelType.SEQ2SEQ_LM
                if isinstance(self.model, Seq2SeqLM) else ModelInfoResponse.ModelType.CAUSAL_LM,
            eos_token=self.model.config.eos_token_id,
            batch_padding=not isinstance(self.model, FlashCausalLM),
        )

    @log_errs
    async def Health(self, request, context):
        if self.model.device.type == "cuda":
            torch.zeros((2, 2)).cuda()
        return generate_pb2.HealthResponse()


    @log_errs
    async def PrefixLookup(self, request: generate_pb2.PrefixLookupRequest, context) -> generate_pb2.PrefixLookupResponse:
        try:
            # This may throw other errors too
            prefix = self.model.prefix_cache.get(request.prefix_id)
        except PrefixNotFound:
            await context.abort(StatusCode.NOT_FOUND, f"prefix id \"{request.prefix_id}\" not found")
        return generate_pb2.PrefixLookupResponse(
            prefix_length=len(prefix) if torch.is_tensor(prefix) else sum(len(t) for t in prefix if t is not None)
        )

    @log_errs
    async def Prefill(self, request: generate_pb2.PrefillRequest, context) -> generate_pb2.PrefillResponse:
        with self.model.context_manager():
            # Prune any existing batches first
            for cbatch in request.to_prune:
                batch_to_prune = self.cache.pop(cbatch.batch_id)
                if batch_to_prune is None:
                    raise ValueError(f"Batch ID {cbatch.batch_id} not found in cache.")

                if cbatch.HasField("status"):
                    pruned_batch = self.model.batch_type.prune(batch_to_prune, cbatch.status.completed_ids)
                    self.cache.set(pruned_batch)

                # Ensure completed batches are garbage collected before calling prefill
                del batch_to_prune

            is_healthcheck = request.batch.id == HEALTHCHECK_BATCH_ID

            if COMPACT_BEFORE_PREFILL and not is_healthcheck:
                self.cache.compact()

            # Construct new batch
            input_token_info = None
            batch, errors = self.model.batch_type.from_pb(
                request.batch,
                tokenizer=self.model.tokenizer,
                device=self.model.device,
                embeddings_lookup=self.model.word_embeddings,
                prefix_cache=self.model.prefix_cache,
                use_position_ids=self.model.use_position_ids,
            )

            batch_id = 0
            if batch is not None:
                for_concat = len(self.cache) > 0
                # Prefill and generate first token
                output_tokens, input_token_info, decode_errors = self.model.generate_token(
                    batch, first=True, for_concat=for_concat,
                )
                if not is_healthcheck:
                    self.cache.set(batch)
                batch_id = batch.get_id()
                if errors:
                    errors.extend(decode_errors)
                else:
                    errors = decode_errors
            else:
                output_tokens = []

            return generate_pb2.PrefillResponse(
                result=generate_pb2.GenerateResult(
                    output_tokens=[
                        output_token.to_pb() for output_token in output_tokens
                    ],
                    errors=[err.to_pb() for err in errors] if errors else None,
                    batch_id=batch_id,
                ),
                input_tokens=[
                    input_tokens.to_pb() for input_tokens in input_token_info
                ] if input_token_info is not None else None,
            )

    @log_errs
    async def NextToken(self, request: generate_pb2.NextTokenRequest, context) -> generate_pb2.NextTokenResponse:
        if len(request.batches) == 0:
            raise ValueError("Must provide at least one batch")

        with self.model.context_manager():
            batches = []
            for cbatch in request.batches:
                batch = self.cache.pop(cbatch.batch_id)
                if cbatch.HasField("status"):
                    if batch is None:
                        raise ValueError(f"Batch ID {cbatch.batch_id} not found in cache.")
                    batch = self.model.batch_type.prune(batch, cbatch.status.completed_ids)
                    if batch is not None:
                        batches.append(batch)

            if len(self.cache) > 0:
                print(f"WARN: Clearing additional batches found in cache: {self.cache.keys()}")
                self.cache.clear()

            if len(batches) == 0:
                # All batches finished, nothing to do
                return generate_pb2.NextTokenResponse()

            batch = batches[0] if len(batches) == 1 else self.model.batch_type.concatenate(batches)

            # Ensure batches are garbage-collected post-concatenation
            del batches

            output_tokens, _, errors = self.model.generate_token(batch)
            self.cache.set(batch)

            return generate_pb2.NextTokenResponse(
                result=generate_pb2.GenerateResult(
                    output_tokens=[
                        output_token.to_pb() for output_token in output_tokens
                    ],
                    errors=[err.to_pb() for err in errors] if errors else None,
                    batch_id=batch.get_id(),
                )
            )


def serve(
    model_name: str,
    revision: Optional[str],
    deployment_framework: str,
    dtype_str: Optional[str],
    max_sequence_length: int,
    max_new_tokens: int,
    max_batch_size: int,
    max_batch_weight: Optional[int],
    sharded: bool,
    cuda_process_memory_fraction: float,
    uds_path: Path,
):
    async def serve_inner(
        model_name: str,
        revision: Optional[str],
        deployment_framework: str,
        dtype_str: Optional[str],
        max_sequence_length: int,
        max_new_tokens: int,
        max_batch_size: int,
        max_batch_weight: Optional[int],
        sharded: bool = False,
    ):
        unix_socket_template = "unix://{}-{}"
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        local_rank = int(os.getenv("RANK", "0"))
        server_urls = [
            unix_socket_template.format(uds_path, rank)
            for rank in range(world_size)
        ]
        local_url = server_urls[local_rank]

        # Set the fraction of cuda/gpu mem available to this process, then load the model
        if torch.cuda.is_available() and cuda_process_memory_fraction < 1:
            torch.cuda.set_per_process_memory_fraction(cuda_process_memory_fraction)

        # Default dtype based on device if not provided
        if dtype_str is None:
            dtype_str = "float16" if torch.cuda.is_available() else "float32"

        model = get_model(model_name, revision, deployment_framework, dtype_str)

        if local_rank == 0:
            print(f"Loaded model as {type(model)}")
            print(f"With model class {type(model.model)}")
            print(f"Using engine {type(model.engine)}")
            device = model.engine.get_device()
            print(f"Using device {device}, dtype {dtype_str}")
            print(model.config.__str__())
            if device.type == "cuda":
                # Log GPU memory stats at startup
                device = model.engine.get_device()
                print(f"Cuda process memory fraction: {cuda_process_memory_fraction}")
                print(torch.cuda.memory_summary(device=device))
                # Start a thread to log GPU usage if configured
                interval = float(os.getenv("LOG_GPU_USAGE_INTERVAL", "0"))
                if interval > 0.0:
                    t = threading.Thread(target=partial(log_gpu_stats, device, interval))
                    t.start()

        if model.compiled:

            # trigger pt2 compile for variety of tensor shapes
            print("Warming up PyTorch 2 compile...")
            warmup_t0 = time.time()
            pt2_compile_warmup(
                model=model,
                max_batch_size=max_batch_size,
                max_new_tokens=max_new_tokens,
                max_sequence_length=max_sequence_length,
                max_batch_weight=max_batch_weight,
                n_samples=10,
                verbose=True,
            )
            warmup_t_elap = (time.time()-warmup_t0)/60.0
            print(f"Time spent during compile warmup: {warmup_t_elap:.2f} minutes")

            # no more compilations can occur after this
            model.freeze_compile()

        server = aio.server()
        generate_pb2_grpc.add_TextGenerationServiceServicer_to_server(
            TextGenerationService(model, Cache(), server_urls), server
        )
        # SERVICE_NAMES = (
        #     generate_pb2.DESCRIPTOR.services_by_name["TextGenerationService"].full_name,
        #     reflection.SERVICE_NAME,
        # )
        # reflection.enable_server_reflection(SERVICE_NAMES, server)
        server.add_insecure_port(local_url)
        await server.start()
        print("Server started at {}".format(local_url))
        try:
            await server.wait_for_termination()
        except KeyboardInterrupt:
            print("Signal received. Shutting down")
            await server.stop(0)

    asyncio.run(
        serve_inner(
            model_name,
            revision,
            deployment_framework,
            dtype_str,
            max_sequence_length,
            max_new_tokens,
            max_batch_size,
            max_batch_weight,
        )
    )


def log_gpu_stats(device="cuda:0", interval: float = 2.0):
    # dedicated thread to log gpu stats every couple of seconds
    while True:
        mem_info = torch.cuda.mem_get_info(device)
        total_used = (mem_info[1] - mem_info[0]) / 1024 / 1024
        percent = mem_info[0] / mem_info[1]
        alloc = torch.cuda.memory_allocated(device) / 1024 / 1024
        alloc_max = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(device) / 1024 / 1024
        reserved_max = torch.cuda.max_memory_reserved(device) / 1024 / 1024
        # print(f"{datetime.now().strftime('%H:%M:%S')} CUDA MEM (MiBs): Alloc: {alloc:.2f} / {alloc_max:.2f}, "
        #       f"Reserved: {reserved:.2f} / {reserved_max:.2f}, Total: {total_used:.2f}, {percent:.1%} free")
        # Log CSV lines for now: X, time, alloc MiB, max alloc MiB, reserved MiB, max reserved MiB, total used MiB, pct free
        print(f"CUDA_MEM,{datetime.now().strftime('%H:%M:%S')},{alloc:.2f},{alloc_max:.2f},"
              f"{reserved:.2f},{reserved_max:.2f},{total_used:.2f},{percent:.3}")
        time.sleep(interval)
