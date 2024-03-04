import os
from datetime import timedelta
from functools import partial
from typing import Any

import torch
import torch.distributed
import torch.distributed as dist

RANK = int(os.getenv("RANK", "0"))


class FakeBarrier:
    def wait(self):
        pass


class FakeGroup:
    def __init__(self, rank, size):
        self._rank = rank
        self._size = size

    def allreduce(self, *args, **kwargs):
        return FakeBarrier()

    def allgather(self, inputs, local_tensor, **kwargs):
        assert (
            len(inputs[0]) == len(local_tensor) == 1
        ), f"{len(inputs[0])} != {len(local_tensor)} != 1, and the FakeGroup is supposed to join on simple tensors"
        for input_ in inputs:
            input_[0].data = local_tensor[0].data
        return FakeBarrier()

    def barrier(self, *args, **kwargs):
        return FakeBarrier()

    def size(self):
        return self._size

    def rank(self):
        return self._rank


def run_rank_n(
    func: partial,
    barrier: bool = False,
    rank: int = 0,
    other_rank_output: Any = None,
) -> Any:
    # runs function on only process with specified rank
    if not dist.is_initialized():
        return func()
    output = func() if dist.get_rank() == rank else other_rank_output
    if barrier:
        dist.barrier()
    return output


def print_rank_n(*values, rank: int = 0) -> None:
    # print on only process with specified rank
    if rank == RANK:
        print(*values)


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dt = getattr(torch, dtype_str, None)
    if type(dt) != torch.dtype:
        raise ValueError(f"Unrecognized data type: {dtype_str}")
    return dt


def initialize_torch_distributed(world_size: int, rank: int):
    if world_size == 1 or os.getenv("DEBUG", None) == "1":
        return FakeGroup(rank, world_size)

    if not torch.distributed.is_initialized():
        if torch.cuda.is_available():
            from torch.distributed import ProcessGroupNCCL

            backend = "nccl"
            options = ProcessGroupNCCL.Options()
            options.is_high_priority_stream = True
            options._timeout = timedelta(seconds=60)
        else:
            backend = "gloo"
            options = None

        # Call the init process.
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=60),
            pg_options=options,
        )
    else:
        print("WARN: torch.distributed is already initialized")

    return torch.distributed.group.WORLD
