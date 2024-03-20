import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import torch
from safetensors import safe_open, SafetensorError
from loguru import logger
import json


QUANTIZE_CONFIG_FILENAME = "quantize_config.json"

def unpack(x, dim, bits=4):
    return unpack_row(x, bits) if dim == 0 else unpack_col(x, bits)

def unpack_col(x, bits):
    mask = 2**bits - 1
    pack_size = 32 // bits
    unpacked_x = torch.zeros((x.shape[0], x.shape[1]*pack_size), dtype=torch.int)
    for i in range(pack_size):
        unpacked_x[:, i::pack_size] = (x >> (i*bits)) & (mask)
    return unpacked_x

def unpack_row(x, bits):
    mask = 2**bits - 1
    pack_size = 32 // bits
    unpacked_x = torch.zeros((x.shape[0]*pack_size, x.shape[1]), dtype=torch.int)
    for i in range(pack_size):
        unpacked_x[i::pack_size] = (x >> (i*bits)) & (mask)
    return unpacked_x


def pack(x, dim, bits=4):
    return pack_row(x, bits) if dim == 0 else pack_col(x, bits)

def pack_col(x, bits):
    mask = 2**bits - 1
    pack_size = 32 // bits
    packed_x = torch.zeros((x.shape[0], x.shape[1]//pack_size), dtype=torch.int)
    for i in range(pack_size):
        packed_x |= (x[:, i::pack_size] & mask) << (i*bits)
    return packed_x

def pack_row(x, bits):
    mask = 2**bits - 1
    pack_size = 32 // bits
    packed_x = torch.zeros((x.shape[0]//pack_size, x.shape[1]), dtype=torch.int)
    for i in range(pack_size):
        packed_x |= (x[i::pack_size] & mask) << (i*bits)
    return packed_x

class Weights:
    def __init__(
        self,
        filenames: List[Path],
        device,
        dtype,
        process_group,
        aliases: Optional[Dict[str, List[str]]] = None,
    ):
        routing = {}
        for filename in filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    if k in routing:
                        raise RuntimeError(
                            f"Key {k} was found in multiple files: {filename} and {routing[k]}"
                        )
                    routing[k] = filename
        if aliases is None:
            aliases = {}
        self.aliases = aliases
        self.routing = routing
        self.device = device
        self.dtype = dtype
        self.process_group = process_group
        self._handles = {}

    def _get_handle(self, filename):
        if filename not in self._handles:
            f = safe_open(filename, framework="pytorch")
            self._handles[filename] = f

        return self._handles[filename]

    def get_filename(self, tensor_name: str) -> (str, str):
        filename = self.routing.get(tensor_name, None)
        if filename is None:
            aliases = self.aliases.get(tensor_name, [])
            for alias in aliases:
                filename = self.routing.get(alias, None)
                if filename is not None:
                    return str(filename), alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return str(filename), tensor_name

    def _get_slice(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        return slice_

    def get_shape(self, tensor_name: str):
        return self._get_slice(tensor_name).get_shape()

    def get_tensor(self, tensor_name: str):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype not in [torch.int32, torch.int64]:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_partial_sharded(self, tensor_name: str, dim: int):
        filename, tensor_name = self.get_filename(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        size = slice_.get_shape()[dim]
        block_size = size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        if dim == 0:
            tensor = slice_[start:stop]
        elif dim == 1:
            tensor = slice_[:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_sharded(self, tensor_name: str, dim: int, perm=None, packed=False):
        filename, tensor_name = self.get_filename(tensor_name)
        f = self._get_handle(filename)
        slice_ = f.get_slice(tensor_name)
        world_size = self.process_group.size()
        size = slice_.get_shape()[dim]
        assert (
            size % world_size == 0
        ), f"The choosen size {size} is not compatible with sharding on {world_size} shards"
        if perm is None:
            return self.get_partial_sharded(tensor_name, dim)
        else:
            return self.get_shuffle_sharded(tensor_name, dim, perm, packed)

    def get_shuffle_sharded(self, tensor_name: str, dim: int, perm, packed: bool):
        filename, tensor_name = self.get_filename(tensor_name)
        world_size = self.process_group.size()
        rank = self.process_group.rank()

        f = self._get_handle(filename)
        tensor = f.get_tensor(tensor_name)
        perm = perm.to(device=tensor.device)
        size = tensor.shape[dim]
        block_size = size // world_size
        start = rank * block_size
        stop = (rank + 1) * block_size

        # TODO: pack-unpack on cuda to speed up this part
        if dim == 0:
            if packed:
                tensor = pack(unpack(tensor, dim)[perm], dim)[start:stop]
            else:
                tensor = tensor[perm][start:stop]
        elif dim == 1:
            if packed:
                tensor = pack(unpack(tensor, dim)[:, perm], dim)[:, start:stop]
            else:
                tensor = tensor[:, perm][:, start:stop]
        else:
            raise NotImplementedError("Let's make that generic when needed")
        # Special case for gptq which shouldn't convert
        # u4 which are disguised as int32
        if tensor.dtype != torch.int32:
            tensor = tensor.to(dtype=self.dtype)
        tensor = tensor.to(device=self.device)
        return tensor

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int, col_perm=None):
        if quantize == "gptq":
            try:
                qweight = torch.cat([self.get_sharded(f"{p}.qweight", dim=1, perm=col_perm, packed=False) for p in prefixes], dim=1)
            except RuntimeError:
                raise RuntimeError("Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`")

            qzeros = torch.cat([self.get_sharded(f"{p}.qzeros", dim=1, perm=col_perm, packed=True) for p in prefixes], dim=1)
            scales = torch.cat([self.get_sharded(f"{p}.scales", dim=1, perm=col_perm, packed=False) for p in prefixes], dim=1)
            w = [self.get_tensor(f"{p}.g_idx") for p in prefixes]
            for w2 in w[1:]:
                torch.testing.assert_close(w2, w[0])
            g_idx = w[0]

            bits, groupsize = self._get_gptq_params()
            use_gptq_cuda = False
            if bits == 4:
                from text_generation_server.utils.layers import HAS_GPTQ_CUDA

                use_gptq_cuda = HAS_GPTQ_CUDA
                if use_gptq_cuda:
                    logger.info(f"Using GPTQ cuda kernels for col {prefixes}")

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_gptq_cuda)
        else:
            w = [self.get_sharded(f"{p}.weight", dim=0) for p in prefixes]
            weight = torch.cat(w, dim=dim)
        return weight

    def get_multi_weights_row(self, prefix: str, quantize: str, row_perm=None, noshard=False):
        if quantize == "gptq":
            bits, groupsize = self._get_gptq_params()

            from text_generation_server.utils.layers import HAS_GPTQ_CUDA
            is_preshuffle = (row_perm != None)
            is_masked_matmul = noshard
            assert (is_preshuffle != is_masked_matmul
                    or not (is_preshuffle or is_masked_matmul)), f"TP-aware optimization can't both be enabled at the same time {is_preshuffle=}, {is_masked_matmul=}"
            use_gptq_cuda = (bits == 4) and HAS_GPTQ_CUDA or (is_preshuffle or is_masked_matmul)
            if self.process_group.rank == 0:
                if use_gptq_cuda:
                    logger.info(f"Using GPTQ cuda kernels for row {prefix}")
                else:
                    logger.warning(
                        "GPTQ cuda kernels (which are faster) could have been used, but are disabled via the DISABLE_EXLLAMA env var,"
                        " or not currently installed, try using BUILD_EXTENSIONS=True"
                    )
            try:
                qweight = self.get_sharded(f"{prefix}.qweight",
                                           dim=0,
                                           perm=row_perm if use_exllama else None,
                                           packed=True,
                ) if not is_masked_matmul else self.get_tensor(f"{prefix}.qweight")
            except RuntimeError:
                raise RuntimeError("Cannot load `gptq` weight, make sure the model is already quantized, or quantize it with `text-generation-server quantize ORIGINAL_MODEL_ID NEW_MODEL_ID`")

            if use_gptq_cuda:
                if groupsize >= 0 and not is_masked_matmul:
                    # Exllama reorders the weights in advance and the activations on the fly, thus
                    # the scales and zero-points do not need to be reordered.
                    qzeros = self.get_sharded(f"{prefix}.qzeros", dim=0)
                    scales = self.get_sharded(f"{prefix}.scales", dim=0)
                else:
                    qzeros = self.get_tensor(f"{prefix}.qzeros")
                    scales = self.get_tensor(f"{prefix}.scales")

                # For tp > 1, at this point we know we do not use act-order
                if (self.process_group.size() == 1 or is_masked_matmul) and not is_preshuffle:
                    g_idx = self.get_tensor(f"{prefix}.g_idx")
                else:
                    g_idx = None
            else:
                # The triton kernel reorders the scales/zero points instead of the weight/activation.
                # Thus, each rank needs the full qzeros/scales.

                qzeros = self.get_tensor(f"{prefix}.qzeros")
                scales = self.get_tensor(f"{prefix}.scales")
                g_idx = self.get_sharded(f"{prefix}.g_idx", dim=0)

            weight = (qweight, qzeros, scales, g_idx, bits, groupsize, use_gptq_cuda)
        else:
            weight = self.get_sharded(f"{prefix}.weight", dim=1) if not noshard else self.get_tensor(f"{prefix}.weight")
        return weight

    def _get_gptq_params(self) -> Tuple[int, int]:
        try:
            bits = self.get_tensor("gptq_bits").item()
            groupsize = self.get_tensor("gptq_groupsize").item()
        except (SafetensorError, RuntimeError) as e:
            try:
                bits = self.gptq_bits
                groupsize = self.gptq_groupsize
            except Exception:
                raise e

        return bits, groupsize

    def _set_gptq_params(self, model_config: Any, model_path: str):
        # Get quantization config from model's configuration
        # or else look for quantize_config.json in the model dir
        config = model_config.to_dict()
        quantize_config = config.get("quantization_config")
        if quantize_config is None:
            filename = os.path.join(model_path, QUANTIZE_CONFIG_FILENAME)
            if not os.path.exists(filename):
                return
            with open(filename, "r") as f:
                quantize_config = json.load(f)

        self.gptq_bits = quantize_config["bits"]
        self.gptq_groupsize = quantize_config["group_size"]
