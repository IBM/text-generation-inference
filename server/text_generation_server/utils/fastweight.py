# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.distributed as dist
from typing import List, Tuple, Optional, Dict
from fastsafetensors.loader import SafeTensorsFileLoader

class FastWeights:
    def __init__(self, filenames:List[str],
                 device: torch.device,
                 dtype: torch.dtype,
                 pg: dist.ProcessGroup,
                 debug_log: bool=False,
                 aliases: Optional[Dict[str, List[str]]] = None,
                 prefix: Optional[str] = None,
                 nogds: bool = False,
                 max_copier_threads: int = 16, # should be same as the number of physical CPUs on a node
                 bbuf_size_kb_total = 160 * 1024, # should be same as L2 cache size
            ):
        self._loader = SafeTensorsFileLoader(pg, device, bbuf_size_kb=bbuf_size_kb_total//pg.size(), max_threads=max_copier_threads, nogds=nogds, debug_log=debug_log)
        rank_filenames: Dict[str, List[str]] = {rank: [] for rank in range(0, pg.size())}
        max_copy_block_size = 1
        total_size = 0
        for idx, filename in enumerate(sorted(filenames, key=lambda x: os.path.basename(x))):
            rank_filenames[idx % pg.size()].append(filename)
            s = os.stat(filename)
            total_size += s.st_size
            if max_copy_block_size < s.st_size:
                max_copy_block_size = s.st_size
        self._loader.add_filenames(rank_filenames)
        if len(filenames) < max_copier_threads:
            max_copy_block_size = total_size // pg.size() // max_copier_threads
            if max_copy_block_size % bbuf_size_kb_total*1024 > 0:
                max_copy_block_size = max_copy_block_size - max_copy_block_size % (bbuf_size_kb_total*1024) + (bbuf_size_kb_total*1024)
        self._fb = self._loader.copy_files_to_device(dtype, max_copy_block_size=max_copy_block_size)
        self.device = device
        self.dtype = dtype
        if aliases is None:
            aliases = {}
        self.prefix = prefix
        self.aliases = aliases
        self.process_group = pg

    def close(self):
        self._fb.close()
        self._loader.close()
        torch.cuda.empty_cache()

    def _get_alias(self, tensor_name: str)->str:
        if self._fb.get_filename(tensor_name) is None:
            for alias in self.aliases[tensor_name]:
                if self._fb.get_filename(alias) is not None:
                    return alias
            raise RuntimeError(f"weight {tensor_name} does not exist")
        return tensor_name

    def get_shape(self, tensor_name: str)->torch.Size:
        return self._fb.get_shape(self._get_alias(tensor_name))

    def get_tensor(self, tensor_name: str)->torch.Tensor:
        return self._fb.get_tensor(self._get_alias(tensor_name), device=self.device, dtype=self.dtype)

    def push_tensor(self, tensor_name: str, dst_rank: int)->torch.Tensor:
        return self._fb.push_tensor(self._get_alias(tensor_name), dst_rank, device=self.device, dtype=self.dtype)

    def get_partial_sharded(self, tensor_name: str, dim: int)->torch.Tensor:
        return self._fb.get_sharded(self._get_alias(tensor_name), dim, device=self.device, dtype=self.dtype)

    def get_sharded(self, tensor_name: str, dim: int=1)->torch.Tensor:
        return self._fb.get_sharded(self._get_alias(tensor_name), dim, device=self.device, dtype=self.dtype)

    def get_multi_weights_col(self, prefixes: List[str], quantize: str, dim: int)->torch.Tensor:
        if quantize in ["gptq", "awq"]:
            raise NotImplementedError("Quantization is not supported yet")
        tensor_names = [self._get_alias(f"{prefix}.weight") for prefix in prefixes]
        return self._fb.get_multi_cols(tensor_names, dim, device=self.device, dtype=self.dtype)

    def get_multi_weights_row(self, prefix: str, quantize: str)->torch.Tensor:
        if quantize in ["gptq", "awq"]:
            raise NotImplementedError("Quantization is not supported yet")
        return self._fb.get_sharded(self._get_alias(f"{prefix}.weight"), 1, device=self.device, dtype=self.dtype)