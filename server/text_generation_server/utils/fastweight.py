# Copyright 2024 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0

import os
import glob
import torch
import torch.distributed as dist
from typing import List, Optional, Dict, Tuple

def get_config(device_index: int) -> Tuple[bool, int, int]:
    auto_config = os.getenv("FST_CONFIG", "auto")
    nogds = os.getenv("FST_NOGDS") # disable GDS if FST_NOGDS==1
    nogds = nogds is not None and nogds == "1"
    max_copier_threads = int(os.getenv("FST_THREADS", "16"))           # number of copy threads at host CPU
    bbuf_size_kb_total = int(os.getenv("FST_BBUF_SIZE_KB", "163840")) # size of bounce buffer at host memory for FST_NOGDS==1
    if auto_config == "auto":
        nogds = not os.path.exists("/run/udev") # udev directory is required for GDS
        from fastsafetensors.common import get_device_numa_node
        node = get_device_numa_node(device_index)
        total_l2_size = 0
        phys_cpus = {}
        failed = False
        for cpudir in glob.glob(f"/sys/devices/system/node/node{node}/cpu[0-9]*"):
            try:
                with open(f"{cpudir}/cache/index2/size") as f: # L2 cache size for a cpu
                    size_str = f.read().strip()
                    if size_str[-1] != "K":
                        raise Exception(f"cannot parse {cpudir}/cache/index2/size")
                    total_l2_size += int(size_str[:-1])
                with open(f"{cpudir}/topology/core_id") as f: # physical core ID
                    phys_cpus[f.read().strip()] = True
            except Exception as e:
                failed = True
                print(f"Failed to auto-configure fastsafetensors. reason: {e}")
                break
        if not failed and total_l2_size > 0:
            bbuf_size_kb_total = total_l2_size
        if not failed and len(phys_cpus) > 0:
            max_copier_threads = len(phys_cpus)
    return (nogds, max_copier_threads, bbuf_size_kb_total)

class FastWeights:
    def __init__(self, filenames:List[str],
                 device: torch.device,
                 dtype: torch.dtype,
                 pg: dist.ProcessGroup,
                 debug_log: bool=False,
                 aliases: Optional[Dict[str, List[str]]] = None,
                 prefix: Optional[str] = None,
            ):
        from fastsafetensors.loader import SafeTensorsFileLoader
        (nogds, max_copier_threads, bbuf_size_kb_total) = get_config(device.index)
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
        msg = f"Fastsafetensors configuration: GDS={not nogds}, maximum number of file copy threads={max_copier_threads}, copy block size={max_copy_block_size}B"
        if nogds:
            msg += f", total bounce buffer size={bbuf_size_kb_total * 1024}B"
        print(msg)
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