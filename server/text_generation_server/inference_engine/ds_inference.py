import glob
import io
import json
import os
from typing import Any, Optional

import torch

import deepspeed
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.dist import run_rank_n


# basic DeepSpeed inference model class for benchmarking
class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: type[_BaseAutoModelClass],
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Optional[Any],
        max_sequence_length: Optional[int],
    ) -> None:
        super().__init__(model_path, model_config)

        if dtype not in [torch.float16, torch.int8]:
            # currently ds-inference only supports fp16 CUDA kernels :(
            raise NotImplementedError(f"{dtype} is not yet supported by deepspeed")

        slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
        slow_but_exact = {"slow_but_exact": True} if slow_but_exact else {}

        # create dummy tensors for allocating space which will be filled with
        # the actual weights while calling deepspeed.init_inference in the
        # following code
        with deepspeed.OnDevice(dtype=torch.float16, device="meta"):
            self.model = model_class.from_config(
                self._config, torch_dtype=torch.bfloat16,
                **slow_but_exact
            )
        self.model = self.model.eval()

        checkpoints_json = os.path.join(model_path, "ds_inference_config.json")
        if not os.path.isfile(checkpoints_json):
            # for bigscience/bloom, sharding is done while loading the model
            # so this is much slower and for this we need to create a
            # checkpoints json
            checkpoints_json = TemporaryCheckpointsJSON(model_path).__enter__()

        self.model = deepspeed.init_inference(
            self.model,
            tensor_parallel={
                "enabled": True,
                "tp_size": self.world_size,
            },
            dtype=dtype,
            base_dir=model_path,
            checkpoint=checkpoints_json,
            replace_with_kernel_inject=True,
            # max_out_tokens=2048,  #TBD - default is 1024 I think
        )

        self.model = self.model.module


class TemporaryCheckpointsJSON:
    def __init__(self, model_path: str):
        self.tmp_directory = "tmp"
        self.tmp_file = os.path.join(self.tmp_directory, "checkpoints.json")
        self.model_path = model_path

    def write_checkpoints_json(self) -> None:
        os.makedirs(self.tmp_directory, exist_ok=True)
        with io.open(self.tmp_file, "w", encoding="utf-8") as f:
            data = {
                "type": "BLOOM",
                # glob in py 3.10 has a root_dir arg which would be useful here!
                "checkpoints": [os.path.basename(f) for f in glob.iglob(f"{self.model_path}/*.bin")],
                "version": 1.0
            }
            json.dump(data, f)

    def __enter__(self):
        run_rank_n(self.write_checkpoints_json, barrier=True)
        return self.tmp_file

    def __exit__(self, type, value, traceback):
        return
