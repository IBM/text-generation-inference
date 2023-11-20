import os
import torch
import intel_extension_for_pytorch as ipex
from transformers.models.auto.auto_factory import _BaseAutoModelClass

from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE
from typing import Any, Optional


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: type[_BaseAutoModelClass],
        dtype: torch.dtype,
        model_config: Optional[Any]
    ) -> None:
        super().__init__(model_path, model_config)

        kwargs = {
            "pretrained_model_name_or_path": model_path,
            "local_files_only": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "torchscript": 'jit',
            "torch_dtype": dtype
        }

        if model_config.model_type == "mpt":
            model_config.init_device = str(self.device)
            kwargs["config"] = model_config

        # if dtype == torch.int8:
        #     # using LLM.int8()
        #     kwargs["load_in_8bit"] = True
        # else:
        #     kwargs["torch_dtype"] = dtype

        try:
            ipex._C.disable_jit_linear_repack()
        except Exception:
            pass

        torch._C._jit_set_texpr_fuser_enabled(False)

        slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
        if slow_but_exact:
            kwargs["slow_but_exact"] = True

        with self.device:
            self.model = model_class.from_pretrained(**kwargs).requires_grad_(False).eval()

            self.model = self.model.to(memory_format=torch.channels_last)
            # self.model = ipex.optimize_transformers(self.model, dtype=dtype)
            self.model = ipex.optimize_transformers(self.model, dtype=dtype, inplace=True)
            print('Intel(R) Extension for PyTorch* enabled')

            self.model.to(self.device)