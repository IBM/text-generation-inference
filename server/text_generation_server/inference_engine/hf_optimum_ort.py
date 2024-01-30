import os
import time

import torch
from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Union, Any, Optional

from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTOptimizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Optional[Any],
        max_sequence_length: Optional[int],
    ) -> None:
        super().__init__(model_path, model_config)

        if model_class == AutoModelForCausalLM:
            model_class = ORTModelForCausalLM
        elif model_class == AutoModelForSeq2SeqLM:
            model_class = ORTModelForSeq2SeqLM

        is_cuda = self.device.type == "cuda"

        # "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"
        provider = "CUDAExecutionProvider" if is_cuda else "CPUExecutionProvider"

        kwargs = {
            "model_id": model_path,
            "local_files_only": True,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "export": False,
            "provider": provider,
        }

        # if dtype == torch.int8:
        #     # using LLM.int8()
        #     kwargs["load_in_8bit"] = True
        # else:
        #     kwargs["torch_dtype"] = dtype

        # 1. Check for onnx file in provided path, assume already converted if there
        model_is_onnx = any(f.endswith(".onnx") for f in os.listdir(model_path))

        if model_is_onnx:
            print("Found .onnx file in model directory, loading ONNX model directly")
            load_start = time.time()
            self.model = model_class.from_pretrained(**kwargs)
            print(f"Load of ONNX model took {time.time() - load_start:.3f}s")
            print(f"Providers: {self.model.providers}")

        else:
            merge_graphs = os.getenv("MERGE_ONNX_GRAPHS", "false").lower() == "true"
            optimize = os.getenv("OPTIMIZE_ONNX_MODEL", "false").lower() == "true"
            # Note it's currently not supported for both of these to be true - optimization will fail
            print(
                f"Converting transformers model to ONNX; merge_graphs={merge_graphs}, optimize={optimize}"
            )
            kwargs["export"] = True
            kwargs["use_merged"] = merge_graphs

            slow_but_exact = (
                os.getenv("BLOOM_SLOW_BUT_EXACT", "false").lower() == "true"
            )
            if slow_but_exact:
                kwargs["slow_but_exact"] = True

            convert_start = time.time()
            self.model = model_class.from_pretrained(**kwargs)
            print(
                f"Conversion to ONNX and initial loading took {time.time() - convert_start:.3f}s"
            )
            print(f"Providers: {self.model.providers}")

            if optimize:
                # TODO auto-cache conversions

                optimized_model_path = "/tmp/onnx_model_optimized"
                os.makedirs(optimized_model_path, exist_ok=True)
                optimize_start = time.time()

                optimizer = ORTOptimizer.from_pretrained(self.model)

                # TODO make optimization level configurable
                if is_cuda:
                    # O4 includes fp16 weight optimization
                    # Shape inference must be disabled for onnxruntime <= 1.14.1 due to a bug
                    optimization_config = AutoOptimizationConfig.O3(
                        for_gpu=True, disable_shape_inference=True
                    )
                else:
                    # For now just O1 for CPU
                    optimization_config = AutoOptimizationConfig.O1()

                print("Starting model optimization step")
                optimizer.optimize(
                    save_dir=optimized_model_path,
                    optimization_config=optimization_config,
                )
                print(
                    f"ORT Model optimization took {time.time() - optimize_start:.3f}s"
                )

                load_start = time.time()
                self.model = model_class.from_pretrained(
                    optimized_model_path, provider=provider, local_files_only=True
                )
                print(f"Load of optimized model took {time.time() - load_start:.3f}s")
