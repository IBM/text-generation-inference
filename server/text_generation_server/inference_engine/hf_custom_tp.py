import os

import torch
import torch.distributed

from typing import List, Optional, Any

from accelerate import init_empty_weights
from safetensors import safe_open
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from transformers.models.bloom.parallel_layers import (
    TensorParallelColumnLinear as BloomTensorParallelColumnLinear,
    TensorParallelEmbedding as BloomTensorParallelEmbedding,
    TensorParallelRowLinear as BloomTensorParallelRowLinear,
)
from transformers.models.t5.parallel_layers import (
    TensorParallelRowLinear as T5TensorParallelRowLinear,
    TensorParallelColumnLinear as T5TensorParallelColumnLinear,
    TensorParallelEmbedding as T5TensorParallelEmbedding,
)
from text_generation_server.models.custom_modeling.flash_llama_modeling import (
    TensorParallelEmbedding as LlamaTensorParallelEmbedding,
    TensorParallelRowLinear as LlamaTensorParallelRowLinear,
    TensorParallelColumnLinear as LlamaTensorParallelColumnLinear,
)

from text_generation_server.utils.layers import (
    TensorParallelColumnLinear,
    TensorParallelRowLinear,
    TensorParallelEmbedding,
)

from text_generation_server.inference_engine import BaseInferenceEngine
from text_generation_server.utils.dist import initialize_torch_distributed
from text_generation_server.utils.hub import local_weight_files


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: type[_BaseAutoModelClass],
        dtype: torch.dtype,
        model_config: Optional[Any],
    ) -> None:
        super().__init__(model_path, model_config)

        pass_process_group = False
        if self._config.model_type == "bloom":
            slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
            self._config.slow_but_exact = slow_but_exact
            load_weights = self.load_weights_bloom
        elif self._config.model_type == "t5":
            load_weights = self.load_weights_t5
        elif self._config.model_type in ["RefinedWeb", "RefinedWebModel"]:
            if model_class.__name__ != "FlashRWForCausalLM":
                raise ValueError(f"RW model TP only supported with FlashAttention")
            pass_process_group = True
            load_weights = self.load_weights_rw
        elif self._config.model_type == "llama":
            if model_class.__name__ != "FlashLlamaForCausalLM":
                raise ValueError(f"Llama model TP only supported with FlashAttention")
            pass_process_group = True
            load_weights = self.load_weights_llama

        else:
            raise ValueError(f"Custom TP currently only supported by bloom and t5-type models")

        quantize = dtype == torch.int8
        if quantize:
            try:
                import bitsandbytes as bnb
                from bitsandbytes.nn import Int8Params
            except:
                raise ImportError(
                    "bitsandbytes is not available on your machine either because it is not installed "
                    "or you don't have a GPU.\n"
                    "You can install it with `pip install bitsandbytes`."
                )

        self.process_group = initialize_torch_distributed(self.world_size, self.rank)
        self.master = self.rank == 0

        self._config.tp_parallel = True

        torch.distributed.barrier(group=self.process_group)
        filenames = local_weight_files(model_path, extension=".safetensors")
        if not filenames:
            raise ValueError("No safetensors weights found")

        kwargs = {"process_group": self.process_group} if pass_process_group else {}

        with init_empty_weights():
            model = model_class.from_config(self._config, **kwargs)

        torch.distributed.barrier(group=self.process_group)
        load_weights(
            model,
            filenames,
            quantize=quantize,
            device=self.device,
            dtype=dtype,
            rank=self.rank,
            world_size=self.world_size,
        )

        self.model = model.eval()  # .to(dtype)
        torch.distributed.barrier(group=self.process_group)

    def process_logits(self, output_logits: torch.Tensor) -> torch.Tensor:
        # Logits are sharded, so we need to gather them
        logits = [torch.empty_like(output_logits) for _ in range(self.world_size)]
        torch.distributed.all_gather(logits, output_logits, group=self.process_group)
        return torch.cat(logits, dim=2)

    @staticmethod
    def load_weights_bloom(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    if name.startswith("transformer.") or name.startswith("lm_head."):
                        full_name = name
                    else:
                        full_name = f"transformer.{name}"

                    module_name, param_name = full_name.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                    current_tensor = parameters[full_name]

                    slice_ = f.get_slice(name)

                    if isinstance(module, BloomTensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, BloomTensorParallelRowLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[1]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[:, start:stop]
                        else:
                            tensor = slice_[:]
                            # XXX: Hack for Rowlinear to add the bias only once.
                            if rank != 0:
                                tensor = torch.zeros_like(tensor)
                    elif isinstance(module, BloomTensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "lm_head.weight":
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        tensor = slice_[:]

                    if current_tensor.shape != tensor.shape:
                        raise ValueError(
                            f"Name {name} -- Current {current_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous().to(dtype)

                    if quantize:
                        tensor = InferenceEngine.quantize_tensor(module, device, param_name, tensor)

                    module._parameters[param_name] = tensor
                    if name == "word_embeddings.weight":
                        model.lm_head._parameters["weight"] = tensor


    @staticmethod
    def load_weights_t5(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    module_name, param_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_name)
                    current_parameter_tensor = parameters.get(name, None)

                    slice_ = f.get_slice(name)

                    if isinstance(module, T5TensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, T5TensorParallelRowLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[1]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[:, start:stop]
                        else:
                            tensor = slice_[:]
                            # XXX: Hack for Rowlinear to add the bias only once.
                            if rank != 0:
                                tensor = torch.zeros_like(tensor)
                    elif isinstance(module, T5TensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "lm_head.weight":
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif "relative_attention_bias.weight" in name:
                        size = slice_.get_shape()[1]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[:, start:stop]
                    else:
                        try:
                            tensor = slice_[:]
                        except:
                            tensor = f.get_tensor(name)

                    if (
                        current_parameter_tensor is not None
                        and current_parameter_tensor.shape != tensor.shape
                    ):
                        raise ValueError(
                            f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous()

                    # See: https://github.com/huggingface/transformers/blob/1fe1e3caa44617047f149bcc0c0b566343b714a7/src/transformers/models/t5/modeling_t5.py#LL316C15-L316C71
                    tensor = tensor.to(torch.float32 if module_name.endswith("wo") else dtype)

                    if quantize and not module_name.endswith("wo"):
                        tensor = InferenceEngine.quantize_tensor(module, device, param_name, tensor)

                    if current_parameter_tensor is not None:
                        module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor


    @staticmethod
    def load_weights_rw(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        parameters = dict(model.named_parameters())
        for file in filenames:
            with safe_open(
                file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    module_name, param_name = name.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    current_parameter_tensor = parameters.get(name, None)

                    slice_ = f.get_slice(name)

                    if isinstance(module, TensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, TensorParallelRowLinear):
                        if param_name == "weight":
                            size = slice_.get_shape()[1]
                            block_size = size // world_size
                            start = rank * block_size
                            stop = (rank + 1) * block_size
                            tensor = slice_[:, start:stop]
                        else:
                            tensor = slice_[:]
                            # XXX: Hack for Rowlinear to add the bias only once.
                            if rank != 0:
                                tensor = torch.zeros_like(tensor)
                    elif isinstance(module, TensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "lm_head.weight" and model.transformer.tp_embeddings:
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        try:
                            tensor = slice_[:]
                        except:
                            tensor = f.get_tensor(name)

                    if (
                        current_parameter_tensor is not None
                        and current_parameter_tensor.shape != tensor.shape
                    ):
                        raise ValueError(
                            f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                        )

                    tensor = tensor.contiguous().to(dtype)

                    if current_parameter_tensor is not None:
                        module._parameters[param_name] = tensor
                    else:
                        module._buffers[param_name] = tensor

        model.post_load_weights(quantize)

    @staticmethod
    def load_weights_llama(
        model,
        filenames: List[str],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
        rank: int,
        world_size: int,
    ):
        for file in filenames:
            with safe_open(
                    file, framework="pt", device=str(device) if not quantize else "cpu"
            ) as f:
                for name in f.keys():
                    slice_ = f.get_slice(name)

                    layer_name = ".".join(name.split(".")[:4])

                    # Fused qkv
                    if "q_proj" in name or "k_proj" in name or "v_proj" in name:
                        final_name = layer_name + ".query_key_value.weight"

                    # Fused gate and up projs
                    elif "gate_proj" in name or "up_proj" in name:
                        final_name = layer_name + ".gate_up_proj.weight"
                    else:
                        final_name = name

                    module_name, param_name = final_name.rsplit(".", 1)
                    module = model.get_submodule(module_name)

                    if isinstance(module, LlamaTensorParallelColumnLinear):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif isinstance(module, LlamaTensorParallelRowLinear):
                        size = slice_.get_shape()[1]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[:, start:stop]
                    elif isinstance(module, LlamaTensorParallelEmbedding):
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    elif name == "lm_head.weight" and model.model.tp_embeddings:
                        size = slice_.get_shape()[0]
                        block_size = size // world_size
                        start = rank * block_size
                        stop = (rank + 1) * block_size
                        tensor = slice_[start:stop]
                    else:
                        try:
                            tensor = slice_[:]
                        except:
                            tensor = f.get_tensor(name)

                    tensor = tensor.contiguous().to(dtype)

                    try:
                        current_parameter_tensor = module._parameters[param_name]
                    except KeyError:
                        current_parameter_tensor = None

                    if current_parameter_tensor is not None:
                        if current_parameter_tensor.device == torch.device("meta"):
                            # Init qkv
                            if "query_key_value" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (tensor.shape[0] * 3, tensor.shape[1])
                                )
                            # Init gate and up proj
                            elif "gate_up_proj" in final_name:
                                module._parameters[param_name] = tensor.new_empty(
                                    (tensor.shape[0] * 2, tensor.shape[1])
                                )

                        # Init gate and up proj
                        if "q_proj" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "k_proj" in name:
                            module._parameters[param_name][
                            tensor.shape[0] : tensor.shape[0] * 2
                            ] = tensor
                        elif "v_proj" in name:
                            module._parameters[param_name][
                            tensor.shape[0] * 2 :
                            ] = tensor
                        elif "gate_proj" in name:
                            module._parameters[param_name][: tensor.shape[0]] = tensor
                        elif "up_proj" in name:
                            module._parameters[param_name][tensor.shape[0] :] = tensor
                        else:
                            if current_parameter_tensor.shape != tensor.shape:
                                raise ValueError(
                                    f"Name {name} -- Current {current_parameter_tensor.shape} and got {tensor.shape}"
                                )

                            module._parameters[param_name] = tensor

                    else:
                        module._buffers[param_name] = tensor

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)


    @staticmethod
    def quantize_tensor(
        module,
        device: torch.device,
        param_name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        if (
            type(module)
            not in [
                TensorParallelRowLinear,
                TensorParallelColumnLinear,
                T5TensorParallelRowLinear,
                T5TensorParallelColumnLinear,
            ]
            or param_name != "weight"
        ):
            return tensor.to(device)

        tensor = Int8Params(
            tensor,
            has_fp16_weights=False,
            requires_grad=False,
        ).to(device)
        state = bnb.MatmulLtState()
        state.threshold = 6.0
        state.has_fp16_weights = False
        state.memory_efficient_backward = False
        state.use_pool = True
        state.CB = tensor.CB
        state.SCB = tensor.SCB
        tensor.CB = None
        tensor.SCB = None

        def replace_linear(state):
            def linear(input, weight, bias):
                out = bnb.matmul(
                    input,
                    weight,
                    state=state,
                    threshold=state.threshold,
                    bias=bias,
                )

                if state.CB is not None:
                    # we converted 8-bit row major to turing/ampere format
                    # in the first inference pass
                    # we no longer need the row-major weight
                    del state.CB
                    weight.data = state.CxB

                return out

            return linear

        module.linear = replace_linear(state)

        return tensor
