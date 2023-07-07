import torch
import torch.distributed

from accelerate import init_empty_weights
from pathlib import Path
from typing import List, Union, Any, Optional

from text_generation_server.inference_engine import BaseInferenceEngine
from text_generation_server.models.custom_modeling.flash_santacoder_modeling import (
    FlashSantacoderForCausalLM,
)
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.utils.hub import get_model_path, local_weight_files


class FlashSantacoder(FlashCausalLM):
    def __init__(
        self,
        model_name: str,
        revision: Optional[str],
        dtype: torch.dtype,
        model_config: Union[Any] = None,
    ):
        self.present_pad = None

        model_path = get_model_path(model_name, revision)

        engine = BaseInferenceEngine(model_path, model_config)
        device = engine.get_device()

        if device.type != "cuda":
            raise NotImplementedError("FlashSantacoder is only available on GPU")

        # We do not use from_pretrained as we modified the model internal module layout
        filenames = local_weight_files(model_path, ".bin")

        with init_empty_weights():
            engine.model = FlashSantacoderForCausalLM(model_config)

        self.load_weights(
            engine.model,
            filenames,
            dtype == torch.int8,
            device,
            dtype,
            model_config.architectures[0].startswith("GPT2"),
        )
        self.model = engine.model.eval().to(device)

        super(FlashCausalLM, self).__init__(engine, dtype)
        self.use_position_ids = True

        if self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.model.config.eos_token_id

    @staticmethod
    def load_weights(
        model: FlashSantacoderForCausalLM,
        filenames: List[Path],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
        transpose: bool,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device if not quantize else "cpu").to(dtype)

                layer_name = ".".join(key.split(".")[:4])

                # Fused qkv
                if "q_attn.weight" in key or "kv_attn.weight" in key:
                    final_key = layer_name + ".c_attn.weight"
                elif "q_attn.bias" in key or "kv_attn.bias" in key:
                    final_key = layer_name + ".c_attn.bias"
                else:
                    final_key = key

                module_name, param_name = final_key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                except KeyError:
                    current_parameter_tensor = None

                if current_parameter_tensor is not None:
                    if transpose and (
                        "c_fc.weight" in key
                        or "c_proj.weight" in key
                        or "q_attn.weight" in key
                        or "kv_attn.weight" in key
                        or "c_attn.weight" in key
                    ):
                        # Tranpose as we use nn.Linear instead of Conv1D
                        value = value.T

                    if current_parameter_tensor.device == torch.device("meta"):
                        # Init qkv
                        if "c_attn.weight" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (
                                    model.transformer.head_size
                                    * (model.transformer.num_heads + 2),
                                    value.shape[1],
                                )
                            )
                        elif "c_attn.bias" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (
                                    model.transformer.head_size
                                    * (model.transformer.num_heads + 2)
                                )
                            )

                    # Copy to correct slice
                    if "q_attn.weight" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "q_attn.bias" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "kv_attn.weight" in key:
                        module._parameters[param_name][
                        model.transformer.head_size * model.transformer.num_heads :
                        ] = value
                    elif "kv_attn.bias" in key:
                        module._parameters[param_name][
                            model.transformer.head_size * model.transformer.num_heads :
                        ] = value
                    else:
                        if current_parameter_tensor.shape != value.shape:
                            raise ValueError(
                                f"Name {final_key} -- Current {current_parameter_tensor.shape} and got {value.shape}"
                            )
                        module._parameters[param_name] = value
                else:
                    module._buffers[param_name] = value

                del value

        torch.cuda.empty_cache()
        model.post_load_weights(quantize)
