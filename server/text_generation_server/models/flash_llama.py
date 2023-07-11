import torch
import torch.distributed

from accelerate import init_empty_weights
from pathlib import Path
from typing import Optional, List, Union, Any

from text_generation_server.inference_engine import BaseInferenceEngine
from text_generation_server.models.custom_modeling.flash_llama_modeling import FlashLlamaForCausalLM
from text_generation_server.models.flash_causal_lm import FlashCausalLM
from text_generation_server.utils import get_model_path, local_weight_files


class FlashLlama(FlashCausalLM):
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
            engine.model = FlashLlamaForCausalLM(model_config)

        self.load_weights(engine.model, filenames, dtype == torch.int8, device, dtype)
        self.model = engine.model.eval().to(device)

        super(FlashCausalLM, self).__init__(engine, dtype)
        self.use_position_ids = True

        if self.model.config.pad_token_id is not None:
            self.tokenizer.pad_token_id = self.model.config.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.model.config.eos_token_id

    @staticmethod
    def load_weights(
        model: FlashLlamaForCausalLM,
        filenames: List[Path],
        quantize: bool,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for filename in filenames:
            state_dict = torch.load(filename, map_location="cpu")
            for key, value in state_dict.items():
                value = value.to(device if not quantize else "cpu").to(dtype)

                layer_name = ".".join(key.split(".")[:4])

                # Fused qkv
                if "q_proj" in key or "k_proj" in key or "v_proj" in key:
                    final_key = layer_name + ".query_key_value.weight"

                # Fused gate and up projs
                elif "gate_proj" in key or "up_proj" in key:
                    final_key = layer_name + ".gate_up_proj.weight"
                else:
                    final_key = key

                module_name, param_name = final_key.rsplit(".", 1)
                module = model.get_submodule(module_name)

                try:
                    current_parameter_tensor = module._parameters[param_name]
                except KeyError:
                    current_parameter_tensor = None

                if current_parameter_tensor is not None:
                    if current_parameter_tensor.device == torch.device("meta"):
                        # Init qkv
                        if "query_key_value" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 3, value.shape[1])
                            )
                        # Init gate and up proj
                        elif "gate_up_proj" in final_key:
                            module._parameters[param_name] = value.new_empty(
                                (value.shape[0] * 2, value.shape[1])
                            )

                    # Copy to correct slice
                    if "q_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "k_proj" in key:
                        module._parameters[param_name][
                        value.shape[0] : value.shape[0] * 2
                        ] = value
                    elif "v_proj" in key:
                        module._parameters[param_name][value.shape[0] * 2 :] = value
                    elif "gate_proj" in key:
                        module._parameters[param_name][: value.shape[0]] = value
                    elif "up_proj" in key:
                        module._parameters[param_name][value.shape[0] :] = value
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
