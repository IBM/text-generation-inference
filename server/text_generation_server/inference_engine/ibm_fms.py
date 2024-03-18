import torch
from typing import Any, Optional, Dict
from loguru import logger

from fms.models import get_model, list_variants, __models as _fms_models
from fms.models.hf import to_hf_api
from fms.utils.activation import __ACT_2_CLS as FMS_ACT_2_CLASS

from text_generation_server.inference_engine.engine import BaseInferenceEngine

# Register FMS model classes with HF
from fms_extras.models.hf import register_fms_models

register_fms_models()


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: None,
        dtype: torch.dtype,
        quantize: Optional[str],
        model_config: Optional[Any],
        max_sequence_length: Optional[int],
    ) -> None:
        # model_config comes in a as a Dict to support the late registration
        if (model_type := model_config["model_type"]) != "gpt_megatron":
            raise ValueError(
                f"Unknown model type {model_type} passed to ibm_fms engine."
            )

        super().__init__(model_path, model_config)

        # only calico supported currently
        fms_architecture_name = "calico"

        # model_config can override what is set by the variant
        fms_config_dict = self._convert_model_config(model_config)

        variant_match = self._find_variant(fms_config_dict, fms_architecture_name)
        if variant_match is None:
            raise ValueError(
                f"Unable to determine model variant for model: {model_path}"
            )

        # get_model does not have a dtype parameter, setting the default dtype
        # reduces memory required to load the model compared to converting afterwards
        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            fms_model = get_model(
                fms_architecture_name,
                variant_match,
                model_path,
                source="megatron",
                device_type=self.device.type,
                **fms_config_dict,
            )
            # get_model does not persist token ids to the HFAdaptedConfig, so
            # pass them along explicitly
            self.model = (
                to_hf_api(
                    fms_model,
                    pad_token_id=model_config["pad_token_id"],
                    bos_token_id=model_config["bos_token_id"],
                    eos_token_id=model_config["eos_token_id"],
                )
                .requires_grad_(False)
                .eval()
            )
        finally:
            torch.set_default_dtype(orig_dtype)

        # update the config to config object instead of a dict
        self._config = self.model.config

    @classmethod
    def _convert_model_config(cls, model_config_dict: Dict) -> Dict:
        # mapping between CalicoConfig attributes and keys in the model_config dict
        act_fn_attr = "activation_fn"
        fms_config_attr_to_config_json_key = {
            "src_vocab_size": "vocab_size",
            "emb_dim": "n_embd",
            "norm_eps": "layer_norm_epsilon",
            "nheads": "n_head",
            "kvheads": "num_key_value_heads",
            "nlayers": "n_layer",
            "pad_id": "pad_token_id",
            # No entry in config.json for hidden_growth_factor
            # No entry in config.json for multiple_of
            act_fn_attr: "activation_function",
            # ignore p_dropout for inference
            "max_expected_seq_len": "n_positions",
        }
        fms_config_dict = {
            attr: model_config_dict[key]
            for attr, key in fms_config_attr_to_config_json_key.items()
            if key in model_config_dict
        }

        # the activation function name may need to be converted
        if act_fn := fms_config_dict.get(act_fn_attr):
            fms_config_dict[act_fn_attr] = cls._convert_activation_function_name(act_fn)

        return fms_config_dict

    @classmethod
    def _convert_activation_function_name(cls, act_name: str) -> str:
        """Attempts to find an FMS compatible activation function name

        gpt_megatron models may use different names for the activation function
        compared to FMS, specifically around whether "GLU" is indicated
        explicitly.

        Refer to the fms.utils.activation module to see supported names
        """
        glu_activation_function_mapping = {
            "geglu": "gelu",
            "miglu": "mish",
            "mishglu": "mish",
            "reglu": "relu",
            "swiglu": "swish",
        }
        if act_name.endswith("_glu"):
            fms_act_name = act_name.rstrip("_glu")
        elif new_name := glu_activation_function_mapping.get(act_name):
            fms_act_name = new_name
        else:
            fms_act_name = act_name

        # ensure the final act name is supported by FMS
        if fms_act_name not in FMS_ACT_2_CLASS:
            raise ValueError(f"Unsupported activation function: {act_name}.")

        return fms_act_name

    @classmethod
    def _find_variant(
        cls, fms_config_dict: Dict, fms_architecture: str
    ) -> Optional[str]:
        # get a list of variant configs to compare against the model_config_dict
        variant_map = {
            # HACK: extract the variant config from the closure created for the factory functions...
            v: _fms_models[fms_architecture][v].__closure__[0].cell_contents
            for v in list_variants(fms_architecture)
        }

        # attributes of the CalicoConfig that must exist and match to find a
        # compatible "variant"
        variant_attrs_to_check = [
            "emb_dim",
            "nheads",
            "kvheads",
            "nlayers",
        ]
        if not all(fms_config_dict.get(attr, None) for attr in variant_attrs_to_check):
            raise ValueError(
                f"Unable to find compatible variant, the following configurations must exist {variant_attrs_to_check}"
            )

        for v_name, v_config in variant_map.items():
            if all(
                fms_config_dict.get(attr) == getattr(v_config, attr)
                for attr in variant_attrs_to_check
            ):
                return v_name
        return None
