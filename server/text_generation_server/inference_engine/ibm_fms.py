import torch
from typing import Any, Optional, Dict

from fms.models import get_model, list_variants, __models as _fms_models
from fms.models.hf import to_hf_api
from fms_extras.models.calico import CalicoConfig

from text_generation_server.inference_engine.engine import BaseInferenceEngine

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
            raise ValueError(f"Unknown model type {model_type} passed to ibm_fms engine.")

        super().__init__(model_path, model_config)

        variant_match = self._find_variant(model_config, 'calico')
        if variant_match is None:
            raise ValueError(f"Unable to determine model variant from model: {model_path}")

        # get_model does not have a dtype parameter, setting the default dtype
        # reduces memory required to load the model compared to converting afterwards
        orig_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            calico_model = get_model("calico", variant_match, model_path, source="megatron", device_type=self.device.type)

            # the CalicoConfig does not persist token ids to the
            # HFAdaptedConfig, so pass them along explicitly
            self.model = to_hf_api(
                calico_model,
                pad_token_id=model_config['pad_token_id'],
                bos_token_id=model_config['bos_token_id'],
                eos_token_id=model_config['eos_token_id'],
            ).requires_grad_(False).eval()
        finally:
            torch.set_default_dtype(orig_dtype)

        # update the config to config object instead of a dict
        self._config = self.model.config

    @classmethod
    def _find_variant(cls, model_config_dict: Dict, fms_architecture) -> Optional[str]:
        # get a list of variant configs to compare against the model_config_dict
        variant_map = {
            # HACK: extract the variant config from the closure created for the factory functions...
            v: _fms_models[fms_architecture][v].__closure__[0].cell_contents
            for v in list_variants(fms_architecture)
        }
        for v, v_config in variant_map.items():
            if cls._is_variant_compatible(model_config_dict, v_config):
                return v
        return None

    @classmethod
    def _is_variant_compatible(cls, model_config_dict: Dict, config: CalicoConfig) -> bool:
        dict_key_to_attr = {
            'vocab_size': 'src_vocab_size',
            'n_embd': 'emb_dim',
            'n_head': 'nheads',
            'num_key_value_heads': 'kvheads',
            'n_layer': 'nlayers',
            'n_positions': 'max_expected_seq_len',
            'pad_token_id': 'pad_id',
        }
        for key, attr in dict_key_to_attr.items():
            if model_config_dict[key] != getattr(config, attr, None):
                return False
        return True
