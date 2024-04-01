import os

import torch
import torch.distributed

from typing import Optional, Any

from transformers.models.auto.auto_factory import _BaseAutoModelClass

from text_generation_server.models import FLASH_ATTENTION, PAGED_ATTENTION
from text_generation_server.utils import Weights

from text_generation_server.inference_engine import BaseInferenceEngine
from text_generation_server.utils.dist import initialize_torch_distributed
from text_generation_server.utils.hub import local_weight_files

NONTP_FLASH_TYPES = ["RefinedWeb", "RefinedWebModel", "gpt_neox", "gpt_bigcode", "llama", "falcon"]
TP_NONFLASH_TYPES = ["bloom", "t5", "gpt_neox"]
TP_FLASH_TYPES = NONTP_FLASH_TYPES  # All flash types currently support TP
NONTP_NONFLASH_TYPES = ["bloom", "t5"]


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

        sharded = self.world_size > 1
        model_type = self._config.model_type
        if model_type == "gpt2" and "--bigcode--" in model_path:  # Hack for starcoder
            model_type = "gpt_bigcode"

        if sharded:
            if FLASH_ATTENTION:
                if model_type not in TP_FLASH_TYPES:
                    raise NotImplementedError(
                        f"TP with flash attention currently supported by the following model types: {TP_FLASH_TYPES}"
                    )
            elif model_type not in TP_NONFLASH_TYPES:
                raise NotImplementedError(
                    f"TP without flash attention currently supported by the following model types: {TP_NONFLASH_TYPES}"
                )
        elif FLASH_ATTENTION:
            if model_type not in NONTP_FLASH_TYPES:
                raise NotImplementedError(
                    f"Flash attention currently only supported by the following model types: {NONTP_FLASH_TYPES}"
                )
        elif model_type not in NONTP_NONFLASH_TYPES:
            raise ValueError("tgis_native engine must be used with FLASH_ATTENTION, num_shards > 1 and/or BLOOM or T5")

        aliases = None

        if model_type == "bloom":
            slow_but_exact = os.getenv('BLOOM_SLOW_BUT_EXACT', 'false').lower() == 'true'
            self._config.slow_but_exact = slow_but_exact
            from text_generation_server.models.custom_modeling.bloom_modeling import BloomForCausalLM
            model_class = BloomForCausalLM

        elif model_type == "t5":
            aliases = {"shared.weight": ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]}
            from text_generation_server.models.custom_modeling.t5_modeling import T5ForConditionalGeneration
            model_class = T5ForConditionalGeneration

        elif model_type == "gpt_neox":
            if FLASH_ATTENTION:
                from text_generation_server.models.custom_modeling.flash_neox_modeling import FlashGPTNeoXForCausalLM
                model_class = FlashGPTNeoXForCausalLM
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                from text_generation_server.models.custom_modeling.neox_modeling import GPTNeoxForCausalLM
                model_class = GPTNeoxForCausalLM

        elif model_type == "gpt_bigcode":
            self._config.transpose = self._config.architectures[0].startswith("GPT2")
            aliases = {"transformer.wte.weight": ["lm_head.weight"]}
            if PAGED_ATTENTION:
                from text_generation_server.models.custom_modeling.paged_santacoder_modeling import PagedSantacoderForCausalLM
                model_class = PagedSantacoderForCausalLM
            else:
                from text_generation_server.models.custom_modeling.flash_santacoder_modeling import FlashSantacoderForCausalLM
                model_class = FlashSantacoderForCausalLM

        elif model_type in ["RefinedWeb", "RefinedWebModel", "falcon"]:
            if sharded and self._config.alibi:
                raise NotImplementedError("TP is not supported for Falcon models using alibi")
            aliases = {
                "transformer.word_embeddings.weight": ["lm_head.weight"],
                "lm_head.weight": ["transformer.word_embeddings.weight"],
            }
            from text_generation_server.models.custom_modeling.flash_rw_modeling import FlashRWForCausalLM
            model_class = FlashRWForCausalLM

        elif model_type == "llama":
            if PAGED_ATTENTION:
                from text_generation_server.models.custom_modeling.paged_llama_modeling import PagedLlamaForCausalLM
                model_class = PagedLlamaForCausalLM
            else:
                from text_generation_server.models.custom_modeling.flash_llama_modeling import FlashLlamaForCausalLM
                model_class = FlashLlamaForCausalLM

        self._config.quantize = quantize

        self.process_group = initialize_torch_distributed(self.world_size, self.rank)
        self.master = self.rank == 0

        torch.distributed.barrier(group=self.process_group)
        filenames = local_weight_files(model_path, extension=".safetensors")
        if not filenames:
            raise ValueError("No safetensors weights found - required for tgis_native engine")

        weights = Weights(
            filenames, device=self.device, dtype=dtype, process_group=self.process_group, aliases=aliases
        )

        if quantize == "gptq":
            weights._set_gptq_params(model_config, model_path)

        model = model_class(self._config, weights)
        torch.distributed.barrier(group=self.process_group)

        if not hasattr(model, "config"):
            model.config = self._config

        self.model = model.eval()  # .to(dtype)
