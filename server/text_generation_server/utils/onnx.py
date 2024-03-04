import os
import time

import torch.cuda
from optimum.onnxruntime import ORTModelForCausalLM, ORTModelForSeq2SeqLM, ORTOptimizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig
from transformers import AutoConfig, AutoTokenizer
from transformers.models.auto import modeling_auto

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE") == "true"


def convert_model(
    model_name: str,
    target_model_path: str,
    revision: str | None = None,
    merge_graphs: bool | None = False,
    optimize: bool | None = False,
    provider: str | None = None,
):
    if provider is None:
        provider = (
            "CUDAExecutionProvider"
            if torch.cuda.is_available()
            else "CPUExecutionProvider"
        )

    print(f"Using provider: {provider}")
    print(f"Merge graphs: {merge_graphs}, optimize: {optimize}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE)
    model_type = config.model_type

    supports_causal_lm = model_type in modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    supports_seq2seq_lm = (
        model_type in modeling_auto.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
    )
    print(
        f"Model type: {model_type}, "
        f"supports_causal_lm = {supports_causal_lm}, supports_seq2seq_lm = {supports_seq2seq_lm}",
    )

    # For now special-casing bart
    if supports_seq2seq_lm and model_type == "bart":
        supports_causal_lm = False

    model_class = ORTModelForCausalLM if supports_causal_lm else ORTModelForSeq2SeqLM

    print("Copying tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(target_model_path)

    print("Converting model")
    model = model_class.from_pretrained(
        model_name,
        revision=revision,
        export=True,
        trust_remote_code=TRUST_REMOTE_CODE,
        provider=provider,
        use_merged=merge_graphs,
    )

    if not optimize:
        save_start = time.time()
        model.save_pretrained(target_model_path)
        print(f"Save of converted model took {time.time() - save_start:.3f}s")
    else:
        optimize_start = time.time()
        optimizer = ORTOptimizer.from_pretrained(model)

        # TODO make optimization level configurable
        if provider != "CPUExecutionProvider":
            # O4 includes fp16 weight optimization
            # Shape inference must be disabled for onnxruntime <= 1.14.1 due to a bug
            optimization_config = AutoOptimizationConfig.O3(
                for_gpu=True,
                disable_shape_inference=True,
            )
        else:
            # For now just O1 for CPU
            optimization_config = AutoOptimizationConfig.O1()

        print("Starting model optimization step")
        optimizer.optimize(
            save_dir=target_model_path,
            optimization_config=optimization_config,
        )
        print(f"ORT Model optimization took {time.time() - optimize_start:.3f}s")
