"""
OpenVINO integration for text-generation-inference.

Usage: set DEPLOYMENT_FRAMEWORK environment variable to hf_optimum_ov or use the
`--deployment-framework=hf_optimum_ov` flag with text-generation-launcher.

Example, including model conversion:

```
pip install optimum[openvino,nncf]
optimum-cli export openvino -m facebook/opt-2.7b data/opt-2.7b-ov
volume=$PWD/data
mkdir $volume
MODEL=/data/opt-2.7b-ov
IMAGE_ID= [text-generation-server IMAGE_ID]
docker run -p 8033:8033 -p 3000:3000 -e OPENVINO_CONFIG=/data/openvino_config.json \
-v $volume:/data $IMAGE_ID text-generation-launcher --model-name $MODEL --deployment-framework hf_optimum_ov

```

OPENVINO_CONFIG is optional. If used, it must point to a json file with an openvino configuration.
If no specific config is specified, it is set to:
{"CACHE_DIR": "", "PERFORMANCE_HINT": "LATENCY", "PERFORMANCE_HINT_NUM_REQUESTS": 1}.

Known limitations:

- Seq2Seq models are not supported yet in this integration.
- Only CPU device is supported at the moment.

"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

import torch
from openvino.runtime import get_version
from optimum.intel import OVModelForCausalLM
from optimum.intel.version import __version__
from text_generation_server.inference_engine.engine import BaseInferenceEngine
from text_generation_server.utils.hub import TRUST_REMOTE_CODE
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


class InferenceEngine(BaseInferenceEngine):
    def __init__(
        self,
        model_path: str,
        model_class: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
        dtype: torch.dtype,
        quantize: Optional[str],  # not used by OpenVINO
        model_config: Optional[Any],
    ) -> None:
        super().__init__(model_path, model_config)
        print(f"Optimum Intel version: {__version__}")
        print(f"OpenVINO version: {get_version()}")
        print("model_path:", model_path)

        if model_class == AutoModelForCausalLM:
            model_class = OVModelForCausalLM
        elif model_class == AutoModelForSeq2SeqLM:
            raise ValueError(
                "Seq2Seq models are not yet supported by the hf_optimum_ov deployment framework"
            )

        ov_config_file = os.getenv("OPENVINO_CONFIG")
        if ov_config_file is not None:
            ov_config = json.loads(Path(ov_config_file).read_text())
        else:
            ov_config = {"CACHE_DIR": ""}

        # Set good default options for latency-optimized workflow
        if "PERFORMANCE_HINT" not in ov_config:
            ov_config["PERFORMANCE_HINT"] = "LATENCY"
        if "NUM_STREAMS" not in ov_config and "PERFORMANCE_HINT_NUM_REQUESTS" not in ov_config:
            ov_config["PERFORMANCE_HINT_NUM_REQUESTS"] = 1

        print(f"ov_config: {ov_config}")

        kwargs = {
            "model_id": model_path,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "export": False,
            "ov_config": ov_config,
            "use_cache": True,
            "load_in_8bit": None,
        }

        model_is_ov = any(f.endswith("_model.xml") for f in os.listdir(model_path))

        if model_is_ov:
            self.model = model_class.from_pretrained(**kwargs)

        else:
            kwargs["export"] = True
            self.model = model_class.from_pretrained(**kwargs)
