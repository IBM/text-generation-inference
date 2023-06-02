import importlib
from typing import Type

from text_generation_server.inference_engine.engine import BaseInferenceEngine


def get_inference_engine_class(deployment_framework: str) -> Type[BaseInferenceEngine]:
    module = importlib.import_module("text_generation_server.inference_engine." + deployment_framework)
    return module.InferenceEngine
