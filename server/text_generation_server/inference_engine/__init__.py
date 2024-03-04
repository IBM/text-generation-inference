import importlib

from text_generation_server.inference_engine.engine import BaseInferenceEngine


def get_inference_engine_class(deployment_framework: str) -> type[BaseInferenceEngine]:
    module = importlib.import_module(
        "text_generation_server.inference_engine." + deployment_framework,
    )
    return module.InferenceEngine
