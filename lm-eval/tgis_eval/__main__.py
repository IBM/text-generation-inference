try:
    from lm_eval.__main__ import cli_evaluate
except ImportError:
    raise ImportError("Could not import lm_eval: Please install ibm-generative-ai[lm-eval] extension.")  # noqa: B904

from .model import initialize_model

initialize_model()

cli_evaluate()
