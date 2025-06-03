import json
from collections import defaultdict
from typing import Any, Iterator, NamedTuple, Optional, Type, cast


import lm_eval.utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Grouper

from tqdm import tqdm
from .pb import generation_pb2_grpc as gpb2, generation_pb2 as pb2
#import proto
import grpc
import numpy as np
from time import time

SERVER = 'localhost'
PORT = 8033
DEFAULT_BATCH_SIZE=64
DEFAULT_MAX_NEW_TOKENS=300

class LogLikelihoodResult(NamedTuple):
    log_likelihood: float
    is_greedy: bool


def initialize_model():
    pass  # model is registered by importing this module

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def option(cls, val):
    return None if val is None else cls(val)

@register_model("tgis_eval")
class TGISLMEval(LM):
    """
    Implementation of LM model interface for evaluating TGIS model with the lm_eval framework.

    See https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md for reference.
    """

    @classmethod
    def create_from_arg_string(
        cls: Type["TGISLMEval"],
        arg_string: str,
        additional_config: Optional[dict] = None,
    ) -> "TGISLMEval":
        """Allow the user to specify model parameters (TextGenerationParameters) in CLI arguments."""
        args = lm_eval.utils.simple_parse_args_string(arg_string)
        print(f"LM args = {args}")
        return cls(parameters=args, additional_config=additional_config)

    def __init__(
        self,
        parameters = None,
        additional_config: Optional[dict] = None,
        show_progressbar: Optional[bool] = True
    ):
        super().__init__()

        additional_config = {} if additional_config is None else additional_config
    
        self.server = parameters.get("server", SERVER)
        self.port = int(parameters.get("port", PORT))

        self.channel = grpc.insecure_channel(f"{self.server}:{self.port}")
        self.stub = gpb2.GenerationServiceStub(self.channel)

        self.model_kind = self.stub.ModelInfo(pb2.ModelInfoRequest()).model_kind

        self._parameters = parameters
        self._show_progressbar = show_progressbar
        self.batch_size = int(additional_config.get("batch_size", DEFAULT_BATCH_SIZE))

        self.decoding_method = parameters.get("decoding_method", "greedy")

        if self.decoding_method == "greedy":
            self.decoding_method = pb2.GREEDY
        elif self.decoding_method == "sample":
            self.decoding_method = pb2.SAMPLE
        else:
            raise ValueError(f"{self.decoding_method} is not valid for parameter decoding_method")
        
        self.sampling_params = pb2.SamplingParameters(
            temperature = option(float,parameters.get("temperature")),
            top_k       = option(int,parameters.get("top_k")),
            top_p       = option(float,parameters.get("top_p")),
            typical_p   = option(float,parameters.get("typical_p")),
            seed        = option(int,parameters.get("seed"))
        )
        start_index = option(int,parameters.get("length_penalty.start_index"))
        decay_factor = option(float,parameters.get("length_penalty.decay_factor"))

        if (start_index is None) != (decay_factor is None):
            raise ValueError(f"length_penalty.{start_index, decay_factor} must both be set or unset")

        length_penalty = pb2.DecodingParameters.LengthPenalty(
            start_index = start_index,
            decay_factor = decay_factor
        ) if start_index is not None else None

        self.decoding_parameters = pb2.DecodingParameters (
            repetition_penalty = option(float,parameters.get("repetition_penalty")),
            length_penalty = length_penalty
        )
    
    def close(self):
        self.channel.close()

    def _tokenize(self, inputs: list[str]) -> Iterator[list[str]]:
        tokenization_request = self.get_tokenization_request(inputs)
        for response in self.stub.Tokenize(tokenization_request).responses:
            yield response.tokens

    def _has_stop_token(self, response_tokens: list[str], context_tokens: list[str]) -> bool:
        context_length = len(context_tokens)

        # workaround difference in tokenization in some models
        for i in range(len(context_tokens)):
            if response_tokens[i] == '<unk>':
                response_tokens[i] = context_tokens[i]

        if response_tokens[: context_length - 1] == context_tokens[:-1]:
            return response_tokens[-1] != context_tokens[-1]  # only last token differs, probably stop sequence (</s>)
        raise RuntimeError(
            f"There is an unexpected difference between tokenizer and model tokens:\n"
            f"context_tokens={context_tokens}\n"
            f"response_tokens={response_tokens[:context_length]}"
        )

    def _check_model_logprobs_support(self):

        if self.model_kind == pb2.ModelInfoResponse.ENCODER_DECODER:
            raise RuntimeError(f"Encoder decoder models don't return logprobs for input tokens and are not supported")

        input_tokens = self.stub.Generate(
            self.get_batch_request(["The best ice cream flavor is:"])
        ).responses[0].input_tokens
        
        if all(token.logprob is None or np.isnan(token.logprob) for token in input_tokens):
            raise RuntimeError(f"The model is not supported: does not return logprobs for input tokens")
        

    def _get_log_likelihood(self, input_tokens: list[pb2.TokenInfo], context_tokens: list[str]) -> LogLikelihoodResult:
        response_tokens: list[str] = [token.text for token in input_tokens]
        context_length = len(context_tokens)

        if self._has_stop_token(response_tokens, context_tokens):
            context_length -= 1

        return LogLikelihoodResult(
            log_likelihood=sum(token.logprob for token in input_tokens[context_length:]),
            is_greedy=all(token.rank == 1 for token in input_tokens[context_length:]),
        )

    @property
    def _log_likelihood_parameters(self):

        return pb2.Parameters(
            method=self.decoding_method,
            sampling=self.sampling_params if self.decoding_method==pb2.SAMPLE else None,
            stopping=pb2.StoppingCriteria(
                min_new_tokens=1,
                max_new_tokens=1,
            ),
            response=pb2.ResponseOptions (
                generated_tokens=True,
                input_tokens=True,
                token_logprobs=True,
                token_ranks=True,
            ),
            decoding=self.decoding_parameters,
        )
    
    def get_batch_request(self, requests, parameters=None):
        params = parameters or self._log_likelihood_parameters
        return pb2.BatchedGenerationRequest(
                model_id="unused",
                params=params,
                requests=[
                    pb2.GenerationRequest(text=request) for request in requests
                ],
            )
    
    def get_tokenization_request(self, requests):
        return pb2.BatchedTokenizeRequest(
            model_id="unused",
            requests = [
                 pb2.TokenizeRequest(text=request) for request in requests
            ],
            return_tokens=True,
        )

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Args:
            requests: Each request contains Instance.args : Tuple[str, str] containing:
                1. an input string to the LM and
                2. a target string on which the loglikelihood of the LM producing this target,
                   conditioned on the input, will be returned.
        Returns:
            tuple (loglikelihood, is_greedy) for each request according to the input order:
                loglikelihood: probability of generating the target string conditioned on the input
                is_greedy: True if and only if the target string would be generated by greedy sampling from the LM
        """
        start = time()
        #print(f"loglikelihood batch size = {len(requests)}")
        self._check_model_logprobs_support()

        results = []

        pb = tqdm(desc="Running text generation", total=len(requests), disable=not self._show_progressbar)

        for batch in chunks(requests, self.batch_size):
            pb.update(len(batch))
            results.extend(self._loglikelihood_batch(batch))
        pb.close()
        print(f"Time elapsed running the loglikelihood requests: {time()-start}s")
        return results
    
    def _loglikelihood_batch(self, requests: list[Instance]) -> list[tuple[float, bool]]:

        #print(f"loglikelihood batch size = {len(requests)}")

        requests = [request.args for request in requests]
        results: list[LogLikelihoodResult] = []

        contexts_tokenized = list(self._tokenize([context for context, _ in requests]))
        generation_inputs = [context + continuation for context, continuation in requests]

        for result, context_tokens in zip(
            self.stub.Generate(self.get_batch_request(generation_inputs)).responses,
            contexts_tokenized,
        ):
            results.append(self._get_log_likelihood(result.input_tokens, context_tokens))

        return cast(list[tuple[float, bool]], results)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Used to evaluate perplexity on a data distribution.

        Args:
            requests: Each request contains Instance.args : tuple[str] containing an input string to the model whose
                entire loglikelihood, conditioned on purely the EOT token, will be calculated.
        Returns:
            tuple (loglikelihood,) for each request according to the input order:
                loglikelihood: solely the probability of producing each piece of text given no starting input.
        """
        start = time()
        self._check_model_logprobs_support()
        results = []
        for batch in chunks(requests, self.batch_size):
            results.extend(self._loglikelihood_rolling_batch(batch))
        print(f"Time elapsed running the loglikelihood_rolling requests: {time()-start}s")
        return results

    def _loglikelihood_rolling_batch(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        generation_inputs = [request.args[0] for request in requests]
        results: list[LogLikelihoodResult] = []
        for result in self.stub.Generate(self.get_batch_request(generation_inputs)).responses:
            results.append(self._get_log_likelihood(result.input_tokens, []))

        return cast(list[tuple[float, bool]], results)


    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        From official model_guide: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md:

        Each request contains Instance.args : Tuple[str, dict] containing:
            1. an input string to the LM and
            2. a dictionary of keyword arguments used to control generation parameters.
        Using this input and these generation parameters, text will be sampled from the language model

        (
            typically until a maximum output length or specific stopping string sequences--for example,
            {"until": ["\n\n", "."], "max_gen_toks": 128}
        ).
        The generated input+output text from the model will then be returned.
        """
        start = time()
        # group requests by their args (e.g. temperature, do_sample, etc.)
        grouper = Grouper(requests, lambda request: json.dumps(request.args[1], sort_keys=True))
        results: dict[str, list[str]] = defaultdict(list)

        pb = tqdm(desc="Running text generation", total=len(requests), disable=not self._show_progressbar)

        for key, requests_group in grouper.get_grouped().items():
            generation_parameters: dict[str, Any] = requests_group[0].args[1]
            inputs = [request.args[0] for request in requests_group]

            # Process parameters
            do_sample = generation_parameters.pop("do_sample", False)
            decoding_method = pb2.DecodingMethod.SAMPLE if do_sample else pb2.DecodingMethod.GREEDY
            until = generation_parameters.pop("until")
            stop_sequences = [until] if isinstance(until, str) else until
            max_new_tokens = generation_parameters.pop("max_gen_toks", DEFAULT_MAX_NEW_TOKENS)
            temperature = generation_parameters.pop("temperature", 0)

            sampling_params = self.sampling_params
            sampling_params.temperature = temperature


            parameters = pb2.Parameters(
                        method=decoding_method,
                        sampling = sampling_params,
                        stopping=pb2.StoppingCriteria(
                            min_new_tokens=1,
                            max_new_tokens=max_new_tokens,
                            stop_sequences=stop_sequences,
                        ),
                        response=pb2.ResponseOptions (
                            generated_tokens=True,
                            input_tokens=True,
                            token_logprobs=True,
                            token_ranks=True,
                        ),
                        decoding=self.decoding_parameters,
                    )
            
            for batch in chunks(inputs, self.batch_size):
                for result in self.stub.Generate(self.get_batch_request(batch, parameters)).responses:
                    results[key].append(result.text)
                pb.update(len(batch))

        pb.close()
        print(f"Time elapsed running the generate_until requests: {time()-start}s")
        return grouper.get_original(results)
