import os
from itertools import chain, repeat
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
    TopKLogitsWarper,
)

from text_generation_server.models.types import TokenInfo, TopToken, InputTokens
from text_generation_server.pb import generate_pb2
from text_generation_server.utils.dist import RANK

FP32_LOGITS = os.getenv("FP32_LOGITS_PROCESS") == "true"

# Constants to avoid repeated allocation
INT_ZEROS = repeat(0)
FLOAT_ZEROS = repeat(0.0)
NONES = repeat(None)
SINGLE_ZERO = [0]
SINGLE_NONE = [None]
SINGLE_NAN = [float("nan")]


class Sampling:
    def __init__(self, seed=None, device=None):
        self.generator = None if seed is None else torch.Generator(device).manual_seed(seed)

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(
            probs, num_samples=1, generator=self.generator
        ).squeeze(-1)
        return next_tokens


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class NextTokenChooser:
    def __init__(
            self, temperature=1.0, top_k=None, top_p=None, seed=None,
            repetition_penalty: Optional[float] = None,
            length_penalty: Optional[Tuple[int, float]] = None,
            min_new_tokens=0, eos_token_id=None, device=None,
    ):
        if min_new_tokens > 0 and eos_token_id is None:
            raise ValueError("Must provide eos_token_id for min_new_tokens > 0")
        self.current_tokens = 0
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.length_penalty = length_penalty if length_penalty is not None and length_penalty[1] > 1.0 else None

        self.warpers = LogitsProcessorList()
        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`

        if repetition_penalty is not None:
            self.warpers.append(RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty)))
        if temperature == 0.0:
            self.choice = Greedy()
        else:
            if temperature is not None and temperature != 1.0:
                self.warpers.append(TemperatureLogitsWarper(float(temperature)))
            if top_k:
                self.warpers.append(TopKLogitsWarper(top_k=top_k))
            if top_p is not None and top_p < 1.0:
                self.warpers.append(TopPLogitsWarper(top_p=top_p))

            self.choice = Sampling(seed, device)

    def __call__(
            self, input_ids, scores, return_logprobs=False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if FP32_LOGITS:
            scores = scores.to(torch.float32)

        # Penalize EOS token if we have not yet generated minimum
        if self.current_tokens < self.min_new_tokens:
            scores[:, self.eos_token_id] = -float("inf")
            self.current_tokens += 1
        # Apply length penalty if applicable
        elif self.length_penalty is not None:
            tokens_past = self.current_tokens - self.length_penalty[0]
            if tokens_past > 0:
                scores[:, self.eos_token_id] = scores[:, self.eos_token_id] * pow(
                    self.length_penalty[1], tokens_past
                )
            self.current_tokens += 1

        # Warp logits
        final_scores = self.warpers(input_ids, scores)

        #self._log_invalid_scores(final_scores, scores)

        # Compute logprobs if requested
        logprobs = torch.log_softmax(final_scores, -1) if return_logprobs else None

        # Choose tokens
        final_scores = final_scores[-1:, :]
        next_ids = self.choice(final_scores)
        return next_ids, final_scores, logprobs

    @classmethod
    def from_pb(cls, pb: generate_pb2.NextTokenChooserParameters, tokenizer: PreTrainedTokenizerBase,
                device: torch.device) -> "NextTokenChooser":
        return NextTokenChooser(
            temperature=pb.temperature,
            top_k=pb.top_k,
            top_p=pb.top_p,
            seed=pb.seed if pb.HasField('seed') else None,
            repetition_penalty=pb.repetition_penalty if pb.HasField('repetition_penalty') else None,
            length_penalty=(pb.length_penalty.start_index, pb.length_penalty.decay_factor)
            if pb.HasField('length_penalty') else None,
            min_new_tokens=pb.min_new_tokens,
            eos_token_id=getattr(tokenizer, 'model_eos_token_id', tokenizer.eos_token_id),
            device=device,
        )

    @staticmethod
    def _log_invalid_scores(scores, pre_warp_scores):
        if RANK == 0:
            # Debugging nan from softmax
            if scores.isposinf().any():
                inf_before = pre_warp_scores.isposinf().any()
                print("WARNING: Found inf in logits, before warp =", bool(inf_before))
            if scores.isnan().any():
                nan_before = pre_warp_scores.isnan().any()
                print("WARNING: Found nan in logits, before warp =", bool(nan_before))


# Extract requested token information from model output
def get_token_info(
        request: generate_pb2.Request,
        scores: torch.Tensor,
        next_token: torch.Tensor,
        logprobs: Optional[torch.Tensor],
) -> TokenInfo:
    next_token = next_token.item()
    token_info = TokenInfo(request_id=request.id, token_id=next_token)

    # logprob of the generated token if requested
    if logprobs is not None:
        token_info.logprob = logprobs[-1, next_token].item()

    # Top n candidates if requested
    return_top_n = request.details.top_n_toks
    if return_top_n:
        flat_scores = scores[-1]
        # Ensure top_n doesn't exceed vocab size
        top_n = min(return_top_n, flat_scores.size(-1))
        # Get nth highest value, ensure it's not -inf (for example if top_n > top_k)
        nth_highest = torch.topk(flat_scores, top_n)[0][-1]
        if nth_highest == -float('inf'):
            nth_highest = torch.finfo(flat_scores.dtype).min
        # Get indices (token ids) of all scores >= nth highest value,
        # cap length at 4 * top_n as a precaution
        top_n_indices = (flat_scores >= nth_highest).nonzero().squeeze(-1)[:(top_n * 4)]
        token_info.top_tokens = [
            #TODO possibly also sort top n in no-logprobs case
            TopToken(token_id=tid.item()) for tid in top_n_indices
        ] if logprobs is None else _sort([
            TopToken(token_id=tid.item(), logprob=logprobs[-1, tid].item()) for tid in top_n_indices
        ])

    # Token ranks if requested
    if request.details.ranks:
        #TODO if we're also returning top_n perhaps search those first
        token_info.rank = (scores > scores[0][next_token]).sum() + 1

    return token_info


# Extract requested input token information from model output
def get_input_tokens_info(request, input_token_ids, all_input_logits) -> InputTokens:

    #TODO optimize this ... can do single gather for chosen and topn logprobs

    # Collect logprobs if requested
    return_logprobs = request.details.logprobs
    if return_logprobs:
        all_input_logprobs = torch.log_softmax(all_input_logits, -1)
        # logprobs of input tokens (except the first one)
        input_logprobs = all_input_logprobs.gather(1, input_token_ids[1:].unsqueeze(-1))

        # Add NaN for the first prompt token
        logprobs_gen = chain(SINGLE_NAN, input_logprobs.squeeze(-1))
    else:
        logprobs_gen = FLOAT_ZEROS

    # Collect ranks if requested
    if request.details.ranks:
        if return_logprobs:
            # Use logprobs that are already gathered
            ranks_gen = chain(SINGLE_ZERO, ((all_input_logprobs > input_logprobs).sum(dim=1) + 1))
        else:
            # Otherwise use logits
            input_logits = all_input_logits.gather(1, input_token_ids[1:].unsqueeze(-1))
            ranks_gen = chain(SINGLE_ZERO, ((all_input_logits > input_logits).sum(dim=1) + 1))
    else:
        ranks_gen = INT_ZEROS

    # Collect top N candidates if requested
    top_n = request.details.top_n_toks
    if top_n:
        # Ensure top_n doesn't exceed vocab size
        top_n = min(top_n, all_input_logits.size(-1))
        # Get the nth highest value for each input token's set of logits
        nth_highest_values = torch.topk(all_input_logits, top_n)[0][..., -1, None]
        # Construct bool tensor marking all scores >= nth highest value for each token
        diff = (all_input_logits >= nth_highest_values)
        # Gather set of marked indices for each token (correspond to top token ids)
        # Slice each tensor to max length of top_n * 4 as a precaution
        max_per_token = top_n * 4
        top_n_indices = [diff[i].nonzero().squeeze(-1)[:max_per_token] for i in range(diff.shape[0])]

        if return_logprobs:
            if len(top_n_indices) != 0:
                combined = torch.nn.utils.rnn.pad_sequence(top_n_indices, batch_first=True)
                top_n_logprobs = all_input_logprobs.gather(1, combined)
                topn_gen = chain(SINGLE_NONE, (
                    (tni := top_n_indices[i], top_n_logprobs[i][:len(tni)])
                    for i in range(len(top_n_indices))
                ))
            else:
                topn_gen = SINGLE_NONE
        else:
            topn_gen = chain(SINGLE_NONE, top_n_indices)
    else:
        topn_gen = NONES

    all_zipped = zip(input_token_ids, logprobs_gen, ranks_gen, topn_gen)

    return InputTokens(
        request_id=request.id,
        tokens=[
            TokenInfo(
                token_id=int(tok_id),
                logprob=float(logprob),
                rank=int(rank),
                top_tokens=None if top_toks is None
                else (
                    #TODO possibly also sort top n in no-logprobs case
                    [TopToken(int(ttid)) for ttid in top_toks] if not return_logprobs
                    else _sort([
                        TopToken(int(ttid), float(ttlp)) for (ttid, ttlp) in zip(*top_toks)
                    ])
                ),
            )
            for (tok_id, logprob, rank, top_toks) in all_zipped
        ]
    )


def _sort(tts: List[TopToken]) -> List[TopToken]:
    tts.sort(reverse=True)
    return tts