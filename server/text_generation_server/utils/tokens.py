import os
from itertools import chain, repeat
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

from text_generation_server.pb import generate_pb2
from text_generation_server.utils.dist import RANK
from text_generation_server.utils.token_types import TokenInfo, TopToken, InputTokens
from text_generation_server.utils.logits_process import (
    static_warper,
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
)

FP32_LOGITS = os.getenv("FP32_LOGITS_PROCESS") == "true"

# Constants to avoid repeated allocation
INT_ZEROS = repeat(0)
FLOAT_ZEROS = repeat(0.0)
NONES = repeat(None)
SINGLE_ZERO = [0]
SINGLE_NONE = [None]
SINGLE_NAN = [float("nan")]


class Sampling:
    def __init__(self, seed: Optional[int] = None, device: str = "cpu"):
        self.generator = None if seed is None else torch.Generator(device).manual_seed(seed)

    def __call__(self, logits):
        probs = torch.nn.functional.softmax(logits, -1)
        # Avoid GPU<->CPU sync done by torch multinomial
        # See: https://github.com/pytorch/pytorch/blob/925a3788ec5c06db62ca732a0e9425a26a00916f/aten/src/ATen/native/Distributions.cpp#L631-L637
        q = torch.empty_like(probs).exponential_(1, generator=self.generator)
        return probs.div_(q).argmax()


class Greedy:
    def __call__(self, logits):
        return logits.argmax(dim=-1)


class NextTokenChooser:
    def __init__(
        self, temperature=1.0, top_k=None, top_p=None, typical_p=None, seed=None,
        repetition_penalty: Optional[float] = None,
        length_penalty: Optional[Tuple[int, float]] = None,
        min_new_tokens=0, eos_token_id=None, device=None,
        return_logprobs=False,
    ):
        if min_new_tokens > 0 and eos_token_id is None:
            raise ValueError("Must provide eos_token_id for min_new_tokens > 0")
        self.return_logprobs = return_logprobs
        self.current_tokens = 0
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id
        self.repetition_processor = (
            RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
            if repetition_penalty is not None else None
        )
        self.length_penalty = length_penalty if length_penalty is not None and length_penalty[1] > 1.0 else None

        if temperature == 0.0:
            self.static_warper = None
            self.choice = Greedy()
        else:
            self.static_warper = static_warper(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                typical_p=typical_p,
                return_logprobs=return_logprobs,
            )
            self.choice = Sampling(seed, device)

    def _process_logits(self, input_ids, scores):
        # Penalize EOS token if we have not yet generated minimum
        if self.current_tokens < self.min_new_tokens:
            scores[:, self.eos_token_id] = -float("inf")
            self.current_tokens += 1
        # Apply length penalty if applicable
        elif self.length_penalty is not None:
            tokens_past = self.current_tokens - self.length_penalty[0]
            if tokens_past > 0:
                eos_scores = scores[:, self.eos_token_id]
                # To support negative logits we compute the penalty of the
                # absolute value and add to the original logit
                scores[:, self.eos_token_id] = eos_scores + torch.abs(eos_scores) * (
                    pow(self.length_penalty[1], tokens_past) - 1)
            self.current_tokens += 1

        # Apply repetition penalty if applicable
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)

        return scores

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if FP32_LOGITS:
            scores = scores.to(torch.float32)

        # Processs and warp logits
        final_scores = self._process_logits(input_ids, scores)

        if self.static_warper is not None:
            final_scores, logprobs = self.static_warper(scores)
        else:
            # Compute logprobs if requested
            logprobs = torch.log_softmax(final_scores, -1) if self.return_logprobs else None

        #self._log_invalid_scores(final_scores, scores)

        # Choose tokens
        final_scores = final_scores[-1:, :]
        next_ids = self.choice(final_scores)
        return next_ids.view(-1), final_scores, logprobs

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.NextTokenChooserParameters,
        return_logprobs: bool,
        tokenizer: PreTrainedTokenizerBase,
        device: torch.device,
    ) -> "NextTokenChooser":
        return NextTokenChooser(
            temperature=pb.temperature,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            seed=pb.seed if pb.HasField('seed') else None,
            repetition_penalty=pb.repetition_penalty if pb.HasField('repetition_penalty') else None,
            length_penalty=(pb.length_penalty.start_index, pb.length_penalty.decay_factor)
            if pb.HasField('length_penalty') else None,
            min_new_tokens=pb.min_new_tokens,
            eos_token_id=getattr(tokenizer, 'model_eos_token_id', tokenizer.eos_token_id),
            device=device,
            return_logprobs=return_logprobs,
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


class HeterogeneousNextTokenChooser:
    """Port of TGI's HeterogeneousNextTokenChooser. Note that we 
    currently don't port the watermark processor.
    """
    def __init__(
        self,
        temperature: List[float],
        top_k: List[float],
        top_p: List[float],
        typical_p: List[float],
        # allow passing in existing Samplings to preserve RNG state
        seeds: List[Optional[Union[int, Sampling]]],
        repetition_penalty: List[float],
        length_penalty: List[Tuple[int, float]],
        min_new_tokens: List[int],
        return_logprobs: List[bool],
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = None,
        # allow passing in existing values to support combining HNTC instances
        current_tokens: Optional[List[int]] = None,
    ):
        warpers = []
        self.repetition_processor = (
            HeterogeneousRepetitionPenaltyLogitsProcessor(
                repetition_penalty, dtype, device,
                # do not penalize the eos token if it is the same id as the pad token
                id_to_exclude = eos_token_id if eos_token_id == pad_token_id else None,
            )
            if any(x != 1.0 for x in repetition_penalty)
            else None
        )

        # Currently, we only build warpers if temperatures are nonzero. This is different
        # from TGI; see NextTokenChooser as a reference.
        do_sample = [x != 0.0 for x in temperature]
        if any(do_sample):
            # Only initialize our warpers if we're doing sampling; we effectively ignore
            # 0 values (greedy) or 1 values (noop) unless we actually are going to mutate logits.
            if any(x != 1.0 for x in temperature):
                corrected_temps = [temp if temp != 0 else 1 for temp in temperature]
                warpers.append(
                    HeterogeneousTemperatureLogitsWarper(corrected_temps, dtype, device)
                )
            if any(x != 0 for x in top_k):
                warpers.append(HeterogeneousTopKLogitsWarper(top_k, device))

            if any(x < 1.0 for x in top_p):
                #assert all(x != 0 for x in top_p)
                warpers.append(HeterogeneousTopPLogitsWarper(top_p, dtype, device))

            if any(x < 1.0 for x in typical_p):
                #assert all(x != 0 for x in typical_p)
                warpers.append(HeterogeneousTypicalLogitsWarper(typical_p, dtype, device))

            self.choice = HeterogeneousSampling(do_sample, seeds, device)
        else:
            self.choice = Greedy()

        self.warpers = warpers
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.length_penalty = length_penalty
        self.min_new_tokens = min_new_tokens
        self.current_tokens = current_tokens if current_tokens is not None else [0] * len(do_sample)
        self.do_sample = do_sample
        self.dtype = dtype
        self.device = device
        self.return_logprobs = return_logprobs

    @property
    def samplings(self):
        if isinstance(self.choice, Greedy):
            return [None]*len(self.do_sample)
        return self.choice.samplings

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor):
        # TODO: Add vectorized implementation for EOS / length penalty; API should be similar to
        # logit processors, e.g., repetition penalty, with some extra args from current tokens.
        # Then, we should be able to .filter() it to prune it like we do for everything else.
        for idx, (current_tokens, min_new_tokens, length_penalty) in enumerate(zip(self.current_tokens, self.min_new_tokens, self.length_penalty)):
            # Penalize EOS token if we have not yet generated the minimum
            if current_tokens < min_new_tokens:
                scores[idx, self.eos_token_id] = -float("inf")
                self.current_tokens[idx] += 1
            # Apply the length penalty
            elif length_penalty is not None:
                tokens_past = current_tokens - length_penalty[0]
                if tokens_past > 0:
                    eos_scores = scores[idx, self.eos_token_id]
                    # To support negative logits we compute the penalty of the
                    # absolute value and add to the original logit
                    scores[idx, self.eos_token_id] = eos_scores + torch.abs(eos_scores) * (
                        pow(length_penalty[1], tokens_past) - 1)
                self.current_tokens[idx] += 1
        # Apply the repetition penalty if we have one
        if self.repetition_processor is not None:
            scores = self.repetition_processor(input_ids, scores)

        # Apply warpers after logit processors
        for warper in self.warpers:
            scores = warper(input_ids, scores)

        next_ids = self.choice(scores)
        # Get the next log probs; we intentionally avoid doing gathering here
        # and postpone it until later in case the request being considered
        # also asks for top k tokens info w/ logprobs
        logprobs = torch.log_softmax(scores, -1) if any(self.return_logprobs) else NONES

        return next_ids, scores, logprobs

    @classmethod
    def from_pb(
        cls,
        pb: List[generate_pb2.NextTokenChooserParameters],
        model_eos_token_id: Optional[int],
        model_pad_token_id: Optional[int],
        return_logprobs: List[bool],
        dtype: torch.dtype,
        device: torch.device,
        # allow passing in internal HNTC state to support combining instances
        samplings: Optional[List[Sampling]] = None,
        current_tokens: Optional[List[int]] = None,
    ) -> "HeterogeneousNextTokenChooser":
        seeds = samplings if samplings else [pb_.seed if pb_.HasField('seed') else None for pb_ in pb]

        return HeterogeneousNextTokenChooser(
            temperature=[pb_.temperature for pb_ in pb],
            repetition_penalty=[pb_.repetition_penalty if pb_.HasField('repetition_penalty') else 1.0 for pb_ in pb],
            top_k=[pb_.top_k for pb_ in pb],
            # Ensure that default (zero) values for top_p and typical_p are converted to 1.0
            # (which corresponds to disabled in both cases)
            top_p=[pb_.top_p if pb_.top_p > 0 else 1.0 for pb_ in pb],
            typical_p=[pb_.typical_p if pb_.typical_p > 0 else 1.0 for pb_ in pb],
            length_penalty=[
                (pb_.length_penalty.start_index, pb_.length_penalty.decay_factor)
                if pb_.HasField('length_penalty') else None for pb_ in pb
            ],
            seeds=seeds,
            min_new_tokens=[pb_.min_new_tokens for pb_ in pb],
            eos_token_id=model_eos_token_id,
            pad_token_id=model_pad_token_id,
            return_logprobs=return_logprobs,
            device=device,
            dtype=dtype,
            current_tokens=current_tokens,
        )

    def filter(self, indices):
        if self.repetition_processor is not None:
            self.repetition_processor = self.repetition_processor.filter(indices)

        filtered_warpers = []
        for warper in self.warpers:
            filtered_warper = warper.filter(indices)
            if filtered_warper is not None:
                filtered_warpers.append(filtered_warper)
        self.warpers = filtered_warpers

        self.do_sample = [self.do_sample[i] for i in indices]
        # TODO: vectorize these & implement filter
        self.current_tokens = [self.current_tokens[i] for i in indices]
        self.min_new_tokens = [self.min_new_tokens[i] for i in indices]
        self.length_penalty = [self.length_penalty[i] for i in indices]
        self.return_logprobs = [self.return_logprobs[i] for i in indices]

        if any(self.do_sample):
            self.choice.filter(indices)
        else:
            self.choice = Greedy()

        return self


class HeterogeneousSampling:
    """
    Mixed greedy and probabilistic sampling. Compute both and pick the right one for each sample.
    """

    def __init__(
            self,
            do_sample: List[bool],
            # allow passing in seeds or existing Samplings to preserve RNG state
            seeds: List[Optional[Union[int, Sampling]]],
            device: torch.device,
        ):
        self.greedy_indices = []
        self.sampling_mapping = {}
        self.samplings = []
        for i, (sample, seed) in enumerate(zip(do_sample, seeds)):
            if sample:
                sampling = seed if isinstance(seed, Sampling) else Sampling(seed, device)
                self.sampling_mapping[i] = sampling
                self.samplings.append(sampling)
            else:
                self.greedy_indices.append(i)
                self.samplings.append(None)

        self.greedy = Greedy()

    def __call__(self, logits):
        out = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        if self.greedy_indices:
            # Computing for all indices is faster than slicing
            torch.argmax(logits, -1, out=out)

        for i, sampling in self.sampling_mapping.items():
            out[i] = sampling(logits[i])
        return out

    def filter(self, indices):
        new_greedy_indices = []
        new_sampling_mapping = {}
        for i, idx in enumerate(indices):
            if idx in self.sampling_mapping:
                new_sampling_mapping[i] = self.sampling_mapping[idx]
            else:
                new_greedy_indices.append(i)

        self.greedy_indices = new_greedy_indices
        self.sampling_mapping = new_sampling_mapping
        self.samplings = [self.samplings[i] for i in indices]
        return self


# Extract requested token information from model output
def get_token_info(
    request: generate_pb2.Request,
    scores: torch.Tensor,  # Assumes shape is [1, vocab_size]
    next_token: torch.Tensor,
    logprobs: Optional[torch.Tensor],  # Assumes shape matches logits
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
        nth_highest = flat_scores.topk(top_n).values[-1]
        torch.nan_to_num_(nth_highest, neginf=torch.finfo(flat_scores.dtype).min)
        # Get indices (token ids) of all scores >= nth highest value,
        # cap length at 4 * top_n as a precaution
        top_n_indices = (flat_scores >= nth_highest).nonzero().squeeze(-1)[:(top_n * 4)]
        token_info.top_tokens = [
            #TODO possibly also sort top n in no-logprobs case
            TopToken(token_id=tid.item()) for tid in top_n_indices
        ] if (logprobs is None) else _sort([
            TopToken(token_id=tid.item(), logprob=logprobs[-1, tid].item()) for tid in top_n_indices
        ])

    # Token ranks if requested
    if request.details.ranks:
        #TODO if we're also returning top_n perhaps search those first
        token_info.rank = (scores > scores[0, next_token]).sum() + 1

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
        nth_highest_values = torch.topk(all_input_logits, top_n).values[..., -1, None]
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
