import pytest
import torch

from text_generation_server.utils.logits_process import (
    HeterogeneousRepetitionPenaltyLogitsProcessor,
    HeterogeneousTemperatureLogitsWarper,
    HeterogeneousTopKLogitsWarper,
    HeterogeneousTopPLogitsWarper,
    HeterogeneousTypicalLogitsWarper,
    StaticWarper
)
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor

##############################################################################
# Tests for comparing vectorized heterogeneous logit processors to their
# sequential implementations. In these tests, we only check valid cases, because
# the vectorized overrides generally don't provide any input validation.

BATCH_SIZE = 2
VOCAB_DIM = 25
# Input IDs of shape (batch_size x logits_dim);
# chosen intentionally to have repetition etc.
INPUT_IDS = torch.tensor([
    [1, 2, 1, 3, 4, 6, 7, 1, 1, 1],
    [1, 7, 0, 3, 4, 6, 7, 1, 1, 1],
], dtype=torch.long)
# NOTE: We assume BATCH_SIZE x VOCAB_DIM instead of BATCH_SIZE x SEQ_LEN x VOCAB_DIM
# because the vectorized operations are designed to work on the last set of logits in
# the sequence. I.e., this is effectively x[:, -1, :] of the 3rd order tensor.
FULL_SCORES = torch.softmax(torch.rand((BATCH_SIZE, VOCAB_DIM), dtype=torch.float32), dim=-1)

def compare_individual_vs_vectorized_scores(s_warped, v_warped):
    """Given scores warped individually, compare to scores warped with a vectorized
    implementation.

    Args:
        s_warped: List[torch.Tensor]
            List of tensors warped as single entries.
        v_warped: torch.Tensor
            Warped tensor mat.
    """
    assert len(s_warped) == v_warped.shape[0]
    for idx, s_warped_scores in enumerate(s_warped):
        v_warped_scores = v_warped[idx]
        assert torch.allclose(s_warped_scores.squeeze(), v_warped_scores)

def test_alignment_repetition_penalty_logits_processor():
    """Ensure that the repetition penalty is consistent when it is/isn't vectorized."""
    # NOTE: 1.0 Tests the case with no penalty
    penalties = [1.0, 2.5]
    # Apply the vectorized repetition logits processor over everything
    # given that we have a homogeneous set of penalties to apply
    vectorized_proc = HeterogeneousRepetitionPenaltyLogitsProcessor(
        penalty=penalties,
        dtype=torch.float32,
        device=None,
    )
    v_warped = vectorized_proc(input_ids=INPUT_IDS, scores=FULL_SCORES)
    # apply each penalty one at a time using the nonvectorized warper
    s_warped = []
    for penalty, logits, ids in zip(penalties, FULL_SCORES, INPUT_IDS):
        single_proc = RepetitionPenaltyLogitsProcessor(penalty=penalty)
        s_warped.append(single_proc(ids.unsqueeze(dim=0), logits.unsqueeze(dim=0)))
    compare_individual_vs_vectorized_scores(s_warped, v_warped)


def test_alignment_temperature_logits_processor():
    """Ensure that the temperature warping is consistent when it is/isn't vectorized."""
    # NOTE: 1.0 Tests the case with no temperature warping
    temperatures = [0.25, 1]
    vectorized_proc = HeterogeneousTemperatureLogitsWarper(
        temperature=temperatures,
        dtype=torch.float32,
        device=None,
    )
    # Vectorized temperature warping happens in place; clone the score tensor!
    score_clone = FULL_SCORES.clone()
    v_warped = vectorized_proc(input_ids=INPUT_IDS, scores=score_clone.view(2, -1))

    s_warped = []
    for temp, logits in zip(temperatures, FULL_SCORES):
        # We are testing alignment with TemperatureLogitsWarper
        # through the StaticWarper wrapper class, both for the no-op case
        # and for the case where we actually modify our scores.
        single_proc = StaticWarper(temperature=temp)
        # NOTE: static warpers return a tuple with scores + logprobs (if enabled);
        # We only care about comparing the first one, i.e., scores, here.
        s_warped.append(single_proc(logits.unsqueeze(dim=0))[0])
    compare_individual_vs_vectorized_scores(s_warped, v_warped)


@pytest.mark.parametrize("top_k", [[0, 3], [1, 3]])
def test_alignment_top_k_logits_processor(top_k):
    """Ensure that the top k warping is consistent when it is/isn't vectorized."""
    vectorized_proc = HeterogeneousTopKLogitsWarper(
        top_k=top_k,
        device=None,
    )
    # top k filling happens in place; clone the score tensor!
    score_clone = FULL_SCORES.clone()
    v_warped = vectorized_proc(input_ids=INPUT_IDS, scores=score_clone)

    s_warped = []
    for k, logits in zip(top_k, FULL_SCORES):
        # We are testing alignment with TopKLogitsWarper
        # through the StaticWarper wrapper class, both when we have
        # things in the batch to ignore, and when we care about everything.
        single_proc = StaticWarper(top_k=k)
        # NOTE: static warpers return a tuple with scores + logprobs (if enabled);
        # We only care about comparing the first one, i.e., scores, here.
        s_warped.append(single_proc(logits.unsqueeze(dim=0))[0])
    compare_individual_vs_vectorized_scores(s_warped, v_warped)


def test_alignment_top_p_logits_processor():
    """Ensure that the top k warping is consistent when it is/isn't vectorized."""
    top_p = [.9, 0]
    vectorized_proc = HeterogeneousTopPLogitsWarper(
        top_p=top_p,
        dtype=torch.float32,
        device=None,
    )
    # top p filtering happens in place; clone the score tensor!
    score_clone = FULL_SCORES.clone()
    v_warped = vectorized_proc(input_ids=INPUT_IDS, scores=score_clone)

    s_warped = []
    for p, logits in zip(top_p, FULL_SCORES):
        # We are testing alignment with TopPLogitsWarper through the StaticWarper
        # wrapper class. Be aware that TopPLogitsWarper is an implementation
        # in TGIS, not in Transformers!
        single_proc = StaticWarper(top_p=p)
        # NOTE: static warpers return a tuple with scores + logprobs (if enabled);
        # We only care about comparing the first one, i.e., scores, here.
        s_warped.append(single_proc(logits.unsqueeze(dim=0))[0])
    compare_individual_vs_vectorized_scores(s_warped, v_warped)


def test_alignment_typical_logits_processor():
    """Ensure that the typical logit warping is consistent when it is/isn't vectorized."""
    masses = [.7, .9]
    vectorized_proc = HeterogeneousTypicalLogitsWarper(
        mass=masses,
        dtype=torch.float32,
        device=None,
    )
    # typical logits filtering happens in place; clone the score tensor!
    score_clone = FULL_SCORES.clone()
    v_warped = vectorized_proc(input_ids=INPUT_IDS, scores=score_clone)

    s_warped = []
    for mass, logits in zip(masses, FULL_SCORES):
        # We are testing alignment with TypicalLogitsWarper through the StaticWarper
        # wrapper class. Be aware that TypicalLogitsWarper is an implementation
        # in TGIS, not in Transformers!
        single_proc = StaticWarper(typical_p=mass)
        # NOTE: static warpers return a tuple with scores + logprobs (if enabled);
        # We only care about comparing the first one, i.e., scores, here.
        s_warped.append(single_proc(logits.unsqueeze(dim=0))[0])
    compare_individual_vs_vectorized_scores(s_warped, v_warped)
