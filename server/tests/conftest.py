import pytest

from transformers import AutoTokenizer

from text_generation_server.pb import generate_pb2


@pytest.fixture
def default_pb_parameters():
    return generate_pb2.NextTokenChooserParameters(
        temperature=0.0,  # Greedy
        top_k=0,
        top_p=1.0,
        #do_sample=False,
    )


@pytest.fixture
def default_pb_stop_parameters():
    return generate_pb2.StoppingCriteriaParameters(stop_sequences=[], max_new_tokens=10)


@pytest.fixture(scope="session")
def bloom_560m_tokenizer():
    return AutoTokenizer.from_pretrained("bigscience/bloom-560m", padding_side="left")


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token_id = 50256
    return tokenizer


@pytest.fixture(scope="session")
def mt0_small_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        "bigscience/mt0-small", padding_side="left"
    )
    tokenizer.bos_token_id = 0
    return tokenizer
