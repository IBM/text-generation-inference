"""Tests for evaluating the prompt cache, with particular focus on making sure
it does LRU eviction in a thread safe way correctly.
"""
import gc
import os
from pathlib import Path

import pytest
from unittest.mock import patch
import torch
import safetensors.torch
from threading import Lock
from text_generation_server import prompt_cache

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.set_default_device(DEVICE)
else:
    DEVICE = None

TESTS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
INTEGRATION_TESTS_DIR = os.path.join(REPO_ROOT, "integration_tests")


@pytest.fixture()
def temp_prompt_store():
    with patch("text_generation_server.prompt_cache.PREFIX_STORE_PATH", Path(os.path.join(INTEGRATION_TESTS_DIR, "prompt_prefixes"))):
        yield


@pytest.fixture()
def tiny_starcoder_decoder_prompt(temp_prompt_store):
    return "tiny_starcoder"


@pytest.fixture()
def tiny_raw_llama_peft_adapter_prompt(temp_prompt_store):
    return "tinyllama_peft_adapter_raw"


@pytest.fixture()
def tiny_llama_peft_adapter_prompt(temp_prompt_store):
    return "tinyllama_peft_adapter"


@pytest.fixture()
def temp_prompt_cache_enc_dec_meta():
    """Build an empty prompt cache for an encoder-decoder model with the 'meta'
        device."""
    dtype = torch.float16
    return prompt_cache.PrefixCache(
        device='meta',
        dtype=dtype,
        max_length=16,
        encoder_decoder=True,
        decoder_start_tok_embedding=torch.rand((1, 8), dtype=dtype, device='meta'),
        return_zero=None,
    )

@pytest.fixture()
def temp_prompt_cache():
    """Build an empty prompt cache that we can test with."""
    return prompt_cache.PrefixCache(
        device=DEVICE,
        dtype=torch.float32,
        max_length=8,
        encoder_decoder=False,
        decoder_start_tok_embedding=None,
        return_zero=None,
    )

### Tests for linked list operations
## Adding new nodes to the list
def test_single_node_list_add_as_head():
    """Ensure that we can create a list with a single node correctly."""
    dll = prompt_cache.DoublyLinkedList()
    node = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    dll.add_node_as_head(node)
    assert dll.head is node
    assert dll.tail is node
    assert dll.head.next is None
    assert dll.head.prev is None
    assert dll.tail.next is None
    assert dll.tail.prev is None

def test_multi_node_list_add_as_head():
    """Ensure that we can create a list with a single node correctly."""
    dll = prompt_cache.DoublyLinkedList()
    node1 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    node2 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="2")
    node3 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="3")
    dll.add_node_as_head(node1)
    dll.add_node_as_head(node2)
    dll.add_node_as_head(node3)
    assert dll.head is node3
    assert dll.tail is node1
    assert node3.prev is None
    assert node3.next is node2
    assert node2.prev is node3
    assert node2.next is node1
    assert node1.next is None
    assert node1.prev is node2

## Removing nodes from the list
def test_remove_tail_from_list_with_one_node():
    """Ensure that we can remove a node from a list with one entry."""
    dll = prompt_cache.DoublyLinkedList()
    node = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    dll.add_node_as_head(node)
    popped_node = dll.pop_tail_node()
    assert dll.head is None
    assert dll.tail is None
    assert popped_node is node

def test_remove_tail_from_multi_node_list():
    """Ensure we can correctly remove the tail from the DLL."""
    dll = prompt_cache.DoublyLinkedList()
    node1 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    node2 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="2")
    dll.add_node_as_head(node1)
    dll.add_node_as_head(node2)
    assert dll.tail is node1
    popped_node = dll.pop_tail_node()
    assert popped_node is node1
    assert dll.head is dll.tail
    assert dll.head is node2
    assert node2.next is None
    assert node2.prev is None

## Moving things within the list
def test_move_to_head_with_one_node():
    """Ensure that moving a node from a list with one entry is a noop."""
    dll = prompt_cache.DoublyLinkedList()
    node = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    dll.add_node_as_head(node)
    dll.move_node_to_head(node)
    assert dll.head is node
    assert dll.tail is node
    assert dll.head.next is None
    assert dll.head.prev is None
    assert dll.tail.next is None
    assert dll.tail.prev is None

def test_move_to_head_multi_node_list():
    """Ensure that moving the head to the front of a multi node list is a noop."""
    dll = prompt_cache.DoublyLinkedList()
    node1 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    node2 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="2")
    # 2 <-> 1
    dll.add_node_as_head(node1)
    dll.add_node_as_head(node2)
    # 2 <-> 1
    dll.move_node_to_head(node2)
    assert dll.head is node2
    assert dll.tail is node1
    assert node2.next is node1
    assert node2.prev is None
    assert node1.prev is node2
    assert node1.next is None

def test_move_to_head_from_tail_multi_node_list():
    """Ensure that we can move the tail of a multinode DLL to the head correctly."""
    dll = prompt_cache.DoublyLinkedList()
    node1 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    node2 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="2")
    # 2 <-> 1
    dll.add_node_as_head(node1)
    dll.add_node_as_head(node2)
    # 1 <-> 2
    dll.move_node_to_head(node1)
    assert dll.head is node1
    assert dll.tail is node2
    assert node1.next is node2
    assert node1.prev is None
    assert node2.prev is node1
    assert node2.next is None

def test_move_to_head_from_middle_multi_node_list():
    """Ensure that we can move a node from the middle of a multinode DLL to the head correctly."""
    dll = prompt_cache.DoublyLinkedList()
    node1 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    node2 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="2")
    node3 = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="3")
    # 3 <-> 2 <-> 1
    dll.add_node_as_head(node1)
    dll.add_node_as_head(node2)
    dll.add_node_as_head(node3)
    # 2 <-> 3 <-> 1
    dll.move_node_to_head(node2)
    assert dll.head is node2
    assert dll.tail is node1
    assert node2.next is node3
    assert node2.prev is None
    assert node3.prev is node2
    assert node3.next is node1
    assert node1.prev is node3
    assert node1.next is None

### Tests for thread lock manager
def test_thread_lock_manager():
    """Ensure that when we enter/exit a lock manager, we correctly lock/unlock."""
    lock = Lock()
    lock_manager = prompt_cache.ThreadLockManager(lock)
    assert not lock.locked()
    with lock_manager:
        assert lock.locked()
    assert not lock.locked()

### Tests for prompt cache node objects
def test_prompt_cache_node_tensor():
    """Verify that our tensor size estimation is correct for a single tensor prompt."""
    gc.collect()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    node = prompt_cache.PromptCacheNode(torch.ones((3, 3)), prefix_id="1")
    expected_memory_allocation = 512 # measured in bytes
    assert node.prompt_size_mb * (1024 ** 2) == expected_memory_allocation
    # Compare to the newly allocated cuda memory if cuda is available
    if initial_memory is not None:
        assert torch.cuda.memory_allocated() - initial_memory == expected_memory_allocation

def test_prompt_cache_node_tuple_all_tensors():
    """Verify that our tensor size estimation is correct for a multitensor prompt."""
    gc.collect()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    node = prompt_cache.PromptCacheNode((torch.ones((3, 3)), torch.ones((3, 3)),), prefix_id="1")
    expected_memory_allocation = 1024 # measured in bytes
    assert node.prompt_size_mb * (1024 ** 2) == expected_memory_allocation
    # Compare to the newly allocated cuda memory if cuda is available
    if initial_memory is not None:
        assert torch.cuda.memory_allocated() - initial_memory == expected_memory_allocation

def test_prompt_cache_node_tuple_with_one_tensor():
    """Ensure our tensor size estimation is correct if we have a None in our prompt tuple."""
    gc.collect()
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else None
    node = prompt_cache.PromptCacheNode((torch.ones((3, 3)), None,), prefix_id="1")
    expected_memory_allocation = 512 # measured in bytes
    assert node.prompt_size_mb * (1024 ** 2) == expected_memory_allocation
    # Compare to the newly allocated cuda memory if cuda is available
    if initial_memory is not None:
        assert torch.cuda.memory_allocated() - initial_memory == expected_memory_allocation

### End to end tests for prompt cache interactions
@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_get_prompt_cache_no_eviction(mock_load_tensors, temp_prompt_cache):
    """Ensure that if we hit a prompt cache hit, its timestamp updates."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    dummy_prompt_id = "prompt1"
    # Prompt cache miss; add the dummy prompt ID to the cache
    t1 = temp_prompt_cache.get(dummy_prompt_id)
    assert len(temp_prompt_cache) == 1
    assert isinstance(t1, torch.Tensor)
    # Prompt cache hit; should retrieve the same tensor object
    t2 = temp_prompt_cache.get(dummy_prompt_id)
    assert len(temp_prompt_cache) == 1
    assert t1 is t2

@patch("text_generation_server.prompt_cache.PromptCacheNode._get_prompt_stats")
@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_get_prompt_cache_with_eviction(mock_load_tensors, mock_get_prompt_size, temp_prompt_cache):
    """Ensure that if we need to make space, we evicted the least recently used tensor."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    mock_get_prompt_size.return_value = (None, (prompt_cache.PROMPT_CACHE_SIZE_MB / 2) - 1)
    temp_prompt_cache.get("prompt1")
    temp_prompt_cache.get("prompt2")
    # Evicts lru prompt ID (prompt1)
    temp_prompt_cache.get("prompt3")
    assert len(temp_prompt_cache) == 2
    assert set(temp_prompt_cache.keys()) == set(["prompt2", "prompt3"])
    # Access our oldest node, updating its timestamp
    temp_prompt_cache.get("prompt2")
    # Then ensure that adding a new prompt ID evicts prompt3 instead of prompt2
    temp_prompt_cache.get("prompt4")
    assert len(temp_prompt_cache) == 2
    assert set(temp_prompt_cache.keys()) == set(["prompt2", "prompt4"])

@patch("text_generation_server.prompt_cache.PromptCacheNode._get_prompt_stats")
@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_get_prompt_cache_tensor_too_large(mock_load_tensors, mock_get_prompt_size, temp_prompt_cache):
    """Ensure that an error is raised if a tensor greater than the cache size is found."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    mock_get_prompt_size.return_value = (None, prompt_cache.PROMPT_CACHE_SIZE_MB + 1)
    with pytest.raises(ValueError):
        temp_prompt_cache.get("prompt1")

@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_clear_cache(mock_load_tensors, temp_prompt_cache):
    """Ensure that we can clear the prompt cache correctly."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    assert len(temp_prompt_cache) == 0
    temp_prompt_cache.get("prompt1")
    assert len(temp_prompt_cache) == 1
    temp_prompt_cache.clear()
    assert len(temp_prompt_cache) == 0

@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_get_cache_keys(mock_load_tensors, temp_prompt_cache):
    """Ensure that we can grab the keys of the prompt cache correctly."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    prompt_ids = set(["prompt1", "prompt2"])
    assert len(temp_prompt_cache) == 0
    for prompt_id in prompt_ids:
        temp_prompt_cache.get(prompt_id)
    assert set(temp_prompt_cache.keys()) == set(prompt_ids)

@patch("text_generation_server.prompt_cache.PrefixCache._load_embedding_tensors")
def test_get_cache_len(mock_load_tensors, temp_prompt_cache):
    """Ensure that we can get the length of the prompt cache correctly."""
    mock_load_tensors.return_value = torch.ones((3, 3))
    assert len(temp_prompt_cache) == 0
    temp_prompt_cache.get("prompt1")
    temp_prompt_cache.get("prompt2")
    assert len(temp_prompt_cache) == 2

### Test code paths for encoder decoder model
# TODO: add more tests here!
@patch("text_generation_server.prompt_cache.PrefixCache._load_torch_file")
def test_prompt_model_device_diff(mock_load, temp_prompt_cache_enc_dec_meta):
    # create prefix tensor on CPU which should be converted to the 'meta' device
    # before the decoder_start_tok_embedding is added to it
    mock_load.return_value = torch.ones((4,8), device='cpu')
    temp_prompt_cache_enc_dec_meta.get("bad_prompt")

### Test cases for invalid prompts
@patch("pathlib.Path.is_file")
def test_prompt_not_exist(mock_is_file, temp_prompt_cache):
    mock_is_file.return_value = False
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0

@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_prompt_with_wrong_dims(mock_is_file, mock_torch_load, temp_prompt_cache):
    mock_is_file.return_value = True

    # one dimension is not enough
    mock_torch_load.return_value = torch.ones((3))
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0

    # three dimensions is too many
    mock_torch_load.return_value = torch.ones((3, 3, 3))
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0

@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_prompt_too_many_virtual_tokens(mock_is_file, mock_torch_load, temp_prompt_cache):
    mock_is_file.return_value = True

    mock_torch_load.return_value = torch.ones((9,16))
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0

@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_prompt_wrong_embed_size(mock_is_file, mock_torch_load, temp_prompt_cache):
    mock_is_file.return_value = True
    # set embed_size to 16
    temp_prompt_cache.embed_size = 16
    mock_torch_load.return_value = torch.ones((1,15))
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0

@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_prompt_with_infinite_after_conversion(mock_is_file, mock_torch_load, temp_prompt_cache):
    mock_is_file.return_value = True
    bad_tensor = torch.ones((3,3), dtype=torch.float64)
    bad_tensor[1, 1] = torch.finfo(torch.float64).max
    mock_torch_load.return_value = bad_tensor
    with pytest.raises(Exception) as e:
        temp_prompt_cache.get("bad_prompt")
    assert e.match("torch.float64 to torch.float32")
    assert len(temp_prompt_cache) == 0

@patch("torch.load")
@patch("pathlib.Path.is_file")
def test_prompt_with_nan(mock_is_file, mock_torch_load, temp_prompt_cache):
    mock_is_file.return_value = True
    bad_tensor = torch.ones((3,3), dtype=torch.float16)
    bad_tensor[1, 1] = torch.nan
    mock_torch_load.return_value = bad_tensor
    with pytest.raises(Exception):
        temp_prompt_cache.get("bad_prompt")
    assert len(temp_prompt_cache) == 0


def test_prompt_cache_decoder_only_load(temp_prompt_cache, tiny_starcoder_decoder_prompt):
    """Simple test that we can load a prompt with a decoder.pt file"""
    # The cache should load this without raising
    prompt = temp_prompt_cache.get(tiny_starcoder_decoder_prompt)

    # Assert this is the same tensor that's in decoder.pt
    decoder_pt_path = os.path.join(prompt_cache.PREFIX_STORE_PATH, tiny_starcoder_decoder_prompt, "decoder.pt")
    decoder = torch.load(decoder_pt_path)
    assert decoder.equal(prompt)


def test_prompt_cache_peft_decoder_load(temp_prompt_cache, tiny_raw_llama_peft_adapter_prompt):
    """Simple test that we can load a prompt for a decoder-only model saved with PEFT directly in adapter_model.bin format"""
    # The cache should load this without raising
    prompt = temp_prompt_cache.get(tiny_raw_llama_peft_adapter_prompt)

    # Assert this is the same tensor that's in adapter_model.bin
    adapter_model_path = os.path.join(prompt_cache.PREFIX_STORE_PATH, tiny_raw_llama_peft_adapter_prompt, "adapter_model.bin")
    adapter_model = torch.load(adapter_model_path, map_location=torch.device('cpu'))
    assert adapter_model["prompt_embeddings"].equal(prompt)


def test_prompt_cache_safetensors_load(temp_prompt_cache, tiny_llama_peft_adapter_prompt):
    """Simple test that we can load a prompt for a decoder-only model saved with PEFT directly in adapter_model.safetensors format"""
    # The cache should load this without raising
    prompt = temp_prompt_cache.get(tiny_llama_peft_adapter_prompt)

    # Assert this is the same tensor that's in adapter_model.safetensors
    adapter_model_path = os.path.join(prompt_cache.PREFIX_STORE_PATH, tiny_llama_peft_adapter_prompt, "adapter_model.safetensors")
    adapter_model = safetensors.torch.load_file(adapter_model_path)
    assert adapter_model["prompt_embeddings"].equal(prompt)
