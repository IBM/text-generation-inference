from loguru import logger
import math
import os
from pathlib import Path
import re
import threading
from typing import Dict, List, Union, Tuple

import torch

PREFIX_STORE_PATH = Path(os.getenv("PREFIX_STORE_PATH", "prompt_prefixes"))

VALID_PREFIX_ID_PATTERN = re.compile("[/\\w\\-]+")
PROMPT_CACHE_SIZE_MB = int(os.getenv("PROMPT_CACHE_SIZE_MB", "512"))
# TODO
# - Make into two-layer LRU cache (CPU, GPU)
# - Include explicit time-based expiry

class PrefixNotFound(Exception):
    pass

class ThreadLockManager:
    """Context manager convenience class wrapping a thread lock; entering an
    initialized instance of this class acquires the encapsulated lock, and
    exiting releases it.
    """
    def __init__(self, lock: threading.Lock):
        self.thread_lock = lock

    def __enter__(self) -> None:
        self.thread_lock.acquire()

    def __exit__(self, *args, **kwargs) -> None:
        self.thread_lock.release()

class PromptCacheNode:
    """A single entry in the prompt cache, which represents both a node
    in the cache map, and a node in the cache doubly linked list.
    """
    def __init__(self,
            prompt: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            prefix_id: str,
            next: "PromptCacheNode"=None,
            prev: "PromptCacheNode"=None
        ) -> None:
        self.prefix_id = prefix_id
        self.prompt = prompt
        self.prompt_virtual_tokens, self.prompt_size_mb = PromptCacheNode._get_prompt_stats(prompt)
        self.next = next
        self.prev = prev

    @staticmethod
    def _get_prompt_stats(prompt: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[int, int]:
        """Get the number of virtual tokens and memory size of a prompt. Note that we round up to the nearest
        increment of 512.

        Args:
            prompt: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                Prompt tuple/tensor we want to take the size of.
        
        Return:
            (prompt virtual token count, prompt size in MiB)
        """
        # In some cases, we may have None, e.g., an encoder / decoder
        # where we don't have prompts to inject for both components
        if prompt is None:
            return 0, 0
        # We either have a Tensor or an iterable of tensors; if it's not
        # a tensor, take the size of all contained tensor objects.
        elif not isinstance(prompt, torch.Tensor):
            return tuple(sum(x) for x in zip(*map(PromptCacheNode._get_prompt_stats, prompt)))
        # Otherwise it's a tensor; round up to nearest 512 increment & convert to mb.
        # See: https://discuss.pytorch.org/t/how-to-know-the-memory-allocated-for-a-tensor-on-gpu/28537/15
        raw_size = prompt.element_size() * prompt.nelement()
        return prompt.shape[0], (math.ceil(raw_size / 512) * 512) / (1024 ** 2)


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add_node_as_head(self, cache_node: PromptCacheNode):
        """Adds a PromptCacheNode, which is not currently part of the linked list,
        to the head of the linked list.

        Args:
            cache_node: PromptCacheNode
                Node containing a prompt being added to the cache.
        """
        # First node in the list
        if self.head is None and self.tail is None:
            self.tail = cache_node
        # If we have an existing head, push it to the second in the list
        elif self.head:
            cache_node.next = self.head
            self.head.prev = cache_node
        self.head = cache_node

    def pop_tail_node(self) -> PromptCacheNode:
        """Deletes the tail from the doubly linked list.

        Returns:
            PromptCacheNode
                Node to be deleted from the doubly linked list.
        """
        if self.head is None and self.tail is None:
            raise ValueError("Cannot remove node from an empty list")
        del_node = self.tail
        # List contains only one node
        if self.head is self.tail:
            self.head = None
            self.tail = None
        # List contains 2+ entries
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        return del_node

    def move_node_to_head(self, cache_node: PromptCacheNode):
        """Moves an existing PromptCacheNode in the linked list to the head,
        which executes when we have a cache hit.

        Args:
            cache_node: PromptCacheNode
                Node in the doubly linked list being moved to the head.
        """
        if cache_node is self.head:
            return
        elif cache_node is self.tail:
            self.tail = self.tail.prev
            self.tail.next = None
        else:
            cache_node.prev.next = cache_node.next
            cache_node.next.prev = cache_node.prev
        # Insert as the head
        self.head.prev = cache_node
        cache_node.next = self.head
        cache_node.prev = None
        self.head = cache_node


class PrefixCache:
    """Holds the cache of injectable prompts for a single model.
    """
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        max_length: int,
        encoder_decoder: bool,
        decoder_start_tok_embedding: torch.Tensor,
    ):
        self.max_length = max_length
        self.embed_size = decoder_start_tok_embedding.shape[1] \
            if decoder_start_tok_embedding is not None else None
        self.device: torch.device = device
        self.dtype = dtype

        self.is_encoder_decoder = encoder_decoder
        self.decoder_start_tok_embedding = decoder_start_tok_embedding

        self.cache_map: Dict[str, PromptCacheNode] = {}
        self.cache_dll: DoublyLinkedList = DoublyLinkedList()
        self.cache_size_mb = 0
        # Initialize a context manager around a thread lock; when we enter
        # the context manager, we acquire the lock, and when we exit, we
        # release it.
        self.requires_lock = ThreadLockManager(threading.Lock())

    def get(self, prefix_id: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve a prompt tensor from the cache dict. If the prefix ID is not part of the
        cache, we attempt to load its tensors from disk making the assumption that:
            - decoder only models have their tensors written to a file named decoder.pt
            - encoder/decoder models have their tensors written to a file named encoder.pt
              and decoder.pt, respectively.
        where the tensor files are located in directory: <PREFIX_STORE_PATH> / <prefix_id>.
        If a prefix_id is not part of the cache, it will be loaded into the cache.

        Args:
            prefix_id: str
                Cache key of the prompt tensor to be retrieved.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                Loaded encoder / decoder prompt tensor for the model under consideration.
        """
        with self.requires_lock:
            cache_node = self._get_from_cache(prefix_id)
        if cache_node is None:
            # Release the lock & load the tensors
            prefix = self._load_embedding_tensors(prefix_id)
            # Relock & add the newly loaded tensor to the cache
            cache_node = self._add_prefix_id_to_cache(prefix_id, prefix)
        return cache_node.prompt

    def _load_embedding_tensors(self, prefix_id: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Load prompt tensors corresponding to a single prefix ID to disk. The return
        value of this function should be what is returned when indexing into the cache
        for prompt injection.

        Args:
            prefix_id: str
                Name of the directory to load prompt tensors from.
        
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                Loaded encoder / decoder prompt tensor for the model under consideration.
        """
        decoder_prefix = self._load_embedding_tensor(prefix_id, "decoder.pt", dtype=self.dtype)
        # For encoder-decoder we store a tuple of (encoder_prefix, decoder_prefix),
        # at least one must be non-None
        if self.is_encoder_decoder:
            encoder_prefix = self._load_embedding_tensor(prefix_id, "encoder.pt", dtype=self.dtype)
            if decoder_prefix is None:
                if encoder_prefix is None:
                    raise PrefixNotFound(f"Prefix id {prefix_id} not found")
            else:
                decoder_prefix = decoder_prefix.to(self.device, non_blocking=True)
                # TODO confirm this cat is correct
                decoder_prefix = torch.cat((decoder_prefix, self.decoder_start_tok_embedding))
            if encoder_prefix is not None:
                encoder_prefix = encoder_prefix.to(self.device, non_blocking=True)
            prefix = encoder_prefix, decoder_prefix
        # For decoder-only we store just the decoder prefix
        elif decoder_prefix is None:
            raise PrefixNotFound(f"Prefix id {prefix_id} not found")
        else:
            prefix = decoder_prefix.to(self.dtype).to(self.device, non_blocking=True)
        return prefix

    def _load_embedding_tensor(self, prefix_id: str, filename: str, dtype: torch.dtype) -> torch.Tensor:
        """Load an embedding tensor from a single file.

        Args:
            prefix_id: str
                Name of the file that we want to load a torch tensor from.
            filename: str
                Name of the file to be loaded.

        Returns:
            torch.Tensor
                Tensor object corresponding to loaded prompt.
        """
        if not VALID_PREFIX_ID_PATTERN.fullmatch(prefix_id):
            raise Exception(f"Invalid prefix id {prefix_id}, must contain only alphanumeric, _ and - and /")
        prefix_path = PREFIX_STORE_PATH / prefix_id / filename
        # Check for path traversal
        if not os.path.normpath(prefix_path).startswith(str(PREFIX_STORE_PATH) + "/"):
            raise Exception(f"Invalid prefix id {prefix_id}")
        if not prefix_path.is_file():
            return None

        logger.info(f"Loading new prefix {prefix_id}/{filename}")
        prefix = torch.load(prefix_path, weights_only=True, map_location=torch.device('cpu'))
        # Verify that it's a tensor of the correct shape
        if not torch.is_tensor(prefix) or len(prefix.shape) != 2:
            raise Exception(f"Invalid prefix embedding tensor")
        if prefix.shape[0] == 0 or prefix.shape[0] > self.max_length:
            raise Exception(f"Invalid prefix embedding length of {prefix.shape[0]}")
        if self.is_encoder_decoder and prefix.shape[0] % 2 != 0:
            raise Exception(f"Encoder/decoder prefix tensor must be of even length")
        if prefix.shape[1] != self.embed_size and self.embed_size is not None:
            raise Exception(
                f"Prefix embedding tensor dim {prefix.shape[1]} does not match model ({self.embed_size})"
            )
        # convert to the desired dtype
        converted_prefix = prefix.to(dtype)
        # detect if we have non-finite elements after the conversion that will
        # cause problems for inference
        if not converted_prefix.isfinite().all():
            # check if the problem was in the pre-converted tensor
            if not prefix.isfinite().all():
                raise Exception(f"Prefix contains non-finite elements")
            raise Exception(f"Prefix contains non-finite elements after conversion from {prefix.dtype} to {dtype}")

        converted_prefix.requires_grad = False
        return converted_prefix

    def _add_prefix_id_to_cache(
        self,
        prefix_id: str,
        prefix: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> PromptCacheNode:
        """Ensure that the prefix ID exists in the prompt cache and return it.
        This function either creates the entry or returns an existing one.

        Args:
            prefix_id: str
                key to be used in the prompt cache.
            prefix: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
                torch tensor or tuple of torch tensors to be mapped to in the cache.

        Returns:
            PromptCacheNode
                Node holding the loaded prompt vectors.
        """
        # Create the new node up front
        new_cache_node = PromptCacheNode(prompt=prefix, prefix_id=prefix_id)
        del_tensors = {}

        new_prompt_virtual_tokens = new_cache_node.prompt_virtual_tokens
        new_prompt_size_mb = new_cache_node.prompt_size_mb
        with self.requires_lock:
            # If we already have it, return the node in the cache.
            # This will release the tensor we just loaded.
            if cache_node := self._get_from_cache(prefix_id):
                return cache_node

            # If this cache node is bigger than the cache, there's no way it'll fit!
            if new_cache_node.prompt_size_mb > PROMPT_CACHE_SIZE_MB:
                raise ValueError(f"Prefix ID object {prefix_id} exceeds the allowed cache size")

            while self.cache_size_mb + new_prompt_size_mb > PROMPT_CACHE_SIZE_MB:
                # Hold a reference to the set of things to be deallocated until we're out of the
                # critical section; then, we can handle the cache keys in a thread safe way
                # without deallocating our tensors in it.
                #
                # As such, our cache will sometimes go over its capacity, but we won't block as
                # long, if we can avoid it.
                del_node = self.cache_dll.pop_tail_node()
                del_tensors[del_node.prefix_id] = del_node
                del self.cache_map[del_node.prefix_id]
                self.cache_size_mb -= del_node.prompt_size_mb
            self.cache_dll.add_node_as_head(new_cache_node)
            self.cache_map[prefix_id] = new_cache_node
            self.cache_size_mb += new_prompt_size_mb
            cache_size_mb = self.cache_size_mb
        if del_tensors:
            logger.info(f"Deleted prefixes {list(del_tensors.keys())} from the prompt cache")

        logger.info(
            f"Added prefix {prefix_id} to the prompt cache, has {new_prompt_virtual_tokens} virtual tokens"
            f", size {new_prompt_size_mb:.3f}MiB, total cache size is now {cache_size_mb:.2f}MiB"
        )
        return new_cache_node

    def _get_from_cache(self, prefix_id: str) -> PromptCacheNode:
        """Gets a node from the cache map using its prefix ID. If it exists, move it
        to the head of the list. It is the responsibility of the method calling this
        one to ensure this operation is executed in a thread-safe way.

        Args:
            prefix_id: str
                ID of the prompt we want to search for.

        Returns:
            PromptCacheNode
                Prompt cache node corresponding to the prefix_id if one is found, or None.
        """
        cache_node = self.cache_map.get(prefix_id)
        if cache_node is not None:
            self.cache_dll.move_node_to_head(cache_node)
        return cache_node

    def clear(self) -> None:
        """Empty the prompt cache."""
        with self.requires_lock:
            self.cache_map.clear()
            self.cache_dll = DoublyLinkedList()
            self.cache_size_mb = 0

    def __len__(self) -> int:
        """Get the size of the prompt cache."""
        with self.requires_lock:
            return len(self.cache_map.keys())

    def keys(self) -> List[str]:
        """Get the keys in the prompt cache."""
        with self.requires_lock:
            return list(self.cache_map.keys())
