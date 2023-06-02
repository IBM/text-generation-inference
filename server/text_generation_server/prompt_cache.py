import os
import re
from pathlib import Path
from typing import Dict, Optional, Union, Tuple

import torch

PREFIX_STORE_PATH = Path(os.getenv("PREFIX_STORE_PATH", "prompt_prefixes"))

VALID_PREFIX_ID_PATTERN = re.compile("[/\\w\\-]+")

# TODO
# - Make into two-layer LRU cache (CPU, GPU)
# - Include explicit time-based expiry
# - Verify cache threadsafety


class PrefixNotFound(Exception):
    pass


class PrefixCache:
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

        self.cache: Dict[str, torch.Tensor] = {}

    def get(self, prefix_id: str) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prefix = self.cache.get(prefix_id)
        if prefix is None:
            decoder_prefix = self._load_embedding_tensor(prefix_id, "decoder.pt")
            # For encoder-decoder we store a tuple of (encoder_prefix, decoder_prefix),
            # at least one must be non-None
            if self.is_encoder_decoder:
                encoder_prefix = self._load_embedding_tensor(prefix_id, "encoder.pt")
                if decoder_prefix is None:
                    if encoder_prefix is None:
                        raise PrefixNotFound(f"Prefix id {prefix_id} not found")
                else:
                    # TODO confirm this cat is correct
                    decoder_prefix = torch.cat((decoder_prefix, self.decoder_start_tok_embedding))
                    decoder_prefix = decoder_prefix.to(self.dtype).to(self.device, non_blocking=True)
                if encoder_prefix is not None:
                    encoder_prefix = encoder_prefix.to(self.dtype).to(self.device, non_blocking=True)
                prefix = encoder_prefix, decoder_prefix
            # For decoder-only we store just the decoder prefix
            elif decoder_prefix is None:
                raise PrefixNotFound(f"Prefix id {prefix_id} not found")
            else:
                prefix = decoder_prefix.to(self.dtype).to(self.device, non_blocking=True)

            self.cache[prefix_id] = prefix

        return prefix

    def _load_embedding_tensor(self, prefix_id: str, filename: str) -> Optional[torch.Tensor]:
        if not VALID_PREFIX_ID_PATTERN.fullmatch(prefix_id):
            raise Exception(f"Invalid prefix id {prefix_id}, must contain only alphanumeric, _ and - and /")
        prefix_path = PREFIX_STORE_PATH / prefix_id / filename
        # Check for path traversal
        if not os.path.normpath(prefix_path).startswith(str(PREFIX_STORE_PATH) + "/"):
            raise Exception(f"Invalid prefix id {prefix_id}")
        if not prefix_path.is_file():
            return None

        print(f"Loading new prefix {prefix_id}/{filename}")
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

        prefix.requires_grad = False
        return prefix

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache.keys())

    def keys(self) -> list:
        return list(self.cache.keys())

