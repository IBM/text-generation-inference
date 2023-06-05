from typing import Dict, Optional, TypeVar

from text_generation_server.models.types import Batch

B = TypeVar("B", bound=Batch)


class Cache:
    def __init__(self):
        self.cache: Dict[int, B] = {}

    def pop(self, batch_id: int) -> Optional[B]:
        return self.cache.pop(batch_id, None)

    def set(self, entry: B):
        if entry is not None:
            self.cache[entry.batch_id] = entry

    def delete(self, batch_id: int):
        del self.cache[batch_id]

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache.keys())

    def keys(self) -> list:
        return list(self.cache.keys())

    def compact(self):
        for batch in self.cache.values():
            batch.compact()
