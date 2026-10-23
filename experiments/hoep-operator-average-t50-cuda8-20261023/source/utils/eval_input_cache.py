"""Bounded CPU cache for deterministic image preprocessing, never features."""
from collections import OrderedDict


class EvalInputCache:
    def __init__(self, max_bytes):
        if max_bytes <= 0:
            raise ValueError('evaluation cache budget must be positive')
        self.max_bytes = int(max_bytes)
        self.bytes_used = 0
        self.hits = self.misses = 0
        self._entries = OrderedDict()

    def get(self, key):
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return value.clone()  # callers cannot mutate the cached input

    def put(self, key, tensor):
        size = tensor.numel() * tensor.element_size()
        if size > self.max_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.bytes_used -= previous.numel() * previous.element_size()
        while self._entries and self.bytes_used + size > self.max_bytes:
            _, oldest = self._entries.popitem(last=False)
            self.bytes_used -= oldest.numel() * oldest.element_size()
        self._entries[key] = tensor.detach().cpu().clone()
        self.bytes_used += size
