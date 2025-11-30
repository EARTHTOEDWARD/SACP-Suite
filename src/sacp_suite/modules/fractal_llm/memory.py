"""Semantic memory helper from Fractal LLM Lab (Bloom filter based)."""

from __future__ import annotations

try:
    from pybloom_live import BloomFilter
except Exception:  # pragma: no cover - optional dependency
    BloomFilter = None  # type: ignore


class SemanticMemory:
    """Small wrapper over BloomFilter to track seen prompts/edits."""

    def __init__(self, capacity: int = 10_000, error_rate: float = 0.01):
        if BloomFilter is None:
            raise ImportError("pybloom-live is required for SemanticMemory")
        self._bf = BloomFilter(capacity, error_rate)

    def seen(self, key: str) -> bool:
        return key in self._bf

    def add(self, key: str) -> None:
        self._bf.add(key)

    def novelty_score(self, key: str) -> float:
        """1.0 if brand-new; 0.0 if definitely seen; 0.5 if maybe."""
        return 0.0 if self.seen(key) else 1.0

