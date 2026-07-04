"""Shared test fixtures.

Real FastEmbed models download ~100MB of weights, so tests inject deterministic
fake embedders: word-hash based, which preserves the property that texts sharing
words are close in both vector spaces. No network, no API keys, fast.
"""

from __future__ import annotations

import math
import re
import zlib

import pytest
from qdrant_client import QdrantClient

from app.store import DENSE_DIM, Embedders, HybridStore


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def fake_dense(texts: list[str]) -> list[list[float]]:
    out = []
    for t in texts:
        vec = [0.0] * DENSE_DIM
        for w in _words(t):
            vec[zlib.crc32(w.encode()) % DENSE_DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        out.append([x / norm for x in vec])
    return out


def fake_sparse(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    out = []
    for t in texts:
        tf: dict[int, float] = {}
        for w in _words(t):
            idx = zlib.crc32(w.encode()) % 50021
            tf[idx] = tf.get(idx, 0.0) + 1.0
        items = sorted(tf.items())
        out.append(([i for i, _ in items], [v for _, v in items]))
    return out


FAKE_EMBEDDERS = Embedders(
    dense_docs=fake_dense,
    dense_query=fake_dense,
    sparse_docs=fake_sparse,
    sparse_query=fake_sparse,
)


@pytest.fixture
def store() -> HybridStore:
    return HybridStore(client=QdrantClient(":memory:"), embedders=FAKE_EMBEDDERS)
