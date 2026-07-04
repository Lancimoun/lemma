"""Reliability evals — measured, not assumed.

Three signals, surfaced live in the UI:

1. Live recall probe  — writes a unique synthetic fact into the REAL index, retrieves it
                        through the REAL pipeline, verifies it comes back, then cleans up.
                        (Optionally also verifies the LLM answer contains the fact.)
2. Canary set         — fixed questions about the bundled handbook doc; retrieval hit-rate.
3. Ops metrics        — rolling latency percentiles, cost per query, cache-hit rate.
"""

from __future__ import annotations

import random
import threading
import time
import uuid
from collections import deque
from typing import Any, Callable, Sequence

from .ingest import Chunk
from .store import Hit, HybridStore

PROBE_PREFIX = "__probe__"
CANARY_DOC = "lemma-handbook.md"

# Substrings must exist in sample_docs/lemma-handbook.md — guarded by a unit test.
CANARIES: list[dict[str, str]] = [
    {
        "question": "Which fusion algorithm does LEMMA use to combine dense and sparse results?",
        "expect_substring": "reciprocal rank fusion",
    },
    {
        "question": "Which embedding model produces LEMMA's dense vectors?",
        "expect_substring": "bge-small",
    },
    {
        "question": "Which algorithm produces LEMMA's sparse keyword vectors?",
        "expect_substring": "BM25",
    },
    {
        "question": "How does the live recall probe verify that retrieval works?",
        "expect_substring": "synthetic fact",
    },
    {
        "question": "How does LEMMA keep its prompt costs low?",
        "expect_substring": "prompt caching",
    },
]

AskFn = Callable[[str, Sequence[Hit]], dict[str, Any]]


def run_recall_probe(store: HybridStore, ask_fn: AskFn | None = None) -> dict[str, Any]:
    """End-to-end self-test on the live index. Always cleans up after itself."""
    probe_id = uuid.uuid4().hex[:8]
    budget = random.randint(10_000, 99_999)
    doc_name = f"{PROBE_PREFIX}{probe_id}"
    fact = (
        f"Project Zephyr-{probe_id} is an internal initiative of the Aurora committee. "
        f"The launch budget approved for Project Zephyr-{probe_id} was exactly {budget} pesos."
    )
    question = f"What was the launch budget of Project Zephyr-{probe_id}?"

    store.add_chunks([Chunk(doc_name=doc_name, chunk_index=0, text=fact)])
    try:
        hits = store.search(question, top_k=5)
        top_hit_is_probe = bool(hits) and hits[0].doc_name == doc_name
        in_top_k = any(h.doc_name == doc_name for h in hits)

        result: dict[str, Any] = {
            "probe_id": probe_id,
            "retrieval_pass": top_hit_is_probe,
            "found_in_top_k": in_top_k,
        }
        if ask_fn is not None and in_top_k:
            response = ask_fn(question, hits)
            result["answer_pass"] = str(budget) in response.get("answer_text", "")
            result["answer_cost_usd"] = response.get("cost_usd")
        result["pass"] = top_hit_is_probe and result.get("answer_pass", True)
        return result
    finally:
        store.delete_doc(doc_name)


def run_canaries(store: HybridStore, top_k: int = 8) -> dict[str, Any]:
    """Retrieval hit-rate over the fixed canary questions (no LLM cost)."""
    results = []
    for canary in CANARIES:
        hits = store.search(canary["question"], top_k=top_k)
        passed = any(
            canary["expect_substring"].lower() in h.text.lower() for h in hits
        )
        results.append({"question": canary["question"], "pass": passed})
    passed_count = sum(1 for r in results if r["pass"])
    return {
        "passed": passed_count,
        "total": len(results),
        "hit_rate": round(passed_count / len(results), 3) if results else None,
        "results": results,
    }


class OpsMetrics:
    """Thread-safe rolling window of per-query operational stats."""

    def __init__(self, maxlen: int = 200):
        self._lock = threading.Lock()
        self._records: deque[dict] = deque(maxlen=maxlen)

    def record(
        self,
        latency_ms: float,
        cost_usd: float | None,
        cache_read_tokens: int,
        input_tokens: int,
    ) -> None:
        with self._lock:
            self._records.append(
                {
                    "ts": time.time(),
                    "latency_ms": latency_ms,
                    "cost_usd": cost_usd,
                    "cache_read_tokens": cache_read_tokens,
                    "input_tokens": input_tokens,
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
        if not records:
            return {"queries": 0}
        latencies = sorted(r["latency_ms"] for r in records)

        def pct(p: float) -> float:
            i = min(len(latencies) - 1, max(0, round(p * (len(latencies) - 1))))
            return latencies[i]

        total_prompt = sum(r["cache_read_tokens"] + r["input_tokens"] for r in records)
        cache_read = sum(r["cache_read_tokens"] for r in records)
        costs = [r["cost_usd"] for r in records if r["cost_usd"] is not None]
        return {
            "queries": len(records),
            "latency_p50_ms": round(pct(0.50), 1),
            "latency_p95_ms": round(pct(0.95), 1),
            "avg_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
            "cache_hit_rate": round(cache_read / total_prompt, 3) if total_prompt else 0.0,
        }
