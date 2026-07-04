"""Claude integration: grounded answers with NATIVE citations.

Retrieved chunks are passed as `document` content blocks with citations enabled,
so the API returns exact cited spans (`cited_text`) per claim — no prompt-engineered
"[1]" markers. Also captures usage for the cost/cache metrics shown in the eval panel.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

import anthropic

from . import config
from .store import Hit

SYSTEM_PROMPT = (
    "You are LEMMA, a research assistant that proves its answers.\n"
    "Answer the user's question using ONLY the provided documents. Every factual claim "
    "must be supported by the documents and cited. If the documents do not contain the "
    "answer, say so plainly — never speculate or use outside knowledge.\n"
    "Be concise and direct: lead with the answer."
)

# NOTE: this prompt is far below the model's minimum cacheable prefix (4096 tokens on
# Opus 4.8), so the cache marker is a placement demo; it starts paying for itself
# automatically if the system prompt grows past the threshold.
_SYSTEM = [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]

_client: anthropic.Anthropic | None = None


class AnswerError(RuntimeError):
    """Raised when the model can't produce a usable answer."""


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise AnswerError("Claude answering is not configured: ANTHROPIC_API_KEY is missing.")
        _client = anthropic.Anthropic()
    return _client


def answer_question(
    question: str,
    hits: Sequence[Hit],
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    model = model or config.MODEL
    max_tokens = max_tokens or config.MAX_TOKENS

    content: list[dict] = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": h.text},
            "title": f"{h.doc_name} · chunk {h.chunk_index}",
            "citations": {"enabled": True},
        }
        for h in hits
    ]
    content.append({"type": "text", "text": question})

    started = time.perf_counter()
    try:
        with get_client().messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": content}],
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise AnswerError(f"Claude API error ({exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise AnswerError(f"Could not reach the Claude API: {exc}") from exc
    except TypeError as exc:
        # The Anthropic SDK raises TypeError when credentials are absent or malformed.
        raise AnswerError(f"Claude client is not configured correctly: {exc}") from exc
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Always check stop_reason before reading content.
    if msg.stop_reason == "refusal":
        raise AnswerError("The model declined to answer this request.")

    segments: list[dict] = []
    for block in msg.content:
        if block.type != "text":
            continue
        citations = [
            {
                "document_title": getattr(c, "document_title", None),
                "document_index": getattr(c, "document_index", None),
                "cited_text": getattr(c, "cited_text", ""),
            }
            for c in (getattr(block, "citations", None) or [])
        ]
        segments.append({"text": block.text, "citations": citations})

    usage = msg.usage
    return {
        "model": msg.model,
        "stop_reason": msg.stop_reason,
        "truncated": msg.stop_reason == "max_tokens",
        "segments": segments,
        "answer_text": "".join(s["text"] for s in segments),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        },
        "cost_usd": estimate_cost(model, usage),
        "latency_ms": round(latency_ms, 1),
    }


def estimate_cost(model: str, usage: Any) -> float | None:
    """Display-only estimate from the published per-MTok prices."""
    prices = config.PRICES.get(model)
    if not prices:
        return None
    in_price, out_price = prices
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cost = (
        usage.input_tokens * in_price
        + cache_read * in_price * 0.1      # cache reads ≈ 0.1× input price
        + cache_write * in_price * 1.25    # 5-min cache writes ≈ 1.25× input price
        + usage.output_tokens * out_price
    ) / 1_000_000
    return round(cost, 6)
