"""Query-complexity classification — the routing signal for adaptive retrieval.

Phase 2 groundwork: production agentic-RAG systems don't run the expensive
iterative-retrieval loop on every query — they route. A cheap classifier sends
simple questions down the one-shot path (LEMMA's current pipeline) and reserves
multi-hop reasoning for questions that actually need it.

This module only *classifies and explains* the decision so the demo can surface
it (LEMMA's transparency thesis: show how hard the system had to think). The
iterative loop itself is a later slice. Stdlib-only, no model call — the routing
decision must be near-free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Signals that a question likely needs more than one retrieval hop.
_MULTIHOP_MARKERS = (
    r"\band\b",
    r"\bthen\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bcompare\b",
    r"\bdifference\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bboth\b",
    r"\brelationship\b",
    r"\bhow does .* affect\b",
    r"\bwhy\b",
)
_COMPARISON = re.compile("|".join(_MULTIHOP_MARKERS), re.IGNORECASE)


@dataclass(frozen=True)
class Complexity:
    label: str  # "simple" | "multi-hop"
    reason: str  # short human-readable justification for the demo


def classify(question: str) -> Complexity:
    """Heuristic, deterministic, no network. Errs toward 'simple' to keep the
    cheap path the default — matching the production consensus that most queries
    are one-shot."""
    q = question.strip()
    words = re.findall(r"[a-zA-Z0-9]+", q)
    question_marks = q.count("?")

    if not words:
        return Complexity("simple", "empty or trivial query")

    signals = []
    if _COMPARISON.search(q):
        signals.append("comparison/multi-part phrasing")
    if question_marks > 1:
        signals.append("multiple questions")
    if len(words) > 25:
        signals.append("long, detailed question")

    if signals:
        return Complexity("multi-hop", "; ".join(signals))
    return Complexity("simple", "single, direct question")
