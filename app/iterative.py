"""Bounded iterative retrieval — search-as-a-tool, hard-capped at three steps.

What this is for
----------------
`complexity.classify` already routes questions to `simple` or `multi-hop`, and
the README is honest that the label has so far been decorative: "both routes
still use the same single-pass hybrid retrieval". This is the loop the
`multi-hop` route was always meant to reach.

A multi-hop question ("who wrote X and what did they publish after?") cannot be
answered from one retrieval, because the second half of the question depends on
what the first half returns. So: search, look at what came back, search again
with what you learned, stop.

Why the bound is the design, not a safety net
--------------------------------------------
An unbounded search agent is the standard failure mode of agentic RAG: it
wanders, each hop costs a model call and a retrieval, and the honest answer to
"when should it stop?" is "sooner than it wants to". The cap is three steps
(README's published promise) and it is enforced here, not left to the planner's
judgement — a planner that never says "done" must still terminate.

Provider-free by construction
-----------------------------
The planner is INJECTED. In production it is a model call that reads the hits so
far and returns either a follow-up query or None. In tests it is a list. That
split is deliberate: the control flow — bounding, dedup, accumulation, stop
reasons — is the part that must never break, and it is exactly the part that
needs no provider to test. Everything here is deterministic given a planner.

Transparency
------------
Every step is recorded. LEMMA's thesis is showing how hard the system had to
think, and a retrieval loop that cannot explain its own hops is a worse demo
than one that never loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Protocol

from .store import Hit

MAX_STEPS = 3  # README's published promise. Changing it changes the docs.


class Planner(Protocol):
    """Decides the next query, or None to stop.

    Given the original question and every hit gathered so far, return a
    follow-up query string, or None when the evidence is sufficient. In
    production this is a model call; the loop does not care.
    """

    def __call__(self, question: str, hits: list[Hit], step: int) -> Optional[str]:
        ...


@dataclass(frozen=True)
class Step:
    """One retrieval hop, recorded for the demo panel."""

    index: int
    query: str
    new_hits: int  # hits this hop contributed that earlier hops had not
    total_hits: int


@dataclass(frozen=True)
class IterativeResult:
    hits: list[Hit]
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def hop_count(self) -> int:
        return len(self.steps)


def _key(hit: Hit) -> tuple[str, int]:
    """Identity of a chunk. Two hops retrieving the same chunk is the normal
    case, not an error — the second hop's query overlaps the first. Dedup on
    (doc, chunk), never on score: the same chunk can come back with a different
    score from a different query, and keeping both would double-count evidence
    and let one chunk dominate the context window."""
    return (hit.doc_name, hit.chunk_index)


def retrieve(
    question: str,
    search: Callable[[str], Iterable[Hit]],
    planner: Planner,
    max_steps: int = MAX_STEPS,
) -> IterativeResult:
    """Search, re-plan, search again — at most `max_steps` times.

    Always performs at least one retrieval with the original question, so a
    planner that immediately returns None degrades to exactly today's one-shot
    behaviour. That is the property that makes this safe to route into: the
    worst case is the current pipeline plus a dict lookup.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    seen: set[tuple[str, int]] = set()
    hits: list[Hit] = []
    steps: list[Step] = []
    query = question
    stop_reason = "planner finished"

    for i in range(max_steps):
        found = list(search(query))
        fresh = 0
        for hit in found:
            k = _key(hit)
            if k in seen:
                continue
            seen.add(k)
            hits.append(hit)
            fresh += 1
        steps.append(Step(index=i + 1, query=query, new_hits=fresh, total_hits=len(hits)))

        if i + 1 >= max_steps:
            # Bound reached. Recorded distinctly from "planner finished" because
            # the two mean opposite things to whoever reads the trace: one is the
            # system deciding it has enough, the other is the system being cut
            # off. Collapsing them would hide the case where the cap is too low.
            stop_reason = "step limit reached"
            break

        follow_up = planner(question, list(hits), i + 1)
        if not follow_up or not follow_up.strip():
            break
        if follow_up.strip() == query.strip():
            # A planner repeating itself would burn the remaining budget on
            # identical searches returning identical hits. Stop instead.
            stop_reason = "planner repeated the query"
            break
        query = follow_up.strip()

    return IterativeResult(hits=hits, steps=steps, stop_reason=stop_reason)


def trace_payload(result: IterativeResult) -> dict:
    """Shape for the API/demo panel — the visible half of the thesis."""
    return {
        "hops": result.hop_count,
        "stop_reason": result.stop_reason,
        "steps": [
            {
                "step": s.index,
                "query": s.query,
                "new_hits": s.new_hits,
                "total_hits": s.total_hits,
            }
            for s in result.steps
        ],
    }
