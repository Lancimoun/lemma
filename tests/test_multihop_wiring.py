"""The `/ask` wiring: multi-hop questions actually loop, simple ones don't.

`app.iterative` (the loop) and `app.complexity` (the router) each have their own
unit tests. What was untested until this file is the *wiring in `main._retrieve`*
that connects them — and an unwired feature is exactly the "decorative label" the
README used to confess to. These tests exercise `_retrieve` directly rather than
the full HTTP stack: the rate limiter and SSE plumbing are not what's under test,
the routing decision is.

The planner and the classifier are both stubbed, on purpose. This proves the
wiring routes correctly given a decision; whether Haiku makes a good decision is
`test_planner`'s job, and whether the classifier labels well is
`test_complexity`'s. Each test owns one seam.
"""

from __future__ import annotations

import pytest

from app import complexity, main
from app.store import Hit


def hit(doc: str, idx: int) -> Hit:
    return Hit(doc_name=doc, chunk_index=idx, text=f"{doc}#{idx}", score=1.0)


class FakeStore:
    """Records every search() call and returns a fresh chunk each time, so hop
    count is visible as len(calls) and accumulation is visible in the hits."""

    def __init__(self):
        self.calls: list[str] = []

    def search(self, query, top_k=8, prefetch_k=20):
        n = len(self.calls)
        self.calls.append(query)
        return [hit("doc", n)]  # a distinct chunk per hop → dedup keeps them all


@pytest.fixture
def store(monkeypatch):
    s = FakeStore()
    monkeypatch.setattr(main, "get_store", lambda: s)
    return s


def _route(monkeypatch, label):
    monkeypatch.setattr(
        main.complexity, "classify",
        lambda q: complexity.Complexity(label, "stubbed route"),
    )


def _planner(monkeypatch, *queries):
    seq = list(queries)
    monkeypatch.setattr(
        main.llm, "plan_follow_up",
        lambda question, hits, step: seq.pop(0) if seq else None,
    )


def test_multihop_question_actually_loops(store, monkeypatch):
    # The feature: a multi-hop label + a planner asking for one follow-up must
    # produce a second retrieval and a two-hop trace. This is the assertion that
    # would have failed every day the label was decorative.
    _route(monkeypatch, "multi-hop")
    _planner(monkeypatch, "the follow-up query")

    hits, route, trace = main._retrieve("who wrote X and what came after?")

    assert route.label == "multi-hop"
    assert store.calls == ["who wrote X and what came after?", "the follow-up query"]
    assert trace is not None and trace["hops"] == 2
    assert len(hits) == 2  # both hops' distinct chunks accumulated


def test_multihop_degrades_to_one_shot_when_planner_declines(store, monkeypatch):
    # The safety property: planner returns None (e.g. no API key → fail-closed).
    # Routing multi-hop must then cost exactly one search, like today's pipeline.
    _route(monkeypatch, "multi-hop")
    _planner(monkeypatch)  # immediately None

    hits, route, trace = main._retrieve("a multi-hop-looking question")

    assert store.calls == ["a multi-hop-looking question"]  # one search only
    assert trace is not None and trace["hops"] == 1
    assert len(hits) == 1


def test_simple_question_never_enters_the_loop(store, monkeypatch):
    # A simple question must not carry a retrieval trace at all, and must never
    # invoke the planner — the one-shot path is unchanged.
    _route(monkeypatch, "simple")
    called = {"planner": False}
    monkeypatch.setattr(
        main.llm, "plan_follow_up",
        lambda *a, **k: called.__setitem__("planner", True) or None,
    )

    hits, route, trace = main._retrieve("what is X?")

    assert store.calls == ["what is X?"]
    assert trace is None
    assert called["planner"] is False


def test_kill_switch_forces_one_shot_even_for_multihop(store, monkeypatch):
    # MULTIHOP_ENABLED=0 must route every question one-shot regardless of label.
    _route(monkeypatch, "multi-hop")
    monkeypatch.setattr(main.config, "MULTIHOP_ENABLED", False)
    _planner(monkeypatch, "a follow-up that must never run")

    hits, route, trace = main._retrieve("multi-hop question with the switch off")

    assert store.calls == ["multi-hop question with the switch off"]
    assert trace is None


def test_empty_question_is_rejected_before_any_search(store, monkeypatch):
    _route(monkeypatch, "simple")
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        main._retrieve("")
    assert exc.value.status_code == 400
    assert store.calls == []
