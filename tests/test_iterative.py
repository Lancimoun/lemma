"""Bounded iterative retrieval — the control flow, with no provider anywhere.

The planner is injected, so every test here is deterministic. What is being
proven is the part that must never break regardless of which model plans the
hops: the loop terminates, it terminates for the right reason, and it does not
double-count evidence.
"""

from __future__ import annotations

import pytest

from app import iterative
from app.store import Hit


def hit(doc: str, idx: int, score: float = 1.0) -> Hit:
    return Hit(doc_name=doc, chunk_index=idx, text=f"{doc}#{idx}", score=score)


def fixed_search(*batches):
    """A search() that returns a different batch per call, then repeats the last."""
    calls = {"n": 0}

    def _search(query):
        i = min(calls["n"], len(batches) - 1)
        calls["n"] += 1
        return batches[i]

    return _search


def planner_from(*queries):
    """A planner that yields each query in turn, then None."""
    seq = list(queries)

    def _planner(question, hits, step):
        return seq.pop(0) if seq else None

    return _planner


class TestTermination:
    def test_a_planner_that_never_stops_still_terminates_at_the_bound(self):
        # The whole reason the cap is in the loop and not in the planner: an
        # agent that always wants one more hop must still stop.
        greedy = lambda question, hits, step: f"another angle {step}"
        result = iterative.retrieve("q", fixed_search([hit("a", 1)]), greedy)
        assert result.hop_count == iterative.MAX_STEPS == 3
        assert result.stop_reason == "step limit reached"

    def test_a_planner_that_stops_immediately_degrades_to_one_shot(self):
        # This is the safety property that makes the loop safe to route into:
        # worst case is today's pipeline plus a dict lookup.
        result = iterative.retrieve("q", fixed_search([hit("a", 1)]), planner_from())
        assert result.hop_count == 1
        assert result.stop_reason == "planner finished"

    def test_always_searches_once_even_before_consulting_the_planner(self):
        # A planner cannot veto the original question — it only gets to propose
        # follow-ups, and it is never asked until after the first retrieval.
        asked = []

        def nosy(question, hits, step):
            asked.append(step)
            return None

        iterative.retrieve("q", fixed_search([hit("a", 1)]), nosy)
        assert asked == [1], "planner must be consulted after the first search, not before"

    def test_empty_follow_up_stops(self):
        for empty in ("", "   ", None):
            result = iterative.retrieve(
                "q", fixed_search([hit("a", 1)]), lambda *a, **k: empty
            )
            assert result.hop_count == 1

    def test_a_planner_repeating_itself_stops_rather_than_burning_the_budget(self):
        # Identical query -> identical hits. Spending two more model calls to
        # rediscover the same chunks is the expensive way to learn nothing.
        result = iterative.retrieve(
            "q", fixed_search([hit("a", 1)]), lambda *a, **k: "q"
        )
        assert result.hop_count == 1
        assert result.stop_reason == "planner repeated the query"

    def test_stop_reasons_distinguish_finished_from_cut_off(self):
        # These mean opposite things to a reader: one is the system deciding it
        # has enough, the other is the system being truncated. If the cap is too
        # low, only this distinction reveals it.
        cut = iterative.retrieve("q", fixed_search([hit("a", 1)]), lambda *a, **k: "more")
        done = iterative.retrieve("q", fixed_search([hit("a", 1)]), planner_from())
        assert cut.stop_reason != done.stop_reason

    def test_max_steps_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            iterative.retrieve("q", fixed_search([hit("a", 1)]), planner_from(), max_steps=0)


class TestEvidence:
    def test_hits_accumulate_across_hops(self):
        result = iterative.retrieve(
            "q",
            fixed_search([hit("a", 1)], [hit("b", 2)]),
            planner_from("second"),
        )
        assert [(h.doc_name, h.chunk_index) for h in result.hits] == [("a", 1), ("b", 2)]

    def test_the_same_chunk_from_two_hops_is_not_double_counted(self):
        # Overlapping hops are the NORMAL case - the follow-up query is related
        # to the first by construction. Double-counting would let one chunk
        # dominate the context window and inflate its apparent support.
        result = iterative.retrieve(
            "q",
            fixed_search([hit("a", 1)], [hit("a", 1, score=0.4), hit("b", 2)]),
            planner_from("second"),
        )
        assert len(result.hits) == 2
        assert result.steps[1].new_hits == 1, "second hop contributed only b#2"

    def test_dedup_is_by_chunk_identity_not_score(self):
        # The same chunk retrieved by a different query legitimately scores
        # differently. Deduping on score would keep both copies.
        result = iterative.retrieve(
            "q",
            fixed_search([hit("a", 1, score=0.9)], [hit("a", 1, score=0.1)]),
            planner_from("second"),
        )
        assert len(result.hits) == 1
        assert result.hits[0].score == 0.9, "first sighting wins; it is not re-scored"

    def test_a_hop_that_finds_nothing_new_is_recorded_honestly(self):
        result = iterative.retrieve(
            "q",
            fixed_search([hit("a", 1)], [hit("a", 1)]),
            planner_from("second"),
        )
        assert result.steps[1].new_hits == 0
        assert result.steps[1].total_hits == 1


class TestTrace:
    def test_every_hop_is_recorded_with_its_query(self):
        result = iterative.retrieve(
            "q",
            fixed_search([hit("a", 1)], [hit("b", 2)]),
            planner_from("who published it"),
        )
        payload = iterative.trace_payload(result)
        assert payload["hops"] == 2
        assert [s["query"] for s in payload["steps"]] == ["q", "who published it"]

    def test_the_first_step_records_the_original_question(self):
        result = iterative.retrieve("original", fixed_search([hit("a", 1)]), planner_from())
        assert result.steps[0].query == "original"

    def test_trace_payload_is_json_shaped(self):
        import json

        result = iterative.retrieve(
            "q", fixed_search([hit("a", 1)], [hit("b", 2)]), planner_from("more")
        )
        json.dumps(iterative.trace_payload(result))  # must not raise
