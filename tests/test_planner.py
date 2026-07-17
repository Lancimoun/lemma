"""llm.plan_follow_up — the provider adapter for bounded iterative retrieval.

The loop's control flow is tested in test_iterative.py with no provider at all.
This file tests the other half: the adapter that turns a model call into the
loop's `Planner` protocol. The client is stubbed, so this still needs no key and
no network.

The contract that matters most here is FAIL CLOSED. Every way this can go wrong
-- missing key, API down, a model that ignores the output format -- must return
None, which stops the loop and degrades to the one-shot behaviour lemma has
today. A retrieval enhancer that can take /ask down with it is a downgrade.
"""

from __future__ import annotations

import pytest

from app import config, llm
from app.store import Hit


def hit(doc="d", idx=0, text="some retrieved text"):
    return Hit(doc_name=doc, chunk_index=idx, text=text, score=1.0)


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeMessage:
    def __init__(self, text):
        self.content = [FakeBlock(text)]


class FakeClient:
    """Records the request so the tests can assert on what was actually sent."""

    def __init__(self, reply="DONE", raises=None):
        self._reply = reply
        self._raises = raises
        self.seen = {}
        self.messages = self

    def create(self, **kwargs):
        self.seen = kwargs
        if self._raises:
            raise self._raises
        return FakeMessage(self._reply)


class TestStopsWhenItShould:
    def test_done_returns_none(self):
        c = FakeClient("DONE")
        assert llm.plan_follow_up("q", [hit()], 1, client=c) is None

    def test_done_is_matched_case_insensitively_and_with_trailing_noise(self):
        # Models append periods and newlines regardless of instructions.
        for reply in ("done", "Done.", "DONE\n", "  done  "):
            c = FakeClient(reply)
            assert llm.plan_follow_up("q", [hit()], 1, client=c) is None, reply

    def test_empty_reply_returns_none(self):
        assert llm.plan_follow_up("q", [hit()], 1, client=FakeClient("")) is None

    def test_a_rambling_reply_is_treated_as_done_not_as_a_query(self):
        # "No explanation" is an instruction, not a guarantee. A 30-word
        # paragraph is a malformed DONE; searching for an essay wastes the hop
        # and pollutes the hits with whatever the essay happens to match.
        essay = " ".join(["the model decided to explain itself at length"] * 5)
        assert llm.plan_follow_up("q", [hit()], 1, client=FakeClient(essay)) is None


class TestFailsClosed:
    def test_api_error_returns_none_rather_than_raising(self):
        c = FakeClient(raises=RuntimeError("API down"))
        assert llm.plan_follow_up("q", [hit()], 1, client=c) is None

    def test_missing_key_returns_none_rather_than_raising(self, monkeypatch):
        # get_client() raises AnswerError without a key. answer_question is
        # allowed to surface that; the planner is not -- retrieval must still
        # work when the enhancement cannot.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setattr(llm, "_client", None)
        assert llm.plan_follow_up("q", [hit()], 1) is None

    def test_a_malformed_response_object_returns_none(self):
        class Broken:
            messages = None

            def create(self, **kwargs):  # pragma: no cover
                raise AttributeError("no messages attr")

        assert llm.plan_follow_up("q", [hit()], 1, client=Broken()) is None


class TestReturnsAQuery:
    def test_a_short_reply_becomes_the_follow_up_query(self):
        c = FakeClient("what did she publish afterwards")
        assert llm.plan_follow_up("q", [hit()], 1, client=c) == "what did she publish afterwards"

    def test_surrounding_quotes_are_stripped(self):
        # Models quote things they were told not to explain.
        assert llm.plan_follow_up("q", [hit()], 1, client=FakeClient('"later works"')) == "later works"
        assert llm.plan_follow_up("q", [hit()], 1, client=FakeClient("'later works'")) == "later works"


class TestRequestShape:
    def test_uses_the_cheap_planner_model_not_the_answering_model(self):
        # The point of the whole design: routing must be near-free. If this ever
        # defaults to config.MODEL, every multi-hop question pays Opus twice.
        c = FakeClient("DONE")
        llm.plan_follow_up("q", [hit()], 1, client=c)
        assert c.seen["model"] == config.PLANNER_MODEL
        assert c.seen["model"] != config.MODEL

    def test_token_budget_is_small(self):
        c = FakeClient("DONE")
        llm.plan_follow_up("q", [hit()], 1, client=c)
        assert c.seen["max_tokens"] <= 256, "a planner emitting prose is a bug, not a feature"

    def test_the_question_and_retrieved_text_both_reach_the_model(self):
        c = FakeClient("DONE")
        llm.plan_follow_up("who wrote it", [hit(text="Ada wrote the notes")], 1, client=c)
        sent = c.seen["messages"][0]["content"]
        assert "who wrote it" in sent
        assert "Ada wrote the notes" in sent

    def test_chunk_text_is_truncated_so_a_long_corpus_cannot_blow_the_budget(self):
        c = FakeClient("DONE")
        llm.plan_follow_up("q", [hit(text="x" * 5000)], 1, client=c)
        sent = c.seen["messages"][0]["content"]
        assert len(sent) < 3000, "planner prompt grows unbounded with chunk size"


def test_matches_the_iterative_planner_protocol():
    """The adapter must be droppable into the loop with no shim."""
    from app import iterative

    c = FakeClient("DONE")
    planner = lambda question, hits, step: llm.plan_follow_up(question, hits, step, client=c)
    result = iterative.retrieve("q", lambda _: [hit()], planner)
    assert result.hop_count == 1
    assert result.stop_reason == "planner finished"
