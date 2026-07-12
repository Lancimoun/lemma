from dataclasses import dataclass

import pytest

from app import llm
from app.llm import AnswerError, estimate_cost


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


def test_estimate_cost_opus_basic():
    # 1M in @ $5 + 1M out @ $25 = $30
    usage = FakeUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_cost("claude-opus-4-8", usage) == 30.0


def test_estimate_cost_counts_cache_reads_and_writes():
    usage = FakeUsage(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=1_000_000,   # 0.1 x $5 = $0.50
        cache_creation_input_tokens=1_000_000,  # 1.25 x $5 = $6.25
    )
    assert estimate_cost("claude-opus-4-8", usage) == 6.75


def test_estimate_cost_unknown_model_returns_none():
    usage = FakeUsage(input_tokens=10, output_tokens=10)
    assert estimate_cost("some-future-model", usage) is None


def test_missing_anthropic_key_raises_controlled_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)

    with pytest.raises(AnswerError, match="ANTHROPIC_API_KEY is missing"):
        llm.get_client()


# --- stream_answer ---------------------------------------------------------


@dataclass
class FakeHit:
    text: str
    doc_name: str
    chunk_index: int


class FakeBlock:
    type = "text"

    def __init__(self, text: str):
        self.text = text
        self.citations = []


class FakeMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.model = "fake-model"
        self.stop_reason = stop_reason
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage(input_tokens=10, output_tokens=5)


class FakeStreamCtx:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn"):
        self._chunks = chunks
        self._stop = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    @property
    def text_stream(self):
        return iter(self._chunks)

    def get_final_message(self):
        return FakeMessage("".join(self._chunks), self._stop)


class FakeClient:
    def __init__(self, chunks: list[str], stop_reason: str = "end_turn"):
        self._ctx = FakeStreamCtx(chunks, stop_reason)
        self.messages = self

    def stream(self, **kwargs):
        return self._ctx


HITS = [FakeHit(text="LEMMA is a RAG assistant.", doc_name="handbook.md", chunk_index=0)]


def test_stream_answer_yields_deltas_then_result(monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeClient(["Hel", "lo"]))

    events = list(llm.stream_answer("what is lemma?", HITS))

    deltas = [payload for kind, payload in events if kind == "delta"]
    assert deltas == ["Hel", "lo"]
    kind, result = events[-1]
    assert kind == "result"
    assert result["answer_text"] == "Hello"
    assert result["truncated"] is False
    assert result["usage"]["input_tokens"] == 10


def test_stream_answer_refusal_raises_after_deltas(monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeClient(["nope"], stop_reason="refusal"))

    with pytest.raises(AnswerError, match="declined"):
        list(llm.stream_answer("q", HITS))


def test_result_flags_uncited_answers_as_ungrounded(monkeypatch):
    monkeypatch.setattr(llm, "_client", FakeClient(["not in the documents"]))
    events = list(llm.stream_answer("q", HITS))
    _, result = events[-1]
    assert result["grounded"] is False


def test_result_flags_cited_answers_as_grounded(monkeypatch):
    class CitedBlock(FakeBlock):
        def __init__(self, text):
            super().__init__(text)
            self.citations = [type("C", (), {"document_title": "handbook.md", "document_index": 0, "cited_text": "LEMMA is"})()]

    msg = FakeMessage("cited answer")
    msg.content = [CitedBlock("cited answer")]
    result = llm._build_result(msg, "fake-model", 12.0)
    assert result["grounded"] is True
