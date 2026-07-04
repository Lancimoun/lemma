from pathlib import Path

from app import evals
from app.ingest import chunk_text, normalize
from app.evals import CANARIES, OpsMetrics, run_canaries, run_recall_probe

HANDBOOK = Path(__file__).resolve().parent.parent / "sample_docs" / "lemma-handbook.md"


def test_canary_substrings_exist_in_handbook():
    """Guard: every canary expectation must actually appear in the shipped sample doc."""
    text = HANDBOOK.read_text(encoding="utf-8").lower()
    for canary in CANARIES:
        assert canary["expect_substring"].lower() in text, canary["expect_substring"]


def test_recall_probe_passes_and_cleans_up(store):
    result = run_recall_probe(store)
    assert result["retrieval_pass"] is True
    assert result["found_in_top_k"] is True
    assert result["pass"] is True
    # probe doc must be deleted afterwards
    names = {d["doc_name"] for d in store.list_docs()}
    assert not any(n.startswith(evals.PROBE_PREFIX) for n in names)


def test_recall_probe_with_llm_answer_check(store):
    captured = {}

    def fake_ask(question, hits):
        captured["question"] = question
        # echo back the probe fact text so the budget number is "answered"
        return {"answer_text": hits[0].text, "cost_usd": 0.001}

    result = run_recall_probe(store, ask_fn=fake_ask)
    assert result["answer_pass"] is True
    assert result["pass"] is True
    assert "Zephyr-" in captured["question"]


def test_recall_probe_fails_when_answer_wrong(store):
    def bad_ask(question, hits):
        return {"answer_text": "I have no idea.", "cost_usd": 0.001}

    result = run_recall_probe(store, ask_fn=bad_ask)
    assert result["answer_pass"] is False
    assert result["pass"] is False


def test_canaries_pass_on_seeded_handbook(store):
    text = normalize(HANDBOOK.read_text(encoding="utf-8"))
    store.add_chunks(chunk_text(evals.CANARY_DOC, text))
    report = run_canaries(store, top_k=8)
    assert report["total"] == len(CANARIES)
    assert report["passed"] == report["total"], report


def test_ops_metrics_percentiles_and_cache_rate():
    m = OpsMetrics()
    assert m.snapshot() == {"queries": 0}
    for latency in [100, 200, 300, 400, 1000]:
        m.record(latency_ms=latency, cost_usd=0.01, cache_read_tokens=50, input_tokens=50)
    snap = m.snapshot()
    assert snap["queries"] == 5
    assert snap["latency_p50_ms"] == 300
    assert snap["latency_p95_ms"] == 1000
    assert snap["avg_cost_usd"] == 0.01
    assert snap["cache_hit_rate"] == 0.5
