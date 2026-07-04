from dataclasses import dataclass

from app.llm import estimate_cost


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
