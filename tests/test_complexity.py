"""The routing classifier must default to 'simple' (cheap path) and only flag
genuinely multi-part questions — a classifier that over-flags would route every
query into the expensive loop, defeating the point.
"""

from app.complexity import classify


def test_single_direct_question_is_simple():
    c = classify("What is LEMMA?")
    assert c.label == "simple"
    assert c.reason


def test_comparison_is_multihop():
    assert classify("What is the difference between RRF and BM25?").label == "multi-hop"


def test_conjunction_is_multihop():
    assert classify("How does ingestion work and how is recall measured?").label == "multi-hop"


def test_multiple_questions_is_multihop():
    assert classify("What is chunking? How big are the chunks?").label == "multi-hop"


def test_long_question_is_multihop():
    q = "walk me through exactly how the retrieval pipeline takes my uploaded document " \
        "and turns it into an answer with citations from start to finish please"
    assert classify(q).label == "multi-hop"


def test_empty_is_simple():
    assert classify("   ").label == "simple"


def test_why_question_routes_multihop():
    # "why" questions typically need reasoning across evidence, not a lookup.
    assert classify("Why does hybrid search beat dense-only retrieval?").label == "multi-hop"
