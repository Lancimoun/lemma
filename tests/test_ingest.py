import random

import pytest

from app.ingest import IngestError, chunk_text, extract_text, normalize


def test_extract_text_txt_and_md():
    assert extract_text("notes.txt", b"hello world") == "hello world"
    assert extract_text("notes.md", "# Title\n\nBody".encode()) == "# Title\n\nBody"


def test_extract_text_rejects_unknown_extension():
    with pytest.raises(IngestError):
        extract_text("malware.exe", b"nope")


def test_extract_text_rejects_empty():
    with pytest.raises(IngestError):
        extract_text("empty.txt", b"   \n  ")


def test_normalize_collapses_blank_runs_and_crlf():
    assert normalize("a\r\n\r\n\r\n\r\nb") == "a\n\nb"


def test_small_text_is_single_chunk():
    chunks = chunk_text("doc.txt", "just a short paragraph", size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_name == "doc.txt"


def test_chunks_respect_size_and_have_sequential_indexes():
    text = "\n\n".join(
        f"Paragraph {i}: " + " ".join(f"word{i}x{j}" for j in range(40)) for i in range(30)
    )
    chunks = chunk_text("doc.txt", text, size=1000, overlap=200)
    assert len(chunks) > 3
    assert all(len(c.text) <= 1000 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_no_content_is_lost_to_chunking():
    words = [f"token{i}" for i in range(1200)]
    text = " ".join(words)
    chunks = chunk_text("doc.txt", text, size=800, overlap=120)
    combined = " ".join(c.text for c in chunks)
    sampled = random.Random(42).sample(words, 40)
    for w in sampled:
        assert w in combined


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("doc.txt", "text", size=100, overlap=100)
