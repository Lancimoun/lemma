"""Text extraction and chunking.

Chunks are character-based (~4 chars per token) and prefer to break on
paragraph, then sentence, then word boundaries so retrieval units stay coherent.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config


class IngestError(ValueError):
    """Raised when a file can't be turned into indexable text."""


@dataclass
class Chunk:
    doc_name: str
    chunk_index: int
    text: str


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise IngestError(f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # pypdf raises many exception types
            raise IngestError(f"Could not read PDF: {exc}") from exc
    else:
        text = data.decode("utf-8", errors="replace")

    text = normalize(text)
    if not text:
        raise IngestError("No extractable text found in the file.")
    return text


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    doc_name: str,
    text: str,
    size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = size or config.CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    if size <= overlap:
        raise ValueError("chunk size must be greater than overlap")

    chunks: list[Chunk] = []
    n = len(text)
    start = 0
    index = 0
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Prefer a natural boundary near the end of the window.
            boundary = _last_boundary(text[start:end])
            if boundary > overlap:  # avoid degenerate tiny chunks
                end = start + boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(doc_name=doc_name, chunk_index=index, text=piece))
            index += 1
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _last_boundary(window: str) -> int:
    """Position just after the last paragraph/sentence/word break in the window."""
    for pattern in ("\n\n", ". ", "\n", " "):
        pos = window.rfind(pattern)
        if pos != -1:
            return pos + len(pattern)
    return len(window)
