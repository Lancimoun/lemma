"""Central configuration — everything tunable lives in environment variables."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("LEMMA_DATA_DIR", str(BASE_DIR / "data")))
QDRANT_PATH = DATA_DIR / "qdrant"

# --- LLM ---
MODEL = os.getenv("MODEL", "claude-opus-4-8")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Planner for bounded iterative retrieval. Deliberately NOT MODEL: deciding
# "do I need another search?" is a routing call, and routing must be near-free
# or the loop costs more than the answer it improves. Same principle as
# complexity.classify, which spends nothing at all. Haiku is 5x cheaper on
# input than Opus and the task is a short judgement over text already retrieved.
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "claude-haiku-4-5")
PLANNER_MAX_TOKENS = int(os.getenv("PLANNER_MAX_TOKENS", "128"))
PUBLIC_BADGE_LABEL = os.getenv("PUBLIC_BADGE_LABEL", "Hybrid RAG + Citations + Live Evals")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "8"))          # chunks sent to Claude
PREFETCH_K = int(os.getenv("PREFETCH_K", "20"))  # per-branch candidates before RRF fusion

# --- Chunking (~500 tokens per chunk at ~4 chars/token) ---
CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# --- Upload limits (abuse protection for the public demo) ---
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
MAX_DOCS = int(os.getenv("MAX_DOCS", "20"))
ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}

# --- Rate limits (per client IP) ---
RATE_LIMIT_ASK = os.getenv("RATE_LIMIT_ASK", "10/minute")
RATE_LIMIT_INGEST = os.getenv("RATE_LIMIT_INGEST", "5/minute")
RATE_LIMIT_EVAL = os.getenv("RATE_LIMIT_EVAL", "4/minute")

# USD per 1M tokens (input, output) — display-only cost estimates.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),   # intro pricing 2.00/10.00 through 2026-08-31
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
