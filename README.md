# LEMMA — a research assistant that *proves* its answers

Upload documents. Ask questions. Every claim in the answer carries a **citation to the
exact passage it came from** — and a built-in **live reliability panel** measures the
system's own recall, latency, cost, and cache efficiency while you use it.

> In mathematics, a *lemma* is a small **proven** result used to build larger proofs.
> LEMMA holds answers to the same standard: no claim without evidence.

## Live demo

- App: https://lemma-production-b84f.up.railway.app
- Health check: https://lemma-production-b84f.up.railway.app/health
- Reliability check: https://lemma-production-b84f.up.railway.app/eval

The public deployment runs on Railway with a persistent `/data` volume for the
embedded Qdrant index. Retrieval and evals work without an LLM key; Claude-backed
answers require `ANTHROPIC_API_KEY` to be set in Railway variables.

## Why this project exists

Most RAG demos stop at "it answered." Production AI systems have to answer a harder
question: **"is it still working?"** LEMMA treats reliability as a feature:

- **Live recall probe** — LEMMA writes a unique synthetic fact into its own live index,
  retrieves it back through the *real* pipeline, verifies it (optionally end-to-end
  through the LLM), and cleans up. Broken retrieval is caught by the system itself,
  not by a confused user.
- **Canary set** — fixed questions with known-good sources; reports retrieval hit-rate
  so chunking/search regressions are visible immediately.
- **Ops metrics** — rolling p50/p95 latency, cost per query, prompt-cache hit-rate.

## Architecture

```
                ┌──────────────────────────── FastAPI ────────────────────────────┐
 upload ──────▶ │ /ingest  extract (txt/md/pdf) → chunk (~500 tok, overlap)        │
                │          → FastEmbed: dense (bge-small) + sparse (BM25)          │
                │          → Qdrant (embedded local mode)                          │
                │                                                                  │
 question ────▶ │ /ask     hybrid search: dense + sparse prefetch → RRF fusion     │
                │          → top-k chunks as `document` blocks (citations enabled) │
                │          → Claude → answer + exact cited spans + usage/cost      │
                │                                                                  │
 reliability ─▶ │ /eval    live recall probe · canary hit-rate · p50/p95 · cost    │
                └──────────────────────────────────────────────────────────────────┘
```

**Design choices (and why):**

| Choice | Reason |
|---|---|
| Qdrant **embedded local mode** | Zero infra to run the demo; identical client API to Qdrant Cloud, so scaling is a one-line change |
| **FastEmbed** local ONNX embeddings | $0 embedding cost, CPU-only; dense (`BAAI/bge-small-en-v1.5`) + sparse (`Qdrant/bm25`, server-side IDF) |
| **RRF fusion** via Qdrant Query API | Semantic recall *and* exact keyword precision (names, codes, IDs) |
| Claude **native citations** | The API returns exact `cited_text` spans per claim — real provenance, not prompt-engineered `[1]` markers |
| **Prompt caching** + usage capture | Cost engineering is visible in the UI per query |
| Injectable embedders | Unit tests run in milliseconds with no model downloads and no API key |
| Rate limits + upload caps | A public demo has to assume abuse |

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # (Linux/macOS: .venv/bin/pip)
copy .env.example .env                               # add your ANTHROPIC_API_KEY
.venv/Scripts/python -m uvicorn app.main:app --port 8000
```

Open http://localhost:8000 — the **LEMMA handbook is pre-indexed**, so ask it about
itself before uploading anything (try: *"How does LEMMA measure reliability?"*).

> First startup downloads ~100 MB of embedding model weights (one-time; the Dockerfile
> pre-bakes them into the image for fast cold starts).

### Run the tests

```bash
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest -q
```

Tests inject deterministic fake embedders — no network, no API key, no model downloads.

## Configuration

Everything is an environment variable (see `.env.example`):

| Var | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | required |
| `MODEL` | `claude-opus-4-8` | set `claude-haiku-4-5` for a low-cost public demo |
| `MAX_TOKENS` | `2048` | answer cap |
| `TOP_K` | `8` | chunks passed to the model |
| `MAX_UPLOAD_MB` / `MAX_DOCS` | `5` / `20` | abuse limits |
| `RATE_LIMIT_ASK` | `10/minute` | per-IP |

## Deploy (Railway)

The included `Dockerfile` pre-downloads embedding weights at build time. Attach a
persistent volume at `/data` (set `LEMMA_DATA_DIR=/data`) so the index survives
restarts, set `ANTHROPIC_API_KEY`, and deploy.

## Honest limits (MVP)

- Single-process by design (embedded Qdrant local mode) — right-sized for a demo;
  the store API is identical to Qdrant Cloud when scale is needed.
- PDF extraction is text-layer only (no OCR).
- Answers are non-streaming to the browser (server-side streaming is used for
  timeout protection); SSE streaming is on the roadmap.

## Roadmap

Agentic multi-hop retrieval (search-as-a-tool) · browser SSE streaming ·
auto-generated gold Q/A eval sets · multi-provider fallback · Langfuse tracing.

---

Built by **Lance Jilliard Galicia** — part of the FORGE family:
**Maxima** (24/7 personal AI), **Axiom** (multi-provider LLM gateway),
**Agent Reliability Arena** (agent eval harness), and **LEMMA**.
