# The LEMMA Handbook

## What is LEMMA?

LEMMA is a research assistant that exposes the evidence behind grounded answers. You
upload documents, ask questions, and cited factual claims point at the exact passage
they came from. If a response has no citation, LEMMA marks it unverified instead of
silently claiming grounding. In mathematics, a lemma is a small proven result used as
a stepping stone toward a larger proof — LEMMA applies that standard as a goal: no
claim without evidence, and no hidden failure to meet it.

LEMMA ships with this handbook pre-indexed, so you can ask it questions about itself
before uploading anything.

## How retrieval works

Every document is split into overlapping chunks of roughly five hundred tokens. Each
chunk is indexed twice, in two different vector spaces:

- **Dense vectors** capture semantic meaning. LEMMA generates them locally with the
  bge-small embedding model (BAAI/bge-small-en-v1.5), a 384-dimension model that runs
  on the CPU through FastEmbed's ONNX runtime — no embedding API, no GPU, no cost.
- **Sparse vectors** capture exact keywords. LEMMA produces them with the BM25
  weighting algorithm, so rare and specific terms (names, codes, identifiers) are
  matched precisely even when semantic similarity would miss them.

At query time both branches are searched in parallel, and the two ranked lists are
merged with reciprocal rank fusion (RRF). Reciprocal rank fusion boosts results that
rank highly in both lists, which combines the semantic recall of dense search with
the keyword precision of sparse search. The vector store is Qdrant running in
embedded local mode, so the whole index lives on disk with zero external services.

## How answers are generated

The top fused chunks are handed to Claude as document content blocks with native
citations enabled. The model is instructed to ground every factual claim in those
documents, and the API returns the exact cited span for each citation — the interface
renders these as citation chips you can inspect. If a response contains no citations,
LEMMA marks it unverified; if the documents do not contain the answer, the model is
instructed to say so instead of guessing.

## How reliability is measured

LEMMA treats reliability as something you measure, not something you assume. The
built-in evaluation panel runs three checks:

1. **Live recall probe.** LEMMA writes a unique synthetic fact into its own live
   index, retrieves it back through the real search pipeline, verifies the fact is
   found (and optionally that the model's answer contains it), and then deletes the
   probe. A failing probe means memory or retrieval is broken — and you find out
   immediately, not from a confused user.
2. **Canary questions.** A fixed set of questions about this handbook runs against
   the index and reports a retrieval hit rate, catching regressions in chunking or
   search configuration.
3. **Operational metrics.** Rolling latency percentiles (p50/p95), average cost per
   query, and the cache hit rate are tracked for every question asked.

## How costs stay low

The retrieval layer is free by design: local embeddings and an embedded vector
database mean the only metered cost is the Claude API call itself. That call is kept
cheap with prompt caching on stable prefixes, a hard cap on response tokens, and
rate limits on every endpoint. The Claude model is configurable through a single
environment variable, so a public demo can run on a fast, low-cost model while the
same code serves a higher-quality model in private use.
