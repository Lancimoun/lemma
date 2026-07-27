"""LEMMA API — FastAPI app wiring ingest, hybrid retrieval, Claude answers, and evals.

Endpoints use sync handlers on purpose: retrieval and the Claude call are blocking,
so FastAPI runs them in its threadpool instead of blocking the event loop.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import json

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import __version__, complexity, config, evals, ingest, iterative, llm
from .store import HybridStore

STATIC_DIR = config.BASE_DIR / "static"
SAMPLE_DOC = config.BASE_DIR / "sample_docs" / "lemma-handbook.md"

limiter = Limiter(key_func=get_remote_address)
metrics = evals.OpsMetrics()
store: HybridStore | None = None


def get_store() -> HybridStore:
    if store is None:  # pragma: no cover — lifespan always sets it
        raise HTTPException(503, "Store not ready yet")
    return store


def _seed_sample_doc(s: HybridStore) -> None:
    existing = {d["doc_name"] for d in s.list_docs()}
    if evals.CANARY_DOC in existing or not SAMPLE_DOC.exists():
        return
    text = ingest.normalize(SAMPLE_DOC.read_text(encoding="utf-8"))
    s.add_chunks(ingest.chunk_text(evals.CANARY_DOC, text))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    store = HybridStore(path=str(config.QDRANT_PATH))
    _seed_sample_doc(store)
    yield
    # Release Qdrant's file lock cleanly (matters on Windows).
    store.client.close()


app = FastAPI(title="LEMMA", version=__version__, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/lemma-social-card.png", include_in_schema=False)
def social_card() -> FileResponse:
    return FileResponse(STATIC_DIR / "lemma-social-card.png", media_type="image/png")


@app.get("/health")
def health() -> dict:
    docs = [d for d in get_store().list_docs() if not d["doc_name"].startswith(evals.PROBE_PREFIX)]
    return {
        "status": "ok",
        "version": __version__,
        "model": config.MODEL,
        "public_badge": config.PUBLIC_BADGE_LABEL,
        "documents": len(docs),
    }


@app.get("/documents")
def documents() -> dict:
    docs = [d for d in get_store().list_docs() if not d["doc_name"].startswith(evals.PROBE_PREFIX)]
    return {"documents": docs}


@app.delete("/documents/{doc_name}")
@limiter.limit(config.RATE_LIMIT_INGEST)
def delete_document(request: Request, doc_name: str) -> dict:
    get_store().delete_doc(doc_name)
    return {"deleted": doc_name}


@app.post("/ingest")
@limiter.limit(config.RATE_LIMIT_INGEST)
def ingest_file(request: Request, file: UploadFile = File(...)) -> dict:
    s = get_store()
    limit_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    data = file.file.read(limit_bytes + 1)
    if len(data) > limit_bytes:
        raise HTTPException(413, f"File exceeds the {config.MAX_UPLOAD_MB} MB limit.")

    doc_names = {d["doc_name"] for d in s.list_docs() if not d["doc_name"].startswith(evals.PROBE_PREFIX)}
    if file.filename not in doc_names and len(doc_names) >= config.MAX_DOCS:
        raise HTTPException(429, f"Document limit reached ({config.MAX_DOCS}). Delete one first.")

    try:
        text = ingest.extract_text(file.filename, data)
    except ingest.IngestError as exc:
        raise HTTPException(400, str(exc)) from exc

    chunks = ingest.chunk_text(file.filename, text)
    s.delete_doc(file.filename)  # re-upload replaces the previous version
    added = s.add_chunks(chunks)
    return {"doc_name": file.filename, "chunks": added}


def _retrieve(question: str):
    """Classify the question, then retrieve for it.

    `multi-hop` questions go through bounded iterative retrieval (search → re-plan
    → search, capped at 3 hops); everything else takes the one-shot pipeline
    unchanged. Returns (hits, route, trace) where trace is the iterative step
    record for the transparency panel, or None for the one-shot path.

    Multi-hop is safe to route into because iterative.retrieve always performs at
    least the original-question search and degrades to exactly the one-shot
    result when the planner returns None (see app.iterative). The classification
    itself is deterministic and costs no model call (see app.complexity).
    """
    if not question:
        raise HTTPException(400, "Question is empty.")
    s = get_store()
    route = complexity.classify(question)

    def _search(q: str):
        return s.search(q, top_k=config.TOP_K, prefetch_k=config.PREFETCH_K)

    trace = None
    if route.label == "multi-hop" and config.MULTIHOP_ENABLED:
        result = iterative.retrieve(question, _search, llm.plan_follow_up)
        hits = result.hits
        trace = iterative.trace_payload(result)
    else:
        hits = list(_search(question))

    if not hits:
        raise HTTPException(404, "No documents indexed yet — upload something first.")
    return hits, route, trace


def _source_list(hits) -> list[dict]:
    return [
        {"doc_name": h.doc_name, "chunk_index": h.chunk_index, "score": round(h.score, 4)}
        for h in hits
    ]


@app.post("/ask/stream")
@limiter.limit(config.RATE_LIMIT_ASK)
def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """SSE variant of /ask: `delta` events per token chunk, one final `result` event.

    Search failures surface as normal HTTP errors before the stream opens;
    failures mid-stream arrive as an `error` event (headers are already sent).
    """
    question = body.question.strip()
    hits, route, trace = _retrieve(question)
    sources = _source_list(hits)

    def events():
        try:
            for kind, payload in llm.stream_answer(question, hits):
                if kind == "delta":
                    yield f"event: delta\ndata: {json.dumps({'text': payload})}\n\n"
                else:
                    payload["sources"] = sources
                    payload["route"] = {"label": route.label, "reason": route.reason}
                    if trace:
                        payload["retrieval"] = trace
                    metrics.record(
                        latency_ms=payload["latency_ms"],
                        cost_usd=payload["cost_usd"],
                        cache_read_tokens=payload["usage"]["cache_read_input_tokens"],
                        input_tokens=payload["usage"]["input_tokens"],
                    )
                    yield f"event: result\ndata: {json.dumps(payload)}\n\n"
        except llm.AnswerError as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.post("/ask")
@limiter.limit(config.RATE_LIMIT_ASK)
def ask(request: Request, body: AskRequest) -> dict:
    question = body.question.strip()
    hits, route, trace = _retrieve(question)

    try:
        result = llm.answer_question(question, hits)
    except llm.AnswerError as exc:
        raise HTTPException(502, str(exc)) from exc

    metrics.record(
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
        cache_read_tokens=result["usage"]["cache_read_input_tokens"],
        input_tokens=result["usage"]["input_tokens"],
    )
    result["sources"] = _source_list(hits)
    result["route"] = {"label": route.label, "reason": route.reason}
    if trace:
        result["retrieval"] = trace
    return result


@app.get("/eval")
@limiter.limit(config.RATE_LIMIT_EVAL)
def run_eval(request: Request, llm_check: bool = False) -> dict:
    """Run the reliability suite. `llm_check=true` also verifies the LLM answer (costs one query)."""
    s = get_store()
    ask_fn = llm.answer_question if llm_check else None
    try:
        probe = evals.run_recall_probe(s, ask_fn=ask_fn)
    except llm.AnswerError as exc:
        raise HTTPException(502, str(exc)) from exc
    canaries = evals.run_canaries(s, top_k=config.TOP_K)
    return {"recall_probe": probe, "canaries": canaries, "ops": metrics.snapshot()}
