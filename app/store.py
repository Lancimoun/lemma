"""Hybrid vector store.

Qdrant (embedded local mode — zero infra) with two vector spaces per chunk:
  - dense:  BAAI/bge-small-en-v1.5 via FastEmbed (384-dim, ONNX on CPU, free)
  - sparse: BM25 term vectors via FastEmbed, with server-side IDF weighting

Queries prefetch both branches and merge with Reciprocal Rank Fusion (RRF),
so results get semantic understanding AND exact keyword precision.

Embedders are injectable so unit tests run without downloading model weights.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Callable, Sequence

from qdrant_client import QdrantClient, models

from .ingest import Chunk

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"
DENSE_DIM = 384
COLLECTION = "lemma_chunks"

DenseFn = Callable[[list[str]], list[list[float]]]
SparseFn = Callable[[list[str]], list[tuple[list[int], list[float]]]]


@dataclass
class Embedders:
    dense_docs: DenseFn
    dense_query: DenseFn
    sparse_docs: SparseFn
    sparse_query: SparseFn


@dataclass
class Hit:
    doc_name: str
    chunk_index: int
    text: str
    score: float


def default_embedders() -> Embedders:
    """Real FastEmbed models. Imported lazily — first use downloads ~100MB of weights."""
    from fastembed import SparseTextEmbedding, TextEmbedding

    dense = TextEmbedding(DENSE_MODEL)
    sparse = SparseTextEmbedding(SPARSE_MODEL)

    def dense_docs(texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in dense.embed(texts)]

    def dense_query(texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in dense.query_embed(texts)]

    def sparse_docs(texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in sparse.embed(texts)]

    def sparse_query(texts: list[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist()) for e in sparse.query_embed(texts)]

    return Embedders(dense_docs, dense_query, sparse_docs, sparse_query)


class HybridStore:
    def __init__(
        self,
        path: str | None = None,
        client: QdrantClient | None = None,
        embedders: Embedders | None = None,
    ):
        if client is not None:
            self.client = client
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self.embedders = embedders or default_embedders()
        # Local-mode Qdrant is single-process; serialize writes defensively.
        self._write_lock = threading.Lock()
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
            )

    # --- writes ---

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        dense = self.embedders.dense_docs(texts)
        sparse = self.embedders.sparse_docs(texts)
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dv,
                    "sparse": models.SparseVector(indices=si, values=sv),
                },
                payload={
                    "doc_name": c.doc_name,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                },
            )
            for c, dv, (si, sv) in zip(chunks, dense, sparse)
        ]
        with self._write_lock:
            self.client.upsert(collection_name=COLLECTION, points=points)
        return len(points)

    def delete_doc(self, doc_name: str) -> None:
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="doc_name", match=models.MatchValue(value=doc_name))]
            )
        )
        with self._write_lock:
            self.client.delete(collection_name=COLLECTION, points_selector=selector)

    # --- reads ---

    def search(self, query: str, top_k: int = 8, prefetch_k: int = 20) -> list[Hit]:
        dense_vec = self.embedders.dense_query([query])[0]
        s_idx, s_val = self.embedders.sparse_query([query])[0]
        response = self.client.query_points(
            collection_name=COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=prefetch_k),
                models.Prefetch(
                    query=models.SparseVector(indices=s_idx, values=s_val),
                    using="sparse",
                    limit=prefetch_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            Hit(
                doc_name=p.payload["doc_name"],
                chunk_index=p.payload["chunk_index"],
                text=p.payload["text"],
                score=p.score,
            )
            for p in response.points
        ]

    def list_docs(self) -> list[dict]:
        counts: dict[str, int] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=COLLECTION,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                name = p.payload["doc_name"]
                counts[name] = counts.get(name, 0) + 1
            if offset is None:
                break
        return [{"doc_name": name, "chunks": n} for name, n in sorted(counts.items())]
