from app.ingest import Chunk


def _seed(store):
    store.add_chunks(
        [
            Chunk("vectors.md", 0, "qdrant vector database stores embeddings for hybrid search"),
            Chunk("fruit.md", 0, "bananas and mango smoothies are delicious tropical drinks"),
            Chunk("deploy.md", 0, "railway deploys python applications with persistent volumes"),
        ]
    )


def test_add_and_search_top_hit(store):
    _seed(store)
    hits = store.search("mango smoothies tropical", top_k=3)
    assert hits, "expected results"
    assert hits[0].doc_name == "fruit.md"


def test_search_returns_payload_fields(store):
    _seed(store)
    hits = store.search("qdrant hybrid embeddings", top_k=2)
    top = hits[0]
    assert top.doc_name == "vectors.md"
    assert top.chunk_index == 0
    assert "qdrant" in top.text
    assert isinstance(top.score, float)


def test_list_docs_counts_chunks(store):
    _seed(store)
    store.add_chunks([Chunk("vectors.md", 1, "another chunk about vectors and search")])
    docs = {d["doc_name"]: d["chunks"] for d in store.list_docs()}
    assert docs["vectors.md"] == 2
    assert docs["fruit.md"] == 1
    assert len(docs) == 3


def test_delete_doc_removes_all_chunks(store):
    _seed(store)
    store.delete_doc("fruit.md")
    docs = {d["doc_name"] for d in store.list_docs()}
    assert "fruit.md" not in docs
    assert len(docs) == 2


def test_empty_add_is_noop(store):
    assert store.add_chunks([]) == 0
