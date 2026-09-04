from unittest.mock import patch

from qdrant_client import QdrantClient

from ingestion.chunker import LegalChunk
from retrieval.vector_store import VietnameseLegalVectorStore


def test_rrf_ranks_items_appearing_in_both_lists_higher():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}, {"chunk_id": "c"}]
    sparse = [{"chunk_id": "b"}, {"chunk_id": "d"}, {"chunk_id": "a"}]
    fused = VietnameseLegalVectorStore._reciprocal_rank_fusion([dense, sparse], top_k=10)
    ids = [f["chunk_id"] for f in fused]
    # "b" xuất hiện hạng cao ở cả 2 list -> phải đứng đầu sau fusion
    assert ids[0] == "b"
    # "d" chỉ xuất hiện ở 1 list -> vẫn phải có mặt (không bị rớt), nhưng hạng thấp hơn "a"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_does_not_double_count_duplicate_ids_within_same_list():
    """Nếu 1 result_list có cùng chunk_id xuất hiện 2 lần (ví dụ do lỗi index
    trùng), RRF không được cộng dồn điểm 2 lần cho cùng 1 item — đó là lỗi
    thiên lệch điểm số, không phải fusion đúng nghĩa."""
    dense = [{"chunk_id": "a"}, {"chunk_id": "a"}, {"chunk_id": "b"}]
    sparse = [{"chunk_id": "b"}]
    fused = VietnameseLegalVectorStore._reciprocal_rank_fusion([dense, sparse], top_k=10)
    scores = {f["chunk_id"]: f["rrf_score"] for f in fused}
    # "a" chỉ được cộng đúng một lần ở dense dù đầu vào bị lặp.
    assert scores["a"] == 1 / 61
    assert scores["b"] > scores["a"]


def test_rrf_respects_top_k():
    dense = [{"chunk_id": str(i)} for i in range(20)]
    fused = VietnameseLegalVectorStore._reciprocal_rank_fusion([dense], top_k=5)
    assert len(fused) == 5


def test_bm25_index_roundtrip_uses_safe_json(tmp_path, monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "qdrant_path", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "index_batch_size", 1)
    client = QdrantClient(":memory:")
    chunks = [
        LegalChunk(f"d::dieu_{i}", "d", "Luật mẫu", None, str(i), "", content, "unknown")
        for i, content in enumerate(("từ_khóa_riêng alpha", "beta gamma", "delta epsilon"), 1)
    ]

    def fake_embed(texts):
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    store = VietnameseLegalVectorStore(client=client)
    with patch("retrieval.vector_store.embed_texts", side_effect=fake_embed):
        store.upsert_chunks(chunks, reset=True)

    assert store._bm25_index_path.suffix == ".gz"
    reloaded = VietnameseLegalVectorStore(client=client)
    reloaded.load_bm25_index()
    assert reloaded._bm25_corpus_ids == [c.chunk_id for c in chunks]
    assert reloaded._sparse_search("từ_khóa_riêng", top_k=3)[0]["chunk_id"] == "d::dieu_1"
