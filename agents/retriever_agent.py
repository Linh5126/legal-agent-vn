"""Retrieve theo nhiều sub-query, merge evidence và giữ trace điểm số."""
from functools import lru_cache

from agents.observability import observe
from agents.state import AgentState
from config import settings
from retrieval.vector_store import VietnameseLegalVectorStore


@lru_cache(maxsize=1)
def get_store() -> VietnameseLegalVectorStore:
    return VietnameseLegalVectorStore()


@observe(name="retriever_node", as_type="retriever")
def retriever_node(state: AgentState) -> dict:
    store = get_store()
    if not store.is_ready():
        raise RuntimeError(
            "Chưa có search index hợp lệ. Hãy chạy `python -m scripts.ingest_data` trước."
        )

    # Giữ evidence tốt từ vòng trước; nếu retry chỉ overwrite khi score mới cao hơn.
    all_docs = {doc["chunk_id"]: doc for doc in state.get("retrieved_docs", [])}
    for sub_q in state["sub_questions"]:
        results = store.hybrid_search(
            sub_q,
            top_k=settings.retrieval_candidate_k,
            rerank_top_k=settings.top_k_rerank,
            candidate_k=settings.retrieval_candidate_k,
            prefer_effective=True,
        )
        # Dữ liệu minh họa chỉ dùng smoke test ingestion, không bao giờ làm
        # căn cứ trả lời pháp lý cho người dùng.
        results = [doc for doc in results if not doc.get("is_sample", False)]
        for doc in results:
            existing = all_docs.get(doc["chunk_id"])
            if not existing or doc.get("final_retrieval_score", 0) > existing.get("final_retrieval_score", 0):
                all_docs[doc["chunk_id"]] = {**doc, "matched_sub_query": sub_q}

    docs = sorted(all_docs.values(), key=lambda d: d.get("final_retrieval_score", 0), reverse=True)
    return {"retrieved_docs": docs[:settings.final_context_k]}
