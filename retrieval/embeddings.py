"""
Wrapper cho embedding model (BGE-M3) và reranker (BGE-reranker-v2-m3).

BGE-M3 được chọn vì hỗ trợ đa ngôn ngữ (bao gồm tiếng Việt), context dài (8192
token - đủ cho các Điều luật dài), và cho ra đồng thời dense vector + sparse
weights trong 1 lần forward — tận dụng được cho hybrid retrieval mà không cần
chạy 2 model riêng biệt.

Lưu ý: lần đầu chạy sẽ tải model (~2-4.5GB) từ HuggingFace Hub, cần kết nối mạng.

QUAN TRỌNG: embedding dùng thư viện `FlagEmbedding`, nhưng reranker dùng
`sentence-transformers` (CrossEncoder) thay vì `FlagEmbedding.FlagReranker`.
Lý do: bản `FlagReranker` gọi `tokenizer.prepare_for_model()` — một API đã bị
gỡ khỏi các bản `transformers` mới (>= khoảng cuối 2025), gây lỗi
`AttributeError: XLMRobertaTokenizer has no attribute 'prepare_for_model'`
trên môi trường có `transformers` mới như Colab. `sentence-transformers`
được bảo trì tốt hơn và không dính lỗi tương thích này, dùng đúng cùng 1
model BGE-reranker-v2-m3.
"""
from functools import lru_cache

import numpy as np

from config import settings


@lru_cache(maxsize=1)
def get_embedding_model():
    """Lazy-load BGE-M3 model, chỉ load 1 lần nhờ cache."""
    import torch
    from FlagEmbedding import BGEM3FlagModel

    # FP16 trên CPU vừa không có lợi vừa có thể gây lỗi ở một số toán tử.
    return BGEM3FlagModel(settings.embedding_model, use_fp16=torch.cuda.is_available())


@lru_cache(maxsize=1)
def get_reranker_model():
    import torch
    from sentence_transformers import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[embeddings] Load reranker trên device: {device}", flush=True)
    return CrossEncoder(settings.reranker_model, max_length=1024, device=device)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Trả về dense embedding cho danh sách văn bản (dùng khi index và khi query)."""
    model = get_embedding_model()
    if not texts:
        return []
    output = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        max_length=1024,
    )
    return output["dense_vecs"].tolist()


def rerank(query: str, candidates: list[str]) -> list[float]:
    """Trả về điểm relevance (càng cao càng liên quan) cho từng candidate ứng với query."""
    if not candidates:
        return []
    reranker = get_reranker_model()
    pairs = [[query, c] for c in candidates]
    raw_scores = reranker.predict(pairs)

    # CrossEncoder trả về raw logit, không tự chuẩn hóa [0,1] như FlagReranker
    # (normalize=True trước đây) — áp sigmoid thủ công để giữ nguyên hành vi
    # điểm số cũ (dùng để so sánh/sort trong hybrid_search).
    scores = 1 / (1 + np.exp(-np.array(raw_scores)))
    return scores.tolist()
