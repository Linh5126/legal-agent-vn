"""Hybrid retrieval: Dense + BM25 + RRF + CrossEncoder reranking.

V2 fixes:
- candidate pool lớn hơn (dense/BM25 30 -> RRF 30 -> rerank 10)
- stable Qdrant point IDs để re-ingest không tạo stale points
- optional effective-status prior
- BM25 index validation
- actual retrieved context được trả ra để evaluation/RAGAS dùng đúng dữ liệu
"""
import gzip
import json
import re
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from config import settings
from retrieval.embeddings import embed_texts, rerank

BM25_SCHEMA_VERSION = 1


class VietnameseLegalVectorStore:
    def __init__(self, collection_name: str | None = None, client: Any | None = None):
        # Qdrant local giữ file lock. Khởi tạo lazy để import module/test hàm
        # thuần túy không tự ý mở DB và để app báo lỗi cấu hình rõ ràng hơn.
        self._client = client
        self.collection_name = collection_name or settings.collection_name
        self._bm25 = None
        self._bm25_corpus_ids: list[str] = []
        self._bm25_payloads: dict[str, dict] = {}
        index_dir = Path(settings.qdrant_path).parent
        self._bm25_index_path = index_dir / f"bm25_index_{self.collection_name}.json.gz"
        self._legacy_bm25_index_path = index_dir / f"bm25_index_{self.collection_name}.pkl"

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            Path(settings.qdrant_path).parent.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=settings.qdrant_path)
        return self._client

    def is_ready(self) -> bool:
        """Index sẵn sàng khi collection dense tồn tại và BM25 đọc được."""
        try:
            dense_ready = self.client.collection_exists(self.collection_name)
        except (OSError, RuntimeError, ValueError):
            return False
        if self._bm25 is None:
            self.load_bm25_index()
        return dense_ready and self._bm25 is not None

    def reset_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        for path in (self._bm25_index_path, self._legacy_bm25_index_path):
            if path.exists():
                path.unlink()
        self._bm25 = None
        self._bm25_corpus_ids = []
        self._bm25_payloads = {}

    def create_collection(self, vector_size: int = 1024):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vn-legal:{chunk_id}"))

    def upsert_chunks(self, chunks: list, reset: bool = False) -> None:
        if not chunks:
            raise ValueError("Không có chunks để index.")
        chunk_ids = [c.chunk_id for c in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("chunk_id bị trùng; dừng index để tránh ghi đè dữ liệu.")
        if reset:
            self.reset_collection()
        elif self.client.collection_exists(self.collection_name):
            raise ValueError(
                "Incremental upsert chưa được hỗ trợ vì sẽ làm BM25 lệch với Qdrant; "
                "hãy gọi với reset=True."
            )

        # Embed/upsert theo batch để corpus hàng chục nghìn Điều không giữ toàn
        # bộ ma trận vector trong RAM cùng lúc.
        batch_size = settings.index_batch_size
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = embed_texts([c.content for c in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding model trả sai số lượng vector.")
            if start == 0:
                self.create_collection(vector_size=len(vectors[0]))
            points = [
                PointStruct(id=self._point_id(c.chunk_id), vector=vec, payload=c.to_payload())
                for c, vec in zip(batch, vectors, strict=True)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        self._build_bm25_index(chunks)

        if reset:
            indexed = self.client.get_collection(self.collection_name).points_count
            if indexed is not None and indexed != len(chunks):
                raise RuntimeError(
                    f"Kiểm tra sau index thất bại: Qdrant={indexed}, chunks={len(chunks)}."
                )

    def _build_bm25_index(self, chunks: list) -> None:
        tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus)
        self._bm25_corpus_ids = [c.chunk_id for c in chunks]
        self._bm25_payloads = {c.chunk_id: c.to_payload() for c in chunks}
        self._bm25_index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": BM25_SCHEMA_VERSION,
            "collection_name": self.collection_name,
            "ids": self._bm25_corpus_ids,
            "payloads": self._bm25_payloads,
            "tokenized_corpus": tokenized_corpus,
        }
        # JSON gzip an toàn hơn pickle (không thực thi mã khi load) và dễ kiểm
        # tra version/index corruption.
        with gzip.open(self._bm25_index_path, "wt", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

    def load_bm25_index(self) -> None:
        if not self._bm25_index_path.exists():
            return
        try:
            with gzip.open(self._bm25_index_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != BM25_SCHEMA_VERSION:
                raise ValueError("BM25 schema version không tương thích")
            ids = data["ids"]
            payloads = data["payloads"]
            corpus = data["tokenized_corpus"]
            if len(ids) != len(corpus) or len(ids) != len(set(ids)):
                raise ValueError("BM25 index có kích thước hoặc ID không hợp lệ")
            if any(cid not in payloads for cid in ids):
                raise ValueError("BM25 index thiếu payload")
            self._bm25 = BM25Okapi(corpus)
            self._bm25_corpus_ids = ids
            self._bm25_payloads = payloads
        except (OSError, EOFError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"BM25 index bị lỗi ({self._bm25_index_path}). Hãy chạy lại ingest."
            ) from exc

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenization ổn định, không phụ thuộc package NLP ngoài.

        Giữ cả unigram và bigram để BM25 nhận diện cụm pháp lý tiếng Việt như
        'hợp đồng lao động', 'đơn phương chấm dứt'.
        """
        words = re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)
        bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        return words + bigrams

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10,
               candidate_k: int = 30, rerank_top_k: int | None = None,
               prefer_effective: bool = True) -> list[dict]:
        """Search with an explicit retrieval stage for ablation studies."""
        query = (query or "").strip()
        if not query:
            raise ValueError("Search query không được để trống.")
        mode = mode.lower()
        if mode == "dense":
            return self._dense_search(query, top_k=top_k)
        if mode == "bm25":
            return self._sparse_search(query, top_k=top_k)
        if mode == "rrf":
            return self._reciprocal_rank_fusion(
                [self._dense_search(query, candidate_k), self._sparse_search(query, candidate_k)],
                top_k=top_k,
            )
        if mode in {"hybrid", "rerank"}:
            return self.hybrid_search(query, top_k=top_k, rerank_top_k=rerank_top_k or top_k,
                                      candidate_k=candidate_k, prefer_effective=prefer_effective,
                                      use_reranker=(mode == "rerank"))
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    def hybrid_search(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
        candidate_k: int | None = None,
        prefer_effective: bool = True,
        use_reranker: bool = True,
    ) -> list[dict]:
        top_k = top_k or settings.top_k_retrieval
        rerank_top_k = rerank_top_k or settings.top_k_rerank
        candidate_k = candidate_k or settings.retrieval_candidate_k

        dense_results = self._dense_search(query, top_k=candidate_k)
        sparse_results = self._sparse_search(query, top_k=candidate_k)
        fused = self._reciprocal_rank_fusion([dense_results, sparse_results], top_k=candidate_k)

        if not fused:
            return []

        if not use_reranker:
            for item in fused:
                item["rerank_score"] = 0.0
                status = item.get("effective_status", "unknown")
                item["status_prior"] = settings.effective_status_boost if prefer_effective and status == "hiệu lực" else 0.0
                item["final_retrieval_score"] = item.get("rrf_score", 0.0) + item["status_prior"]
            fused.sort(key=lambda x: x["final_retrieval_score"], reverse=True)
            return fused[:rerank_top_k]

        texts = [item["content"] for item in fused]
        scores = rerank(query, texts)
        for item, score in zip(fused, scores, strict=True):
            item["rerank_score"] = float(score)
            status = item.get("effective_status", "unknown")
            item["status_prior"] = settings.effective_status_boost if prefer_effective and status == "hiệu lực" else 0.0
            item["final_retrieval_score"] = item["rerank_score"] + item["status_prior"]

        fused.sort(key=lambda x: x["final_retrieval_score"], reverse=True)
        return fused[:rerank_top_k]

    def _dense_search(self, query: str, top_k: int) -> list[dict]:
        if not self.client.collection_exists(self.collection_name):
            return []
        query_vector = embed_texts([query])[0]
        hits = self.client.query_points(collection_name=self.collection_name, query=query_vector, limit=top_k).points
        return [{"chunk_id": h.payload["chunk_id"], **h.payload, "dense_score": float(h.score)} for h in hits]

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        if self._bm25 is None:
            self.load_bm25_index()
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            (
                (cid, score)
                for cid, score in zip(self._bm25_corpus_ids, scores, strict=True)
                if score > 0
            ),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [{"chunk_id": cid, **self._bm25_payloads[cid], "bm25_score": float(score)} for cid, score in ranked]

    @staticmethod
    def _reciprocal_rank_fusion(result_lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
        fused_scores: dict[str, float] = {}
        payload_by_id: dict[str, dict] = {}
        for result_list in result_lists:
            seen_in_list: set[str] = set()
            for rank_i, item in enumerate(result_list):
                cid = item["chunk_id"]
                if cid in seen_in_list:
                    continue
                seen_in_list.add(cid)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (k + rank_i + 1)
                payload_by_id[cid] = {**payload_by_id.get(cid, {}), **item}
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)[:top_k]
        return [payload_by_id[cid] | {"rrf_score": fused_scores[cid]} for cid in ranked_ids]
