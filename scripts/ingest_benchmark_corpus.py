"""
Nạp corpus benchmark (từ YuITC retrieval dataset) vào MỘT collection Qdrant
RIÊNG so với collection chính (vn_legal_docs) — để đo recall@k thuần túy của
retriever bằng nhãn thật, không lẫn với corpus UTS_VLC dùng cho agent demo.

Chạy sau: python scripts/download_hf_datasets.py --dataset retrieval_benchmark
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import LegalChunk
from retrieval.vector_store import VietnameseLegalVectorStore

BENCHMARK_COLLECTION = "vn_legal_retrieval_benchmark"


def load_benchmark_chunks(path: str = "data/raw_benchmark/benchmark_corpus.jsonl") -> list[LegalChunk]:
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(LegalChunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                doc_title=row["doc_title"],
                chapter=None,
                article_number=row["article_number"],
                article_title="",
                content=row["content"],
                effective_status=row["effective_status"],
            ))
    return chunks


def main():
    chunk_path = Path("data/raw_benchmark/benchmark_corpus.jsonl")
    if not chunk_path.exists():
        print(f"Không tìm thấy {chunk_path}. Chạy trước: "
              f"python scripts/download_hf_datasets.py --dataset retrieval_benchmark")
        return

    chunks = load_benchmark_chunks(str(chunk_path))
    print(f"Đã load {len(chunks)} chunk benchmark, đang embed và nạp vào collection "
          f"'{BENCHMARK_COLLECTION}'...")

    store = VietnameseLegalVectorStore(collection_name=BENCHMARK_COLLECTION)
    store.upsert_chunks(chunks, reset=True)

    print(
        "Hoàn tất. Chạy tiếp: python -m evaluation.evaluate_retrieval "
        "--testset data/testset/yuitc_retrieval_testset.json "
        "--collection vn_legal_retrieval_benchmark"
    )


if __name__ == "__main__":
    main()
