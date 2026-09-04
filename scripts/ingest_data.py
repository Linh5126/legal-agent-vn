"""Validate -> chunk -> embed -> index. Re-ingest mặc định RESET collection."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker import split_into_articles, split_long_article, validate_chunk_id_uniqueness
from ingestion.loader import load_raw_documents
from retrieval.vector_store import VietnameseLegalVectorStore


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus pháp luật vào Qdrant/BM25")
    parser.add_argument(
        "--include-samples",
        action="store_true",
        help="cho phép index dữ liệu minh họa; tuyệt đối không dùng để tư vấn thật",
    )
    args = parser.parse_args()

    t0 = time.time()
    raw_docs = load_raw_documents(validate=True)
    if not args.include_samples:
        raw_docs = [doc for doc in raw_docs if not doc.is_sample]
    if not raw_docs:
        raise RuntimeError(
            "Không có văn bản pháp luật thật để ingest. Hãy thêm corpus có metadata/source_url; "
            "chỉ dùng --include-samples để smoke test."
        )

    all_chunks = []
    for doc in raw_docs:
        articles = split_into_articles(
            doc.raw_text,
            doc.doc_id,
            doc.doc_title,
            doc.effective_status,
            metadata={
                "doc_type": doc.doc_type,
                "issue_date": doc.issue_date,
                "source_url": doc.source_url,
                "is_sample": doc.is_sample,
                "license": doc.license,
            },
        )
        for article in articles:
            all_chunks.extend(split_long_article(article))

    print(f"Validated {len(raw_docs)} documents -> {len(all_chunks)} chunks")
    if len(all_chunks) < len(raw_docs):
        raise RuntimeError("Số chunk bất thường: ingest bị hủy.")

    # Lớp bảo vệ cuối chống mất dữ liệu âm thầm. Chunker tự loại dòng mục lục
    # rõ ràng và thêm hậu tố occurrence cho Điều/Khoản lặp hợp lệ trong luật
    # sửa đổi; nếu vẫn còn ID trùng thì đó là lỗi lập trình/dữ liệu chưa được
    # xử lý và ingest phải dừng trước khi Qdrant ghi đè point.
    dup_report = validate_chunk_id_uniqueness(all_chunks)
    if dup_report:
        details = "\n".join(
            f"- {cid}: '{a.doc_title}' — 2 nội dung khác nhau đủ dài, không thể tự động chọn "
            f"(thân bài {len(a.content)} và {len(b.content)} ký tự)"
            for cid, a, b in dup_report
        )
        raise RuntimeError(
            "Phát hiện chunk_id trùng lặp sau chunking — KHÔNG ingest để tránh mất dữ liệu "
            "âm thầm trong Qdrant. Hãy kiểm tra thủ công các văn bản sau:\n" + details
        )

    store = VietnameseLegalVectorStore()
    store.upsert_chunks(all_chunks, reset=True)
    print(f"Ingest completed in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
