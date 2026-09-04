"""Kiểm tra nhanh cấu hình mà không tải embedding/reranker model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from ingestion.loader import _read_documents, analyze_document  # noqa: E402
from retrieval.vector_store import VietnameseLegalVectorStore  # noqa: E402


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    supported_python = (3, 11) <= sys.version_info[:2] < (3, 13)
    checks.append(("Python 3.11/3.12", supported_python, sys.version.split()[0]))

    checks.append((
        "Gemini API key",
        bool(settings.gemini_api_key),
        "đã cấu hình" if settings.gemini_api_key else "thiếu GEMINI_API_KEY",
    ))

    try:
        docs = _read_documents(settings.raw_data_dir)
        reports = [analyze_document(doc) for doc in docs]
        real_valid = sum(report["valid"] and not doc.is_sample for doc, report in zip(docs, reports, strict=True))
        corpus_ok = real_valid > 0
        corpus_detail = f"{real_valid} văn bản thật hợp lệ; {len(docs)} tệp tổng cộng"
    except (OSError, ValueError) as exc:
        corpus_ok = False
        corpus_detail = str(exc)
    checks.append(("Corpus", corpus_ok, corpus_detail))

    try:
        index_ok = VietnameseLegalVectorStore().is_ready()
        index_detail = "sẵn sàng" if index_ok else "chưa ingest hoặc index không đồng bộ"
    except RuntimeError as exc:
        index_ok = False
        index_detail = str(exc)
    checks.append(("Qdrant + BM25", index_ok, index_detail))

    for name, ok, detail in checks:
        print(f"{'OK' if ok else 'FAIL':4} | {name:18} | {detail}")

    if all(ok for _, ok, _ in checks):
        print("\nHệ thống sẵn sàng chạy.")
        return 0
    print("\nHãy xử lý các mục FAIL; xem README.md để biết lệnh cài/ingest.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

