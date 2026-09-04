"""Load + validate raw legal documents before indexing.

The loader treats corpus integrity as a hard gate. A legal document with no
real Article markers, suspicious metadata/content mismatch, or obviously wrong
content is rejected before embedding/indexing.
"""
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ingestion.chunker import ARTICLE_PATTERN, normalize_legal_text


@dataclass
class RawDocument:
    doc_id: str
    doc_title: str
    raw_text: str
    effective_status: str
    doc_type: str = "unknown"
    issue_date: str | None = None
    source_url: str | None = None
    path: str | None = None
    is_sample: bool = False
    license: str | None = None


def analyze_document(doc: RawDocument) -> dict:
    text = normalize_legal_text(doc.raw_text)
    article_count = len(list(ARTICLE_PATTERN.finditer(text)))
    issues: list[str] = []
    warnings: list[str] = []

    if not text or len(text) < 100:
        issues.append(f"too_short:{len(text)}")

    # Real laws/codes should expose Article headings. This catches the known
    # UTS_VLC 45/2019/QH14 corruption before it reaches the vector store.
    if doc.doc_type in {"law", "code", "constitution", "ordinance"} and article_count == 0:
        issues.append("no_article_markers")

    # A handful of cheap semantic fingerprints for high-risk documents. This
    # is intentionally conservative: absence means warning except for known
    # critical anchor phrases.
    title = doc.doc_title.lower()
    fingerprint_map = {
        "bộ luật lao động": ["người lao động", "người sử dụng lao động", "hợp đồng lao động"],
        "luật bảo hiểm xã hội": ["bảo hiểm xã hội", "người lao động"],
        "luật doanh nghiệp": ["doanh nghiệp", "công ty"],
    }
    for key, anchors in fingerprint_map.items():
        if key in title:
            missing = [a for a in anchors if a not in text.lower()]
            if len(missing) == len(anchors):
                issues.append(f"semantic_mismatch:{key}")
            elif missing:
                warnings.append(f"missing_anchor:{','.join(missing)}")

    if doc.doc_id and doc.doc_id.lower() not in text.lower():
        warnings.append("doc_id_not_in_body")
    if doc.doc_title and doc.doc_title.lower() not in text.lower():
        warnings.append("title_not_in_body")
    if doc.is_sample:
        warnings.append("sample_document_not_for_legal_answers")
    if doc.issue_date:
        try:
            date.fromisoformat(doc.issue_date)
        except (TypeError, ValueError):
            issues.append("invalid_issue_date")
    if not doc.source_url and not doc.is_sample:
        warnings.append("missing_source_url")

    return {
        "doc_id": doc.doc_id,
        "doc_title": doc.doc_title,
        "doc_type": doc.doc_type,
        "path": doc.path,
        "chars": len(text),
        "articles": article_count,
        "issues": issues,
        "warnings": warnings,
        "valid": not issues,
    }


def _read_documents(raw_dir: str = "./data/raw") -> list[RawDocument]:
    raw_path = Path(raw_dir)
    documents: list[RawDocument] = []
    files = sorted(set(raw_path.glob("*.md")) | set(raw_path.glob("*.txt")))
    for text_file in files:
        meta_file = text_file.with_suffix(".meta.json")
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Metadata không hợp lệ: {meta_file}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"Metadata phải là JSON object: {meta_file}")
        raw_text = text_file.read_text(encoding="utf-8", errors="replace")
        documents.append(RawDocument(
            doc_id=meta.get("doc_id", text_file.stem),
            doc_title=meta.get("doc_title", text_file.stem),
            raw_text=raw_text,
            effective_status=meta.get("effective_status", "unknown"),
            doc_type=meta.get("doc_type", "unknown"),
            issue_date=meta.get("issue_date"),
            source_url=meta.get("source_url"),
            path=str(text_file),
            is_sample=bool(meta.get("is_sample", False)),
            license=meta.get("license"),
        ))
    ids = [doc.doc_id for doc in documents]
    if len(ids) != len(set(ids)):
        duplicates = sorted({doc_id for doc_id in ids if ids.count(doc_id) > 1})
        raise ValueError(f"doc_id bị trùng trong corpus: {', '.join(duplicates)}")
    return documents


def load_raw_documents(raw_dir: str = "./data/raw", validate: bool = True, strict: bool = True) -> list[RawDocument]:
    documents = _read_documents(raw_dir)
    if not validate:
        return documents

    invalid = []
    for doc in documents:
        report = analyze_document(doc)
        if not report["valid"]:
            invalid.append(report)
        else:
            print(
                f"[loader] {doc.doc_id}: {report['articles']} Điều, "
                f"{report['chars']:,} chars, status={doc.effective_status}"
            )

    if invalid and strict:
        details = "\n".join(
            f"- {r['doc_id']} | {r['path']} | {'; '.join(r['issues'])}"
            for r in invalid
        )
        raise ValueError(
            "Corpus validation failed. Không được ingest dữ liệu lỗi:\n" + details +
            "\nHãy chạy `python -m scripts.audit_corpus` và sửa/quarantine các file BAD trước khi ingest."
        )
    return documents
