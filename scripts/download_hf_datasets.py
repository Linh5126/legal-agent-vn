"""Download and validate legal datasets.

UTS_VLC is treated as an upstream source, not as ground truth. Every record is
validated before it is written to data/raw. Known-bad records are quarantined.
For 45/2019/QH14 (Labor Code), `--repair_known` can fetch the official
Government PDF and extract it to markdown before validation.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW_DIR = Path("data/raw")
QUARANTINE_DIR = Path("data/quarantine")
BENCHMARK_DIR = Path("data/raw_benchmark")
TESTSET_DIR = Path("data/testset")
OFFICIAL_LABOR_PDF = "https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf"
OFFICIAL_LABOR_HTML = "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=139264&Keyword=45%2F2019%2FQH14"
SECONDARY_LABOR_HTML = "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-Luat-lao-dong-2019-333670.aspx"
MIRROR_LABOR_PDF = "https://phongtochuc.hub.edu.vn/DATA/DOCUMENT/2022/11/24/lu%E1%BA%ADt%20lao%20%C4%91%E1%BB%99ng%20m%E1%BB%9Bi%20nh%E1%BA%A5t%202019.pdf"


def _safe_dataset_filename(value: object) -> str | None:
    """Chỉ chấp nhận basename .md/.txt; dataset là input không tin cậy."""
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.name != value or path.suffix.lower() not in {".md", ".txt"}:
        return None
    if not re.fullmatch(r"[\wÀ-ỹ(). -]+\.(?:md|txt)", value, flags=re.IGNORECASE):
        return None
    return value


def uts_vlc_record_to_meta(record: dict) -> dict:
    return {
        "doc_id": record["id"],
        "doc_title": record["title"],
        "doc_type": record["type"],
        # Dataset snapshot không tự chứng minh tình trạng hiệu lực hiện tại.
        "effective_status": "unknown",
        "status_verified_at": None,
        "issue_date": None,
        "source_url": "https://huggingface.co/datasets/undertheseanlp/UTS_VLC",
        "upstream_dataset": "undertheseanlp/UTS_VLC",
        "license": "kiểm tra dataset card trước khi phân phối lại",
    }


def _quick_validate(record: dict) -> list[str]:
    text = str(record.get("content", ""))
    doc_type = record.get("type", "unknown")
    issues = []
    if _safe_dataset_filename(record.get("filename")) is None:
        issues.append("invalid_filename")
    article_count = len(re.findall(r"(?im)^\s*(?:\*\*|__)?Điều\s+\d+[a-zA-Z]?\s*(?:[.:]|$)", text))
    if doc_type in {"law", "code", "constitution", "ordinance"} and article_count == 0:
        issues.append("no_article_markers")
    title = str(record.get("title", "")).lower()
    low = text.lower()
    if "bộ luật lao động" in title and all(k not in low for k in ["người lao động", "người sử dụng lao động", "hợp đồng lao động"]):
        issues.append("semantic_mismatch:bo_luat_lao_dong")
    return issues


def _quarantine(record: dict, issues: list[str]) -> None:
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    raw_name = str(record.get("filename") or record.get("id") or "record")
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(raw_name).name) or "record"
    (QUARANTINE_DIR / safe).write_text(record.get("content", ""), encoding="utf-8")
    # Remove stale local copies from a previous run before writing the quarantine.
    stale_raw = RAW_DIR / safe
    stale_meta = stale_raw.with_suffix(".meta.json")
    for stale in (stale_raw, stale_meta):
        if stale.exists():
            stale.unlink()
    (QUARANTINE_DIR / f"{safe}.meta.json").write_text(json.dumps({
        "id": record.get("id"), "title": record.get("title"), "type": record.get("type"),
        "issues": issues, "upstream": "undertheseanlp/UTS_VLC"
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def repair_known_labor_code() -> None:
    """Repair 45/2019/QH14 from official HTML first, PDF second, secondary HTML last.

    The Government PDF is a signed 83-page PDF whose text layer can be nearly empty in
    some environments (observed extraction: < 500 chars). The official VBPL full-text
    page is therefore preferred because it is already machine-readable. PDF extraction
    remains as a fallback. A secondary full-text source is used only as a last resort.
    """
    import requests

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    anchors = ["BỘ LUẬT LAO ĐỘNG", "45/2019/QH14", "Điều 1", "Điều 35", "Điều 137", "Điều 138"]

    def normalized(s: str) -> str:
        s = s.replace("\u00a0", " ")
        return re.sub(r"\s+", " ", s).strip().lower()

    def missing_for(text: str) -> list[str]:
        compact = normalized(text)
        return [a for a in anchors if normalized(a) not in compact]

    def extract_html(url: str) -> str:
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise RuntimeError("Cần cài beautifulsoup4 để dùng HTML fallback") from e
        r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        for node in soup(["script", "style", "noscript", "svg"]):
            node.decompose()
        return soup.get_text("\n")

    candidates: list[tuple[str, str]] = []

    # 1) Official VBPL machine-readable full text.
    print("Trying official VBPL full text:", OFFICIAL_LABOR_HTML)
    try:
        official_html_text = extract_html(OFFICIAL_LABOR_HTML)
        candidates.append(("official_vbpl_html", official_html_text))
        missing = missing_for(official_html_text)
        if not missing:
            selected_source = "official_vbpl_html"
            selected_text = official_html_text
        else:
            selected_source = None
            selected_text = ""
            print(f"Official VBPL HTML validation failed: {missing}")
    except Exception as e:
        selected_source = None
        selected_text = ""
        print(f"Official VBPL HTML fetch failed: {e}")

    # 2) Machine-readable mirror PDF.
    # This copy is a text-based 47-page PDF and is used only as a recovery
    # source when the signed Government PDF has an unusable text layer.
    print("Trying machine-readable Labor Code PDF mirror:", MIRROR_LABOR_PDF)
    try:
        tmp_pdf = QUARANTINE_DIR / "45-2019-QH14-mirror.pdf"
        r = requests.get(MIRROR_LABOR_PDF, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            raise RuntimeError("mirror response is not a PDF")
        tmp_pdf.write_bytes(r.content)
        try:
            import pymupdf
        except ImportError:
            import fitz as pymupdf
        doc = pymupdf.open(str(tmp_pdf))
        try:
            mirror_text = "\n\n".join(page.get_text("text") or "" for page in doc)
        finally:
            doc.close()
        mirror_missing = missing_for(mirror_text)
        print(f"Mirror PDF extraction chars: {len(mirror_text):,}")
        if not mirror_missing:
            selected_source, selected_text = "machine_readable_mirror_pdf", mirror_text
        else:
            print(f"Machine-readable mirror validation failed: {mirror_missing}")
    except Exception as e:
        print(f"Machine-readable mirror fetch failed: {e}")

    # 3) Official Government PDF. Some signed PDFs expose only a tiny text layer.
    if not selected_text:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise RuntimeError("Cần cài pypdf để chạy --repair_known") from e
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError as e:
                raise RuntimeError("Cần cài PyMuPDF (gói `pymupdf`) để chạy --repair_known") from e

        tmp_pdf = QUARANTINE_DIR / "45-2019-QH14-official.pdf"
        print("Downloading official Labor Code PDF:", OFFICIAL_LABOR_PDF)
        r = requests.get(OFFICIAL_LABOR_PDF, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        tmp_pdf.write_bytes(r.content)

        pypdf_pages = []
        reader = PdfReader(str(tmp_pdf))
        for page in reader.pages:
            pypdf_pages.append(page.extract_text() or "")
        pypdf_text = "\n\n".join(pypdf_pages)
        if not missing_for(pypdf_text):
            selected_source, selected_text = "official_government_pdf_pypdf", pypdf_text

        if not selected_text:
            doc = pymupdf.open(str(tmp_pdf))
            try:
                pymupdf_text = "\n\n".join(page.get_text("text") or "" for page in doc)
            finally:
                doc.close()
            if not missing_for(pymupdf_text):
                selected_source, selected_text = "official_government_pdf_pymupdf", pymupdf_text
            print(f"Official PDF extraction chars: pypdf={len(pypdf_text):,}, pymupdf={len(pymupdf_text):,}")

    # 4) Last resort: a secondary full-text source. Keep explicit provenance.
    if not selected_text:
        print("Trying secondary full-text source:", SECONDARY_LABOR_HTML)
        try:
            secondary_text = extract_html(SECONDARY_LABOR_HTML)
            missing = missing_for(secondary_text)
            if not missing:
                selected_source, selected_text = "secondary_thuvienphapluat_html", secondary_text
            else:
                print(f"Secondary source validation failed: {missing}")
        except Exception as e:
            print(f"Secondary source fetch failed: {e}")

    if not selected_text:
        missing_reports = [f"{source}:{missing_for(text)}" for source, text in candidates]
        raise RuntimeError(
            "Official extraction failed validation. "
            f"Tried official VBPL HTML + machine-readable mirror PDF + official PDF + secondary HTML; details={missing_reports}"
        )

    # Clean obvious navigation/header noise while retaining the legal text.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in selected_text.splitlines()]
    selected_text = "\n\n".join(line for line in lines if line)

    target = RAW_DIR / "code-2019-bo-luat-lao-dong.md"
    target.write_text(selected_text, encoding="utf-8")
    meta = {
        "doc_id": "45/2019/QH14",
        "doc_title": "Bộ luật Lao động",
        "doc_type": "code",
        "effective_status": "unknown",
        "issue_date": "2019-11-20",
        "status_verified_at": None,
        "source_url": "https://vanban.chinhphu.vn/?docid=198540&lang=vi&pageid=27160",
        "source_attachment": OFFICIAL_LABOR_PDF,
        "full_text_source": (OFFICIAL_LABOR_HTML if selected_source == "official_vbpl_html" else MIRROR_LABOR_PDF if selected_source == "machine_readable_mirror_pdf" else SECONDARY_LABOR_HTML if selected_source == "secondary_thuvienphapluat_html" else OFFICIAL_LABOR_PDF),
        "source_note": f"Repaired from {selected_source} after upstream corpus content mismatch.",
        "license": "kiểm tra điều kiện sử dụng của nguồn trước khi phân phối lại",
    }
    (RAW_DIR / "code-2019-bo-luat-lao-dong.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Repaired: {target} | source={selected_source} | chars={len(selected_text):,}")

def download_uts_vlc(limit: int | None = None, doc_type: str | None = None, repair_known: bool = False) -> None:
    from datasets import load_dataset
    print("Downloading undertheseanlp/UTS_VLC split=2026...")
    ds = load_dataset("undertheseanlp/UTS_VLC", split="2026")
    if doc_type:
        ds = ds.filter(lambda x: x["type"] == doc_type)
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    accepted = rejected = 0
    for record in ds:
        issues = _quick_validate(record)
        if issues:
            rejected += 1
            _quarantine(record, issues)
            print(f"[QUARANTINE] {record.get('id')} -> {issues}")
            continue
        filename = _safe_dataset_filename(record.get("filename"))
        if filename is None:  # đã bị _quick_validate loại; giữ guard phòng thủ
            raise RuntimeError("Tên file dataset không hợp lệ lọt qua validation")
        (RAW_DIR / filename).write_text(record["content"], encoding="utf-8")
        meta_filename = str(Path(filename).with_suffix(".meta.json"))
        (RAW_DIR / meta_filename).write_text(
            json.dumps(uts_vlc_record_to_meta(record), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        accepted += 1

    if repair_known and not limit and doc_type is None:
        repair_known_labor_code()

    print(f"UTS_VLC accepted={accepted}, quarantined={rejected}")


def retrieval_records_to_benchmark(records: list[dict]) -> tuple[list[dict], list[dict]]:
    chunks_by_id = {}
    testset = []
    for record in records:
        qid = record["qid"]
        question = record["question"]
        cids = record["cid"]
        contexts = record["context_list"]
        relevant_ids = []
        for cid, context_text in zip(cids, contexts, strict=True):
            chunk_id = f"yuitc::{cid}"
            relevant_ids.append(chunk_id)
            if chunk_id not in chunks_by_id:
                article_match = re.search(r"Điều\s+(\d+[a-zA-Z]?)", context_text)
                chunks_by_id[chunk_id] = {
                    "chunk_id": chunk_id, "doc_id": "yuitc_benchmark_corpus",
                    "doc_title": "YuITC Vietnamese Legal Retrieval Benchmark",
                    "article_number": article_match.group(1) if article_match else "unknown",
                    "content": context_text, "effective_status": "unknown",
                }
        testset.append({"id": f"yuitc_q{qid}", "question": question, "ground_truth_answer": "",
                         "relevant_chunk_ids": relevant_ids, "type": "retrieval_benchmark_real"})
    return list(chunks_by_id.values()), testset


def download_retrieval_benchmark(limit: int | None = None) -> None:
    from datasets import load_dataset
    print("Downloading YuITC/Vietnamese-Legal-Doc-Retrieval-Data...")
    ds = load_dataset("YuITC/Vietnamese-Legal-Doc-Retrieval-Data", split="train")
    if limit is not None and limit <= 0:
        raise ValueError("limit phải lớn hơn 0")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    chunks, testset = retrieval_records_to_benchmark(list(ds))
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    with open(BENCHMARK_DIR / "benchmark_corpus.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    TESTSET_DIR.mkdir(parents=True, exist_ok=True)
    (TESTSET_DIR / "yuitc_retrieval_testset.json").write_text(
        json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved benchmark chunks={len(chunks)}, questions={len(testset)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["uts_vlc", "retrieval_benchmark", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--type", default=None)
    ap.add_argument("--repair_known", action="store_true", help="repair known-bad UTS_VLC Labor Code from official Government PDF")
    args = ap.parse_args()
    if args.dataset in ("uts_vlc", "all"):
        download_uts_vlc(args.limit, args.type, args.repair_known)
    if args.dataset in ("retrieval_benchmark", "all"):
        download_retrieval_benchmark(args.limit)


if __name__ == "__main__":
    main()
