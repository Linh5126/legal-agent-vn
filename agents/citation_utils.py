"""Parse và kiểm tra trích dẫn pháp lý trong câu trả lời.

BUG THẬT đã phát hiện khi chạy main.py trên câu hỏi thật: phần "NGUỒN TRÍCH
DẪN" liệt kê TẤT CẢ retrieved_docs (Điều 137, 138, 29, 35, 36, 37), trong khi
nội dung câu trả lời chỉ thực sự trích dẫn 3/6 (Điều 138, 35, 29) — 3 nguồn
còn lại chưa từng xuất hiện trong câu trả lời nhưng vẫn bị hiển thị như thể
đã được dùng làm căn cứ. Với một trợ lý pháp lý, đây là lỗi tin cậy nghiêm
trọng: người dùng không thể phân biệt "nguồn đã đọc" và "nguồn đã dùng".

Trước khi sửa, evaluation/evaluate.py::citation_coverage() đã làm ĐÚNG việc
này (đối chiếu marker "Điều X" + doc_title có xuất hiện trong answer không),
nhưng finalize.py (chạy trong production) lại làm khác. Module này là nguồn
sự thật DUY NHẤT cho logic đối chiếu, dùng chung cho cả hai nơi để tránh lệch
pha giữa cái được đo (evaluation) và cái người dùng thực sự thấy (finalize).
"""
import re
import unicodedata

CITATION_PATTERN = re.compile(
    r"\[\s*"
    r"(?:Điểm\s+[^,\]\n]+\s*,\s*)?"
    r"(?:Khoản\s+\d+[a-zA-Z]?\s*,\s*)?"
    r"Điều\s+(?P<article>\d+[a-zA-Z]?)\s*,\s*"
    r"(?P<title>[^\]\n]+?)\s*\]",
    flags=re.IGNORECASE,
)


def _normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    return " ".join(value.split()).casefold()


def extract_citation_markers(answer: str) -> list[tuple[str, str]]:
    """Trả về các cặp ``(số Điều, tên văn bản)`` được viết đúng cú pháp."""
    return [
        (m.group("article"), " ".join(m.group("title").split()))
        for m in CITATION_PATTERN.finditer(answer or "")
    ]


def extract_cited_doc_records(answer: str, docs: list[dict]) -> list[dict]:
    """Trả về đúng các record đã được cite, không match chuỗi con mơ hồ.

    Ví dụ ``Điều 3`` không được coi là trích dẫn của ``Điều 35``. So khớp
    tên văn bản không phân biệt hoa/thường và khoảng trắng nhưng phải trùng
    toàn bộ tên.
    """
    markers = {
        (_normalize(article), _normalize(title))
        for article, title in extract_citation_markers(answer)
    }
    result = []
    seen: set[tuple[str, str]] = set()
    for doc in docs:
        citation_key = (
            _normalize(str(doc.get("article_number", ""))),
            _normalize(doc.get("doc_title", "")),
        )
        source_key = (
            _normalize(doc.get("doc_id", "") or doc.get("doc_title", "")),
            citation_key[0],
        )
        if citation_key in markers and source_key not in seen:
            result.append(doc)
            seen.add(source_key)
    return result


def unsupported_citations(answer: str, docs: list[dict]) -> list[str]:
    """Liệt kê marker có trong câu trả lời nhưng không tồn tại trong context."""
    allowed = {
        (_normalize(str(doc.get("article_number", ""))), _normalize(doc.get("doc_title", "")))
        for doc in docs
    }
    invalid = []
    for article, title in extract_citation_markers(answer):
        if (_normalize(article), _normalize(title)) not in allowed:
            invalid.append(f"Điều {article}, {title}")
    return list(dict.fromkeys(invalid))


def extract_cited_docs(answer: str, docs: list[dict]) -> list[str]:
    """Trả về danh sách "Điều X, <doc_title>" cho những doc THỰC SỰ được
    nhắc tới trong `answer` (cả marker Điều lẫn tên văn bản đều phải xuất
    hiện), sắp xếp và loại trùng.
    """
    return list(dict.fromkeys(
        f"Điều {d['article_number']}, {d['doc_title']}"
        for d in extract_cited_doc_records(answer, docs)
    ))


def citation_coverage(answer: str, docs: list[dict]) -> float:
    """Tỉ lệ retrieved_docs thực sự được trích dẫn trong answer. Nếu không
    có doc nào được retrieve, coi là "phủ" đủ khi model biết từ chối đúng
    cách (nói "chưa đủ căn cứ") thay vì bịa.
    """
    if not docs:
        return 1.0 if "chưa đủ căn cứ" in answer.lower() else 0.0
    unique_sources = {
        (
            _normalize(doc.get("doc_id", "") or doc.get("doc_title", "")),
            _normalize(str(doc.get("article_number", ""))),
        )
        for doc in docs
    }
    return len(extract_cited_doc_records(answer, docs)) / len(unique_sources)
