"""Robust chunking cho văn bản pháp luật tiếng Việt.

Mục tiêu: mỗi chunk bám vào một Điều (và nếu quá dài thì tách theo Khoản),
nhưng tuyệt đối không âm thầm biến một văn bản không parse được thành một
chunk khổng lồ. Dữ liệu pháp luật sai cấu trúc phải được phát hiện ở ingest.
"""
import re
from dataclasses import dataclass, field

# Hỗ trợ markdown bold, không có dấu chấm, dấu ':' và Điều 12a.
ARTICLE_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*|__)?Điều\s+(\d+[a-zA-Z]?)\s*(?:[.:]\s*)?(.*?)\s*(?:\*\*|__)?\s*$"
)
CHAPTER_PATTERN = re.compile(
    r"(?im)^\s*(?:\*\*|__)?Chương\s+([IVXLCDM\d]+)\s*\.?\s*(.*?)\s*(?:\*\*|__)?\s*$"
)
CLAUSE_PATTERN = re.compile(r"(?m)^\s*(\d+)\.\s+")

# Một 'Điều' thật luôn có nội dung pháp lý thực chất. Một dòng mục lục
# (ví dụ "Điều 5. Phạm vi điều chỉnh" liệt kê trong MỤC LỤC trước khi vào nội
# dung thật) cũng khớp ARTICLE_PATTERN nhưng gần như không có "thân bài" phía
# sau trước khi gặp dòng "Điều" kế tiếp. Ngưỡng này dùng để phân biệt 2 loại.
MIN_ARTICLE_BODY_CHARS = 40


@dataclass
class LegalChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    chapter: str | None
    article_number: str
    article_title: str
    content: str
    effective_status: str = "unknown"
    metadata: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "chapter": self.chapter,
            "article_number": self.article_number,
            "article_title": self.article_title,
            "content": self.content,
            "effective_status": self.effective_status,
            **self.metadata,
        }


def normalize_legal_text(text: str) -> str:
    """Normalize markdown/whitespace mà không phá cấu trúc Điều/Khoản."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # BOM và zero-width characters
    text = text.replace("\ufeff", "").replace("\u200b", "")
    # HTML comment (thường xuất hiện trong file dataset/sample)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    # Chuẩn hóa các heading markdown nhưng giữ text
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    # Bỏ bold quanh marker Điều/Chương nhưng không cần bỏ bold trong nội dung
    text = re.sub(r"^\s*(\*\*|__)(Điều\s+\d+[a-zA-Z]?[^\n]*?)(\*\*|__)\s*$", r"\2", text, flags=re.M | re.I)
    text = re.sub(r"^\s*(\*\*|__)(Chương\s+[IVXLCDM\d]+[^\n]*?)(\*\*|__)\s*$", r"\2", text, flags=re.M | re.I)
    # Không collapse toàn bộ newline: newline là tín hiệu cấu trúc quan trọng.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _drop_toc_duplicate_matches(matches: list, raw_text: str, doc_id: str) -> list:
    """Loại các match của ARTICLE_PATTERN thực chất là dòng mục lục, không
    phải heading Điều thật.

    BUG THẬT đã phát hiện khi chạy trên corpus thật (306 văn bản): sau ingest,
    số point trong Qdrant thấp hơn ~12.6% so với số chunk đã sinh ra, do
    'Điều N' bị match 2 lần trong cùng 1 văn bản (1 lần ở mục lục, 1 lần ở
    heading thật) -> cả 2 tạo ra CÙNG chunk_id (xem vector_store.py
    _point_id, dùng uuid5 tất định theo chunk_id) -> upsert sau ĐÈ upsert
    trước, mất nội dung mà không có bất kỳ cảnh báo nào.

    Chỉ tự động bỏ một match trùng số Điều nếu thân bài của nó (khoảng cách
    tới match kế tiếp) ngắn hơn MIN_ARTICLE_BODY_CHARS -- tức về mặt cấu trúc
    gần như chắc chắn là một dòng mục lục. Các lần xuất hiện còn lại được giữ
    nguyên; split_into_articles() sẽ cấp ID có hậu tố occurrence cho chúng.
    Điều này cần thiết với luật sửa đổi, nơi cùng một số Điều có thể xuất hiện
    nhiều lần trong các phần trích dẫn/thay thế và đều là dữ liệu hợp lệ.
    """
    by_number: dict[str, list] = {}
    for m in matches:
        by_number.setdefault(m.group(1), []).append(m)
    dup_numbers = {num for num, ms in by_number.items() if len(ms) > 1}
    if not dup_numbers:
        return matches

    keep = []
    dropped = []
    for i, m in enumerate(matches):
        num = m.group(1)
        if num not in dup_numbers:
            keep.append(m)
            continue
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body_len = body_end - m.end()
        if body_len < MIN_ARTICLE_BODY_CHARS:
            dropped.append((num, body_len))
            continue
        keep.append(m)

    if dropped:
        preview = ", ".join(f"Điều {number} ({length} ký tự)" for number, length in dropped)
        print(f"[chunker] {doc_id}: bỏ {len(dropped)} dòng nghi là mục lục trùng số Điều: {preview}", flush=True)

    return keep


def split_into_articles(raw_text: str, doc_id: str, doc_title: str,
                        effective_status: str = "unknown",
                        metadata: dict | None = None) -> list[LegalChunk]:
    raw_text = normalize_legal_text(raw_text)
    matches = list(ARTICLE_PATTERN.finditer(raw_text))
    if not matches:
        raise ValueError(
            f"Không parse được 'Điều' trong document '{doc_id}'. "
            "Không được fallback thành một chunk toàn văn; hãy kiểm tra source/format."
        )
    matches = _drop_toc_duplicate_matches(matches, raw_text, doc_id)

    # Luật sửa đổi có thể trích dẫn/thay thế cùng một số Điều nhiều lần.
    # Không được loại tùy tiện hay để các lần này dùng chung chunk_id, vì
    # Qdrant sẽ ghi đè point có ID trùng. Hậu tố occurrence vừa giữ đủ dữ
    # liệu vừa có tính tất định trong một lần snapshot corpus.
    article_counts: dict[str, int] = {}
    for match in matches:
        number = match.group(1)
        article_counts[number] = article_counts.get(number, 0) + 1
    article_occurrences: dict[str, int] = {}

    chunks: list[LegalChunk] = []
    current_chapter = None
    chapter_matches = list(CHAPTER_PATTERN.finditer(raw_text))
    chapter_idx = 0

    for i, m in enumerate(matches):
        while chapter_idx + 1 < len(chapter_matches) and chapter_matches[chapter_idx + 1].start() < m.start():
            chapter_idx += 1
        if chapter_matches and chapter_matches[chapter_idx].start() < m.start():
            cm = chapter_matches[chapter_idx]
            current_chapter = f"Chương {cm.group(1)} - {cm.group(2)}".strip(" -")

        article_number = m.group(1)
        article_occurrences[article_number] = article_occurrences.get(article_number, 0) + 1
        article_title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        header = f"Điều {article_number}. {article_title}".rstrip()
        full_content = f"{header}\n{body}".strip()

        chunk_id = f"{doc_id}::dieu_{article_number}"
        if article_counts[article_number] > 1:
            chunk_id += f"::occ_{article_occurrences[article_number]}"

        chunks.append(LegalChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_title=doc_title,
            chapter=current_chapter,
            article_number=article_number,
            article_title=article_title,
            content=full_content,
            effective_status=effective_status,
            metadata=dict(metadata or {}),
        ))

    return chunks


def validate_chunk_id_uniqueness(chunks: list[LegalChunk]) -> list[tuple[str, LegalChunk, LegalChunk]]:
    """Trả về danh sách (chunk_id, chunk_đầu, chunk_trùng) cho mọi chunk_id bị
    trùng sau toàn bộ bước chunking. Đây là lớp bảo vệ CUỐI CÙNG trước khi
    upsert vào Qdrant: vector_store.py tạo point ID tất định từ chunk_id
    (uuid5), nên bất kỳ trùng lặp nào lọt qua đây sẽ khiến 1 chunk bị ghi đè
    và biến mất khỏi index một cách âm thầm. Gọi hàm này ngay sau khi build
    xong all_chunks, TRƯỚC khi upsert (xem scripts/ingest_data.py).
    """
    seen: dict[str, LegalChunk] = {}
    dups: list[tuple[str, LegalChunk, LegalChunk]] = []
    for c in chunks:
        if c.chunk_id in seen:
            dups.append((c.chunk_id, seen[c.chunk_id], c))
        else:
            seen[c.chunk_id] = c
    return dups


def split_long_article(chunk: LegalChunk, max_chars: int = 2200) -> list[LegalChunk]:
    """Tách Điều dài theo Khoản nhưng giữ header Điều trong mọi sub-chunk."""
    if len(chunk.content) <= max_chars:
        return [chunk]

    matches = list(CLAUSE_PATTERN.finditer(chunk.content))
    if len(matches) < 2:
        return [chunk]

    header = chunk.content[:matches[0].start()].strip()
    clause_counts: dict[str, int] = {}
    for match in matches:
        number = match.group(1)
        clause_counts[number] = clause_counts.get(number, 0) + 1
    clause_occurrences: dict[str, int] = {}
    sub_chunks: list[LegalChunk] = []
    for i, m in enumerate(matches):
        clause_number = m.group(1)
        clause_occurrences[clause_number] = clause_occurrences.get(clause_number, 0) + 1
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(chunk.content)
        clause_text = chunk.content[start:end].strip()
        chunk_id = f"{chunk.chunk_id}::khoan_{clause_number}"
        if clause_counts[clause_number] > 1:
            chunk_id += f"::occ_{clause_occurrences[clause_number]}"
        sub_chunks.append(LegalChunk(
            chunk_id=chunk_id,
            doc_id=chunk.doc_id,
            doc_title=chunk.doc_title,
            chapter=chunk.chapter,
            article_number=chunk.article_number,
            article_title=chunk.article_title,
            content=f"{header}\n{clause_text}".strip(),
            effective_status=chunk.effective_status,
            metadata={
                **chunk.metadata,
                "parent_chunk_id": chunk.chunk_id,
                "clause_number": clause_number,
            },
        ))
    return sub_chunks
