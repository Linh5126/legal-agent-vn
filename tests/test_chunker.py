from ingestion.chunker import split_into_articles, split_long_article, validate_chunk_id_uniqueness


def test_markdown_article_formats():
    text = """# Luật mẫu\n\n**Điều 1.** Phạm vi điều chỉnh\nNội dung 1.\n\nĐiều 2: Đối tượng\nNội dung 2.\n"""
    chunks = split_into_articles(text, "X", "Luật mẫu", "hiệu lực")
    assert [c.article_number for c in chunks] == ["1", "2"]


def test_toc_duplicate_articles_are_auto_dropped():
    """Bug thật đã phát hiện trên corpus 306 văn bản (~12.6% chunk bị mất
    trong Qdrant): mục lục liệt kê 'Điều N' rồi heading thật lặp lại 'Điều N'
    -> cùng chunk_id -> upsert sau đè upsert trước. Dòng mục lục (thân bài
    gần như rỗng) phải bị tự động loại, chỉ giữ heading thật."""
    text = """MỤC LỤC
Điều 1. Phạm vi điều chỉnh
Điều 2. Đối tượng áp dụng

Chương I. QUY ĐỊNH CHUNG

Điều 1. Phạm vi điều chỉnh
Đây là nội dung thật của Điều 1, đủ dài để vượt ngưỡng MIN_ARTICLE_BODY_CHARS.

Điều 2. Đối tượng áp dụng
Đây là nội dung thật của Điều 2, cũng đủ dài để không bị coi là mục lục.
"""
    chunks = split_into_articles(text, "X", "Luật mẫu", "hiệu lực")
    assert [c.article_number for c in chunks] == ["1", "2"]
    assert "nội dung thật" in chunks[0].content
    assert validate_chunk_id_uniqueness(chunks) == []


def test_repeated_article_number_is_preserved_with_unique_occurrence_ids():
    """Luật sửa đổi có thể chứa nhiều lần xuất hiện hợp lệ của cùng số Điều.
    Phải giữ đủ nội dung và cấp ID riêng thay vì chặn toàn bộ ingest."""
    text = """Điều 1. Tiêu đề A
Đây là một đoạn nội dung khá dài của điều 1 phiên bản thứ nhất, đủ dài để không bị coi là mục lục.

Điều 1. Tiêu đề B
Đây là một đoạn nội dung khá dài khác của điều 1 phiên bản thứ hai, cũng đủ dài để không bị coi là mục lục.
"""
    chunks = split_into_articles(text, "X", "Luật mẫu lỗi", "hiệu lực")
    assert [chunk.chunk_id for chunk in chunks] == [
        "X::dieu_1::occ_1",
        "X::dieu_1::occ_2",
    ]
    assert validate_chunk_id_uniqueness(chunks) == []


def test_repeated_clause_number_is_preserved_with_unique_occurrence_ids():
    text = """Điều 1. Sửa đổi nhiều nội dung
1. Nội dung thứ nhất đủ dài để phần Điều vượt quá giới hạn tách nhỏ.
2. Nội dung thứ hai đủ dài để phần Điều vượt quá giới hạn tách nhỏ.
1. Một nội dung khác cũng được đánh số một trong phần sửa đổi tiếp theo.
""" + ("Bổ sung nội dung pháp lý. " * 120)
    article = split_into_articles(text, "X", "Luật sửa đổi", "hiệu lực")[0]
    chunks = split_long_article(article, max_chars=100)
    assert [chunk.chunk_id for chunk in chunks] == [
        "X::dieu_1::khoan_1::occ_1",
        "X::dieu_1::khoan_2",
        "X::dieu_1::khoan_1::occ_2",
    ]
    assert validate_chunk_id_uniqueness(chunks) == []
