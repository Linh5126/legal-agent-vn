from agents.citation_utils import citation_coverage, extract_cited_docs, unsupported_citations

DOCS = [
    {"article_number": "138", "doc_title": "Bộ luật Lao động"},
    {"article_number": "35", "doc_title": "Bộ luật Lao động"},
    {"article_number": "999", "doc_title": "Văn bản không liên quan"},
]


def test_extract_cited_docs_only_returns_docs_actually_mentioned():
    answer = "Theo [Điều 138, Bộ luật Lao động], người lao động nữ mang thai được quyền..."
    cited = extract_cited_docs(answer, DOCS)
    assert cited == ["Điều 138, Bộ luật Lao động"]


def test_extract_cited_docs_requires_both_article_and_title():
    """Chỉ có số Điều trùng số nhưng khác văn bản thì KHÔNG được tính là trích dẫn."""
    answer = "Điều 138 của luật khác quy định..."
    cited = extract_cited_docs(answer, DOCS)
    assert cited == []


def test_citation_coverage_full_when_all_docs_cited():
    answer = "[Điều 138, Bộ luật Lao động] và [Điều 35, Bộ luật Lao động] và [Điều 999, Văn bản không liên quan]"
    assert citation_coverage(answer, DOCS) == 1.0


def test_citation_coverage_zero_docs_but_proper_abstention():
    assert citation_coverage("Chưa đủ căn cứ để khẳng định.", []) == 1.0


def test_citation_coverage_zero_docs_without_abstention_language():
    assert citation_coverage("Có, chắc chắn là được.", []) == 0.0


def test_article_number_must_not_match_prefix():
    docs = [{"article_number": "3", "doc_title": "Bộ luật Lao động"}]
    answer = "Theo [Điều 35, Bộ luật Lao động], thời hạn là 45 ngày."
    assert extract_cited_docs(answer, docs) == []


def test_unsupported_citation_is_reported():
    answer = "Theo [Điều 999, Văn bản bịa], nội dung này đúng."
    assert unsupported_citations(answer, DOCS) == ["Điều 999, Văn bản bịa"]


def test_citation_accepts_clause_and_point_prefix():
    answer = "Theo [Điểm đ, Khoản 2, Điều 35, Bộ luật Lao động], không cần báo trước."
    assert extract_cited_docs(answer, DOCS) == ["Điều 35, Bộ luật Lao động"]


def test_split_clause_chunks_count_as_one_citation_source():
    docs = [
        {
            "chunk_id": "d::dieu_1::khoan_1",
            "doc_id": "d",
            "article_number": "1",
            "doc_title": "Luật A",
        },
        {
            "chunk_id": "d::dieu_1::khoan_2",
            "doc_id": "d",
            "article_number": "1",
            "doc_title": "Luật A",
        },
    ]
    assert citation_coverage("[Khoản 1, Điều 1, Luật A]", docs) == 1.0
