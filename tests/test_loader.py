from ingestion.loader import RawDocument, analyze_document


def test_invalid_law_without_articles():
    doc = RawDocument("45/2019/QH14", "Bộ luật Lao động", "Kế hoạch của UBND tỉnh Lào Cai", "hiệu lực", "code")
    report = analyze_document(doc)
    assert report["valid"] is False
    assert "no_article_markers" in report["issues"]
    assert any("semantic_mismatch" in x for x in report["issues"])
