import os

# Môi trường CI có thể khai báo SOCKS proxy nhưng không cài optional socksio;
# UI local không cần proxy để được import/kiểm thử.
for key in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)

from app import _format_citations  # noqa: E402


def test_format_citations_links_only_http_sources():
    result = {"citation_details": [
        {"label": "Điều 1, Luật A", "source_url": "https://example.com/a", "effective_status": "hiệu lực"},
        {"label": "Điều 2, Luật B", "source_url": "javascript:alert(1)", "effective_status": "unknown"},
    ]}
    rendered = _format_citations(result)
    assert "[Điều 1, Luật A](https://example.com/a)" in rendered
    assert "javascript:" not in rendered
