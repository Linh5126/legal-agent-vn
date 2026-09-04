import json

import pytest

from tools import legal_tools


def test_check_document_effective_status_only_takes_doc_id(tmp_path, monkeypatch):
    """Đây chính là bug đã sửa: trước đây tool đòi thêm tham số
    `effective_status_lookup` mà LLM không thể cung cấp qua tool_input, khiến
    tool luôn lỗi TypeError khi gọi thật. Giờ chỉ cần doc_id — đúng như những
    gì TOOL_DESCRIPTIONS mô tả cho LLM."""
    meta_file = tmp_path / "45-2019-QH14.meta.json"
    meta_file.write_text(json.dumps({"doc_id": "45/2019/QH14", "effective_status": "hiệu lực"}),
                          encoding="utf-8")

    from config import settings
    monkeypatch.setattr(settings, "raw_data_dir", str(tmp_path))
    legal_tools._load_effective_status_lookup.cache_clear()

    result = legal_tools.check_document_effective_status(doc_id="45/2019/QH14")
    assert "hiệu lực" in result

    legal_tools._load_effective_status_lookup.cache_clear()


def test_check_document_effective_status_unknown_doc(tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "raw_data_dir", str(tmp_path))
    legal_tools._load_effective_status_lookup.cache_clear()

    result = legal_tools.check_document_effective_status(doc_id="khong-ton-tai")
    assert "Không tìm thấy" in result

    legal_tools._load_effective_status_lookup.cache_clear()


def test_calculate_social_insurance_basic():
    result = legal_tools.calculate_social_insurance(10_000_000)
    assert "800,000" in result or "800.000" in result


@pytest.mark.parametrize("salary", [0, -1, float("inf"), True, "10000000"])
def test_calculate_social_insurance_rejects_invalid_salary(salary):
    with pytest.raises((TypeError, ValueError)):
        legal_tools.calculate_social_insurance(salary)
