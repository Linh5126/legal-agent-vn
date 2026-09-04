"""
Bộ công cụ (tools) mà Tool-calling agent có thể gọi. Đây là các hàm tính toán/
tra cứu xác định (deterministic) mà LLM không nên "tự tính" vì dễ sai số —
đúng tinh thần tool-calling: LLM quyết định GỌI gì, tool thực thi logic chính xác.

Mở rộng: thêm tool mới chỉ cần viết hàm + đăng ký vào TOOL_REGISTRY bên dưới.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

from config import settings


def calculate_social_insurance(monthly_salary: float) -> str:
    """
    Ước tính mức đóng bảo hiểm xã hội bắt buộc phía người lao động (8% lương).
    Đây là công thức đơn giản hóa cho mục đích demo — số liệu thật cần đối chiếu
    quy định hiện hành và mức lương tối đa đóng BHXH.
    """
    if isinstance(monthly_salary, bool) or not isinstance(monthly_salary, (int, float)):
        raise TypeError("monthly_salary phải là số")
    if not math.isfinite(float(monthly_salary)) or not 0 < monthly_salary <= 1_000_000_000_000:
        raise ValueError("monthly_salary phải lớn hơn 0 và không vượt quá 1.000 tỷ VNĐ")

    employee_rate = 0.08
    amount = monthly_salary * employee_rate
    return (
        f"Với lương {monthly_salary:,.0f} VNĐ, mức đóng BHXH bắt buộc phía "
        f"người lao động (8%) ước tính là {amount:,.0f} VNĐ/tháng. "
        "Đây chỉ là phép tính minh họa; mức làm căn cứ đóng và mức trần phải "
        "được xác định từ văn bản đang có hiệu lực trong corpus."
    )


@lru_cache(maxsize=1)
def _load_effective_status_lookup() -> dict:
    """Nạp bảng tra cứu hiệu lực từ toàn bộ data/raw/*.meta.json, cache 1 lần
    (invalidate bằng _load_effective_status_lookup.cache_clear() nếu re-ingest
    trong cùng process — chủ yếu dùng trong test).
    """
    lookup = {}
    raw_dir = Path(settings.raw_data_dir)
    for meta_file in sorted(raw_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        doc_id = meta.get("doc_id")
        if doc_id:
            lookup[doc_id] = meta.get("effective_status", "unknown")
    return lookup


def check_document_effective_status(doc_id: str) -> str:
    """
    Kiểm tra tình trạng hiệu lực của một văn bản luật dựa trên bảng tra cứu
    nạp từ metadata (data/raw/*.meta.json — xem ingestion/loader.py).

    BUG THẬT đã sửa: bản trước yêu cầu tham số `effective_status_lookup: dict`
    do caller truyền vào. Nhưng caller thực tế (tool_agent.py) gọi tool bằng
    `TOOL_REGISTRY[tool_name](**tool_input)` với `tool_input` do LLM tự sinh
    từ TOOL_DESCRIPTIONS — LLM không có (và không nên có) cách nào tạo ra
    một dict tra cứu toàn corpus. Kết quả: MỌI lần gọi tool này đều rơi vào
    nhánh TypeError trong tool_agent.py và trả về lỗi "tham số không hợp lệ"
    — tool coi như không bao giờ hoạt động được. Sửa bằng cách tự nạp lookup
    nội bộ (cache 1 lần) thay vì đòi hỏi tham số mà LLM không thể cung cấp.
    """
    if not isinstance(doc_id, str) or not doc_id.strip() or len(doc_id) > 200:
        raise ValueError("doc_id không hợp lệ")
    doc_id = doc_id.strip()
    status = _load_effective_status_lookup().get(doc_id, "unknown")
    if status == "unknown":
        return f"Không tìm thấy thông tin hiệu lực cho văn bản '{doc_id}'."
    return f"Văn bản '{doc_id}' hiện đang ở trạng thái: {status}."


TOOL_REGISTRY = {
    "calculate_social_insurance": calculate_social_insurance,
    "check_document_effective_status": check_document_effective_status,
}

# Mô tả tool để đưa vào prompt cho LLM quyết định có nên gọi hay không
TOOL_DESCRIPTIONS = """\
- calculate_social_insurance(monthly_salary: float): tính mức đóng BHXH bắt buộc theo lương tháng.
- check_document_effective_status(doc_id: str): kiểm tra một văn bản luật còn hiệu lực hay không.
"""
