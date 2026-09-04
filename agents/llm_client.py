"""
Wrapper gọi Gemini dùng chung cho tất cả agent, có retry + fallback model khi
gặp lỗi quá tải (503) hoặc bị khai tử đột ngột (404).

Dùng Google AI Studio (Gemini Developer API) — chỉ cần API key, có free tier,
KHÔNG cần tài khoản GCP/thanh toán như Vertex AI. Lấy key tại:
https://aistudio.google.com/apikey (khác với gói Gemini Pro sinh viên trong app).

Lỗi 503 UNAVAILABLE ("high demand") từ Gemini là lỗi TẠM THỜI phía Google, hay
gặp ở free tier giờ cao điểm — không phải bug. Chiến lược 2 tầng:
  1. Retry nhiều lần trên CÙNG model (đa số lỗi thoáng qua sẽ tự hết)
  2. Nếu vẫn lỗi sau khi hết số lần retry, CHUYỂN SANG model dự phòng nhẹ hơn
     (thường ít bị nghẽn hơn model chính) thay vì để cả graph sập theo.
"""
import json
import logging
import re

from google import genai
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from agents.observability import observe
from config import settings

# BUG THẬT đã sửa: trước đây `genai.Client(...)` được gọi ngay ở import-time
# (module-level). Bất kỳ module nào import agents.llm_client — kể cả gián
# tiếp qua agents.verifier hay agents.tool_agent — sẽ CRASH ngay khi collect
# nếu process chưa có GEMINI_API_KEY, kể cả khi test đó không hề gọi LLM
# thật (ví dụ test_verifier.py chỉ test should_continue_loop, hàm thuần túy
# không cần LLM). Điều này chặn đứng toàn bộ `pytest` trên máy CI/máy dev
# chưa cấu hình `.env`. Sửa bằng lazy singleton: chỉ khởi tạo client ở lần
# gọi LLM đầu tiên, không phải ở lúc import module.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY chưa được cấu hình (xem .env.example). "
                "Cần key thật để thực sự gọi LLM — import module này để test "
                "logic thuần túy (không gọi LLM) vẫn hoạt động bình thường."
            )
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


# Model chính lấy từ settings.llm_model (mặc định "gemini-3.5-flash", xem
# config.py — đồng bộ giá trị này nếu đổi default ở đó).
# Model dự phòng lấy từ settings.llm_fallback_model — nhẹ hơn, thường có
# nhiều capacity hơn model chính đang được nhiều người dùng free tier gọi
# cùng lúc. Có thể thêm nhiều tầng dự phòng hơn nếu vẫn hay gặp 503.
LOGGER = logging.getLogger(__name__)
FALLBACK_MODELS = list(dict.fromkeys(
    model.strip() for model in (settings.llm_model, settings.llm_fallback_model) if model.strip()
))


def _status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    match = re.search(r"\b(4\d\d|5\d\d)\b", str(exc))
    return int(match.group(1)) if match else None


def _is_retryable(exc: Exception) -> bool:
    """Chỉ retry lỗi tạm thời; không chờ vô ích khi thiếu key/400/401."""
    return _status_code(exc) in {408, 429, 500, 502, 503, 504}


def _can_fallback(exc: Exception) -> bool:
    # 404 có thể là model vừa bị rút; chuyển model ngay nhưng không retry model cũ.
    return _is_retryable(exc) or _status_code(exc) == 404

# 4 lần thử trên MỖI model, chờ tăng dần tới tối đa 30s trước khi chuyển
# sang model dự phòng tiếp theo (reraise=True để giữ nguyên loại lỗi gốc,
# dễ đọc log hơn khi in ra ở fallback loop).
_retry_single_model = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


@_retry_single_model
def _generate_once(model_name: str, system_prompt: str, user_prompt: str,
                    max_tokens: int, response_mime_type: str | None = None) -> str:
    config_kwargs = dict(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        # QUAN TRỌNG: các model Gemini 2.5+/3.x có "thinking" (suy luận nội bộ)
        # bật mặc định, tiêu tốn một phần max_output_tokens cho việc suy nghĩ
        # trước khi sinh câu trả lời — với các tác vụ ngắn gọn (JSON có cấu
        # trúc như planner/verifier), điều này dễ khiến output bị CẮT CỤT giữa
        # chừng trước khi JSON kịp đóng ngoặc (lỗi "Unterminated string").
        # Tắt hẳn thinking cho các lệnh gọi này vì không cần suy luận sâu.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    if response_mime_type:
        config_kwargs["response_mime_type"] = response_mime_type

    response = _get_client().models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text or ""


def _generate_with_fallback(system_prompt: str, user_prompt: str, max_tokens: int,
                             response_mime_type: str | None = None) -> str:
    last_error: Exception | None = None

    for i, model_name in enumerate(FALLBACK_MODELS):
        try:
            return _generate_once(model_name, system_prompt, user_prompt, max_tokens, response_mime_type)
        except Exception as e:
            last_error = e
            is_last = i == len(FALLBACK_MODELS) - 1
            if not _can_fallback(e) or is_last:
                raise
            LOGGER.warning(
                "Model %s lỗi (%s); chuyển sang model dự phòng %s",
                model_name,
                type(e).__name__,
                FALLBACK_MODELS[i + 1],
            )

    # Hết cả danh sách fallback mà vẫn lỗi -> báo lỗi thật, không giả vờ thành công
    if last_error is not None:
        raise last_error
    raise RuntimeError("Chưa cấu hình model Gemini nào.")


@observe(name="call_llm", as_type="generation")
def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    return _generate_with_fallback(system_prompt, user_prompt, max_tokens)


@observe(name="call_llm_json", as_type="generation")
def call_llm_json(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> dict:
    """Gọi LLM và ép trả về JSON hợp lệ bằng response_mime_type của Gemini
    (đáng tin cậy hơn tự parse chuỗi bằng tay).

    Nếu JSON vẫn bị cắt cụt (ví dụ model fallback không tắt được thinking),
    tự động thử lại 1 lần với max_tokens gấp đôi trước khi báo lỗi thật.
    """
    for attempt_tokens in (max_tokens, max_tokens * 2):
        raw = _generate_with_fallback(system_prompt, user_prompt, attempt_tokens, response_mime_type="application/json")
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("JSON trả về phải là object")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            last_raw = raw
    raise ValueError(f"LLM không trả JSON hợp lệ sau khi tăng max_tokens: {last_raw[:300]}") from last_error
