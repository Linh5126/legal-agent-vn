"""Bootstrap tracing Langfuse cho toàn bộ agent.

TRƯỚC ĐÂY: langfuse_public_key/secret_key có trong config.py/.env.example và
langfuse nằm trong requirements.txt, nhưng KHÔNG có một lệnh gọi Langfuse
nào trong toàn bộ codebase — quan sát được (observability) chỉ là trang trí,
không hoạt động thật. Module này implement tracing thật, nhưng tuyệt đối
không được là hard dependency: nếu người dùng chưa cấu hình Langfuse key
(trường hợp mặc định), hệ thống phải chạy y hệt như không có module này.

Langfuse SDK (>=4.x) đọc credentials từ biến môi trường process
(LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST) khi client tự khởi tạo lần đầu, không
đọc trực tiếp từ pydantic Settings của ta — nên cần "bắc cầu" ở đây để .env
vẫn là nguồn cấu hình duy nhất.
"""
import os

from config import settings

_have_keys = bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _noop_observe(*args, **kwargs):
    """Decorator giữ nguyên chữ ký của langfuse.observe nhưng không làm gì —
    dùng khi Langfuse chưa được cấu hình hoặc chưa cài đặt, để agents/* không
    cần biết (và không cần import có điều kiện) tracing có bật hay không.
    """
    def _decorator(func):
        return func
    if len(args) == 1 and callable(args[0]) and not kwargs:
        return args[0]
    return _decorator


if _have_keys:
    os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
    try:
        from langfuse import observe
    except ImportError:
        print(
            "[observability] Đã cấu hình LANGFUSE_*_KEY nhưng chưa cài package "
            "`langfuse` (pip install -r requirements.txt) — bỏ qua tracing.",
            flush=True,
        )
        observe = _noop_observe
else:
    observe = _noop_observe
