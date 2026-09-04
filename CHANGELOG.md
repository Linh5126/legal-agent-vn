# Changelog

## 0.3.1

- Giữ các Điều/Khoản lặp hợp lệ trong văn bản sửa đổi bằng `occ_N` thay vì
  chặn ingest toàn corpus vì `chunk_id` trùng.
- Vẫn loại các dòng mục lục ngắn và giữ kiểm tra uniqueness trước khi index.
- Retrieval evaluation ghi thêm độ trễ trung bình của từng cấu hình để phục
  vụ bảng ablation CS419.
- Notebook Colab bao phủ toàn bộ quy trình tải dữ liệu, ingest, backup/restore,
  retrieval evaluation, end-to-end evaluation và demo Gradio.

## 0.3.0 — 2026-09-03

- Harden citation parsing/verification, prompt boundaries và sample-data isolation.
- Chuyển BM25 index từ pickle sang versioned JSON gzip.
- Batch embedding/upsert; kiểm tra index consistency; lazy Qdrant lifecycle.
- Validate query, iteration, tool inputs và LLM judge outputs.
- Chỉ retry lỗi Gemini tạm thời; giữ fallback model cho lỗi phù hợp.
- Sửa Gradio 6 UI, error handling, source links và local-only defaults.
- Sửa evaluation collection/output handling và empty-dataset guards.
- Thêm README, setup checker, dependency profiles, pyproject và regression tests.
- Làm sạch notebook Colab (outputs/user metadata) và cập nhật workflow index mới.
