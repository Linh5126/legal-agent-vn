# Báo cáo đánh giá và cải tiến Legal Agent VN

Ngày rà soát: 2026-09-03  
Phiên bản sau cải tiến: 0.3.0

## Kết luận

Dự án có kiến trúc RAG nhiều bước hợp lý cho đồ án/demo: tách planner, retrieval, tool, verifier và finalize; có hybrid search, test và evaluation. Bản ban đầu đã có 22 unit test chạy qua, nhưng chưa đạt trạng thái “clone/cài/chạy” ổn định và có một số rủi ro đáng kể đối với sản phẩm pháp lý.

Bản 0.3.0 đã củng cố khả năng vận hành, an toàn nguồn/citation, quản lý index, UI và tài liệu. Dự án hiện phù hợp để tiếp tục làm **prototype có kiểm soát**. Chưa nên gọi là production legal assistant cho đến khi có corpus thật được quản trị hiệu lực, kiểm thử chuyên gia và lớp bảo mật triển khai.

## Chấm điểm

| Hạng mục | Ban đầu | Sau cải tiến | Nhận xét |
|---|---:|---:|---|
| Kiến trúc và phân tách module | 7.5/10 | 8.5/10 | Luồng agent rõ; bổ sung lazy resource, giữ evidence qua vòng retry |
| Correctness và chống lỗi | 6.0/10 | 8.2/10 | Validate input/output, citation exact, batch/index consistency |
| Bảo mật và legal safety | 5.0/10 | 8.0/10 | Không public UI mặc định, bỏ pickle load, guard prompt/citation/sample |
| Khả năng cài đặt/vận hành | 3.5/10 | 8.0/10 | Có README, dependency profiles, setup checker, CLI lỗi rõ ràng |
| Test và đánh giá | 6.5/10 | 8.0/10 | 39 test; sửa collection evaluation; thêm regression cho lỗi mới |
| Production readiness | 3.0/10 | 5.5/10 | Vẫn thiếu corpus governance, auth, Qdrant server và legal review |

Điểm tổng hợp tham khảo: **5.4/10 → 7.7/10**. Điểm production bị giới hạn có chủ đích vì chất lượng pháp lý không thể được chứng minh chỉ bằng unit test kỹ thuật.

## Phát hiện chính và cách xử lý

### Mức nghiêm trọng cao

1. **BM25 dùng `pickle.load`.** File index bị thay thế có thể thực thi mã khi load. Đã chuyển sang JSON gzip có `schema_version`, validation độ dài, ID và payload. Bản cũ cần re-ingest.
2. **UI bật `share=True` và `debug=True` mặc định.** Có nguy cơ công khai endpoint/câu hỏi pháp lý và log lỗi. Đã đổi sang localhost, không share/debug; cấu hình qua `.env`.
3. **Dữ liệu mẫu được gắn “hiệu lực”.** Có thể được ưu tiên và dùng nhầm làm căn cứ. Đã đánh dấu `is_sample`, loại khỏi ingest mặc định và loại tiếp ở retriever.
4. **Dataset snapshot bị gắn toàn bộ là “hiệu lực”.** Snapshot không chứng minh trạng thái hiện tại. Downloader giờ ghi `unknown` và yêu cầu bước xác minh riêng.
5. **Citation dùng substring.** `Điều 3` có thể match nhầm `Điều 35`; citation không có trong context vẫn có thể lọt qua LLM judge. Đã parse chính xác marker, hỗ trợ Điểm/Khoản, đối chiếu tất định và dedupe các chunk cùng Điều.

### Mức nghiêm trọng trung bình

1. Vector store mở ở import-time, dễ giữ lock Qdrant và làm test/import thất bại. Đã lazy-load và có preflight.
2. Ingest giữ toàn bộ vector trong RAM. Đã embed/upsert theo batch và kiểm số point sau reset.
3. Incremental Qdrant upsert có thể làm BM25 lệch collection. Đã chặn rõ cho đến khi có thiết kế incremental đồng bộ.
4. RRF cộng điểm lặp nếu một list chứa duplicate ID. Đã dedupe theo từng list.
5. BM25 trả cả kết quả điểm 0. Đã lọc để tránh đưa context hoàn toàn không khớp.
6. Retry LLM áp dụng cả lỗi thiếu key/401/400. Đã chỉ retry lỗi tạm thời; 404 chuyển model ngay.
7. Verifier tin kiểu dữ liệu LLM (`"false"` có thể truthy, score có thể ngoài 0..1). Đã strict bool, ép/clamp score và thêm deterministic gate.
8. Evaluation retrieval mặc định dùng collection không nhất quán với testset. Đã cho phép chọn `--collection`, sửa default và hướng dẫn benchmark riêng.
9. Gradio không có trong requirements và API `Chatbot(type=...)` không tương thích Gradio 6. Đã khai báo/pin dải tương thích và sửa UI.
10. Đường dẫn `.env`/data phụ thuộc current working directory. Đã resolve từ project root.

### Chất lượng dự án

- Bổ sung README tiếng Việt, sơ đồ kiến trúc, hướng dẫn Windows/Linux/Colab, dữ liệu, test và migration.
- Tách runtime/dev/data/evaluation/observability dependencies.
- Bổ sung `pyproject.toml`, cấu hình pytest/ruff và Python 3.11–3.12.
- Thêm `scripts.check_setup` để kiểm key, corpus và index mà không tải model.
- Notebook Colab đã xóa output cũ, user metadata và cập nhật `.json.gz`/collection arguments.
- Bổ sung source URL/status ở citation detail để UI có thể mở nguồn.

## Bằng chứng kiểm tra

| Kiểm tra | Kết quả |
|---|---|
| Python compileall | Qua |
| Ruff lint | Qua, 0 lỗi |
| Pytest | **43 passed** |
| Corpus audit trên dữ liệu đi kèm | 1 sample hợp lệ về cấu trúc, 0 văn bản thật |
| Import/build Gradio UI | Qua với Gradio 6.26 |
| Secret scan | Không phát hiện API key hardcode |

Chưa chạy end-to-end generation/retrieval với model thật vì gói đính kèm không có `GEMINI_API_KEY`, không có corpus pháp luật thật và không có index đã ingest hợp lệ. Đây là giới hạn dữ liệu/cấu hình, không nên che giấu bằng mock khi đánh giá chất lượng pháp lý.

## Việc bắt buộc trước buổi demo

1. Thêm corpus thật, mỗi văn bản có URL cụ thể, ngày ban hành, trạng thái và ngày kiểm tra trạng thái.
2. Điền key trong `.env`, chạy `python -m scripts.audit_corpus`, rồi ingest lại.
3. Chạy `python -m scripts.check_setup`; chỉ demo khi mọi mục đều `OK`.
4. Chạy retrieval benchmark trên đúng collection và lưu kết quả có timestamp/model version.
5. Chuẩn bị tối thiểu 30–50 câu hỏi do người hiểu pháp luật gán nhãn, gồm câu ngoài phạm vi và văn bản hết hiệu lực.

## Roadmap production đề xuất

### P0 — Legal correctness

- Registry văn bản có `valid_from`, `valid_to`, `status_verified_at`, văn bản sửa đổi/thay thế/bãi bỏ và bản hợp nhất.
- Lưu hash/snapshot nội dung để câu trả lời có thể audit theo đúng phiên bản nguồn.
- Ground-truth set do chuyên gia duyệt; regression gate trước mỗi lần đổi model/corpus.

### P1 — Security và vận hành

- Xác thực, phân quyền theo corpus/tenant, rate limit, quota và audit log.
- PII/secrets redaction trước khi gọi LLM; chính sách retention và xóa dữ liệu.
- Qdrant server/Cloud có authentication, TLS, backup; không dùng local mode cho nhiều process/corpus lớn.
- Timeout/circuit breaker/metrics cho Gemini, embedding, reranker và Qdrant.

### P2 — Hiệu năng và sản phẩm

- Cache embedding/query, batching async và streaming UI.
- Phân loại lĩnh vực/phạm vi trước retrieval; metadata filter theo thời điểm, loại văn bản và thẩm quyền.
- Hiển thị trích đoạn được dùng, đường dẫn nguồn, ngày kiểm tra hiệu lực và lý do từ chối.
- So sánh chi phí/latency/accuracy giữa các model và reranker nhỏ hơn.
