"""
Giao diện demo dạng "sản phẩm" cho hệ thống multi-agent — thay vì gõ
`!python main.py "..."`, chạy file này để có giao diện web thật.

CÁCH CHẠY TRÊN COLAB (khuyến nghị — không phụ thuộc Gradio share tunnel,
tunnel này hay bị chặn mạng dẫn tới lỗi "Could not create share link"):

    Chạy TRỰC TIẾP trong 1 cell Colab (KHÔNG dùng !python app.py):

        !pip install -q gradio
        from app import demo
        demo.launch(share=False, prevent_thread_lock=True)

        import time; time.sleep(2)
        from google.colab.output import eval_js
        print(eval_js("google.colab.kernel.proxyPort(7860)"))

    Bấm vào URL in ra (dạng https://xxxxx-7860-colab.googleusercontent.com/)
    — đây là proxy port CỦA CHÍNH COLAB, không phụ thuộc server bên ngoài
    của Gradio nên ổn định hơn nhiều trong môi trường Colab.

CÁCH CHẠY TRÊN MÁY CÁ NHÂN (không phải Colab):
    pip install -r requirements.txt
    python app.py
    Mặc định chỉ mở localhost. Chỉ đặt APP_SHARE=true khi thật sự cần và đã
    hiểu rủi ro công khai endpoint.

Thiết kế UI cố tình phơi bày TOÀN BỘ pipeline (sub-questions, nguồn trích dẫn,
điểm faithfulness, số vòng lặp) thay vì chỉ hiện câu trả lời cuối — vì đây là
đồ án minh họa kiến trúc multi-agent, "hộp trắng" có sức thuyết phục hơn nhiều
so với chatbot "hộp đen" thông thường khi trình bày trước hội đồng.
"""
import logging
import time
from urllib.parse import urlparse

import gradio as gr

from agents.graph import run_query
from config import settings

LOGGER = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = [
    "Người lao động nữ mang thai có được đơn phương chấm dứt hợp đồng không?",
    "Thời gian báo trước khi đơn phương chấm dứt hợp đồng lao động là bao lâu?",
    "Viên chức nữ có thai có quyền gì khi phải nghỉ việc theo chỉ định của bác sĩ?",
]


def answer_question(question: str, history: list):
    """Chạy toàn bộ graph, trả về câu trả lời + metadata để hiển thị chi tiết."""
    if not question or not question.strip():
        return history, "", "", ""

    start = time.time()
    try:
        result = run_query(question)
    except Exception:
        LOGGER.exception("Không thể xử lý câu hỏi")
        error_message = (
            "Hệ thống chưa thể xử lý câu hỏi. Hãy kiểm tra GEMINI_API_KEY, "
            "chạy bước ingest và xem log máy chủ."
        )
        history = [
            *(history or []),
            {"role": "user", "content": question},
            {"role": "assistant", "content": error_message},
        ]
        return history, "", "_Không có nguồn trích dẫn._", "⚠️ Pipeline gặp lỗi cấu hình hoặc runtime."
    elapsed = time.time() - start

    # --- Câu trả lời hiển thị trong khung chat ---
    history = [
        *(history or []),
        {"role": "user", "content": question},
        {"role": "assistant", "content": result["final_answer"]},
    ]

    # --- Panel "Nguồn trích dẫn" ---
    citations_md = _format_citations(result)

    # --- Panel "Chi tiết pipeline" (điểm khác biệt của demo "hộp trắng") ---
    verification = result["verification"]
    is_sufficient = "✅ Đủ căn cứ" if verification["is_sufficient"] else "⚠️ Chưa đủ căn cứ"
    detail_md = f"""\
**Số vòng lặp tra cứu:** {result['iteration']}
**Faithfulness score (verifier tự chấm):** {verification['faithfulness_score']:.2f}
**Kết luận verifier:** {is_sufficient}
**Thời gian xử lý:** {elapsed:.1f}s

**Lý do verifier (`reasoning`):**
> {verification['reasoning'] or '(không có)'}
"""
    if not verification["is_sufficient"]:
        detail_md += f"\n**Thông tin còn thiếu:**\n> {verification['missing_info']}"

    return history, "", citations_md, detail_md


def _format_citations(result: dict) -> str:
    lines = []
    for item in result.get("citation_details", []):
        label = str(item.get("label", "Nguồn")).replace("[", "\\[").replace("]", "\\]")
        source_url = item.get("source_url")
        parsed = urlparse(source_url) if isinstance(source_url, str) else None
        rendered = f"[{label}]({source_url})" if parsed and parsed.scheme in {"http", "https"} else label
        status = item.get("effective_status", "unknown")
        lines.append(f"- {rendered} — trạng thái: `{status}`")
    if not lines:
        lines = [f"- {c}" for c in result.get("citations", [])]
    return "\n".join(lines) or "_Không có nguồn trích dẫn._"


with gr.Blocks(title="Trợ lý pháp luật lao động VN — Demo Multi-Agent RAG") as demo:
    gr.Markdown(
        """
        # ⚖️ Trợ lý tra cứu pháp luật lao động Việt Nam
        **Demo kiến trúc Multi-Agent RAG** — Planner → Retriever (Hybrid Search) → Tool-calling → Verifier → Finalize

        ⚠️ *Đây là đồ án học thuật, không phải dịch vụ tư vấn pháp lý chính thức. Câu trả lời có thể chưa đầy đủ hoặc cần kiểm chứng thêm.*
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            # Gradio 6 dùng message dictionaries mặc định (tham số `type` đã bỏ).
            chatbot = gr.Chatbot(label="Hội thoại", height=450)
            question_box = gr.Textbox(
                label="Câu hỏi của bạn",
                placeholder="Nhập câu hỏi pháp luật lao động...",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Gửi câu hỏi", variant="primary")
                clear_btn = gr.Button("Xóa hội thoại")

            gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=question_box, label="Câu hỏi mẫu")

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Nguồn trích dẫn")
            citations_output = gr.Markdown("_Chưa có câu hỏi nào được gửi._")

            gr.Markdown("### 🔍 Chi tiết pipeline (verifier)")
            detail_output = gr.Markdown("_Chưa có câu hỏi nào được gửi._")

    submit_btn.click(
        answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box, citations_output, detail_output],
    )
    question_box.submit(
        answer_question,
        inputs=[question_box, chatbot],
        outputs=[chatbot, question_box, citations_output, detail_output],
    )
    clear_btn.click(lambda: ([], "", "", ""), outputs=[chatbot, question_box, citations_output, detail_output])

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        share=settings.app_share,
        debug=settings.app_debug,
        server_name=settings.app_server_name,
        server_port=settings.app_server_port,
    )
