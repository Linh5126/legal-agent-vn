"""
Node cuối cùng của graph: chốt câu trả lời để trả về người dùng.

Nếu verifier xác nhận đủ căn cứ -> dùng draft_answer làm final_answer.
Nếu hết số vòng lặp cho phép mà VẪN chưa đủ căn cứ -> KHÔNG được im lặng trả
lời như thể chắc chắn; phải thêm cảnh báo rõ ràng để người dùng biết giới hạn
của câu trả lời (đúng nguyên tắc "biết từ chối khi không chắc" đã đặt ra trong
problem statement).
"""
from agents.citation_utils import extract_cited_doc_records, extract_cited_docs
from agents.observability import observe
from agents.state import AgentState


@observe(name="finalize_node")
def finalize_node(state: AgentState) -> dict:
    verification = state["verification"]
    # V2 fix: chỉ liệt kê nguồn THỰC SỰ xuất hiện trong draft_answer, không
    # phải toàn bộ retrieved_docs (xem agents/citation_utils.py để biết lý
    # do — đã xác nhận bằng bằng chứng chạy thật trên main.py).
    citations = extract_cited_docs(state["draft_answer"], state["retrieved_docs"])
    citation_details = [
        {
            "label": f"Điều {doc['article_number']}, {doc['doc_title']}",
            "doc_id": doc.get("doc_id", ""),
            "source_url": doc.get("source_url"),
            "effective_status": doc.get("effective_status", "unknown"),
        }
        for doc in extract_cited_doc_records(state["draft_answer"], state["retrieved_docs"])
    ]

    final_answer = state["draft_answer"]

    if not verification["is_sufficient"]:
        final_answer += (
            "\n\n⚠️ Lưu ý: hệ thống chưa tìm đủ căn cứ pháp lý chắc chắn cho "
            f"toàn bộ câu hỏi sau {state['iteration']} lượt tra cứu "
            f"(thiếu: {verification['missing_info'] or 'không xác định rõ'}). "
            "Khuyến nghị tham khảo thêm nguồn chính thức hoặc chuyên gia pháp lý."
        )

    return {
        "final_answer": final_answer,
        "citations": citations,
        "citation_details": citation_details,
    }
