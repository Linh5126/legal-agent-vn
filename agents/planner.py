"""Planner V2: tạo query đa góc + ưu tiên thuật ngữ pháp lý."""
from agents.llm_client import call_llm_json
from agents.observability import observe
from agents.state import AgentState
from config import settings

SYSTEM_PROMPT = """Bạn là planner cho hệ thống RAG pháp luật Việt Nam.
Hãy tạo 1-4 truy vấn tìm kiếm bổ sung, không trả lời câu hỏi.
Ưu tiên giữ nguyên các thực thể pháp lý: người lao động, loại hợp đồng,
điều kiện, quyền/nghĩa vụ, Điều luật, luật/nghị định, thời điểm hiệu lực.
Nếu có feedback từ verifier, tạo ít nhất một query nhắm trực tiếp vào phần thiếu.
Nếu câu hỏi có thể liên quan đến hiệu lực/sửa đổi/thay thế, tạo thêm query về
văn bản hiện hành và quan hệ sửa đổi.
Chỉ JSON: {"sub_questions": ["..."]}
"""


@observe(name="planner_node", as_type="chain")
def planner_node(state: AgentState) -> dict:
    feedback = state.get("planner_feedback", "")
    prompt = f"Câu hỏi gốc: {state['original_query']}"
    if feedback:
        prompt += f"\nThông tin verifier nói còn thiếu: {feedback}"
    result = call_llm_json(SYSTEM_PROMPT, prompt, max_tokens=600)
    qs = result.get("sub_questions")
    if not isinstance(qs, list):
        qs = []
    qs = [
        q.strip()
        for q in qs
        if isinstance(q, str) and q.strip() and len(q.strip()) <= settings.max_query_chars
    ]
    # deterministic fallback: query gốc luôn phải được retrieval
    if state["original_query"] not in qs:
        qs = [state["original_query"], *qs]
    # dedupe + cap
    dedup = list(dict.fromkeys(qs))
    return {"sub_questions": dedup[:4]}
