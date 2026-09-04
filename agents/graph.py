"""
Dựng đồ thị LangGraph theo đúng kiến trúc trong problem statement:

    START -> planner -> retriever -> tool_agent -> verifier
                ^                                      |
                |______________ retry ________________ |
                                                         |
                                                    finalize -> END

LangGraph được chọn (thay vì CrewAI) chính vì cần conditional edge tạo vòng
lặp có điều kiện này — CrewAI xử lý workflow tuyến tính/role-based tốt hơn,
nhưng kém linh hoạt hơn cho vòng lặp "chưa đủ căn cứ thì tra lại".
"""
from langgraph.graph import END, StateGraph

from agents.finalize import finalize_node
from agents.observability import observe
from agents.planner import planner_node
from agents.retriever_agent import get_store, retriever_node
from agents.state import AgentState
from agents.tool_agent import tool_agent_node
from agents.verifier import should_continue_loop, verifier_node


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("tool_agent", tool_agent_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "tool_agent")
    graph.add_edge("tool_agent", "verifier")

    # Conditional edge — đây chính là vòng lặp tự kiểm tra trong sơ đồ
    graph.add_conditional_edges(
        "verifier",
        should_continue_loop,
        {
            "retry": "planner",
            "finalize": "finalize",
        },
    )

    graph.add_edge("finalize", END)

    # recursion_limit đề phòng loop bất thường; max_iterations trong state đã
    # tự giới hạn số vòng, đây là lớp an toàn thứ hai ở tầng graph
    return graph.compile()


@observe(name="run_query", as_type="agent")
def run_query(query: str, max_iterations: int | None = None) -> dict:
    from config import settings

    query = (query or "").strip()
    if not query:
        raise ValueError("Câu hỏi không được để trống.")
    if len(query) > settings.max_query_chars:
        raise ValueError(
            f"Câu hỏi quá dài ({len(query)} ký tự); giới hạn là {settings.max_query_chars}."
        )
    if max_iterations is None:
        max_iterations = settings.max_verification_loops
    if not 1 <= max_iterations <= 5:
        raise ValueError("max_iterations phải nằm trong khoảng 1..5.")

    issues = []
    if not settings.gemini_api_key:
        issues.append("thiếu GEMINI_API_KEY")
    if not get_store().is_ready():
        issues.append("chưa có Qdrant/BM25 index hợp lệ (chạy `python -m scripts.ingest_data`)")
    if issues:
        raise RuntimeError("Cấu hình chưa sẵn sàng: " + "; ".join(issues) + ".")

    app = build_graph()
    initial_state = {
        "original_query": query,
        "sub_questions": [],
        "planner_feedback": "",
        "retrieved_docs": [],
        "tool_results": [],
        "draft_answer": "",
        "verification": {"is_sufficient": False, "faithfulness_score": 0.0, "missing_info": "", "reasoning": ""},
        "iteration": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
        "citations": [],
        "citation_details": [],
    }
    result = app.invoke(initial_state, config={"recursion_limit": 25})
    return result
