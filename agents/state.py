"""
State được truyền xuyên suốt graph. Mỗi node đọc/ghi vào state này.
Dùng TypedDict theo đúng convention của LangGraph.
"""
from typing import TypedDict


class RetrievedDoc(TypedDict):
    chunk_id: str
    doc_title: str
    article_number: str
    content: str
    effective_status: str
    rerank_score: float


class ToolCallResult(TypedDict):
    tool_name: str
    tool_input: dict
    tool_output: str


class VerificationResult(TypedDict):
    is_sufficient: bool
    faithfulness_score: float  # 0-1, câu trả lời bám sát nguồn tới đâu
    missing_info: str          # mô tả thông tin còn thiếu, dùng để re-plan
    reasoning: str


class CitationDetail(TypedDict):
    label: str
    doc_id: str
    source_url: str | None
    effective_status: str


class AgentState(TypedDict):
    # input
    original_query: str

    # planner output
    sub_questions: list[str]
    planner_feedback: str  # feedback từ verifier ở vòng lặp trước (nếu có)

    # retriever output
    retrieved_docs: list[RetrievedDoc]

    # tool-calling output
    tool_results: list[ToolCallResult]

    # draft answer trước khi verify
    draft_answer: str

    # verifier output
    verification: VerificationResult

    # điều khiển vòng lặp
    iteration: int
    max_iterations: int

    # kết quả cuối
    final_answer: str
    citations: list[str]
    citation_details: list[CitationDetail]
