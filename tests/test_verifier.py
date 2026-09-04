from unittest.mock import patch

from agents.verifier import should_continue_loop, verifier_node


def _state(is_sufficient, iteration, max_iterations=2):
    return {
        "verification": {"is_sufficient": is_sufficient, "faithfulness_score": 0.0,
                          "missing_info": "", "reasoning": ""},
        "iteration": iteration,
        "max_iterations": max_iterations,
    }


def test_finalize_when_sufficient_even_if_iterations_remain():
    assert should_continue_loop(_state(is_sufficient=True, iteration=1)) == "finalize"


def test_retry_when_insufficient_and_iterations_remain():
    assert should_continue_loop(_state(is_sufficient=False, iteration=1)) == "retry"


def test_finalize_when_max_iterations_reached_even_if_insufficient():
    """Đây chính là cơ chế 'biết từ chối khi không chắc': không được lặp vô
    hạn dù verifier vẫn nói chưa đủ căn cứ."""
    assert should_continue_loop(_state(is_sufficient=False, iteration=2, max_iterations=2)) == "finalize"


def test_finalize_when_iteration_exceeds_max():
    """Lớp an toàn: nếu vì lý do gì đó iteration > max_iterations, vẫn phải
    dừng chứ không được coi là điều kiện chưa khớp."""
    assert should_continue_loop(_state(is_sufficient=False, iteration=5, max_iterations=2)) == "finalize"


def test_verifier_rejects_hallucinated_citation_even_if_llm_accepts():
    state = {
        "original_query": "Hỏi luật",
        "draft_answer": "Có. [Điều 999, Văn bản bịa]",
        "retrieved_docs": [{
            "chunk_id": "x::dieu_1", "article_number": "1", "doc_title": "Luật thật",
            "content": "Điều 1. Nội dung", "effective_status": "hiệu lực",
        }],
        "tool_results": [],
        "iteration": 0,
    }
    fake = {"is_sufficient": True, "faithfulness_score": 1.0, "missing_info": "", "reasoning": "OK"}
    with patch("agents.verifier.call_llm_json", return_value=fake):
        result = verifier_node(state)
    assert result["verification"]["is_sufficient"] is False
    assert result["verification"]["faithfulness_score"] <= 0.3


def test_verifier_does_not_treat_string_false_as_true():
    state = {
        "original_query": "Hỏi luật",
        "draft_answer": "Chưa đủ căn cứ để khẳng định.",
        "retrieved_docs": [],
        "tool_results": [],
        "iteration": 0,
    }
    fake = {"is_sufficient": "false", "faithfulness_score": 2, "missing_info": None, "reasoning": None}
    with patch("agents.verifier.call_llm_json", return_value=fake):
        result = verifier_node(state)
    assert result["verification"]["is_sufficient"] is False
    assert result["verification"]["faithfulness_score"] == 1.0
