from unittest.mock import patch

from agents import tool_agent


def _state(query="Lương 10 triệu đóng BHXH bao nhiêu?"):
    return {
        "original_query": query,
        "retrieved_docs": [],
    }


def test_guardrail_blocks_tool_not_in_registry():
    """LLM (hoặc kẻ tấn công qua prompt injection) đòi gọi tool không tồn tại
    trong TOOL_REGISTRY -> phải bị bỏ qua hoàn toàn, KHÔNG thực thi."""
    fake_decision = {"tool_calls": [{"tool_name": "delete_all_files", "tool_input": {}}]}
    with patch.object(tool_agent, "call_llm_json", return_value=fake_decision):
        results = tool_agent._decide_and_call_tools(_state())
    assert results == []


def test_guardrail_caps_number_of_tool_calls(monkeypatch):
    """Dù LLM đòi gọi nhiều tool hợp lệ hơn giới hạn, chỉ tối đa
    settings.max_tool_calls_per_turn cuộc gọi được thực thi."""
    from config import settings
    monkeypatch.setattr(settings, "max_tool_calls_per_turn", 1)

    fake_decision = {"tool_calls": [
        {"tool_name": "calculate_social_insurance", "tool_input": {"monthly_salary": 10_000_000}},
        {"tool_name": "calculate_social_insurance", "tool_input": {"monthly_salary": 20_000_000}},
    ]}
    with patch.object(tool_agent, "call_llm_json", return_value=fake_decision):
        results = tool_agent._decide_and_call_tools(_state())
    assert len(results) == 1


def test_invalid_tool_input_does_not_crash_pipeline():
    """Tool tồn tại nhưng tham số sai kiểu -> lỗi phải được bắt và trả về
    dưới dạng tool_output mô tả lỗi, không được để exception văng ra ngoài
    (một node LangGraph lỗi sẽ làm sập cả graph)."""
    fake_decision = {"tool_calls": [
        {"tool_name": "calculate_social_insurance", "tool_input": {"wrong_param": 1}},
    ]}
    with patch.object(tool_agent, "call_llm_json", return_value=fake_decision):
        results = tool_agent._decide_and_call_tools(_state())
    assert len(results) == 1
    assert "Lỗi gọi tool" in results[0]["tool_output"]
