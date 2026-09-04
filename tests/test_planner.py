from unittest.mock import patch

from agents.planner import planner_node


def test_planner_falls_back_when_llm_returns_wrong_shape():
    state = {"original_query": "Câu hỏi gốc", "planner_feedback": ""}
    with patch("agents.planner.call_llm_json", return_value={"sub_questions": "không phải list"}):
        assert planner_node(state)["sub_questions"] == ["Câu hỏi gốc"]


def test_planner_deduplicates_and_caps_queries():
    state = {"original_query": "Q0", "planner_feedback": ""}
    generated = {"sub_questions": ["Q1", "Q1", "Q2", "Q3", "Q4"]}
    with patch("agents.planner.call_llm_json", return_value=generated):
        assert planner_node(state)["sub_questions"] == ["Q0", "Q1", "Q2", "Q3"]

