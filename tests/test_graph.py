import pytest

from agents.graph import run_query
from config import settings


def test_run_query_rejects_blank_input_before_runtime_calls():
    with pytest.raises(ValueError, match="để trống"):
        run_query("   ")


def test_run_query_rejects_oversized_input_before_runtime_calls():
    with pytest.raises(ValueError, match="quá dài"):
        run_query("x" * (settings.max_query_chars + 1))


def test_run_query_rejects_invalid_iteration_limit():
    with pytest.raises(ValueError, match="1..5"):
        run_query("Câu hỏi", max_iterations=0)


def test_run_query_reports_missing_runtime_requirements():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        run_query("Câu hỏi hợp lệ")
