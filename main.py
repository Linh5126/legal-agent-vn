"""
Chạy: python main.py "câu hỏi của bạn"

Ví dụ:
    python main.py "Người lao động nữ mang thai có được đơn phương chấm dứt hợp đồng không?"
"""
import argparse
import sys

from agents.graph import run_query


def main():
    parser = argparse.ArgumentParser(description="Trợ lý tra cứu pháp luật Việt Nam")
    parser.add_argument("query", help="câu hỏi pháp lý")
    parser.add_argument("--max-iterations", type=int, default=None, help="số vòng tra cứu (1-5)")
    args = parser.parse_args()

    query = args.query
    print(f"\nCâu hỏi: {query}\n{'-'*60}")

    try:
        result = run_query(query, max_iterations=args.max_iterations)
    except (ValueError, RuntimeError) as exc:
        print(f"\nKhông thể chạy: {exc}", file=sys.stderr)
        return 2

    print(f"\nSố vòng lặp tra cứu: {result['iteration']}")
    print(f"Faithfulness score: {result['verification']['faithfulness_score']}")
    print(f"\n=== CÂU TRẢ LỜI ===\n{result['final_answer']}")
    print("\n=== NGUỒN TRÍCH DẪN ===")
    for c in result["citations"]:
        print(f"  - {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
