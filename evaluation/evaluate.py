"""End-to-end evaluation for the legal agent.

Keeps retrieval correctness separate from generation faithfulness and measures
latency, sufficiency, citation coverage, and abstention behavior.
"""
import json
import math
import time
from pathlib import Path
from statistics import mean

from agents.citation_utils import citation_coverage
from agents.graph import run_query


def load_testset(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def retrieval_recall_at_k(docs, relevant, k=8):
    if not relevant:
        return math.nan
    ids = {d["chunk_id"] for d in docs[:k]}
    return len(ids & set(relevant)) / len(set(relevant))


def run_benchmark(testset_path, output_path="evaluation/results_final.json"):
    testset = load_testset(testset_path)
    if not testset:
        raise ValueError("Testset rỗng; không thể tính metric.")
    rows = []
    for item in testset:
        t0 = time.perf_counter()
        result = run_query(item["question"])
        latency = time.perf_counter() - t0
        docs = result.get("retrieved_docs", [])
        answer = result.get("final_answer", "")
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "type": item.get("type", "unknown"),
            "ground_truth_answer": item.get("ground_truth_answer", ""),
            "final_answer": answer,
            "retrieved_docs": docs,
            "contexts": [d["content"] for d in docs],
            "citations": result.get("citations", []),
            "retrieval_recall@5": retrieval_recall_at_k(docs, item.get("relevant_chunk_ids", []), 5),
            "retrieval_recall@8": retrieval_recall_at_k(docs, item.get("relevant_chunk_ids", []), 8),
            "faithfulness_score": result.get("verification", {}).get("faithfulness_score", 0.0),
            "is_sufficient": result.get("verification", {}).get("is_sufficient", False),
            "missing_info": result.get("verification", {}).get("missing_info", ""),
            "iterations_used": result.get("iteration", 0),
            "citation_coverage": citation_coverage(answer, docs),
            "latency_sec": round(latency, 3),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(rows)
    print(f"Saved: {output_path}")
    return rows


def _print_summary(rows):
    labeled = [r for r in rows if not math.isnan(r["retrieval_recall@8"])]
    print("=" * 64)
    print(f"END-TO-END | n={len(rows)}")
    if labeled:
        print(f"Recall@5:            {mean(r['retrieval_recall@5'] for r in labeled):.4f}")
        print(f"Recall@8:            {mean(r['retrieval_recall@8'] for r in labeled):.4f}")
    print(f"Faithfulness judge:   {mean(r['faithfulness_score'] for r in rows):.4f}")
    print(f"Sufficient rate:      {mean(float(r['is_sufficient']) for r in rows):.2%}")
    print(f"Citation coverage:    {mean(r['citation_coverage'] for r in rows):.2%}")
    print(f"Avg iterations:       {mean(r['iterations_used'] for r in rows):.2f}")
    print(f"Avg latency:          {mean(r['latency_sec'] for r in rows):.2f}s")


def run_ragas_eval(results_path="evaluation/results_final.json"):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("ground_truth_answer") and r.get("contexts")]
    if not rows:
        raise ValueError("Không có sample đủ ground truth + context để chạy RAGAS.")
    ds = Dataset.from_list([
        {"question": r["question"], "answer": r["final_answer"],
         "contexts": r["contexts"], "reference": r["ground_truth_answer"]}
        for r in rows
    ])
    scores = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    print(scores)
    return scores


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("testset", nargs="?", default="data/testset/sample_testset.json")
    ap.add_argument("--output", default="evaluation/results_final.json")
    args = ap.parse_args()
    run_benchmark(args.testset, args.output)
