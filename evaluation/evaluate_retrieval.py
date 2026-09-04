"""Retrieval evaluation with ablation study.

Measures each stage independently: Dense, BM25, RRF and RRF+CrossEncoder.
This prevents a single end-to-end score from hiding which component helps.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from retrieval.vector_store import VietnameseLegalVectorStore

MODES = ["dense", "bm25", "rrf", "rerank"]


def recall_at_k(ids, relevant, k):
    if not relevant:
        return math.nan
    return len(set(ids[:k]) & set(relevant)) / len(set(relevant))


def hit_at_k(ids, relevant, k):
    if not relevant:
        return math.nan
    return float(bool(set(ids[:k]) & set(relevant)))


def reciprocal_rank(ids, relevant):
    relevant = set(relevant)
    if not relevant:
        return math.nan
    for i, cid in enumerate(ids, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def metrics(ids, relevant):
    return {
        "recall@1": recall_at_k(ids, relevant, 1),
        "recall@5": recall_at_k(ids, relevant, 5),
        "recall@10": recall_at_k(ids, relevant, 10),
        "recall@30": recall_at_k(ids, relevant, 30),
        "hit@5": hit_at_k(ids, relevant, 5),
        "hit@10": hit_at_k(ids, relevant, 10),
        "mrr": reciprocal_rank(ids, relevant),
    }


def evaluate_retrieval(testset_path, sample_size=300, modes=None, collection_name=None,
                       output_path="evaluation/retrieval_ablation_results.json"):
    data = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    labeled = [x for x in data if x.get("relevant_chunk_ids")]
    if sample_size and len(labeled) > sample_size:
        random.seed(42)
        labeled = random.sample(labeled, sample_size)
    modes = modes or MODES
    if not labeled:
        raise ValueError("Testset không có relevant_chunk_ids để đánh giá retrieval.")

    store = VietnameseLegalVectorStore(collection_name=collection_name or settings.collection_name)
    if not store.is_ready():
        raise RuntimeError("Collection/index chưa sẵn sàng; hãy ingest corpus trước khi evaluate.")
    results_by_mode = {m: [] for m in modes}
    t0 = time.time()

    for i, item in enumerate(labeled, 1):
        for mode in modes:
            mode_started = time.perf_counter()
            results = store.search(
                item["question"], mode=mode, top_k=30, candidate_k=30, rerank_top_k=30, prefer_effective=False
            )
            latency_ms = (time.perf_counter() - mode_started) * 1000
            ids = [r["chunk_id"] for r in results]
            row = {"id": item.get("id"), "question": item["question"], **metrics(ids, item["relevant_chunk_ids"]),
                   "retrieved_ids": ids[:30], "latency_ms": latency_ms}
            results_by_mode[mode].append(row)
        if i % 10 == 0:
            print(f"[{i}/{len(labeled)}]")

    summary = {}
    keys = ["recall@1","recall@5","recall@10","recall@30","hit@5","hit@10","mrr"]
    for mode, rows in results_by_mode.items():
        summary[mode] = {}
        for key in keys:
            vals = [r[key] for r in rows if not math.isnan(r[key])]
            summary[mode][key] = mean(vals) if vals else math.nan
        summary[mode]["avg_latency_ms"] = mean(r["latency_ms"] for r in rows)

    out = {"summary": summary, "n": len(labeled), "seed": 42, "results": results_by_mode}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"RETRIEVAL ABLATION | n={len(labeled)} | {time.time()-t0:.1f}s")
    for mode in modes:
        s = summary[mode]
        print(
            f"{mode:8s} | R@5={s['recall@5']:.4f} | "
            f"R@10={s['recall@10']:.4f} | MRR={s['mrr']:.4f} | "
            f"latency={s['avg_latency_ms']:.1f}ms"
        )
    print(f"Saved: {path}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="data/testset/sample_testset.json")
    ap.add_argument("--sample_size", type=int, default=300)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    ap.add_argument("--collection", default=None)
    ap.add_argument("--output", default="evaluation/retrieval_ablation_results.json")
    args = ap.parse_args()
    evaluate_retrieval(args.testset, args.sample_size, args.modes, args.collection, args.output)
