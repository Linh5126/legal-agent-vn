"""Audit raw corpus before ingest.

Outputs a machine-readable JSON report and human-readable summary. The audit
never silently fixes a bad file; it tells you exactly which source to repair or
quarantine.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingestion.loader import _read_documents, analyze_document


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--output", default="evaluation/corpus_audit.json")
    args = ap.parse_args()

    docs = _read_documents(args.raw_dir)
    reports = []
    for doc in docs:
        r = analyze_document(doc)
        reports.append(r)
        status = "OK" if r["valid"] else "BAD"
        print(
            f"{status:3s} {r['doc_id']:40s} | articles={r['articles']:4d} "
            f"| chars={r['chars']:8,d}"
            + (f" | issues={','.join(r['issues'])}" if r['issues'] else "")
            + (f" | warnings={','.join(r['warnings'])}" if r['warnings'] else "")
        )

    bad = [r for r in reports if not r["valid"]]
    out = {
        "documents": len(reports),
        "valid_documents": len(reports) - len(bad),
        "invalid_documents": len(bad),
        "total_articles": sum(r["articles"] for r in reports if r["valid"]),
        "reports": reports,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTOTAL documents={len(reports)}, valid={len(reports)-len(bad)}, bad={len(bad)}")
    print(f"Saved audit: {args.output}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
