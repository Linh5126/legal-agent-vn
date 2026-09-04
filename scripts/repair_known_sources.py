"""Repair known upstream corpus issues from authoritative sources.

Currently fixes UTS_VLC record 45/2019/QH14 using the official Government PDF.
Use only after reviewing the quarantine/audit report.
"""
from scripts.download_hf_datasets import repair_known_labor_code

if __name__ == "__main__":
    repair_known_labor_code()
