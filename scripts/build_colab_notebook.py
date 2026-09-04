"""Build the self-contained Google Colab notebook shipped with the project."""

import json
from pathlib import Path
from textwrap import dedent


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(source).strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    markdown(
        """
        # Legal Agent VN — quy trình Google Colab đầy đủ

        Notebook này bao phủ toàn bộ quy trình của đồ án CS419:

        1. Cài môi trường và cấu hình Gemini API.
        2. Tải **toàn bộ** UTS_VLC và YuITC retrieval benchmark.
        3. Ingest `vn_legal_docs` và `vn_legal_retrieval_benchmark`.
        4. Lưu/khôi phục Qdrant + BM25 bằng một archive trên Google Drive.
        5. Đánh giá Dense, BM25, RRF, Rerank.
        6. Đánh giá End-to-End Legal Agent.
        7. Chạy giao diện Gradio qua proxy của Colab.

        **Trước khi chạy:** chọn `Runtime → Change runtime type → T4 GPU`, sau đó
        thêm Colab Secret tên chính xác `GEMINI_API_KEY` và bật quyền notebook.

        - **Lần đầu:** chạy mục A, sau đó lưu index.
        - **Từ lần sau:** chạy các cell chuẩn bị, bỏ qua mục A và chạy mục B để khôi phục.
        - Chỉ chạy demo Gradio sau khi evaluation hoàn tất; Qdrant local không nên bị
          mở đồng thời bởi nhiều process.
        """
    ),
    markdown("## 1. Mount Drive và đặt project vào Drive"),
    code(
        """
        from pathlib import Path
        import os
        import zipfile

        from google.colab import drive, files

        drive.mount("/content/drive")

        PROJECT_DIR = Path("/content/drive/MyDrive/legal-agent-vn")

        # Lần đầu project chưa tồn tại: notebook tự yêu cầu upload ZIP.
        # Khi có ZIP phiên bản mới: đổi thành True để ghi đè mã nguồn cũ.
        UPDATE_PROJECT_FROM_ZIP = False

        if UPDATE_PROJECT_FROM_ZIP or not (PROJECT_DIR / "config.py").exists():
            print("Hãy upload file legal-agent-vn.zip...")
            uploaded = files.upload()
            zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
            if len(zip_names) != 1:
                raise RuntimeError("Cần upload đúng một file ZIP của project.")

            zip_path = Path("/content") / zip_names[0]
            destination = PROJECT_DIR.resolve()
            destination.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zip_path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if target != destination and destination not in target.parents:
                        raise ValueError(f"ZIP chứa đường dẫn không an toàn: {member.filename}")
                archive.extractall(destination)

        if not (PROJECT_DIR / "config.py").exists():
            raise FileNotFoundError("Không tìm thấy config.py trong PROJECT_DIR.")

        os.chdir(PROJECT_DIR)
        print("PROJECT_DIR:", PROJECT_DIR)
        """
    ),
    markdown("## 2. Cài dependency"),
    code(
        """
        %cd $PROJECT_DIR
        !python -m pip install -q --upgrade pip
        !python -m pip install -q -r requirements-data.txt
        """
    ),
    markdown("## 3. Cấu hình API và đường dẫn index local"),
    code(
        """
        import os
        from google.colab import userdata

        try:
            gemini_key = userdata.get("GEMINI_API_KEY")
        except Exception as exc:
            raise RuntimeError(
                "Hãy tạo Colab Secret tên GEMINI_API_KEY và bật Notebook access."
            ) from exc

        if not gemini_key:
            raise RuntimeError("GEMINI_API_KEY đang rỗng.")

        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["QDRANT_PATH"] = "/content/qdrant_db_local"
        os.environ["LLM_MODEL"] = "gemini-3.5-flash"
        os.environ["LLM_FALLBACK_MODEL"] = "gemini-3.1-flash-lite"

        print("GEMINI_API_KEY: OK")
        print("QDRANT_PATH:", os.environ["QDRANT_PATH"])
        print("LLM_MODEL:", os.environ["LLM_MODEL"])
        """
    ),
    markdown("## 4. Kiểm tra GPU và API Gemini"),
    code(
        """
        import torch

        print("Python/Torch:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))

        !nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
        """
    ),
    code(
        """
        from google import genai
        from google.genai import types

        api_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        api_response = api_client.models.generate_content(
            model=os.environ["LLM_MODEL"],
            contents="Chỉ trả lời đúng một từ: OK",
            config=types.GenerateContentConfig(
                max_output_tokens=20,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        print("Gemini API:", api_response.text.strip())
        """
    ),
    markdown(
        """
        # A. Lần chạy đầu: tải full data và ingest

        Chỉ chạy các cell mục A khi chưa có `legal_index_v03.tar.gz`, hoặc khi muốn
        xây lại index. Dữ liệu thô nằm trong `PROJECT_DIR/data` nên được lưu trực tiếp
        trên Drive. Qdrant được xây ở `/content` để nhanh hơn.
        """
    ),
    markdown("## A1. Tải toàn bộ corpus chính và retrieval benchmark"),
    code(
        """
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.download_hf_datasets",
                "--dataset",
                "all",
                "--repair_known",
            ],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## A2. Audit corpus chính"),
    code(
        """
        subprocess.run(
            [sys.executable, "-m", "scripts.audit_corpus"],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## A3. Ingest corpus chính → `vn_legal_docs`"),
    code(
        """
        # Có thể chạy lâu do phải embed khoảng 38 nghìn chunks bằng BGE-M3.
        # Thành công khi dòng cuối là: Ingest completed in ...s
        subprocess.run(
            [sys.executable, "-m", "scripts.ingest_data"],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## A4. Ingest benchmark → `vn_legal_retrieval_benchmark`"),
    code(
        """
        # Tạo khoảng 55.953 points. Cảnh báo Qdrant local >20.000 points không phải lỗi.
        subprocess.run(
            [sys.executable, "-m", "scripts.ingest_benchmark_corpus"],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## A5. Kiểm tra đủ hai collection và hai BM25 index"),
    code(
        """
        import warnings
        from pathlib import Path
        from qdrant_client import QdrantClient

        warnings.filterwarnings(
            "ignore",
            message="Local mode is not recommended.*",
            category=UserWarning,
        )

        qdrant_path = Path(os.environ["QDRANT_PATH"])
        client = QdrantClient(path=str(qdrant_path))
        try:
            current_collections = {
                item.name for item in client.get_collections().collections
            }
            for name in sorted(current_collections):
                count = client.count(collection_name=name, exact=True).count
                print(f"{name}: {count:,} points")
        finally:
            client.close()

        bm25_paths = sorted(qdrant_path.parent.glob("bm25_index_*.json.gz"))
        for path in bm25_paths:
            print(f"{path.name}: {path.stat().st_size / 1024**2:.1f} MB")

        required_collections = {
            "vn_legal_docs",
            "vn_legal_retrieval_benchmark",
        }
        required_bm25 = {
            "bm25_index_vn_legal_docs.json.gz",
            "bm25_index_vn_legal_retrieval_benchmark.json.gz",
        }

        assert required_collections <= current_collections, (
            "Thiếu collection: " + str(required_collections - current_collections)
        )
        assert required_bm25 <= {path.name for path in bm25_paths}, (
            "Thiếu BM25: " + str(required_bm25 - {path.name for path in bm25_paths})
        )
        print("INDEX ĐẦY ĐỦ: OK")
        """
    ),
    markdown("## A6. Lưu chung Qdrant + hai BM25 index lên Drive"),
    code(
        """
        from pathlib import Path
        import hashlib
        import os
        import shutil
        import tarfile

        from qdrant_client import QdrantClient

        LOCAL_QDRANT = Path(os.environ["QDRANT_PATH"])
        SAVED_INDEX = PROJECT_DIR / "data" / "legal_index_v03.tar.gz"
        SAVED_HASH = Path(str(SAVED_INDEX) + ".sha256")
        BUILD_ARCHIVE = Path("/content/legal_index_v03.build.tar.gz")
        DRIVE_TEMP = Path(str(SAVED_INDEX) + ".tmp")

        REQUIRED_COLLECTIONS = {
            "vn_legal_docs",
            "vn_legal_retrieval_benchmark",
        }
        REQUIRED_BM25_NAMES = {
            "bm25_index_vn_legal_docs.json.gz",
            "bm25_index_vn_legal_retrieval_benchmark.json.gz",
        }

        def calculate_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        if not LOCAL_QDRANT.exists():
            raise FileNotFoundError("Chưa tìm thấy Qdrant index.")

        client = QdrantClient(path=str(LOCAL_QDRANT))
        try:
            existing_collections = {
                item.name for item in client.get_collections().collections
            }
        finally:
            client.close()

        missing_collections = REQUIRED_COLLECTIONS - existing_collections
        if missing_collections:
            raise RuntimeError(
                "Qdrant còn thiếu collection: " + ", ".join(sorted(missing_collections))
            )

        bm25_files = sorted(
            path
            for path in LOCAL_QDRANT.parent.glob("bm25_index_*.json.gz")
            if path.name in REQUIRED_BM25_NAMES
        )
        existing_bm25 = {path.name for path in bm25_files}
        missing_bm25 = REQUIRED_BM25_NAMES - existing_bm25
        if missing_bm25:
            raise FileNotFoundError(
                "Còn thiếu BM25 index: " + ", ".join(sorted(missing_bm25))
            )

        SAVED_INDEX.parent.mkdir(parents=True, exist_ok=True)
        BUILD_ARCHIVE.unlink(missing_ok=True)
        DRIVE_TEMP.unlink(missing_ok=True)

        def exclude_runtime_files(info):
            return None if Path(info.name).name == ".lock" else info

        print("Collections:", sorted(existing_collections))
        print("BM25 indexes:", sorted(existing_bm25))
        print("Đang đóng gói index ở local Colab...")

        with tarfile.open(BUILD_ARCHIVE, "w:gz") as archive:
            archive.add(
                LOCAL_QDRANT,
                arcname="qdrant_db_local",
                filter=exclude_runtime_files,
            )
            for bm25_file in bm25_files:
                archive.add(bm25_file, arcname=bm25_file.name)

        archive_hash = calculate_sha256(BUILD_ARCHIVE)
        print("Đang sao chép archive lên Drive...")
        shutil.copy2(BUILD_ARCHIVE, DRIVE_TEMP)
        DRIVE_TEMP.replace(SAVED_INDEX)
        SAVED_HASH.write_text(archive_hash + "\\n", encoding="utf-8")
        BUILD_ARCHIVE.unlink(missing_ok=True)

        print("Đã lưu:", SAVED_INDEX)
        print(f"Kích thước: {SAVED_INDEX.stat().st_size / 1024**2:.1f} MB")
        print("SHA-256:", archive_hash)
        """
    ),
    markdown(
        """
        # B. Từ lần chạy sau: khôi phục index

        Trong runtime Colab mới, chạy lại các mục 1–4, **bỏ qua toàn bộ mục A** và
        chạy cell B1. Archive chứa cả hai collection và cả hai BM25 index.
        """
    ),
    markdown("## B1. Khôi phục index từ Drive về `/content`"),
    code(
        """
        from pathlib import Path
        import hashlib
        import os
        import shutil
        import tarfile
        import tempfile

        from qdrant_client import QdrantClient

        os.environ["QDRANT_PATH"] = "/content/qdrant_db_local"

        SAVED_INDEX = PROJECT_DIR / "data" / "legal_index_v03.tar.gz"
        SAVED_HASH = Path(str(SAVED_INDEX) + ".sha256")
        LOCAL_QDRANT = Path(os.environ["QDRANT_PATH"])
        REQUIRED_BM25_NAMES = {
            "bm25_index_vn_legal_docs.json.gz",
            "bm25_index_vn_legal_retrieval_benchmark.json.gz",
        }
        REQUIRED_COLLECTIONS = {
            "vn_legal_docs",
            "vn_legal_retrieval_benchmark",
        }

        def calculate_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as file:
                for block in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        if not SAVED_INDEX.exists():
            raise FileNotFoundError(
                "Chưa có legal_index_v03.tar.gz. Hãy chạy mục A trong lần đầu."
            )

        occupied = [LOCAL_QDRANT]
        occupied.extend(LOCAL_QDRANT.parent / name for name in REQUIRED_BM25_NAMES)
        occupied = [path for path in occupied if path.exists()]
        if occupied:
            raise RuntimeError(
                "Runtime không còn sạch: " + ", ".join(map(str, occupied))
                + ". Nếu đây là lần chạy mới, hãy restart runtime rồi khôi phục lại."
            )

        actual_hash = calculate_sha256(SAVED_INDEX)
        if SAVED_HASH.exists():
            expected_hash = SAVED_HASH.read_text(encoding="utf-8").strip()
            if actual_hash != expected_hash:
                raise RuntimeError("Archive sai SHA-256 hoặc chưa sao chép đầy đủ.")

        extract_root = Path(tempfile.mkdtemp(prefix="legal-index-", dir="/content"))
        try:
            with tarfile.open(SAVED_INDEX, "r:gz") as archive:
                for member in archive.getmembers():
                    target = (extract_root / member.name).resolve()
                    if target != extract_root and extract_root not in target.parents:
                        raise ValueError(f"Archive có path traversal: {member.name}")
                    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                        raise ValueError(f"Archive có member không an toàn: {member.name}")
                archive.extractall(extract_root)

            restored_qdrant = extract_root / "qdrant_db_local"
            restored_bm25 = [extract_root / name for name in REQUIRED_BM25_NAMES]
            if not restored_qdrant.is_dir():
                raise RuntimeError("Archive thiếu qdrant_db_local.")
            missing = [path.name for path in restored_bm25 if not path.is_file()]
            if missing:
                raise RuntimeError("Archive thiếu BM25: " + ", ".join(sorted(missing)))

            shutil.move(str(restored_qdrant), str(LOCAL_QDRANT))
            for source in restored_bm25:
                shutil.move(str(source), str(LOCAL_QDRANT.parent / source.name))
        finally:
            shutil.rmtree(extract_root, ignore_errors=True)

        client = QdrantClient(path=str(LOCAL_QDRANT))
        try:
            restored_collections = {
                item.name for item in client.get_collections().collections
            }
            missing = REQUIRED_COLLECTIONS - restored_collections
            if missing:
                raise RuntimeError(
                    "Archive thiếu collection: " + ", ".join(sorted(missing))
                )
            for name in sorted(restored_collections):
                count = client.count(collection_name=name, exact=True).count
                print(f"{name}: {count:,} points")
        finally:
            client.close()

        print("Khôi phục thành công.")
        print("SHA-256:", actual_hash)
        """
    ),
    markdown("## 5. Kiểm tra hệ thống trước khi evaluation/demo"),
    code(
        """
        %cd $PROJECT_DIR
        !python -m scripts.check_setup
        """
    ),
    markdown(
        """
        ## 6. Retrieval Evaluation: Dense, BM25, RRF, Rerank

        `300` câu phù hợp cho bảng thực nghiệm đồ án. Đổi thành `50` để smoke test;
        đổi thành `0` để chạy toàn bộ testset, nhưng Rerank toàn bộ có thể mất nhiều giờ.
        """
    ),
    code(
        """
        import subprocess
        import sys

        RETRIEVAL_SAMPLE_SIZE = 300
        RETRIEVAL_RESULT = PROJECT_DIR / "evaluation" / "retrieval_ablation_results.json"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.evaluate_retrieval",
                "--testset",
                "data/testset/yuitc_retrieval_testset.json",
                "--collection",
                "vn_legal_retrieval_benchmark",
                "--sample_size",
                str(RETRIEVAL_SAMPLE_SIZE),
                "--modes",
                "dense",
                "bm25",
                "rrf",
                "rerank",
                "--output",
                str(RETRIEVAL_RESULT),
            ],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## 7. Hiển thị bảng và biểu đồ Retrieval Evaluation"),
    code(
        """
        import json
        import matplotlib.pyplot as plt
        import pandas as pd
        from IPython.display import display

        retrieval_data = json.loads(RETRIEVAL_RESULT.read_text(encoding="utf-8"))
        retrieval_df = pd.DataFrame(retrieval_data["summary"]).T
        retrieval_df.index.name = "Phương pháp"
        retrieval_df = retrieval_df[
            ["recall@1", "recall@5", "recall@10", "recall@30", "hit@5", "hit@10", "mrr", "avg_latency_ms"]
        ]

        display(retrieval_df.style.format({
            "recall@1": "{:.4f}",
            "recall@5": "{:.4f}",
            "recall@10": "{:.4f}",
            "recall@30": "{:.4f}",
            "hit@5": "{:.4f}",
            "hit@10": "{:.4f}",
            "mrr": "{:.4f}",
            "avg_latency_ms": "{:.1f}",
        }))

        chart_data = retrieval_df[["recall@5", "recall@10", "mrr"]]
        ax = chart_data.plot(kind="bar", figsize=(10, 5), rot=0)
        ax.set_title(f"Retrieval Ablation — n={retrieval_data['n']}")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        plt.tight_layout()

        chart_path = PROJECT_DIR / "evaluation" / "retrieval_ablation_chart.png"
        plt.savefig(chart_path, dpi=180, bbox_inches="tight")
        plt.show()
        print("Đã lưu biểu đồ:", chart_path)
        """
    ),
    markdown(
        """
        ## 8. End-to-End Agent Evaluation

        Bộ `sample_testset.json` có 4 câu, phù hợp để smoke test pipeline. Muốn đưa
        E2E vào kết luận chính thức, nên mở rộng thành 30–50 câu do người có chuyên
        môn gán nhãn. Cell này gọi Gemini nên có thể chịu rate limit của API key.
        """
    ),
    code(
        """
        E2E_RESULT = PROJECT_DIR / "evaluation" / "results_final.json"

        subprocess.run(
            [
                sys.executable,
                "-m",
                "evaluation.evaluate",
                "data/testset/sample_testset.json",
                "--output",
                str(E2E_RESULT),
            ],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown("## 9. Hiển thị kết quả End-to-End"),
    code(
        """
        e2e_rows = json.loads(E2E_RESULT.read_text(encoding="utf-8"))
        e2e_df = pd.DataFrame(e2e_rows)

        display(e2e_df[[
            "id",
            "type",
            "retrieval_recall@5",
            "retrieval_recall@8",
            "faithfulness_score",
            "is_sufficient",
            "citation_coverage",
            "iterations_used",
            "latency_sec",
        ]])

        csv_path = PROJECT_DIR / "evaluation" / "results_final_summary.csv"
        e2e_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("Đã lưu bảng CSV:", csv_path)
        """
    ),
    markdown("## 10. Smoke test một câu hỏi bằng CLI"),
    code(
        """
        subprocess.run(
            [
                sys.executable,
                "main.py",
                "Người lao động phải báo trước bao lâu khi đơn phương chấm dứt hợp đồng lao động không xác định thời hạn?",
            ],
            cwd=PROJECT_DIR,
            env=os.environ.copy(),
            check=True,
        )
        """
    ),
    markdown(
        """
        ## 11. Demo Gradio — chạy cuối cùng

        Sau khi cell chạy, mở URL Colab proxy được in ra. Không chạy retrieval/E2E
        evaluation ở process khác trong lúc demo đang mở vì Qdrant local giữ file lock.
        """
    ),
    code(
        """
        import time
        from IPython.display import HTML, display
        from google.colab.output import eval_js

        from app import demo

        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            prevent_thread_lock=True,
            show_error=True,
        )

        time.sleep(3)
        demo_url = eval_js("google.colab.kernel.proxyPort(7860)")
        print("Mở demo:", demo_url)
        display(HTML(f'<a href="{demo_url}" target="_blank"><b>MỞ LEGAL AGENT DEMO</b></a>'))
        """
    ),
    markdown("## 12. Dừng demo khi cần"),
    code(
        """
        demo.close()
        print("Đã dừng Gradio. Nếu cần chạy evaluation lại, nên restart runtime rồi khôi phục index.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "Legal Agent VN — Full Pipeline.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


output = Path(__file__).resolve().parent.parent / "colab" / "colab.ipynb"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(output)
