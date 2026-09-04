"""Cấu hình tập trung, độc lập với thư mục chạy hiện tại."""
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    llm_model: str = "gemini-3.5-flash"
    llm_fallback_model: str = "gemini-3.1-flash-lite"

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    qdrant_path: str = str(BASE_DIR / "data" / "qdrant_db")
    collection_name: str = "vn_legal_docs"
    raw_data_dir: str = str(BASE_DIR / "data" / "raw")

    max_verification_loops: int = Field(default=2, ge=1, le=5)
    max_tool_calls_per_turn: int = Field(default=2, ge=0, le=5)
    top_k_retrieval: int = Field(default=10, ge=1, le=100)
    top_k_rerank: int = Field(default=8, ge=1, le=50)
    final_context_k: int = Field(default=8, ge=1, le=30)
    retrieval_candidate_k: int = Field(default=30, ge=1, le=200)
    effective_status_boost: float = Field(default=0.05, ge=0.0, le=0.5)
    embedding_batch_size: int = Field(default=12, ge=1, le=128)
    index_batch_size: int = Field(default=64, ge=1, le=512)
    max_query_chars: int = Field(default=2_000, ge=100, le=20_000)
    max_context_chars: int = Field(default=36_000, ge=2_000, le=200_000)
    max_doc_context_chars: int = Field(default=6_000, ge=500, le=30_000)

    app_share: bool = False
    app_debug: bool = False
    app_server_name: str = "127.0.0.1"
    app_server_port: int = Field(default=7860, ge=1, le=65535)

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
