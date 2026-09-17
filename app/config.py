from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = "Enterprise RAG Assistant"
    app_env: str = "development"
    log_level: str = "INFO"

    aws_region: str = "us-east-1"

    bedrock_chat_model_id: str = (
        "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )

    bedrock_embedding_model_id: str = (
        "amazon.titan-embed-text-v2:0"
    )

    vector_store_path: str = "./storage/vector_store"

    documents_path: str = "./data/sample_enterprise_docs"

    top_k: int = 4

    chunk_size: int = 800

    chunk_overlap: int = 120

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
