from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # HuggingFace
    hf_api_token: str = ""

    # SEC EDGAR
    sec_edgar_user_agent: str = "Sentinel sentinel@example.com"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "sentinel_docs"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "sentinel"

    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    generation_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    asr_model: str = "openai/whisper-large-v3"
    ner_model: str = "dslim/bert-base-NER"
    classifier_model: str = "facebook/bart-large-mnli"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 10
    rerank_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
