"""Application configuration loaded from environment / .env."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    hf_api_token: str = ""
    hf_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Gatekeeper — Eq. 1 weights
    weight_frequency: float = 0.3
    weight_confidence: float = 0.4
    weight_emotion: float = 0.3
    importance_threshold: float = 0.35

    # Forgetting — Eq. 7
    decay_lambda: float = 0.05
    decay_interval_hours: int = 24
    forget_floor: float = 0.05

    # Memory
    active_context_size: int = 20
    retrieval_top_k: int = 5
    retrieval_sim_weight: float = 0.6
    retrieval_imp_weight: float = 0.4

    # Storage
    chroma_persist_dir: str = "./data/chroma"
    sqlite_path: str = "./data/persona.db"

    # App
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
