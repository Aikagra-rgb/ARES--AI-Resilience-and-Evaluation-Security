from pathlib import Path
from typing import List
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Look for .env in root or backend dir
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env" if (ROOT_DIR / ".env").exists() else Path(__file__).resolve().parent.parent / ".env"

# Well-known development credential. Accepted ONLY when ENVIRONMENT is development/test.
DEV_API_KEY = "ares-dev-secret-key-42"
_DEV_ENVIRONMENTS = frozenset({"development", "dev", "local", "test", "testing"})
_MIN_PRODUCTION_KEY_LENGTH = 24


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_PATH if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # API Server
    ares_api_base_url: str = Field(default="http://localhost:8000", alias="ARES_API_BASE_URL")

    # Groq (Primary)
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_default_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_DEFAULT_MODEL")

    # Google Gemini (Secondary)
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai", alias="GEMINI_BASE_URL")
    gemini_default_model: str = Field(default="gemini-3.7-flash", alias="GEMINI_DEFAULT_MODEL")

    # NVIDIA NIM (Tertiary)
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    nvidia_default_model: str = Field(default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", alias="NVIDIA_DEFAULT_MODEL")

    # OpenAI
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_default_model: str = Field(default="gpt-4o-mini", alias="OPENAI_DEFAULT_MODEL")

    # Ollama (Local on RTX GPU)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_default_model: str = Field(default="llama3.1:8b", alias="OLLAMA_DEFAULT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    prefer_local_embeddings: bool = Field(default=False, alias="PREFER_LOCAL_EMBEDDINGS")

    # Vector DB (Qdrant)
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="ares_attacks", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=768, alias="QDRANT_VECTOR_SIZE")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")

    # Role Assignments
    attacker_provider: str = Field(default="groq", alias="ATTACKER_PROVIDER")
    victim_provider: str = Field(default="groq", alias="VICTIM_PROVIDER")
    judge_provider: str = Field(default="gemini", alias="JUDGE_PROVIDER")
    hardener_provider: str = Field(default="nvidia", alias="HARDENER_PROVIDER")
    hardener_model: str = Field(
        default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        alias="HARDENER_MODEL",
    )

    # Resilience & Fallback Settings
    fallback_order: List[str] = Field(default_factory=lambda: ["groq", "gemini", "nvidia", "ollama"])
    max_retries: int = 3
    retry_min_wait_seconds: float = 0.5
    retry_max_wait_seconds: float = 4.0
    request_timeout_seconds: float = 30.0

    # API Security & Authentication
    # Auth is ON by default. It can only be switched off explicitly, and only in a
    # development/test environment (enforced by `_enforce_production_security`).
    environment: str = Field(default="development", alias="ENVIRONMENT")
    api_key: str = Field(default=DEV_API_KEY, alias="ARES_API_KEY")
    auth_enabled: bool = Field(default=True, alias="ARES_AUTH_ENABLED")

    # Runtime Enforcement Gateway
    gateway_block_threshold: float = Field(default=0.82, alias="GATEWAY_BLOCK_THRESHOLD")
    gateway_sanitize_threshold: float = Field(default=0.68, alias="GATEWAY_SANITIZE_THRESHOLD")
    gateway_canary_token: str = Field(default="CANARY_ARES_SECRET_42", alias="GATEWAY_CANARY_TOKEN")

    # Embedding-space safety. Each embedding model owns its own Qdrant collection (see
    # ares.vectordb.profiles); vectors from different models are NEVER compared.
    embedding_fallback_to_gemini: bool = Field(default=True, alias="EMBEDDING_FALLBACK_TO_GEMINI")

    @property
    def is_development(self) -> bool:
        return self.environment.strip().lower() in _DEV_ENVIRONMENTS

    @model_validator(mode="after")
    def _enforce_production_security(self) -> "Settings":
        """Fail closed: outside development, auth must be on and the dev key is forbidden."""
        if self.is_development:
            return self
        if not self.auth_enabled:
            raise ValueError(
                f"ARES_AUTH_ENABLED=false is not allowed when ENVIRONMENT={self.environment!r}."
            )
        if self.api_key == DEV_API_KEY or len(self.api_key) < _MIN_PRODUCTION_KEY_LENGTH:
            raise ValueError(
                "ARES_API_KEY must be set to a unique secret of at least "
                f"{_MIN_PRODUCTION_KEY_LENGTH} characters when ENVIRONMENT={self.environment!r}."
            )
        return self


settings = Settings()

