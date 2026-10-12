"""
Unified ARES Configuration Module.

This is the single source of truth for all configuration across the ARES system.
Consolidates settings from both app/ and ares/ packages.
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Constants
# =============================================================================

_DEV_ENVIRONMENTS = frozenset({"development", "dev", "local", "test", "testing"})
_INSECURE_FINGERPRINT_SECRET = "development-only-change-me"
_DEV_API_KEY = "ares-dev-secret-key-42"
_MIN_PRODUCTION_KEY_LENGTH = 24

# Look for .env in root or backend dir
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / ".env" if (ROOT_DIR / ".env").exists() else Path(__file__).resolve().parent.parent / ".env"


# =============================================================================
# Unified Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Unified configuration for the entire ARES system.
    
    All settings are loaded from environment variables with the specified aliases.
    Production security is enforced via model validators.
    """
    
    model_config = SettingsConfigDict(
        env_file=ENV_PATH if ENV_PATH.exists() else None,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )
    
    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    # PostgreSQL only - no SQLite fallback. Must be set via DATABASE_URL env var.
    # Format: postgresql+psycopg://user:pass@host:port/dbname
    database_url: str = Field(..., alias="DATABASE_URL")
    auto_create_schema: bool = Field(default=True, alias="AUTO_CREATE_SCHEMA")
    
    # -------------------------------------------------------------------------
    # Environment & Security
    # -------------------------------------------------------------------------
    environment: str = Field(default="development", alias="ENVIRONMENT")
    
    # Header-only identity (no API key). Honoured exclusively in a development environment;
    # a non-development ENVIRONMENT refuses to start with this enabled.
    allow_dev_identity: bool = Field(default=False, alias="ALLOW_DEV_IDENTITY")
    
    prompt_fingerprint_secret: str = Field(default=_INSECURE_FINGERPRINT_SECRET, alias="PROMPT_FINGERPRINT_SECRET")
    execution_payload_encryption_key: SecretStr | None = Field(default=None, alias="EXECUTION_PAYLOAD_ENCRYPTION_KEY")
    
    # API Security & Authentication
    # Auth is ON by default. It can only be switched off explicitly, and only in a
    # development/test environment (enforced by `_enforce_production_security`).
    api_key: str = Field(default=_DEV_API_KEY, alias="ARES_API_KEY")
    auth_enabled: bool = Field(default=True, alias="ARES_AUTH_ENABLED")
    
    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    openai_default_model: str = Field(default="gpt-4o-mini", alias="OPENAI_DEFAULT_MODEL")
    
    # Groq (Primary)
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_default_model: str = Field(default="openai/gpt-oss-120b", alias="GROQ_DEFAULT_MODEL")
    
    # Google Gemini (Secondary)
    gemini_api_key: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_base_url: str = Field(default="https://generativelanguage.googleapis.com/v1beta/openai", alias="GEMINI_BASE_URL")
    gemini_default_model: str = Field(default="gemini-3.7-flash", alias="GEMINI_DEFAULT_MODEL")
    
    # NVIDIA NIM (Tertiary)
    nvidia_api_key: SecretStr | None = Field(default=None, alias="NVIDIA_API_KEY")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1", alias="NVIDIA_BASE_URL")
    nvidia_default_model: str = Field(default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", alias="NVIDIA_DEFAULT_MODEL")
    
    # Ollama (Local on RTX GPU)
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_default_model: str = Field(default="llama3.1:8b", alias="OLLAMA_DEFAULT_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    prefer_local_embeddings: bool = Field(default=False, alias="PREFER_LOCAL_EMBEDDINGS")
    
    # -------------------------------------------------------------------------
    # Vector DB (Qdrant)
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="ares_attacks", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=768, alias="QDRANT_VECTOR_SIZE")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    
    # -------------------------------------------------------------------------
    # Role Assignments
    # -------------------------------------------------------------------------
    attacker_provider: str = Field(default="groq", alias="ATTACKER_PROVIDER")
    victim_provider: str = Field(default="groq", alias="VICTIM_PROVIDER")
    judge_provider: str = Field(default="gemini", alias="JUDGE_PROVIDER")
    hardener_provider: str = Field(default="nvidia", alias="HARDENER_PROVIDER")
    hardener_model: str = Field(
        default="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        alias="HARDENER_MODEL",
    )
    
    # -------------------------------------------------------------------------
    # Resilience & Fallback Settings
    # -------------------------------------------------------------------------
    fallback_order: List[str] = Field(default_factory=lambda: ["groq", "gemini", "nvidia", "ollama"], alias="FALLBACK_ORDER")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    retry_min_wait_seconds: float = Field(default=0.5, alias="RETRY_MIN_WAIT_SECONDS")
    retry_max_wait_seconds: float = Field(default=4.0, alias="RETRY_MAX_WAIT_SECONDS")
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT_SECONDS")
    provider_timeout_seconds: float = Field(default=60, alias="PROVIDER_TIMEOUT_SECONDS")
    
    # -------------------------------------------------------------------------
    # Runtime Enforcement Gateway
    # -------------------------------------------------------------------------
    gateway_block_threshold: float = Field(default=0.82, alias="GATEWAY_BLOCK_THRESHOLD")
    gateway_sanitize_threshold: float = Field(default=0.68, alias="GATEWAY_SANITIZE_THRESHOLD")
    gateway_canary_token: str = Field(default="CANARY_ARES_SECRET_42", alias="GATEWAY_CANARY_TOKEN")
    
    # -------------------------------------------------------------------------
    # Embedding-space safety
    # -------------------------------------------------------------------------
    # When True, allows deterministic hash fallback when external embedding APIs fail.
    # Default: False (fail closed - report degraded instead of generating pseudo-vectors)
    embedding_allow_hash_fallback: bool = Field(default=False, alias="EMBEDDING_ALLOW_HASH_FALLBACK")
    
    # -------------------------------------------------------------------------
    # CORS
    # -------------------------------------------------------------------------
    cors_origins: str = Field(
        default="http://localhost:5000,https://localhost:5001,http://localhost:5149,https://localhost:7149,http://localhost:5180,https://localhost:7180",
        alias="CORS_ORIGINS"
    )
    
    # -------------------------------------------------------------------------
    # Worker
    # -------------------------------------------------------------------------
    run_worker: bool = Field(default=True, alias="RUN_WORKER")
    
    # -------------------------------------------------------------------------
    # API Server
    # -------------------------------------------------------------------------
    ares_api_base_url: str = Field(default="http://localhost:8000", alias="ARES_API_BASE_URL")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def is_development(self) -> bool:
        return self.environment.strip().lower() in _DEV_ENVIRONMENTS
    
    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
    
    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    
    @model_validator(mode="after")
    def _validate_database_url(self) -> "Settings":
        """Fail fast if DATABASE_URL is unset or invalid."""
        if not self.database_url or not self.database_url.strip():
            raise ValueError(
                "DATABASE_URL is required. Set it to a PostgreSQL connection string, "
                "e.g., postgresql+psycopg://user:pass@host:port/dbname"
            )
        
        # Must be PostgreSQL
        if not self.database_url.startswith("postgresql"):
            raise ValueError(
                f"DATABASE_URL must be a PostgreSQL connection string (postgresql+psycopg://...), "
                f"got: {self.database_url}"
            )
        
        return self

    @model_validator(mode="after")
    def _enforce_production_security(self) -> "Settings":
        """Fail closed: outside development, auth must be on and the dev key is forbidden."""
        if self.is_development:
            return self
        
        # Auth must be enabled in production
        if not self.auth_enabled:
            raise ValueError(
                f"ARES_AUTH_ENABLED=false is not allowed when ENVIRONMENT={self.environment!r}."
            )
        
        # Dev identity must be disabled in production
        if self.allow_dev_identity:
            raise ValueError(f"ALLOW_DEV_IDENTITY=true is not allowed when ENVIRONMENT={self.environment!r}.")
        
        # Prompt fingerprint secret must be changed from default
        if self.prompt_fingerprint_secret == _INSECURE_FINGERPRINT_SECRET:
            raise ValueError("PROMPT_FINGERPRINT_SECRET must be changed from its development default.")
        
        # Encryption key required in production
        if self.execution_payload_encryption_key is None:
            raise ValueError("EXECUTION_PAYLOAD_ENCRYPTION_KEY is required outside development.")
        
        # API key must be unique and strong
        if self.api_key == _DEV_API_KEY or len(self.api_key) < _MIN_PRODUCTION_KEY_LENGTH:
            raise ValueError(
                "ARES_API_KEY must be set to a unique secret of at least "
                f"{_MIN_PRODUCTION_KEY_LENGTH} characters when ENVIRONMENT={self.environment!r}."
            )
        
        # Canary token must be explicitly set in production
        if not self.gateway_canary_token:
            raise ValueError(
                "GATEWAY_CANARY_TOKEN must be explicitly set in production environment. "
                "Generate a secure random token (e.g., using 'openssl rand -hex 32')."
            )
        
        return self


# =============================================================================
# Cached Settings Accessor
# =============================================================================

@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# =============================================================================
# Legacy Compatibility
# =============================================================================

# For backward compatibility with code using app.config.get_settings
def get_app_settings() -> Settings:
    """Legacy compatibility: returns the same unified settings."""
    return get_settings()


# Global settings instance for backward compatibility
settings = Settings()


# Re-export the Settings class from ares
# This ensures any code using `from app.config import Settings` gets the unified class
Settings = Settings