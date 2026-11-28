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
    # PostgreSQL is required outside development/test; local SQLite is allowed when ENVIRONMENT is development/dev/local/test/testing.
    # Format: postgresql+psycopg://user:pass@host:port/dbname (or sqlite:///path/to/ares.db in dev/test)
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
    
    # HMAC Request Integrity & Signing (None resolves to 'permissive' in dev/test, 'enforcing_critical' outside dev)
    hmac_enforce_mode: Optional[str] = Field(default=None, alias="HMAC_ENFORCE_MODE")
    hmac_secret: str = Field(default=_DEV_API_KEY, alias="HMAC_SECRET")
    hmac_max_clock_skew: int = Field(default=300, alias="HMAC_MAX_CLOCK_SKEW")
    
    # HTTP Gateway Rate Limiting & Redis Backend
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_requests: int = Field(default=120, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(default=60, alias="RATE_LIMIT_WINDOW_SECONDS")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    tenant_api_keys: str = Field(default="", alias="ARES_TENANT_API_KEYS")
    
    # Gateway Resilience & Circuit Breaker (None resolves to False in dev/test, True outside dev)
    gateway_fail_closed_on_outage: Optional[bool] = Field(default=None, alias="GATEWAY_FAIL_CLOSED_ON_OUTAGE")
    
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
    ollama_optimizer_model: str = Field(default="ares-optimizer:8b", alias="OLLAMA_OPTIMIZER_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    prefer_local_embeddings: bool = Field(default=True, alias="PREFER_LOCAL_EMBEDDINGS")
    
    # -------------------------------------------------------------------------
    # Native LoRA Inference Server & Optimizer Thresholds
    # -------------------------------------------------------------------------
    lora_server_url: str = Field(default="http://localhost:11435/v1", alias="ARES_LORA_SERVER_URL")
    use_lora_server: bool = Field(default=True, alias="ARES_USE_LORA_SERVER")
    class_fidelity_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "coding_technical": 0.85,
            "creative_writing": 0.87,
            "analytical_qa": 0.86,
            "data_extraction_json": 0.80,
            "system_directive": 0.80,
        },
        alias="CLASS_FIDELITY_THRESHOLDS",
    )
    
    # -------------------------------------------------------------------------
    # Vector DB (Qdrant)
    # -------------------------------------------------------------------------
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="ares_attacks", alias="QDRANT_COLLECTION")
    qdrant_vector_size: int = Field(default=768, alias="QDRANT_VECTOR_SIZE")
    qdrant_timeout_seconds: float = Field(default=5.0, alias="QDRANT_TIMEOUT_SECONDS")
    gemini_embedding_model: str = Field(default="models/gemini-embedding-001", alias="GEMINI_EMBEDDING_MODEL")
    
    # -------------------------------------------------------------------------
    # Role Assignments
    # -------------------------------------------------------------------------
    attacker_provider: str = Field(default="ollama", alias="ATTACKER_PROVIDER")
    victim_provider: str = Field(default="ollama", alias="VICTIM_PROVIDER")
    judge_provider: str = Field(default="ollama", alias="JUDGE_PROVIDER")
    hardener_provider: str = Field(default="ollama", alias="HARDENER_PROVIDER")
    hardener_model: str = Field(
        default="llama3.1:8b",
        alias="HARDENER_MODEL",
    )
    
    # -------------------------------------------------------------------------
    # Resilience & Fallback Settings
    # -------------------------------------------------------------------------
    fallback_order: List[str] = Field(default_factory=lambda: ["ollama", "groq", "gemini", "nvidia"], alias="FALLBACK_ORDER")
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
    # Red-Team Evidence Governance & Dynamic Attacker Policy
    # -------------------------------------------------------------------------
    redteam_findings_retention_days: int = Field(default=30, ge=1, le=365, alias="REDTEAM_FINDINGS_RETENTION_DAYS")
    redteam_allow_encrypted_evidence: bool = Field(default=False, alias="REDTEAM_ALLOW_ENCRYPTED_EVIDENCE")
    enable_dynamic_redteam: bool = Field(default=False, alias="ARES_ENABLE_DYNAMIC_REDTEAM")
    redteam_target_allowlist: str = Field(default="", alias="ARES_REDTEAM_TARGET_ALLOWLIST")
    redteam_dynamic_max_attempts: int = Field(default=10, ge=1, le=25, alias="ARES_REDTEAM_DYNAMIC_MAX_ATTEMPTS")
    redteam_dynamic_rate_limit_per_hour: int = Field(
        default=5, ge=1, le=100, alias="ARES_REDTEAM_DYNAMIC_RATE_LIMIT_PER_HOUR"
    )
    redteam_allowed_providers: str = Field(default="ollama", alias="ARES_REDTEAM_ALLOWED_PROVIDERS")
    allow_optimizer_harmful_override: bool = Field(default=False, alias="ARES_ALLOW_OPTIMIZER_HARMFUL_OVERRIDE")
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def is_development(self) -> bool:
        return self.environment.strip().lower() in _DEV_ENVIRONMENTS
    
    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def redteam_allowed_target_hosts(self) -> set[str]:
        """Allowed hostnames for outbound red-team target probing."""
        return {
            h.strip().lower()
            for h in (self.redteam_target_allowlist or "").split(",")
            if h and h.strip()
        }

    @property
    def redteam_allowed_provider_set(self) -> set[str]:
        """Allowed LLM providers for controlled dynamic red-team workflows."""
        return {
            p.strip().lower()
            for p in (self.redteam_allowed_providers or "").split(",")
            if p and p.strip()
        }

    @property
    def parsed_tenant_api_keys(self) -> dict[str, str]:
        """Parse optional `ARES_TENANT_API_KEYS` (JSON object or comma-separated `org_id:secret`)."""
        raw = (self.tenant_api_keys or "").strip()
        if not raw:
            return {}
        if raw.startswith("{"):
            try:
                import json as _json

                loaded = _json.loads(raw)
                if isinstance(loaded, dict):
                    return {
                        str(k).strip(): str(v).strip()
                        for k, v in loaded.items()
                        if str(k).strip() and str(v).strip()
                    }
            except Exception:
                return {}
        mapping: dict[str, str] = {}
        for part in raw.split(","):
            item = part.strip()
            if not item or ":" not in item:
                continue
            oid, key_val = item.split(":", 1)
            oid_clean = oid.strip()
            key_clean = key_val.strip()
            if oid_clean and key_clean:
                mapping[oid_clean] = key_clean
        return mapping
    
    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    
    @model_validator(mode="after")
    def _validate_database_url(self) -> "Settings":
        """Fail fast if DATABASE_URL is unset or invalid; allow SQLite only in dev/test."""
        if not self.database_url or not self.database_url.strip():
            raise ValueError(
                "DATABASE_URL is required. Set it to a PostgreSQL connection string, "
                "e.g., postgresql+psycopg://user:pass@host:port/dbname"
            )

        url = self.database_url.strip()
        is_postgres = url.startswith("postgresql")
        is_sqlite = url.startswith("sqlite")
        if self.is_development:
            if not (is_postgres or is_sqlite):
                raise ValueError(
                    "DATABASE_URL must be PostgreSQL (postgresql+psycopg://...) or SQLite (sqlite:///...) "
                    f"in development/test, got: {self.database_url}"
                )
        elif not is_postgres:
            raise ValueError(
                "DATABASE_URL must be a PostgreSQL connection string (postgresql+psycopg://...) "
                f"outside development, got: {self.database_url}"
            )

        return self

    @model_validator(mode="after")
    def _normalize_lora_server_url(self) -> "Settings":
        if self.lora_server_url:
            cleaned = self.lora_server_url.strip().rstrip("/")
            if not cleaned.endswith("/v1"):
                cleaned = f"{cleaned}/v1"
            object.__setattr__(self, "lora_server_url", cleaned)
        return self

    @model_validator(mode="after")
    def _resolve_fail_policy_defaults(self) -> "Settings":
        """Apply explicit fail-closed defaults outside development/test while honoring overrides."""
        resolved_mode = self.hmac_enforce_mode
        if resolved_mode is None or not str(resolved_mode).strip():
            resolved_mode = "permissive" if self.is_development else "enforcing_critical"
        else:
            resolved_mode = str(resolved_mode).strip().lower()
        allowed_modes = {"enforcing", "enforcing_critical", "permissive", "disabled"}
        if resolved_mode not in allowed_modes:
            raise ValueError(
                f"Invalid HMAC_ENFORCE_MODE={resolved_mode!r}. Expected one of {sorted(allowed_modes)}."
            )
        object.__setattr__(self, "hmac_enforce_mode", resolved_mode)

        if self.gateway_fail_closed_on_outage is None:
            object.__setattr__(self, "gateway_fail_closed_on_outage", not self.is_development)

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

        # HMAC request signing cannot be disabled in production and must use a non-dev secret
        if self.hmac_enforce_mode == "disabled":
            raise ValueError(f"HMAC_ENFORCE_MODE='disabled' is not allowed when ENVIRONMENT={self.environment!r}.")

        if self.hmac_secret == _DEV_API_KEY or len(self.hmac_secret) < _MIN_PRODUCTION_KEY_LENGTH:
            raise ValueError(
                "HMAC_SECRET must be set to a unique secret of at least "
                f"{_MIN_PRODUCTION_KEY_LENGTH} characters when ENVIRONMENT={self.environment!r}."
            )
        
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