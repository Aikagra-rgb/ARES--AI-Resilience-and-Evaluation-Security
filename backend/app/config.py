from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_ENVIRONMENTS = frozenset({"development", "dev", "local", "test", "testing"})
_INSECURE_FINGERPRINT_SECRET = "development-only-change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./ares.db"
    environment: str = "development"
    auto_create_schema: bool = True
    # Header-only identity (no API key). Honoured exclusively in a development environment;
    # a non-development ENVIRONMENT refuses to start with this enabled.
    allow_dev_identity: bool = False
    prompt_fingerprint_secret: str = _INSECURE_FINGERPRINT_SECRET
    execution_payload_encryption_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    qdrant_url: str = "https://c8cc3800-bb1b-48df-b26a-2de8dd1e35a4.ca-central-1-0.aws.cloud.qdrant.io"
    qdrant_api_key: SecretStr | None = None
    run_worker: bool = True
    provider_timeout_seconds: float = 60
    ollama_base_url: str = "http://localhost:11434"
    ollama_default_model: str = "llama3.1:8b"
    cors_origins: str = "http://localhost:5000,https://localhost:5001,http://localhost:5149,https://localhost:7149,http://localhost:5180,https://localhost:7180"

    @property
    def is_development(self) -> bool:
        return self.environment.strip().lower() in _DEV_ENVIRONMENTS

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @model_validator(mode="after")
    def _enforce_production_security(self) -> "Settings":
        if self.is_development:
            return self
        if self.allow_dev_identity:
            raise ValueError(f"ALLOW_DEV_IDENTITY=true is not allowed when ENVIRONMENT={self.environment!r}.")
        if self.prompt_fingerprint_secret == _INSECURE_FINGERPRINT_SECRET:
            raise ValueError("PROMPT_FINGERPRINT_SECRET must be changed from its development default.")
        if self.execution_payload_encryption_key is None:
            raise ValueError("EXECUTION_PAYLOAD_ENCRYPTION_KEY is required outside development.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
