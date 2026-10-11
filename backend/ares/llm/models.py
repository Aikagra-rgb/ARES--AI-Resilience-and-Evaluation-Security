from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"

    @classmethod
    def from_str(cls, value: str) -> "LLMProvider":
        val = value.lower().strip()
        if val in ("groq", "g"):
            return cls.GROQ
        elif val in ("gemini", "google"):
            return cls.GEMINI
        elif val in ("nvidia", "nim", "nemotron"):
            return cls.NVIDIA
        elif val in ("ollama", "local"):
            return cls.OLLAMA
        raise ValueError(f"Unknown provider '{value}'. Expected 'groq', 'gemini', 'nvidia', or 'ollama'.")


class ChatMessage(BaseModel):
    role: str = Field(description="Role of the message author (system, user, assistant)")
    content: str = Field(description="Content of the message")


class CompletionRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
    provider: Optional[LLMProvider] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: Optional[float] = None
    stop: Optional[List[str]] = None
    fallback_on_rate_limit: bool = True


class CompletionResponse(BaseModel):
    text: str = Field(description="Generated completion text")
    provider_used: str = Field(description="Provider that handled the request")
    model_used: str = Field(description="Model used for completion")
    latency_ms: float = Field(description="Latency in milliseconds")
    prompt_tokens: Optional[int] = Field(default=None, description="Prompt token count")
    completion_tokens: Optional[int] = Field(default=None, description="Completion token count")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens used")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider response payload")


class LLMError(Exception):
    """Base exception for LLM operations."""
    def __init__(self, message: str, provider: Optional[str] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMRateLimitError(LLMError):
    """Rate limit exceeded (HTTP 429)."""
    pass


class LLMAuthenticationError(LLMError):
    """Authentication or authorization failed (HTTP 401/403)."""
    pass


class LLMAllProvidersFailedError(LLMError):
    """All providers in the fallback chain failed."""
    def __init__(self, errors: Dict[str, Exception]):
        msg = "All LLM providers failed: " + "; ".join(f"{prov}: {err}" for prov, err in errors.items())
        super().__init__(msg)
        self.errors = errors
