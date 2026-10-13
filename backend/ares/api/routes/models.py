"""
Model catalog endpoints — powers model selection in the ARES UI.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter

from ares.api.schemas import AiProviderEnum, ModelConfigResponse

router = APIRouter(prefix="/api/models", tags=["models"])

_MODELS: List[ModelConfigResponse] = [
    ModelConfigResponse(
        provider=AiProviderEnum.groq,
        id="llama-3.1-8b-instant",
        display_name="Llama 3.1 8B Instant",
        context_window=128000,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.groq,
        id="llama-3.1-70b-versatile",
        display_name="Llama 3.1 70B Versatile",
        context_window=128000,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.groq,
        id="mixtral-8x7b-32768",
        display_name="Mixtral 8×7B",
        context_window=32768,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.gemini,
        id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        context_window=1048576,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.nvidia_nim,
        id="nvidia/nemotron-4-340b-instruct",
        display_name="Nemotron 4 340B",
        context_window=4096,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.ollama,
        id="llama3.1:8b",
        display_name="Llama 3.1 8B (Local RTX)",
        context_window=128000,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.ollama,
        id="mistral:latest",
        display_name="Mistral 7B (Local RTX)",
        context_window=32768,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.ollama,
        id="qwen2.5:7b",
        display_name="Qwen 2.5 7B (Local RTX)",
        context_window=128000,
        is_available=True,
    ),
    ModelConfigResponse(
        provider=AiProviderEnum.ollama,
        id="deepseek-r1:8b",
        display_name="DeepSeek R1 8B (Local Reasoning)",
        context_window=65536,
        is_available=True,
    ),
]


@router.get("", response_model=List[ModelConfigResponse])
@router.get("/", response_model=List[ModelConfigResponse])
async def get_models(provider: Optional[str] = None) -> List[ModelConfigResponse]:
    """Return catalog of supported LLM models, optionally filtered by provider."""
    if provider:
        return [m for m in _MODELS if m.provider.value == provider]
    return _MODELS
