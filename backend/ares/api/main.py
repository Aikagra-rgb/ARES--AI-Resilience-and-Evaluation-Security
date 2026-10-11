"""
ARES FastAPI application entry point.

Mounts all route routers and configures CORS for the Blazor Server frontend.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ares.api.routes import dashboard, tests, harden, search, stats, corpus, gateway

app = FastAPI(
    title="ARES — AI Resilience & Evaluation Security System",
    description="REST API powering the ARES Blazor dashboard. Provides red-team evaluation, "
                "prompt hardening, semantic corpus search, and ML analytics endpoints.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Blazor Server frontend (runs on 5000/5001/5180)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "https://localhost:5001",
        "http://localhost:5180",
        "http://localhost:7180",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers
app.include_router(dashboard.router)
app.include_router(tests.router)
app.include_router(harden.router)
app.include_router(search.router)
app.include_router(stats.router)
app.include_router(corpus.router)
app.include_router(gateway.router)

# Also expose /api/models (simple static list)
from fastapi import APIRouter as _APIRouter
from ares.api.schemas import AiProviderEnum, ModelConfigResponse

_models_router = _APIRouter(prefix="/api", tags=["models"])

_MODELS = [
    ModelConfigResponse(provider=AiProviderEnum.groq, id="llama-3.1-8b-instant",      display_name="Llama 3.1 8B Instant",       context_window=128000, is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.groq, id="llama-3.1-70b-versatile",   display_name="Llama 3.1 70B Versatile",    context_window=128000, is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.groq, id="mixtral-8x7b-32768",        display_name="Mixtral 8×7B",               context_window=32768,  is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.gemini, id="gemini-2.0-flash",         display_name="Gemini 2.0 Flash",           context_window=1048576, is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.nvidia_nim, id="nvidia/nemotron-4-340b-instruct", display_name="Nemotron 4 340B", context_window=4096, is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.ollama, id="llama3.1:8b", display_name="Llama 3.1 8B (Local RTX)", context_window=128000, is_available=True),
    ModelConfigResponse(provider=AiProviderEnum.ollama, id="mistral:latest", display_name="Mistral 7B (Local RTX)", context_window=32768, is_available=True),
]

@_models_router.get("/models", response_model=list[ModelConfigResponse])
async def get_models(provider: str | None = None) -> list[ModelConfigResponse]:
    if provider:
        return [m for m in _MODELS if m.provider.value == provider]
    return _MODELS

app.include_router(_models_router)


@app.get("/healthz", tags=["health"])
async def healthz() -> dict:
    """Liveness probe."""
    return {"status": "ok", "service": "ARES API"}
