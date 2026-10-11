import asyncio
import json
import httpx
import pytest

from app.config import Settings
from app.providers.base import ProviderExecutionError
from app.providers.ollama_provider import OllamaAdapter
from app.providers.registry import ProviderRegistry


def test_ollama_adapter_executes_successfully() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Ollama local model response."}}],
                "usage": {"total_tokens": 42},
            },
        )

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(
        Settings(ollama_base_url="http://localhost:11434", ollama_default_model="llama3.1:8b"),
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )
    result = asyncio.run(
        adapter.execute(
            {
                "model": "llama3.1:8b",
                "system_prompt": "You are a local security evaluator.",
                "user_prompt": "Evaluate prompt robustness",
                "temperature": 0.0,
                "max_response_tokens": 256,
            },
            "user-123",
        )
    )

    body = json.loads(seen["request"].content)
    assert seen["request"].url == "http://localhost:11434/v1/chat/completions"
    assert body["model"] == "llama3.1:8b"
    assert body["messages"][0]["content"] == "You are a local security evaluator."
    assert body["messages"][1]["content"] == "Evaluate prompt robustness"
    assert result.output_text == "Ollama local model response."
    assert result.token_estimate == 42


def test_ollama_adapter_timeout_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Connection timed out")

    transport = httpx.MockTransport(handler)
    adapter = OllamaAdapter(
        Settings(ollama_base_url="http://localhost:11434"),
        client_factory=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs),
    )

    with pytest.raises(ProviderExecutionError) as exc_info:
        asyncio.run(
            adapter.execute(
                {
                    "model": "mistral:latest",
                    "system_prompt": "Sys",
                    "user_prompt": "User",
                },
                "user-123",
            )
        )
    assert "timed out" in str(exc_info.value).lower()


def test_ollama_registered_in_provider_registry() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Registry response."}}],
                "usage": {"total_tokens": 10},
            },
        )

    transport = httpx.MockTransport(handler)
    settings = Settings(ollama_base_url="http://localhost:11434")
    registry = ProviderRegistry(settings)
    # Substitute transport in adapter
    registry._ollama._client_factory = lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs)

    registry.ensure_configured("ollama")
    result = asyncio.run(
        registry.execute("ollama", {"model": "qwen2.5:7b", "user_prompt": "test"}, "user-1")
    )
    assert result.output_text == "Registry response."
