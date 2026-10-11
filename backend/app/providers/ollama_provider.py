"""
Ollama Provider Adapter for ARES.

Enables local GPU model execution via Ollama (Llama 3.1 8B, Mistral 7B,
Qwen 2.5 7B, DeepSeek R1 8B, etc.) using Ollama's OpenAI-compatible endpoint.
"""

from collections.abc import Callable
import httpx

from app.config import Settings
from app.providers.base import ProviderExecutionError, ProviderNotConfiguredError, ProviderResult


class OllamaAdapter:
    """Server-side local Ollama API adapter for on-premise / edge model execution."""

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[..., httpx.AsyncClient] = httpx.AsyncClient,
    ):
        self._settings = settings
        self._client_factory = client_factory

    @property
    def endpoint(self) -> str:
        return f"{self._settings.ollama_base_url.rstrip('/')}/v1/chat/completions"

    def is_configured(self) -> bool:
        return bool(self._settings.ollama_base_url)

    async def execute(self, payload: dict, user_id: str) -> ProviderResult:
        if not self.is_configured():
            raise ProviderNotConfiguredError("Ollama is not configured on this server.")

        model = payload.get("model") or self._settings.ollama_default_model
        messages = [
            {"role": "system", "content": payload.get("system_prompt", "")},
            {"role": "user", "content": payload.get("user_prompt", "")},
        ]
        body = {
            "model": model,
            "messages": messages,
            "temperature": payload.get("temperature", 0.2),
            "max_tokens": payload.get("max_response_tokens", 512),
        }
        headers = {
            "Content-Type": "application/json",
        }

        try:
            async with self._client_factory(timeout=self._settings.provider_timeout_seconds) as client:
                response = await client.post(self.endpoint, json=body, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderExecutionError(f"Ollama local model '{model}' timed out.") from exc
        except httpx.HTTPError as exc:
            raise ProviderExecutionError(
                f"Ollama local daemon could not be reached at {self._settings.ollama_base_url}."
            ) from exc

        if response.is_error:
            raise ProviderExecutionError(f"Ollama API error ({response.status_code}): {response.text[:200]}")

        data = response.json()
        choices = data.get("choices") or []
        text = choices[0]["message"]["content"] if choices else ""
        usage = data.get("usage") or {}
        return ProviderResult(output_text=text, token_estimate=int(usage.get("total_tokens") or 0))
