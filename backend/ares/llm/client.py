import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from ares.config import Settings, settings as default_settings
from ares.llm.models import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMAllProvidersFailedError,
    LLMAuthenticationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified OpenAI-compatible adapter for Groq, Google Gemini, and NVIDIA NIM.
    Includes exponential retry logic and cross-provider fallback (Groq -> Gemini -> NVIDIA).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.settings = settings or default_settings
        self._http_client = http_client
        self._owns_http_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.request_timeout_seconds)
            )
            self._owns_http_client = True
        return self._http_client

    async def close(self) -> None:
        """Close internal HTTP client if owned."""
        if self._owns_http_client and self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    def _get_provider_config(self, provider: LLMProvider) -> Tuple[str, str, str]:
        """
        Returns (base_url, api_key, default_model) for the given provider.
        """
        if provider == LLMProvider.GROQ:
            return (
                self.settings.groq_base_url.rstrip("/"),
                self.settings.groq_api_key.get_secret_value() if self.settings.groq_api_key else "",
                self.settings.groq_default_model,
            )
        elif provider == LLMProvider.GEMINI:
            return (
                self.settings.gemini_base_url.rstrip("/"),
                self.settings.gemini_api_key.get_secret_value() if self.settings.gemini_api_key else "",
                self.settings.gemini_default_model,
            )
        elif provider == LLMProvider.NVIDIA:
            return (
                self.settings.nvidia_base_url.rstrip("/"),
                self.settings.nvidia_api_key.get_secret_value() if self.settings.nvidia_api_key else "",
                self.settings.nvidia_default_model,
            )
        elif provider == LLMProvider.OLLAMA:
            return (
                self.settings.ollama_base_url.rstrip("/") + "/v1",
                "ollama",
                self.settings.ollama_default_model,
            )
        elif provider == LLMProvider.OPENAI:
            return (
                self.settings.openai_base_url.rstrip("/"),
                self.settings.openai_api_key.get_secret_value() if self.settings.openai_api_key else "",
                self.settings.openai_default_model,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _format_messages(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
    ) -> List[Dict[str, str]]:
        """Converts prompt or messages into OpenAI-standard list of message dicts."""
        if messages:
            formatted = []
            for msg in messages:
                if isinstance(msg, ChatMessage):
                    formatted.append({"role": msg.role, "content": msg.content})
                elif isinstance(msg, dict):
                    formatted.append({"role": msg["role"], "content": msg["content"]})
                else:
                    raise ValueError(f"Invalid message type: {type(msg)}")
            return formatted
        elif prompt is not None:
            return [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

    async def _call_provider_with_retry(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> CompletionResponse:
        """Executes completion call to a specific provider with exponential backoff retry."""
        base_url, api_key, default_model = self._get_provider_config(provider)
        target_model = model or default_model

        if not api_key and provider != LLMProvider.OLLAMA:
            raise LLMAuthenticationError(
                f"Missing API key for provider '{provider.value}'. Check your .env file.",
                provider=provider.value,
                status_code=401,
            )

        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop

        client = await self._get_client()
        retries = 0
        last_exception: Optional[Exception] = None

        while retries <= self.settings.max_retries:
            start_time = time.perf_counter()
            try:
                response = await client.post(endpoint, json=payload, headers=headers)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if not choices:
                        raise LLMError(f"No choices returned from {provider.value}", provider=provider.value)

                    msg_dict = choices[0].get("message", {})
                    content = msg_dict.get("content") or ""
                    if not content and "reasoning" in msg_dict:
                        content = msg_dict.get("reasoning") or ""

                    usage = data.get("usage", {})

                    return CompletionResponse(
                        text=content,
                        provider_used=provider.value,
                        model_used=target_model,
                        latency_ms=round(latency_ms, 2),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        raw_response=data,
                    )

                elif response.status_code == 429:
                    error_msg = f"Rate limit (429) hit on provider '{provider.value}': {response.text}"
                    logger.warning(error_msg)
                    last_exception = LLMRateLimitError(error_msg, provider=provider.value, status_code=429)
                    if retries < self.settings.max_retries:
                        wait = min(
                            self.settings.retry_max_wait_seconds,
                            self.settings.retry_min_wait_seconds * (2 ** retries),
                        )
                        await asyncio.sleep(wait)
                        retries += 1
                        continue
                    else:
                        raise last_exception

                elif response.status_code in (401, 403):
                    error_msg = f"Authentication error ({response.status_code}) on provider '{provider.value}': {response.text}"
                    logger.error(error_msg)
                    raise LLMAuthenticationError(error_msg, provider=provider.value, status_code=response.status_code)

                elif response.status_code in (500, 502, 503, 504):
                    error_msg = f"Server error ({response.status_code}) from '{provider.value}': {response.text}"
                    logger.warning(error_msg)
                    last_exception = LLMError(error_msg, provider=provider.value, status_code=response.status_code)
                    if retries < self.settings.max_retries:
                        wait = min(
                            self.settings.retry_max_wait_seconds,
                            self.settings.retry_min_wait_seconds * (2 ** retries),
                        )
                        await asyncio.sleep(wait)
                        retries += 1
                        continue
                    else:
                        raise last_exception

                else:
                    error_msg = f"Unexpected HTTP status {response.status_code} from '{provider.value}': {response.text}"
                    logger.error(error_msg)
                    raise LLMError(error_msg, provider=provider.value, status_code=response.status_code)

            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                error_msg = f"Network/Timeout exception on provider '{provider.value}': {str(exc)}"
                logger.warning(error_msg)
                last_exception = LLMError(error_msg, provider=provider.value)
                if retries < self.settings.max_retries:
                    wait = min(
                        self.settings.retry_max_wait_seconds,
                        self.settings.retry_min_wait_seconds * (2 ** retries),
                    )
                    await asyncio.sleep(wait)
                    retries += 1
                    continue
                else:
                    raise last_exception

        if last_exception:
            raise last_exception
        raise LLMError(f"Failed to complete request with provider {provider.value}", provider=provider.value)

    async def complete(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
        provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        fallback_on_rate_limit: bool = True,
    ) -> CompletionResponse:
        """
        Unified completion method.
        If provider is specified, attempts that provider first.
        If provider rate-limits / fails and fallback_on_rate_limit is True,
        falls back sequentially through the fallback order (Groq -> Gemini -> NVIDIA).
        """
        formatted_messages = self._format_messages(prompt=prompt, messages=messages)

        # Determine target provider and fallback list
        target_provider: Optional[LLMProvider] = None
        if provider is not None:
            if isinstance(provider, str):
                target_provider = LLMProvider.from_str(provider)
            else:
                target_provider = provider

        # Build execution order
        provider_order: List[LLMProvider] = []
        if target_provider:
            provider_order.append(target_provider)
            if fallback_on_rate_limit:
                for prov_str in self.settings.fallback_order:
                    p = LLMProvider.from_str(prov_str)
                    if p not in provider_order:
                        provider_order.append(p)
        else:
            for prov_str in self.settings.fallback_order:
                provider_order.append(LLMProvider.from_str(prov_str))

        errors: Dict[str, Exception] = {}
        for prov in provider_order:
            try:
                current_model = model if (target_provider is None or prov == target_provider) else None
                response = await self._call_provider_with_retry(
                    provider=prov,
                    messages=formatted_messages,
                    model=current_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stop=stop,
                )
                return response
            except (LLMRateLimitError, LLMError, LLMAuthenticationError) as exc:
                logger.warning(f"Provider '{prov.value}' failed: {exc}. Trying fallback...")
                errors[prov.value] = exc
                if not fallback_on_rate_limit:
                    raise exc
                continue

        raise LLMAllProvidersFailedError(errors)

    def complete_sync(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Union[ChatMessage, Dict[str, str]]]] = None,
        provider: Optional[Union[str, LLMProvider]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        fallback_on_rate_limit: bool = True,
    ) -> CompletionResponse:
        """Synchronous wrapper for complete()."""
        return asyncio.run(
            self.complete(
                prompt=prompt,
                messages=messages,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
                fallback_on_rate_limit=fallback_on_rate_limit,
            )
        )
