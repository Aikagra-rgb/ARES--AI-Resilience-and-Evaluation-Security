import pytest
import httpx
from unittest.mock import AsyncMock, patch

from ares.config import Settings
from ares.llm.client import LLMClient
from ares.llm.models import (
    ChatMessage,
    CompletionResponse,
    LLMAllProvidersFailedError,
    LLMAuthenticationError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
)


@pytest.fixture
def mock_settings():
    return Settings(
        groq_api_key="mock-groq-key",
        groq_base_url="https://api.groq.com/openai/v1",
        groq_default_model="openai/gpt-oss-120b",
        gemini_api_key="mock-gemini-key",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        gemini_default_model="gemini-3.7-flash",
        nvidia_api_key="mock-nvidia-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_default_model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        fallback_order=["groq", "gemini", "nvidia"],
        max_retries=2,
        retry_min_wait_seconds=0.01,
        retry_max_wait_seconds=0.05,
    )


def make_mock_openai_response(content: str = "Test response", model: str = "test-model"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123456789,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }


@pytest.mark.asyncio
async def test_llm_client_successful_groq_call(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Hello from Groq", model="openai/gpt-oss-120b"),
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await client.complete(prompt="Hi", provider="groq")

        assert isinstance(res, CompletionResponse)
        assert res.text == "Hello from Groq"
        assert res.provider_used == "groq"
        assert res.model_used == "openai/gpt-oss-120b"
        assert res.prompt_tokens == 10
        assert res.completion_tokens == 20
        assert res.total_tokens == 30
        assert res.latency_ms > 0

        # Check call args
        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == "https://api.groq.com/openai/v1/chat/completions"
        call_headers = mock_post.call_args[1]["headers"]
        assert call_headers["Authorization"] == "Bearer mock-groq-key"


@pytest.mark.asyncio
async def test_llm_client_successful_ollama_call(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Hello from Ollama", model="llama3.1:8b"),
        request=httpx.Request("POST", "http://localhost:11434/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await client.complete(prompt="Hi", provider="ollama")

        assert isinstance(res, CompletionResponse)
        assert res.text == "Hello from Ollama"
        assert res.provider_used == "ollama"
        assert res.model_used == "llama3.1:8b"

        mock_post.assert_called_once()
        call_url = mock_post.call_args[0][0]
        assert call_url == "http://localhost:11434/v1/chat/completions"


@pytest.mark.asyncio
async def test_llm_client_successful_gemini_call(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Hello from Gemini", model="gemini-3.7-flash"),
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await client.complete(prompt="Hi", provider="gemini")

        assert res.text == "Hello from Gemini"
        assert res.provider_used == "gemini"
        assert res.model_used == "gemini-3.7-flash"


@pytest.mark.asyncio
async def test_llm_client_successful_nvidia_call(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Hello from NVIDIA", model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        res = await client.complete(prompt="Hi", provider="nvidia")

        assert res.text == "Hello from NVIDIA"
        assert res.provider_used == "nvidia"


@pytest.mark.asyncio
async def test_llm_client_retry_on_429_then_success(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp_429 = httpx.Response(
        status_code=429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    mock_resp_200 = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Success after retry"),
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [mock_resp_429, mock_resp_200]

        res = await client.complete(prompt="Retry test", provider="groq")

        assert res.text == "Success after retry"
        assert res.provider_used == "groq"
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_llm_client_fallback_groq_to_gemini(mock_settings):
    """When Groq fails with 429 on all retries, it should fall back to Gemini."""
    client = LLMClient(settings=mock_settings)
    mock_resp_429 = httpx.Response(
        status_code=429,
        text="Groq Rate limit exceeded",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    mock_resp_gemini_200 = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("Gemini fallback success", model="gemini-3.7-flash"),
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = [
            mock_resp_429,
            mock_resp_429,
            mock_resp_429,
            mock_resp_gemini_200,
        ]

        res = await client.complete(prompt="Fallback test", provider="groq", fallback_on_rate_limit=True)

        assert res.text == "Gemini fallback success"
        assert res.provider_used == "gemini"


@pytest.mark.asyncio
async def test_llm_client_fallback_groq_to_gemini_to_nvidia(mock_settings):
    """When Groq and Gemini fail, fallback to NVIDIA."""
    client = LLMClient(settings=mock_settings)
    mock_resp_429 = httpx.Response(
        status_code=429,
        text="Rate limit exceeded",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    mock_resp_503 = httpx.Response(
        status_code=503,
        text="Service unavailable",
        request=httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"),
    )
    mock_resp_nvidia_200 = httpx.Response(
        status_code=200,
        json=make_mock_openai_response("NVIDIA fallback success", model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
        request=httpx.Request("POST", "https://integrate.api.nvidia.com/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = (
            [mock_resp_429] * 3 +
            [mock_resp_503] * 3 +
            [mock_resp_nvidia_200]
        )

        res = await client.complete(prompt="NVIDIA fallback test", provider="groq", fallback_on_rate_limit=True)

        assert res.text == "NVIDIA fallback success"
        assert res.provider_used == "nvidia"


@pytest.mark.asyncio
async def test_llm_client_all_providers_fail_raises_error(mock_settings):
    client = LLMClient(settings=mock_settings)
    mock_resp_500 = httpx.Response(
        status_code=500,
        text="Internal Error",
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp_500

        with pytest.raises(LLMAllProvidersFailedError) as exc_info:
            await client.complete(prompt="All fail test", fallback_on_rate_limit=True)

        assert "All LLM providers failed" in str(exc_info.value)
        assert len(exc_info.value.errors) == 3


def test_chat_messages_formatting(mock_settings):
    client = LLMClient(settings=mock_settings)

    # From string prompt
    msgs1 = client._format_messages(prompt="Hello")
    assert msgs1 == [{"role": "user", "content": "Hello"}]

    # From ChatMessage list
    msgs2 = client._format_messages(messages=[
        ChatMessage(role="system", content="Be concise"),
        ChatMessage(role="user", content="Ping"),
    ])
    assert msgs2 == [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Ping"},
    ]


# =========================================================================
# Live Integration Tests (using configured API keys)
# =========================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_groq_completion():
    from ares.config import settings
    if not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY not set")

    client = LLMClient()
    response = await client.complete(
        prompt="Respond with exactly one word: 'ONLINE'.",
        provider="groq",
        temperature=0.0,
        max_tokens=256,
    )
    print(f"\n[Live Groq] Used: {response.provider_used} | Model: {response.model_used} | Latency: {response.latency_ms}ms | Text: {response.text.strip()}")
    assert response.provider_used == "groq"
    assert "ONLINE" in response.text.upper()
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_gemini_completion():
    from ares.config import settings
    if not settings.gemini_api_key:
        pytest.skip("GEMINI_API_KEY not set")

    client = LLMClient()
    response = await client.complete(
        prompt="Respond with exactly one word: 'ONLINE'.",
        provider="gemini",
        temperature=0.0,
        max_tokens=256,
    )
    print(f"\n[Live Gemini] Used: {response.provider_used} | Model: {response.model_used} | Latency: {response.latency_ms}ms | Text: {response.text.strip()}")
    assert response.provider_used in ("gemini", "groq")
    assert "ONLINE" in response.text.upper()
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_live_nvidia_completion():
    from ares.config import settings
    if not settings.nvidia_api_key:
        pytest.skip("NVIDIA_API_KEY not set")

    client = LLMClient()
    response = await client.complete(
        prompt="Respond with exactly one word: 'ONLINE'.",
        provider="nvidia",
        temperature=0.0,
        max_tokens=256,
    )
    print(f"\n[Live NVIDIA] Used: {response.provider_used} | Model: {response.model_used} | Latency: {response.latency_ms}ms | Text: {response.text.strip()}")
    assert response.provider_used in ("nvidia", "groq")
    assert "ONLINE" in response.text.upper()
    await client.close()
