from app.config import Settings
from app.providers.base import ProviderExecutionError, ProviderNotConfiguredError, ProviderResult
from app.providers.groq_provider import GroqAdapter
from app.providers.ollama_provider import OllamaAdapter
from app.providers.openai_responses import OpenAIResponsesAdapter


class ProviderRegistry:
    def __init__(self, settings: Settings):
        self._openai = OpenAIResponsesAdapter(settings)
        self._groq = GroqAdapter(settings)
        self._ollama = OllamaAdapter(settings)

    def ensure_configured(self, provider: str) -> None:
        p_lower = provider.lower()
        if p_lower in ("openai", "open_ai"):
            if not self._openai.is_configured():
                raise ProviderNotConfiguredError("OpenAI is not configured on this server.")
            return
        if p_lower == "groq":
            if not self._groq.is_configured():
                raise ProviderNotConfiguredError("Groq is not configured on this server.")
            return
        if p_lower in ("ollama", "local"):
            if not self._ollama.is_configured():
                raise ProviderNotConfiguredError("Ollama is not configured on this server.")
            return
        raise ProviderNotConfiguredError(f"{provider} is not enabled on this server.")

    async def execute(self, provider: str, payload: dict, user_id: str) -> ProviderResult:
        self.ensure_configured(provider)
        p_lower = provider.lower()
        if p_lower in ("openai", "open_ai"):
            return await self._openai.execute(payload, user_id)
        if p_lower == "groq":
            return await self._groq.execute(payload, user_id)
        if p_lower in ("ollama", "local"):
            return await self._ollama.execute(payload, user_id)
        raise ProviderExecutionError("No adapter is available for this provider.")
