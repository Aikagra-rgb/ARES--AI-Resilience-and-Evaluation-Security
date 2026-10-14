"""
ARES Python SDK Client
"""
import httpx
from typing import Any, Dict, Optional
from ares_sdk.config import AresConfig
from ares_sdk.exceptions import AresAPIError, AresAuthenticationError, AresRateLimitError

class AresClient:
    """Async Client for interfacing with the ARES Prompt Compiler & Resilience Engine."""

    def __init__(self, api_key: str, base_url: str = "http://localhost:8000"):
        self.config = AresConfig(api_key=api_key, base_url=base_url)
        self.client = httpx.AsyncClient(
            headers={"X-Ares-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=self.config.timeout,
        )

    async def optimize_prompt(
        self,
        prompt: str,
        domain: Optional[str] = None,
        target_fidelity: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Optimize and compress a user prompt via the ARES API.
        """
        payload = {
            "prompt": prompt,
            "domain": domain,
            "target_fidelity": target_fidelity,
        }
        try:
            response = await self.client.post(
                f"{self.config.base_url}/api/v1/optimizer/optimize",
                json=payload,
            )
            if response.status_code == 401 or response.status_code == 403:
                raise AresAuthenticationError(f"Authentication failed: {response.text}")
            elif response.status_code == 429:
                raise AresRateLimitError(f"Rate limit exceeded: {response.text}")
            elif response.status_code != 200:
                raise AresAPIError(f"Optimization failed ({response.status_code}): {response.text}")

            return response.json()
        except httpx.RequestError as exc:
            raise AresAPIError(f"Network error connecting to ARES API: {exc}") from exc

    async def health(self) -> Dict[str, Any]:
        """Check system health status."""
        response = await self.client.get(f"{self.config.base_url}/health")
        return response.json()

    async def close(self):
        """Close underlying HTTP client connection."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
