"""HuggingFace provider for UniversalAI."""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import httpx

from universal_ai.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
)
from universal_ai.models.common import (
    FinishReason,
    ProviderCapabilities,
    Usage,
)
from universal_ai.models.response import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.models.request import ChatRequest, EmbedRequest

logger = logging.getLogger("universal_ai.providers.huggingface")


class HuggingFaceProvider:
    name = "huggingface"
    version = "1.0.0"
    default_model = "HuggingFaceH4/zephyr-7b-beta"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=False,
        audio=False,
        embeddings=True,
        tools=False,
        json_mode=False,
        system_message=True,
        max_tokens=2048,
        context_window=2048,
        supported_models=[],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY")
        self.api_base = api_base or os.getenv("HF_API_BASE", "https://api-inference.huggingface.co")
        if not self.api_key:
            msg = "huggingface"
            raise AuthenticationError(msg, "HF_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,
            )
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = []
        for m in request.messages:
            messages.append({"role": m.role.value, "content": m.content})
        body = {
            "inputs": messages,
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens or 512,
                "return_full_text": False,
            },
        }
        client = self._get_client()
        model = request.model or self.default_model
        resp = await client.post(f"/models/{model}", json=body)
        if resp.status_code == 403:
            msg = "huggingface"
            raise AuthenticationError(msg, "Invalid API key or model access denied")
        if resp.status_code == 404:
            msg = "huggingface"
            raise ModelNotFoundError(msg, f"Model {model} not found")
        if resp.status_code == 429:
            msg = "huggingface"
            raise RateLimitError(msg, "Rate limit exceeded")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            data = data[0]
        generated_text = data.get("generated_text", "")
        return ChatResponse(
            content=generated_text,
            model=model,
            provider="huggingface",
            usage=Usage(),
            finish_reason=FinishReason.stop,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = []
        for m in request.messages:
            messages.append({"role": m.role.value, "content": m.content})
        body = {
            "inputs": messages,
            "parameters": {
                "temperature": request.temperature,
                "max_new_tokens": request.max_tokens or 512,
                "return_full_text": False,
            },
            "options": {"use_cache": False},
        }
        client = self._get_client()
        model = request.model or self.default_model
        async with client.stream("POST", f"/models/{model}", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        data = data[0]
                    text = data.get("generated_text", "")
                    if text:
                        yield ChatChunk(content=text, delta=text, model=model)
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        inputs = request.input if isinstance(request.input, str) else request.input[0]
        client = self._get_client()
        model = request.model
        resp = await client.post(
            f"/models/{model}",
            json={"inputs": inputs},
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            vector = data[0]
        elif isinstance(data, dict):
            vector = data.get("embedding", [])
        else:
            vector = []
        return EmbedResponse(
            vector=vector,
            model=model,
            provider="huggingface",
            usage=Usage(),
            raw=data,
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/models")
            return {"healthy": resp.status_code == 200}
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.3

    def calculate_cost(self, estimated_tokens: int) -> float:
        return 0.0

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"HuggingFaceProvider(base_url={self.api_base!r})"
