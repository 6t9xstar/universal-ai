"""Ollama provider for UniversalAI."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

from universal_ai.exceptions import ProviderError
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

logger = logging.getLogger("universal_ai.providers.ollama")


class OllamaProvider:
    name = "ollama"
    version = "1.0.0"
    default_model = "llama3"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=False,
        audio=False,
        embeddings=True,
        tools=False,
        json_mode=False,
        system_message=True,
        max_tokens=8192,
        context_window=8192,
        supported_models=[],
    )

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or "http://localhost:11434"
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        return self._client

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            body["options"]["num_predict"] = request.max_tokens
        client = self._get_client()
        resp = await client.post("/api/chat", json=body)
        if resp.status_code == 404:
            msg = "ollama"
            raise ProviderError(
                msg,
                f"Model {body['model']} not found. Run: ollama pull {body['model']}",
            )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")
        usage_data = data.get("usage", {})
        prompt_tokens = usage_data.get("prompt_eval_count", 0)
        completion_tokens = usage_data.get("eval_count", 0)
        usage = Usage(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        done = data.get("done", True)
        finish_reason = FinishReason.stop if done else FinishReason.length
        return ChatResponse(
            content=content,
            model=data.get("model", body["model"]),
            provider="ollama",
            usage=usage,
            finish_reason=finish_reason,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            body["options"]["num_predict"] = request.max_tokens
        client = self._get_client()
        async with client.stream("POST", "/api/chat", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    message = data.get("message", {})
                    content = message.get("content", "")
                    done = data.get("done", False)
                    if content:
                        yield ChatChunk(
                            content=content,
                            delta=content,
                            model=data.get("model", ""),
                        )
                    if done:
                        usage_data = data.get("usage", {})
                        prompt_tokens = usage_data.get("prompt_eval_count", 0)
                        completion_tokens = usage_data.get("eval_count", 0)
                        usage = Usage(
                            input_tokens=prompt_tokens,
                            output_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )
                        yield ChatChunk(finish_reason=FinishReason.stop, usage=usage)
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        body = {
            "model": request.model,
            "prompt": request.input if isinstance(request.input, str) else request.input[0],
        }
        client = self._get_client()
        resp = await client.post("/api/embeddings", json=body)
        resp.raise_for_status()
        data = resp.json()
        vector = data.get("embedding", [])
        return EmbedResponse(
            vector=vector,
            model=data.get("model", request.model),
            provider="ollama",
            usage=Usage(),
            raw=data,
        )

    async def health_check(self) -> dict:
        try:
            client = self._get_client()
            resp = await client.get("/api/tags")
            models = resp.json().get("models", [])
            return {
                "healthy": resp.status_code == 200,
                "models": [m.get("name", "") for m in models],
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    async def get_latency(self) -> float:
        return 0.05

    def calculate_cost(self, estimated_tokens: int) -> float:
        return 0.0

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"OllamaProvider(base_url={self.base_url!r})"
