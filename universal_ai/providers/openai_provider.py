"""OpenAI provider for UniversalAI."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import os
from typing import TYPE_CHECKING, Any

import httpx

from universal_ai.exceptions import (
    AuthenticationError,
    ModelNotFoundError,
    RateLimitError,
)
from universal_ai.models.common import (
    FinishReason,
    MessageRole,
    ProviderCapabilities,
    Usage,
)
from universal_ai.models.response import (
    ChatChunk,
    ChatResponse,
    EmbedResponse,
    ToolCall,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.models.request import ChatRequest, EmbedRequest

logger = logging.getLogger("universal_ai.providers.openai")


class OpenAIProvider:
    name = "openai"
    version = "1.0.0"
    default_model = "gpt-4o-mini"
    capabilities = ProviderCapabilities(
        chat=True,
        streaming=True,
        vision=True,
        audio=False,
        embeddings=True,
        tools=True,
        json_mode=True,
        system_message=True,
        max_tokens=128000,
        context_window=128000,
        supported_models=[
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ],
    )

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        if not self.api_key:
            msg = "openai"
            raise AuthenticationError(msg, "OPENAI_API_KEY not set")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.api_base,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
        return self._client

    def _encode_image(self, image: str | bytes) -> str:
        if isinstance(image, bytes):
            mime_type = mimetypes.guess_type("image")[0] or "image/png"
            data = base64.b64encode(image).decode()
            return f"data:{mime_type};base64,{data}"
        if image.startswith(("http://", "https://")):
            return image
        if image.startswith("data:"):
            return image
        path = os.path.expanduser(image)
        if not os.path.exists(path):
            msg = f"Image file not found: {path}"
            raise FileNotFoundError(msg)
        mime_type, _ = mimetypes.guess_type(path)
        mime_type = mime_type or "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"data:{mime_type};base64,{data}"

    def _build_messages(self, request: ChatRequest) -> tuple[list[dict], str | None]:
        messages: list[dict] = []
        system_content: str | None = None
        for m in request.messages:
            if m.role == MessageRole.system:
                system_content = m.content
            else:
                msg: dict = {"role": m.role.value, "content": m.content}
                messages.append(msg)
        if request.image:
            try:
                image_url = self._encode_image(request.image)
                if messages:
                    messages[-1]["content"] = [
                        {"type": "text", "text": messages[-1]["content"]},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ]
            except Exception as e:
                logger.warning("Failed to encode image: %s", e)
        return messages, system_content

    async def chat(self, request: ChatRequest) -> ChatResponse:
        messages, system_content = self._build_messages(request)
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if system_content:
            body["messages"].insert(0, {"role": "system", "content": system_content})
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        if request.stop:
            body["stop"] = request.stop
        if request.frequency_penalty:
            body["frequency_penalty"] = request.frequency_penalty
        if request.presence_penalty:
            body["presence_penalty"] = request.presence_penalty
        if request.top_p != 1.0:
            body["top_p"] = request.top_p
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        if request.tools:
            body["tools"] = [
                t._tool_definition if hasattr(t, "_tool_definition") else t for t in request.tools
            ]
        client = self._get_client()
        resp = await client.post("/chat/completions", json=body)
        if resp.status_code == 401:
            msg = "openai"
            raise AuthenticationError(msg, "Invalid API key")
        if resp.status_code == 404:
            msg = "openai"
            raise ModelNotFoundError(msg, f"Model {body['model']} not found")
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", "60"))
            msg = "openai"
            raise RateLimitError(msg, "Rate limit exceeded", retry_after)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        finish_map = {
            "stop": FinishReason.stop,
            "length": FinishReason.length,
            "tool_calls": FinishReason.tool_calls,
            "content_filter": FinishReason.content_filter,
        }
        finish_reason = finish_map.get(choice.get("finish_reason", "stop"), FinishReason.unknown)
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )
        content = choice.get("message", {}).get("content") or ""
        tool_calls: list[ToolCall] = []
        for tc in choice.get("message", {}).get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tc["id"],
                    function_name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
            )
        return ChatResponse(
            content=content,
            model=data.get("model", body["model"]),
            provider="openai",
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw=data,
        )

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[ChatChunk]:
        messages, system_content = self._build_messages(request)
        body: dict = {
            "model": request.model or self.default_model,
            "messages": messages,
            "temperature": request.temperature,
            "stream": True,
        }
        if system_content:
            body["messages"].insert(0, {"role": "system", "content": system_content})
        if request.max_tokens:
            body["max_tokens"] = request.max_tokens
        client = self._get_client()
        async with client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    content = delta.get("content", "")
                    finish_reason = None
                    if choice.get("finish_reason"):
                        finish_reason = FinishReason(choice["finish_reason"])
                    yield ChatChunk(
                        content=content,
                        delta=content,
                        model=data.get("model", ""),
                        finish_reason=finish_reason,
                    )
                except json.JSONDecodeError:
                    continue

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        body = {"model": request.model, "input": request.input}
        client = self._get_client()
        resp = await client.post("/embeddings", json=body)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data["data"]
        if len(embeddings) == 1:
            vector = embeddings[0]["embedding"]
        else:
            vector = [e["embedding"] for e in embeddings]
        usage_data = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=0,
            total_tokens=usage_data.get("total_tokens", 0),
        )
        return EmbedResponse(
            vector=vector if isinstance(vector, list) else vector[0],
            model=data.get("model", request.model),
            provider="openai",
            usage=usage,
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
        return 0.1

    def calculate_cost(self, estimated_tokens: int) -> float:
        return estimated_tokens * 0.000001

    async def transcribe(self, audio: str | bytes, model: str = "whisper-1", **kwargs: Any) -> str:
        client = self._get_client()
        if isinstance(audio, bytes):
            resp = await client.post(
                "/audio/transcriptions",
                data={"model": model, **kwargs},
                files={"file": ("audio.wav", audio, "audio/wav")},
            )
        else:
            with open(audio, "rb") as f:
                resp = await client.post(
                    "/audio/transcriptions",
                    data={"model": model, **kwargs},
                    files={"file": (os.path.basename(audio), f, "audio/wav")},
                )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "")

    async def speak(self, text: str, model: str = "tts-1", voice: str = "alloy", **kwargs: Any) -> bytes:
        client = self._get_client()
        resp = await client.post(
            "/audio/speech",
            json={"model": model, "input": text, "voice": voice, **kwargs},
        )
        resp.raise_for_status()
        return resp.content

    async def generate_image(self, prompt: str, model: str = "dall-e-3", size: str = "1024x1024", n: int = 1, **kwargs: Any) -> list[str]:
        client = self._get_client()
        resp = await client.post(
            "/images/generations",
            json={"model": model, "prompt": prompt, "size": size, "n": n, **kwargs},
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["url"] for item in data.get("data", [])]

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def __repr__(self) -> str:
        return f"OpenAIProvider(base_url={self.api_base!r})"
