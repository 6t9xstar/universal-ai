from __future__ import annotations

from typing import TYPE_CHECKING, Any

from universal_ai.events.emitter import EventEmitter
from universal_ai.models.response import ChatChunk, ChatResponse, Usage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class StreamManager:
    def __init__(self, provider: Any, event_emitter: EventEmitter | None = None) -> None:
        self._provider = provider
        self._event_emitter = event_emitter or EventEmitter()

    async def stream(self, request: Any) -> AsyncIterator[ChatChunk]:
        if self._provider is None:
            msg = "No provider set for StreamManager"
            raise ValueError(msg)
        self._event_emitter.emit("stream.start", {"request": request})
        async for chunk in self._provider.chat_stream(request):
            self._event_emitter.emit("stream.chunk", chunk)
            yield chunk
        self._event_emitter.emit("stream.complete", None)


class StreamBuffer:
    def __init__(self) -> None:
        self._chunks: list[ChatChunk] = []

    async def collect(self, stream: AsyncIterator[ChatChunk]) -> ChatResponse:
        content = ""
        model = ""
        provider = ""
        usage = Usage()
        finish_reason = None
        async for chunk in stream:
            self._chunks.append(chunk)
            content += chunk.delta
            model = chunk.model or model
            provider = getattr(chunk, "provider", "") or provider
            if chunk.usage:
                usage = chunk.usage
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason
        return ChatResponse(
            content=content,
            model=model,
            provider=provider,
            usage=usage,
            finish_reason=finish_reason,
        )

    @property
    def chunks(self) -> list[ChatChunk]:
        return list(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
