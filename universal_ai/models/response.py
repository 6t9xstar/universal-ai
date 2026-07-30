from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from universal_ai.models.common import FinishReason, ToolCall, Usage


class ChatResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: Usage
    finish_reason: FinishReason = FinishReason.stop
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.content

    @property
    def tokens(self) -> int:
        return self.usage.total_tokens

    @property
    def cost(self) -> float:
        return self.usage.estimated_cost

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"ChatResponse(model={self.model!r}, provider={self.provider!r}, "
            f"tokens={self.tokens}, content={preview!r})"
        )


class ChatChunk(BaseModel):
    content: str = ""
    delta: str = ""
    model: str = ""
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def __repr__(self) -> str:
        delta_preview = self.delta[:30] + "..." if len(self.delta) > 30 else self.delta
        return f"ChatChunk(delta={delta_preview!r}, model={self.model!r})"


class EmbedResponse(BaseModel):
    vector: list[float]
    model: str
    provider: str
    usage: Usage
    raw: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EmbedResponse(model={self.model!r}, provider={self.provider!r}, "
            f"dimensions={len(self.vector)})"
        )


class ToolResponse(BaseModel):
    content: str
    model: str
    provider: str
    usage: Usage
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = FinishReason.tool_calls
    raw: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ToolResponse(model={self.model!r}, provider={self.provider!r}, "
            f"tool_calls={len(self.tool_calls)})"
        )


class ImageResponse(BaseModel):
    images: list[str]
    model: str
    provider: str
    usage: Usage
    raw: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ImageResponse(model={self.model!r}, provider={self.provider!r}, "
            f"images={len(self.images)})"
        )


class AudioResponse(BaseModel):
    audio: bytes
    model: str
    provider: str
    usage: Usage
    format: str = "mp3"

    def __repr__(self) -> str:
        return (
            f"AudioResponse(model={self.model!r}, provider={self.provider!r}, "
            f"bytes={len(self.audio)})"
        )
