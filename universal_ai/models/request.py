from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from universal_ai.models.common import Message, Tool


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    top_p: float = 1.0
    tools: list[Tool] = Field(default_factory=list)
    stream: bool = False
    json_mode: bool = False
    json_schema: type | None = None
    image: str | bytes | None = None
    stop: list[str] = Field(default_factory=list)
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ChatRequest(model={self.model!r}, messages={len(self.messages)}, "
            f"temperature={self.temperature}, stream={self.stream})"
        )


class ChatStreamRequest(ChatRequest):
    stream: bool = True


class EmbedRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: str = "float"
    dimensions: int | None = None

    def __repr__(self) -> str:
        text = self.input if isinstance(self.input, str) else f"[{len(self.input)} items]"
        preview = text[:50] + "..." if len(text) > 50 else text
        return f"EmbedRequest(model={self.model!r}, input={preview!r})"


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    size: str | None = None
    quality: str | None = None
    style: str | None = None
    n: int = 1


class AudioTranscribeRequest(BaseModel):
    audio: str | bytes
    model: str = "whisper-1"
    response_format: str = "json"
    temperature: float = 0.0
    language: str | None = None
    prompt: str | None = None


class AudioSpeakRequest(BaseModel):
    text: str
    model: str = "tts-1"
    voice: str = "alloy"
    response_format: str = "mp3"
    speed: float = 1.0
