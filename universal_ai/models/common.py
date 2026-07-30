from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel


class MessageRole(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class FinishReason(str, Enum):
    stop = "stop"
    length = "length"
    tool_calls = "tool_calls"
    content_filter = "content_filter"
    error = "error"
    unknown = "unknown"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0
    cached_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"Usage(input={self.input_tokens}, output={self.output_tokens}, "
            f"total={self.total_tokens}, cost=${self.estimated_cost:.6f})"
        )


class ToolCall(BaseModel):
    id: str
    function_name: str
    arguments: str

    def __repr__(self) -> str:
        return f"ToolCall(id={self.id!r}, function={self.function_name!r})"


class Tool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = {}
    strict: bool = False

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r})"


class Message(BaseModel):
    role: MessageRole
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    images: list[str] = []
    audio: str | None = None

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Message(role={self.role.value}, content={preview!r})"


class ProviderCapabilities(BaseModel):
    chat: bool = True
    streaming: bool = True
    vision: bool = False
    audio: bool = False
    embeddings: bool = False
    tools: bool = False
    json_mode: bool = False
    system_message: bool = True
    max_tokens: int = 4096
    context_window: int = 4096
    supported_models: list[str] = []
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0
