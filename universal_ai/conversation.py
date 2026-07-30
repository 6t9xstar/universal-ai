from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from universal_ai.models.common import Message, MessageRole

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.models.response import ChatChunk

logger = logging.getLogger("universal_ai.conversation")


class Conversation:
    def __init__(
        self,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        ai: Any | None = None,
        tools: list[Any] | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._ai = ai
        self._tools = tools or []
        self._history: list[Message] = []
        self._turn_count = 0
        self._total_cost: float = 0.0
        self._total_tokens: int = 0

    @property
    def history(self) -> list[Message]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def add_user_message(self, content: str) -> None:
        self._history.append(Message(role=MessageRole.user, content=content))

    def add_user(self, content: str) -> None:
        self.add_user_message(content)

    def add_assistant_message(self, content: str) -> None:
        self._history.append(Message(role=MessageRole.assistant, content=content))

    def add_assistant(self, content: str) -> None:
        self.add_assistant_message(content)

    def add_system_message(self, content: str) -> None:
        self._history.append(Message(role=MessageRole.system, content=content))

    def add_system(self, content: str) -> None:
        self.add_system_message(content)

    def _build_messages(self) -> list[Message]:
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role=MessageRole.system, content=self._system_prompt))
        messages.extend(self._history)
        return messages

    async def send(
        self,
        ai: Any | None = None,
        message: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if ai is None:
            ai = self._ai
        if ai is None:
            msg = "No AI instance provided"
            raise ValueError(msg)
        if message is not None:
            self.add_user_message(message)
        if self._max_turns and self._turn_count >= self._max_turns:
            msg = f"Max turns ({self._max_turns}) reached for this conversation"
            raise ValueError(msg)
        messages = self._build_messages()
        tool_list = tools if tools is not None else self._tools
        response = await ai.chat(messages, tools=tool_list, **kwargs)
        self.add_assistant_message(response.content)
        self._turn_count += 1
        if hasattr(response, "usage") and response.usage:
            self._total_tokens += response.usage.total_tokens
            self._total_cost += getattr(response.usage, "estimated_cost", 0.0) or 0.0
        return response

    async def stream(
        self,
        ai: Any | None = None,
        message: str | None = None,
        tools: list[Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        if ai is None:
            ai = self._ai
        if ai is None:
            msg = "No AI instance provided"
            raise ValueError(msg)
        if message is not None:
            self.add_user_message(message)
        if self._max_turns and self._turn_count >= self._max_turns:
            msg = f"Max turns ({self._max_turns}) reached for this conversation"
            raise ValueError(msg)
        messages = self._build_messages()
        tool_list = tools if tools is not None else self._tools
        content = ""
        async for chunk in ai.stream(messages, tools=tool_list, **kwargs):
            yield chunk
            content += chunk.delta
            if chunk.usage:
                self._total_tokens += chunk.usage.total_tokens
                self._total_cost += getattr(chunk.usage, "estimated_cost", 0.0) or 0.0
        self.add_assistant_message(content)
        self._turn_count += 1

    def reset(self) -> None:
        self._history.clear()
        self._turn_count = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def truncate(self, keep_last: int = 10) -> None:
        if len(self._history) > keep_last:
            self._history = self._history[-keep_last:]

    @property
    def tools(self) -> list[Any]:
        return list(self._tools)

    @tools.setter
    def tools(self, value: list[Any]) -> None:
        self._tools = list(value)

    def last(self, n: int = 1) -> list[Message]:
        return self._history[-n:]

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"Conversation(turns={self._turn_count}, messages={len(self._history)})"
