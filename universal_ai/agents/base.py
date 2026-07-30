from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from universal_ai.models import ChatRequest, Message, MessageRole

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from universal_ai.ai import AI

logger = logging.getLogger("universal_ai.agents")


class Agent:
    def __init__(
        self,
        name: str = "agent",
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = "auto",
        tools: list[Any] | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.tools = tools or []
        self._ai: AI | None = None
        self._history: list[Message] = []

    def set_ai(self, ai: AI) -> None:
        self._ai = ai

    async def run(self, message: str, ai: AI | None = None) -> Any:
        if ai is None:
            ai = self._ai
        if ai is None:
            msg = "No AI instance provided. Call set_ai() first."
            raise RuntimeError(msg)
        if not self._history:
            self._history.append(Message(role=MessageRole.system, content=self.system_prompt))
        self._history.append(Message(role=MessageRole.user, content=message))
        request = ChatRequest(
            messages=self._history,
            model=self.model,
            tools=self.tools,
        )
        if self.tools:
            request.tools = self.tools
        response = await ai.chat(request)
        self._history.append(Message(role=MessageRole.assistant, content=response.content))
        return response

    async def stream(self, message: str, ai: AI | None = None) -> AsyncIterator[Any]:
        if ai is None:
            ai = self._ai
        if ai is None:
            msg = "No AI instance provided."
            raise RuntimeError(msg)
        if not self._history:
            self._history.append(Message(role=MessageRole.system, content=self.system_prompt))
        self._history.append(Message(role=MessageRole.user, content=message))
        request = ChatRequest(
            messages=self._history,
            model=self.model,
            tools=self.tools,
            stream=True,
        )
        async for chunk in ai.stream(request):
            yield chunk

    def reset(self) -> None:
        self._history.clear()

    def last(self, n: int = 1) -> list[Message]:
        return self._history[-n:]

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, model={self.model!r})"


class Coordinator:
    def __init__(self, agents: list[Agent] | None = None) -> None:
        self.agents = agents or []

    def add_agent(self, agent: Agent) -> None:
        self.agents.append(agent)

    def remove_agent(self, agent_name: str) -> bool:
        for i, a in enumerate(self.agents):
            if a.name == agent_name:
                self.agents.pop(i)
                return True
        return False

    def get_agent(self, name: str) -> Agent | None:
        for a in self.agents:
            if a.name == name:
                return a
        return None


class Team:
    def __init__(self, *agents: Agent) -> None:
        self.agents = list(agents)
        self._coordinator = Coordinator(self.agents)

    def add(self, agent: Agent) -> None:
        self.agents.append(agent)

    def remove(self, name: str) -> bool:
        return self._coordinator.remove_agent(name)

    async def run(self, ai: AI, message: str) -> str:
        tasks = [agent.run(message, ai) for agent in self.agents]
        responses = await asyncio.gather(*tasks)
        return "\n".join(f"## {r.provider}\n{r.text}" for r in responses if hasattr(r, "text"))

    def __repr__(self) -> str:
        return f"Team(agents={[a.name for a in self.agents]})"
