from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from universal_ai.models.common import Message, MessageRole

logger = logging.getLogger("universal_ai.agents.memory")


class AgentMemory:
    def __init__(self, backend: str = "memory", max_turns: int = 100) -> None:
        self.backend = backend
        self.max_turns = max_turns
        self._memory: dict[str, list[Message]] = {}

    def add(self, agent_name: str, message: Message) -> None:
        if agent_name not in self._memory:
            self._memory[agent_name] = []
        self._memory[agent_name].append(message)
        if len(self._memory[agent_name]) > self.max_turns:
            self._memory[agent_name] = self._memory[agent_name][-self.max_turns :]

    def get(self, agent_name: str, n: int = 10) -> list[Message]:
        return self._memory.get(agent_name, [])[-n:]

    def get_all(self, agent_name: str) -> list[Message]:
        return list(self._memory.get(agent_name, []))

    def clear(self, agent_name: str | None = None) -> None:
        if agent_name:
            self._memory.pop(agent_name, None)
        else:
            self._memory.clear()

    def save(self, path: str | None = None) -> None:
        path = path or os.path.join(Path.home(), ".config", "universal-ai", "memory.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            name: [{"role": m.role.value, "content": m.content} for m in messages]
            for name, messages in self._memory.items()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str | None = None) -> None:
        path = path or os.path.join(Path.home(), ".config", "universal-ai", "memory.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for name, messages in data.items():
                self._memory[name] = [
                    Message(role=MessageRole(m.get("role", "user")), content=m.get("content", ""))
                    for m in messages
                ]
        except Exception as e:
            logger.warning("Failed to load memory: %s", e)
