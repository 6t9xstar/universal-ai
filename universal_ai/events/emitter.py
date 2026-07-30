from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("universal_ai.events")


class EventEmitter:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def off(self, event: str, handler: Callable) -> bool:
        handlers = self._handlers.get(event, [])
        if handler in handlers:
            handlers.remove(handler)
            return True
        return False

    def emit(self, event: str, data: Any = None) -> None:
        handlers = self._handlers.get(event, [])
        for handler in handlers:
            try:
                result = handler(data)
                if asyncio.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(result)
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.exception("Error in event handler for %s: %s", event, e)

    def clear(self, event: str | None = None) -> None:
        if event:
            self._handlers.pop(event, None)
        else:
            self._handlers.clear()

    def handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))

    def events(self) -> list[str]:
        return list(self._handlers.keys())

    def __repr__(self) -> str:
        total = sum(len(h) for h in self._handlers.values())
        return f"EventEmitter(events={len(self._handlers)}, handlers={total})"
