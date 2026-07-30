from __future__ import annotations

import logging
from typing import Any

from universal_ai.tools import ToolExecutor

logger = logging.getLogger("universal_ai.agents.tool_router")


class ToolRouter:
    def __init__(self) -> None:
        self._executor = ToolExecutor()
        self._routing_table: dict[str, str] = {}

    def register_tool(self, func: Any) -> Any:
        self._executor.register(func)
        return func

    async def route(self, tool_name: str, arguments: str) -> str:
        import json

        func = self._executor.get_tool(tool_name)
        if func is None:
            return f"Error: tool '{tool_name}' not found"
        try:
            try:
                args = json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            if callable(func):
                import inspect

                if inspect.iscoroutinefunction(func):
                    result = await func(**args)
                else:
                    result = func(**args)
                return str(result)
            return str(func)
        except Exception as e:
            return f"Error: {e}"

    def get_available_tools(self) -> list[str]:
        return list(self._executor._tools.keys())

    def __repr__(self) -> str:
        return f"ToolRouter(tools={len(self._executor._tools)})"
