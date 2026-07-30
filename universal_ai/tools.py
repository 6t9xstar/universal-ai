from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

from universal_ai.models.common import Tool, ToolCall

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("universal_ai.tools")


def tool(func: Callable) -> Callable:
    func._universal_ai_tool = True

    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):
            continue
        param_info: dict[str, Any] = {"type": "string"}
        if param.annotation != inspect.Parameter.empty:
            annotation = param.annotation
            if hasattr(annotation, "__name__"):
                annotation = annotation.__name__
            elif hasattr(annotation, "__origin__"):
                annotation = str(annotation)
            type_map = {
                "str": "string",
                "int": "integer",
                "float": "number",
                "bool": "boolean",
                "list": "array",
                "dict": "object",
            }
            param_info["type"] = type_map.get(annotation, annotation)
        if param.default == inspect.Parameter.empty:
            required.append(name)
        else:
            param_info["description"] = f"Default: {param.default!r}"
        properties[name] = param_info

    func._tool_definition = Tool(
        name=func.__name__,
        description=func.__doc__ or "",
        parameters={"type": "object", "properties": properties, "required": required},
    )
    func._tool_definition_dict = {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }
    func._is_tool = True
    func._universal_ai_tool = True
    return func


class ToolExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {}

    def register(self, func: Callable) -> Callable:
        if hasattr(func, "_universal_ai_tool"):
            self._tools[func.__name__] = func
        return func

    def register_tool(self, func: Callable) -> None:
        self._tools[func.__name__] = func

    def get_tool_definitions(self) -> list[Tool]:
        tools: list[Tool] = []
        for name, func in self._tools.items():
            if hasattr(func, "_tool_definition"):
                tools.append(func._tool_definition)
            else:
                tools.append(
                    Tool(
                        name=name,
                        description=func.__doc__ or "",
                        parameters={"type": "object", "properties": {}},
                    )
                )
        return tools

    def get_tool(self, name: str) -> Callable | None:
        return self._tools.get(name)

    async def execute(self, tool_call: ToolCall) -> str:
        func = self._tools.get(tool_call.function_name)
        if func is None:
            return f"Error: tool '{tool_call.function_name}' not found"
        try:
            import json

            try:
                arguments = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)
            return str(result)
        except Exception as e:
            logger.exception("Tool execution failed for %s: %s", tool_call.function_name, e)
            return f"Error: {e}"

    async def execute_all(self, tool_calls: list[ToolCall]) -> list[str]:
        import asyncio

        return list(await asyncio.gather(*[self.execute(tc) for tc in tool_calls]))

    def clear(self) -> None:
        self._tools.clear()
