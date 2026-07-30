"""Providers CLI commands."""

from __future__ import annotations


async def handle_providers_command(args) -> None:
    from universal_ai.utils.plugin_registry import PluginRegistry

    registry = PluginRegistry()
    providers = registry.list_providers()
    if not providers:
        print("No providers available")
        return
    print("Available providers:")
    for name in providers:
        print(f"  - {name}")
