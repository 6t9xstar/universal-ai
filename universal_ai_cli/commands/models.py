"""Models management CLI commands."""

from __future__ import annotations

import sys


async def cmd_models_list() -> None:
    from universal_ai.utils.plugin_registry import PluginRegistry

    registry = PluginRegistry()
    providers = registry.list_providers()
    if not providers:
        print("No providers available. Install a provider package.")
        return
    print("Available models by provider:\n")
    for provider_name in providers:
        try:
            provider = registry.get_provider(provider_name)
            if provider is None:
                continue
            print(f"  {provider_name}:")
            if hasattr(provider, "supported_models"):
                models = provider.supported_models
                if models:
                    for model in models[:10]:
                        print(f"    - {model}")
                    if len(models) > 10:
                        print(f"    ... and {len(models) - 10} more")
                else:
                    print(f"    (default: {getattr(provider, 'default_model', 'unknown')})")
            else:
                default = getattr(provider, "default_model", "unknown")
                print(f"    (default: {default})")
        except Exception as e:
            print(f"  {provider_name}: Error listing models - {e}")
    print("\nUse 'uai models info <model>' for details on a specific model.")


async def cmd_models_pull(model: str, provider: str | None = None) -> None:
    from universal_ai.utils.plugin_registry import PluginRegistry

    if provider is None:
        provider = "ollama"
    registry = PluginRegistry()
    provider_instance = registry.get_provider(provider)
    if provider_instance is None:
        print(f"Error: Provider '{provider}' not found", file=sys.stderr)
        sys.exit(1)
    if not hasattr(provider_instance, "pull_model"):
        print(f"Error: Provider '{provider}' does not support model pulling", file=sys.stderr)
        print("Model pulling is supported by: ollama, llamacpp")
        sys.exit(1)
    print(f"Pulling model '{model}' via {provider}...")
    try:
        async for progress in provider_instance.pull_model(model):
            if isinstance(progress, dict):
                status = progress.get("status", "")
                completed = progress.get("completed", 0)
                total = progress.get("total", 0)
                if total > 0:
                    percent = (completed / total) * 100
                    print(f"\r  {status}: {percent:.1f}% ({completed}/{total})", end="", flush=True)
                else:
                    print(f"\r  {status}", end="", flush=True)
            else:
                print(f"\r  {progress}", end="", flush=True)
        print("\n\nModel pulled successfully!")
    except Exception as e:
        print(f"\n\nError pulling model: {e}", file=sys.stderr)
        sys.exit(1)


async def handle_models_command(args) -> None:
    action = getattr(args, "action", "list") or "list"
    if action == "list":
        await cmd_models_list()
    elif action == "pull":
        name = getattr(args, "name", None)
        if not name:
            print("Error: Model name required for pull", file=sys.stderr)
            sys.exit(1)
        provider = getattr(args, "provider", None)
        await cmd_models_pull(name, provider)
    elif action == "remove":
        name = getattr(args, "name", None)
        if not name:
            print("Error: Model name required for remove", file=sys.stderr)
            sys.exit(1)
        print(f"Removing model '{name}' is not yet implemented.")
    elif action == "info":
        name = getattr(args, "name", None)
        if not name:
            print("Error: Model name required for info", file=sys.stderr)
            sys.exit(1)
        print(f"Model information: {name}\n(Model details are provider-specific.)")


def create_models_parser(subparsers) -> None:
    p = subparsers.add_parser("models", help="Manage AI models")
    p.add_argument("action", nargs="?", default="list", choices=["list", "pull", "remove", "info"])
    p.add_argument("name", nargs="?", help="Model name")
    p.add_argument("-p", "--provider", help="Provider to use")
    return p
