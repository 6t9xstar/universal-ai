"""Configuration management CLI commands."""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import sys
from pathlib import Path

from universal_ai.config import UniversalAIConfig


def get_config_path() -> Path:
    config_dir = Path.home() / ".config" / "universalai"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config() -> dict:
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(config: dict) -> None:
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


async def cmd_config_show() -> None:
    config = load_config()
    env_config = UniversalAIConfig.load()
    print("UniversalAI Configuration")
    print("=" * 50)
    print("\nFile-based config (~/.config/universal-ai/config.json):")
    if config:
        for key, value in config.items():
            if "key" in key.lower() and value:
                display_value = str(value)[:8] + "..." if len(str(value)) > 8 else "***"
            else:
                display_value = value
            print(f"  {key}: {display_value}")
    else:
        print("  (empty)")
    print("\nEnvironment-based config:")
    print(f"  provider: {env_config.provider or '(not set)'}")
    print(f"  model: {env_config.model}")
    print(f"  temperature: {env_config.temperature}")
    print(f"  max_tokens: {env_config.max_tokens}")


async def cmd_config_set(key: str, value: str) -> None:
    config = load_config()
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            with contextlib.suppress(ValueError):
                value = float(value)
    config[key] = value
    save_config(config)
    print(f"Set {key} = {value}")


async def cmd_config_set_api_key(provider: str) -> None:
    provider = provider.lower()
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "huggingface": "HF_API_KEY",
        "hf": "HF_API_KEY",
    }
    env_var = env_var_map.get(provider)
    if not env_var:
        print(f"Error: Unknown provider '{provider}'", file=sys.stderr)
        sys.exit(1)
    print(f"Setting API key for {provider}")
    api_key = getpass.getpass(f"Enter API key for {provider}: ")
    if not api_key:
        print("Cancelled.")
        return
    confirm = getpass.getpass("Confirm API key: ")
    if api_key != confirm:
        print("Error: API keys do not match", file=sys.stderr)
        sys.exit(1)
    os.environ[env_var] = api_key
    config = load_config()
    config[f"{provider}_api_key"] = api_key
    save_config(config)
    print(f"\nAPI key for {provider} saved successfully!")


async def cmd_config_unset(key: str) -> None:
    config = load_config()
    if key in config:
        del config[key]
        save_config(config)
        print(f"Removed config key: {key}")
    else:
        print(f"Key '{key}' not found in config")


async def cmd_config_reset() -> None:
    confirm = input("Are you sure you want to reset all configuration? (y/N): ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return
    config_path = get_config_path()
    if config_path.exists():
        config_path.unlink()
    print("Configuration reset to defaults.")


async def handle_config_command(args) -> None:
    action = getattr(args, "action", "show") or "show"
    if action == "show":
        await cmd_config_show()
    elif action == "set":
        key = getattr(args, "key", None)
        value = getattr(args, "value", None)
        if not key or not value:
            print("Error: Both key and value required for 'config set'", file=sys.stderr)
            sys.exit(1)
        await cmd_config_set(key, value)
    elif action == "set-api-key":
        key = getattr(args, "key", None)
        if not key:
            print("Error: Provider name required for 'config set-api-key'", file=sys.stderr)
            sys.exit(1)
        await cmd_config_set_api_key(key)
    elif action == "unset":
        key = getattr(args, "key", None)
        if not key:
            print("Error: Key required for 'config unset'", file=sys.stderr)
            sys.exit(1)
        await cmd_config_unset(key)
    elif action == "reset":
        await cmd_config_reset()


def create_config_parser(subparsers) -> None:
    p = subparsers.add_parser("config", help="Manage configuration")
    p.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "set", "set-api-key", "unset", "reset"],
    )
    p.add_argument("key", nargs="?", help="Config key or provider name")
    p.add_argument("value", nargs="?", help="Config value")
    return p
