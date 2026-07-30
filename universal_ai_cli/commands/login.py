"""Login/logout CLI commands for provider authentication."""

from __future__ import annotations

import getpass
import json
import os
import sys
from pathlib import Path

PROVIDER_ENV_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "huggingface": "HF_API_KEY",
    "hf": "HF_API_KEY",
}

PROVIDER_DESCRIPTIONS = {
    "openai": "OpenAI (GPT-4, GPT-3.5)",
    "anthropic": "Anthropic (Claude)",
    "gemini": "Google Gemini",
    "groq": "Groq (fast inference)",
    "mistral": "Mistral AI",
    "openrouter": "OpenRouter (multi-provider)",
    "huggingface": "HuggingFace Inference API",
}


def get_credentials_path() -> Path:
    creds_dir = Path.home() / ".config" / "universal-ai"
    creds_dir.mkdir(parents=True, exist_ok=True)
    return creds_dir / "credentials.json"


def load_credentials() -> dict:
    creds_path = get_credentials_path()
    if creds_path.exists():
        try:
            with open(creds_path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_credentials(creds: dict) -> None:
    creds_path = get_credentials_path()
    with open(creds_path, "w") as f:
        json.dump(creds, f, indent=2)
    if os.name != "nt":
        os.chmod(creds_path, 0o600)


def setenv_if_not_set(env_var: str, value: str) -> None:
    if env_var not in os.environ:
        os.environ[env_var] = value


async def cmd_login(provider: str) -> None:
    provider = provider.lower()
    if provider not in PROVIDER_ENV_MAP:
        print(f"Error: Unknown provider '{provider}'", file=sys.stderr)
        print("\nSupported providers:")
        for name, desc in PROVIDER_DESCRIPTIONS.items():
            env_var = PROVIDER_ENV_MAP.get(name)
            status = "local" if env_var is None else "requires API key"
            print(f"  {name:<15} {desc} ({status})")
        sys.exit(1)
    env_var = PROVIDER_ENV_MAP[provider]
    print(f"Login to {provider}")
    print(f"Provider: {PROVIDER_DESCRIPTIONS.get(provider, provider)}")
    print()
    existing = os.getenv(env_var)
    if existing:
        masked = existing[:8] + "..." if len(existing) > 8 else "***"
        print(f"Already authenticated (key: {masked})")
        confirm = input("Do you want to replace it? (y/N): ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    api_key = getpass.getpass(f"Enter API key for {provider}: ")
    if not api_key:
        print("Cancelled.")
        return
    confirm = getpass.getpass("Confirm API key: ")
    if api_key != confirm:
        print("Error: API keys do not match", file=sys.stderr)
        sys.exit(1)
    creds = load_credentials()
    creds[provider] = api_key
    save_credentials(creds)
    os.environ[env_var] = api_key
    print(f"\nSuccessfully logged in to {provider}!")
    try:
        from universal_ai import AI

        ai = AI()
        await ai.set_provider(provider)
        status = await ai.provider_status()
        if provider in status and status[provider].get("healthy"):
            print(f"Connection verified for {provider}")
        else:
            print(f"Could not verify connection for {provider}")
        await ai.close()
    except Exception as e:
        print(f"Could not verify connection: {e}")


async def cmd_logout(provider: str) -> None:
    provider = provider.lower()
    if provider not in PROVIDER_ENV_MAP:
        print(f"Error: Unknown provider '{provider}'", file=sys.stderr)
        sys.exit(1)
    env_var = PROVIDER_ENV_MAP[provider]
    creds = load_credentials()
    if provider in creds:
        del creds[provider]
        save_credentials(creds)
    if env_var in os.environ:
        del os.environ[env_var]
    print(f"Logged out from {provider}.")


async def cmd_login_status() -> None:
    print("Provider Login Status")
    print("=" * 50)
    creds = load_credentials()
    for provider, env_var in PROVIDER_ENV_MAP.items():
        desc = PROVIDER_DESCRIPTIONS.get(provider, provider)
        in_env = bool(os.getenv(env_var))
        in_creds = provider in creds
        if in_env:
            status = "ENV"
        elif in_creds:
            status = "CREDS"
        else:
            status = "NOT LOGGED IN"
        print(f"  {provider:<15} {desc:<30} [{status}]")
    print("\nUse 'uai login <provider>' to authenticate.")


async def handle_login_command(args) -> None:
    if getattr(args, "status", False):
        await cmd_login_status()
    else:
        provider = getattr(args, "provider", None)
        if not provider:
            await cmd_login_status()
        else:
            await cmd_login(provider)


async def handle_logout_command(args) -> None:
    provider = getattr(args, "provider", None)
    if not provider:
        print("Error: Provider name required", file=sys.stderr)
        sys.exit(1)
    await cmd_logout(provider)


def create_login_parser(subparsers) -> None:
    p = subparsers.add_parser("login", help="Login to a provider")
    p.add_argument("provider", nargs="?", help="Provider name")
    p.add_argument("--status", action="store_true", help="Show login status")
    return p


def create_logout_parser(subparsers) -> None:
    p = subparsers.add_parser("logout", help="Logout from a provider")
    p.add_argument("provider", help="Provider name")
    return p
