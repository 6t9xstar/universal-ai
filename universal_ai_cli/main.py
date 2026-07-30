"""UniversalAI CLI - uai command."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uai",
        description="UniversalAI - The Requests library for AI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    p = subparsers.add_parser("chat", help="Chat with an AI model")
    p.add_argument("message", nargs="?", help="Message to send")
    p.add_argument("-m", "--model", help="Model to use")
    p.add_argument("-p", "--provider", help="Provider to use")
    p.add_argument("--stream", action="store_true", help="Stream the response")

    p = subparsers.add_parser("providers", help="List available providers")
    p.add_argument("action", nargs="?", choices=["status"], help="Provider action")

    p = subparsers.add_parser("models", help="Manage AI models")
    p.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=["list", "pull", "remove", "info"],
        help="Model action (default: list)",
    )
    p.add_argument("name", nargs="?", help="Model name")
    p.add_argument("-p", "--provider", help="Provider to use")

    subparsers.add_parser("doctor", help="Run system diagnostics")

    p = subparsers.add_parser("benchmark", help="Benchmark AI providers")
    p.add_argument("--model", help="Model to benchmark")
    p.add_argument("-p", "--provider", help="Provider to benchmark")
    p.add_argument("--compare", action="store_true", help="Compare multiple providers")
    p.add_argument("--iterations", type=int, default=3, help="Number of iterations")
    p.add_argument("--prompt", default="Write a short poem about technology.", help="Prompt to use")

    p = subparsers.add_parser("config", help="Manage configuration")
    p.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "set", "set-api-key", "unset", "reset"],
        help="Config action (default: show)",
    )
    p.add_argument("key", nargs="?", help="Config key or provider name")
    p.add_argument("value", nargs="?", help="Config value")

    p = subparsers.add_parser("login", help="Login to a provider")
    p.add_argument("provider", nargs="?", help="Provider name")
    p.add_argument("--status", action="store_true", help="Show login status")

    p = subparsers.add_parser("logout", help="Logout from a provider")
    p.add_argument("provider", nargs="?", help="Provider name")

    p = subparsers.add_parser("serve", help="Start local API server")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    p.add_argument("--port", type=int, default=8000, help="Port to bind to")
    p.add_argument("-p", "--provider", help="Default provider to use")

    return parser


async def cmd_chat(args: argparse.Namespace) -> None:
    from universal_ai import AI

    ai = AI()
    if args.provider:
        await ai.set_provider(args.provider)
    message = args.message or input("You: ")

    try:
        if args.stream:
            print("AI: ", end="", flush=True)
            async for chunk in ai.stream(message, model=args.model or "auto"):
                print(chunk.delta, end="", flush=True)
            print()
        else:
            response = await ai.chat(message, model=args.model or "auto")
            print(f"AI: {response.text}")
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        await ai.close()


async def cmd_providers(args: argparse.Namespace) -> None:
    from universal_ai import AI

    ai = AI()
    try:
        if args.action == "status":
            statuses = await ai.provider_status()
            if not statuses:
                print("No providers found")
                return
            for name, status in statuses.items():
                ok = status.get("healthy", False)
                icon = "OK" if ok else "FAIL"
                extra = ""
                if "models" in status:
                    models = status["models"][:3]
                    extra = f" ({', '.join(models)})"
                if "error" in status:
                    extra = f" ({status['error']})"
                print(f"  [{icon}] {name}{extra}")
        else:
            providers_list = await ai.list_providers()
            if not providers_list:
                print("No providers available")
                return
            print("Available providers:")
            for p in providers_list:
                name = p.get("name", str(p)) if isinstance(p, dict) else p
                print(f"  - {name}")
    finally:
        await ai.close()


async def cmd_doctor(args: argparse.Namespace) -> None:
    from universal_ai.utils.detection import ProviderDetector

    print("UniversalAI Diagnostics")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")

    try:
        from universal_ai import __version__

        print(f"UniversalAI: {__version__}")
    except ImportError:
        print("UniversalAI: 1.0.0")

    try:
        import pydantic

        print(f"Pydantic: {pydantic.__version__}")
    except ImportError:
        print("Pydantic: NOT INSTALLED")

    try:
        import httpx

        print(f"httpx: {httpx.__version__}")
    except ImportError:
        print("httpx: NOT INSTALLED")

    providers = await ProviderDetector.detect(parallel=True)
    detected = ", ".join(providers) if providers else "None"
    print(f"\nDetected providers: {detected}")

    if not providers:
        print("\nTip: Install Ollama or set an API key (e.g., OPENAI_API_KEY)")
    else:
        for name in providers:
            print(f"  [OK] {name}")

    try:
        from universal_ai import AI

        ai = AI()
        try:
            statuses = await ai.provider_status()
            if statuses:
                print("\nProvider health:")
                for name, status in statuses.items():
                    ok = status.get("healthy", False)
                    icon = "OK" if ok else "FAIL"
                    extra = ""
                    if "error" in status:
                        extra = f" ({status['error']})"
                    print(f"  [{icon}] {name}{extra}")
        finally:
            await ai.close()
    except BaseException:
        print("\nCould not check provider health.")


async def main_async() -> None:
    parser = create_parser()
    args = parser.parse_args()

    commands: dict[str, Any] = {
        "chat": cmd_chat,
        "providers": cmd_providers,
        "doctor": cmd_doctor,
    }

    if args.command == "models":
        from universal_ai_cli.commands.models import handle_models_command

        await handle_models_command(args)
    elif args.command == "benchmark":
        from universal_ai_cli.commands.benchmark import handle_benchmark_command

        await handle_benchmark_command(args)
    elif args.command == "config":
        from universal_ai_cli.commands.config import handle_config_command

        await handle_config_command(args)
    elif args.command == "login":
        from universal_ai_cli.commands.login import handle_login_command

        await handle_login_command(args)
    elif args.command == "logout":
        from universal_ai_cli.commands.login import handle_logout_command

        await handle_logout_command(args)
    elif args.command == "serve":
        from universal_ai_cli.commands.serve import handle_serve_command

        await handle_serve_command(args)
    elif args.command in commands:
        await commands[args.command](args)
    else:
        parser.print_help()


def main() -> None:
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(130)
    except BaseException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
