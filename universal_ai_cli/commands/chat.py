"""Chat CLI command - interactive chat with AI providers."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


async def cmd_chat(args: argparse.Namespace) -> None:
    from universal_ai import AI

    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    message = getattr(args, "message", None)

    kwargs = {}
    if provider:
        kwargs["provider"] = provider
    if model:
        kwargs["model"] = model

    ai = AI(**kwargs)

    if message:
        response = await ai.chat(message)
        print(response.content)
        return

    print("UniversalAI Interactive Chat")
    print("Type 'exit' or 'quit' to stop, 'clear' to reset conversation")
    print("=" * 50)

    conv = ai.conversation()

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                conv.reset()
                print("Conversation cleared.")
                continue

            response = await conv.send(message=user_input)
            print(f"\nAssistant: {response.content}")

    finally:
        await ai.close()


def create_chat_parser(subparsers) -> None:
    p = subparsers.add_parser("chat", help="Chat with an AI provider")
    p.add_argument("message", nargs="?", help="Message to send (non-interactive mode)")
    p.add_argument("-p", "--provider", help="Provider to use (e.g., openai, anthropic)")
    p.add_argument("-m", "--model", help="Model to use")
    p.set_defaults(func=cmd_chat)
