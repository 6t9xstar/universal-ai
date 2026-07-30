"""UniversalAI Basic Usage Example."""

from __future__ import annotations

import asyncio

from universal_ai import AI
from universal_ai.config import UniversalAIConfig


async def basic_chat() -> None:
    ai = AI()
    try:
        response = await ai.chat("What is the meaning of life?")
        print(f"Response: {response.text}")
        print(f"Tokens: {response.tokens}")
    except Exception as exc:
        print(f"No provider available: {exc}")


async def streaming_chat() -> None:
    ai = AI()
    try:
        print("Streaming response:")
        async for chunk in ai.stream("Write a short poem"):
            print(chunk.delta, end="", flush=True)
        print()
    except Exception as exc:
        print(f"Streaming not available: {exc}")


async def with_config() -> None:
    config = UniversalAIConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=100,
    )
    ai = AI(config=config)
    try:
        response = await ai.chat("Hello!")
        print(f"Config response: {response.text}")
    except Exception as exc:
        print(f"Config example error: {exc}")


async def providers_info() -> None:
    ai = AI()
    try:
        providers = await ai.list_providers()
        print(f"Available providers: {providers}")
    except Exception as exc:
        print(f"Provider listing error: {exc}")


async def main() -> None:
    print("=")
    await basic_chat()
    print()
    await streaming_chat()
    print()
    await with_config()
    print()
    await providers_info()


if __name__ == "__main__":
    asyncio.run(main())
