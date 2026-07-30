from __future__ import annotations

import asyncio

from universal_ai import AI


async def async_context_example():
    async with AI(provider="openai") as ai:
        response = await ai.chat("What is machine learning?")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    asyncio.run(async_context_example())
