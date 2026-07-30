from __future__ import annotations

import asyncio

from universal_ai import AI


async def embedding_example():
    ai = AI()
    try:
        vector = await ai.embed("Hello, world!")
        print(f"Vector dimensions: {len(vector.vector)}")
        print(f"Model: {vector.model}")
    except Exception as e:
        print(f"Embedding error: {e}")


if __name__ == "__main__":
    asyncio.run(embedding_example())
