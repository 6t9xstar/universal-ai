from __future__ import annotations

import asyncio

from universal_ai import AI


async def advanced_rag_example():
    ai = AI()
    try:
        from universal_ai.rag import RAG

        rag = RAG()
        rag.add_folder("./documents")

        response = await ai.chat(
            "What is the company's policy on remote work?",
            knowledge=rag,
        )
        print(f"RAG Response: {response.text}")
    except Exception as e:
        print(f"RAG example error: {e}")


async def main():
    await advanced_rag_example()


if __name__ == "__main__":
    asyncio.run(main())
