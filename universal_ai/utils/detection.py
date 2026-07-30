from __future__ import annotations

import asyncio
import logging
import os
import shutil

logger = logging.getLogger("universal_ai.detection")


class ProviderDetector:
    DETECTION_ORDER = [
        ("ollama", "_detect_ollama"),
        ("lmstudio", "_detect_lmstudio"),
        ("llamacpp", "_detect_llamacpp"),
        ("vllm", "_detect_vllm"),
        ("openai", "_detect_openai"),
        ("anthropic", "_detect_anthropic"),
        ("gemini", "_detect_gemini"),
        ("groq", "_detect_groq"),
        ("mistral", "_detect_mistral"),
        ("openrouter", "_detect_openrouter"),
        ("huggingface", "_detect_huggingface"),
        ("azure", "_detect_azure"),
    ]

    @classmethod
    async def detect(cls, parallel: bool = True) -> list[str]:
        if parallel:
            return await cls._detect_parallel()
        return await cls._detect_sequential()

    @classmethod
    async def _detect_parallel(cls) -> list[str]:
        async def _try_detect(name: str, method_name: str) -> str | None:
            method = getattr(cls, method_name)
            try:
                if await asyncio.wait_for(method(), timeout=2.0):
                    return name
            except asyncio.TimeoutError:
                logger.debug("Detection timeout for %s", name)
            except Exception as e:
                logger.debug("Detection failed for %s: %s", name, e)
            return None

        tasks = [_try_detect(name, method_name) for name, method_name in cls.DETECTION_ORDER]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    @classmethod
    async def _detect_sequential(cls) -> list[str]:
        available = []
        for name, method_name in cls.DETECTION_ORDER:
            method = getattr(cls, method_name)
            try:
                if await method():
                    available.append(name)
            except Exception:
                pass
        return available

    @staticmethod
    async def _detect_ollama() -> bool:
        if shutil.which("ollama") is None:
            return False
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
                return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    async def _detect_lmstudio() -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 1234), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    @staticmethod
    async def _detect_llamacpp() -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 8080), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    @staticmethod
    async def _detect_vllm() -> bool:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", 8000), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    @staticmethod
    async def _detect_openai() -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    @staticmethod
    async def _detect_anthropic() -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    async def _detect_gemini() -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    @staticmethod
    async def _detect_groq() -> bool:
        return bool(os.getenv("GROQ_API_KEY"))

    @staticmethod
    async def _detect_mistral() -> bool:
        return bool(os.getenv("MISTRAL_API_KEY"))

    @staticmethod
    async def _detect_openrouter() -> bool:
        return bool(os.getenv("OPENROUTER_API_KEY"))

    @staticmethod
    async def _detect_huggingface() -> bool:
        return bool(os.getenv("HF_API_KEY") or os.getenv("HUGGINGFACE_API_KEY"))

    @staticmethod
    async def _detect_azure() -> bool:
        return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_API_BASE"))
