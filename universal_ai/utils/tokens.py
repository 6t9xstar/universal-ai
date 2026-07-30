from __future__ import annotations

import re


class TokenCounter:
    _WORD_SPLIT = re.compile(r"\s+")
    _SPECIAL = re.compile(r"[^\w\s]")

    @classmethod
    def count(cls, text: str) -> int:
        if not text:
            return 0
        text = text.strip()
        if not text:
            return 0
        words = cls._WORD_SPLIT.split(text)
        token_count = 0
        for word in words:
            if not word:
                continue
            token_count += 1
            length = len(word)
            if length <= 4:
                token_count += 0
            elif length <= 7:
                token_count += 1
            else:
                token_count += length // 4
            specials = cls._SPECIAL.findall(word)
            token_count += len(specials)
        token_count += max(0, len(words) - 1)
        return max(1, token_count)

    @classmethod
    def count_messages(cls, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += cls.count(part.get("text", ""))
                    else:
                        total += cls.count(str(part))
            else:
                total += cls.count(str(content))
            total += 4
        total += 2
        return total

    @classmethod
    def fits_in_context(cls, messages: list[dict], max_tokens: int) -> bool:
        estimated = cls.count_messages(messages)
        return estimated < max_tokens
