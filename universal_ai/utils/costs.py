from __future__ import annotations


class CostCalculator:
    RATES: dict[tuple[str, str], tuple[float, float]] = {
        ("openai", "gpt-4o"): (0.00001, 0.00003),
        ("openai", "gpt-4o-mini"): (0.0000015, 0.000006),
        ("openai", "gpt-4-turbo"): (0.00001, 0.00003),
        ("openai", "gpt-3.5-turbo"): (0.000001, 0.000002),
        ("openai", "text-embedding-3-small"): (0.0000001, 0.0),
        ("openai", "text-embedding-3-large"): (0.00000013, 0.0),
        ("anthropic", "claude-3-opus-20240229"): (0.000015, 0.000075),
        ("anthropic", "claude-3-sonnet-20240229"): (0.000003, 0.000015),
        ("anthropic", "claude-3-haiku-20240307"): (0.0000025, 0.0000125),
        ("gemini", "gemini-pro"): (0.0000005, 0.000002),
        ("gemini", "gemini-1.5-pro"): (0.0000035, 0.0000105),
        ("gemini", "gemini-1.5-flash"): (0.00000035, 0.00000105),
        ("groq", "mixtral-8x7b-32768"): (0.0000005, 0.0000007),
        ("mistral", "mistral-large-latest"): (0.000008, 0.000024),
        ("mistral", "mistral-small-latest"): (0.000001, 0.000003),
        ("openrouter", "openai/gpt-4o"): (0.00001, 0.00003),
        ("openrouter", "anthropic/claude-3-opus"): (0.000015, 0.000075),
        ("huggingface", "default"): (0.0, 0.0),
        ("ollama", "default"): (0.0, 0.0),
        ("llamacpp", "default"): (0.0, 0.0),
        ("lmstudio", "default"): (0.0, 0.0),
        ("vllm", "default"): (0.0, 0.0),
    }
    DEFAULT_RATES = (0.000001, 0.000002)

    @classmethod
    def calculate(cls, provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
        input_rate, output_rate = cls.get_rates(provider, model)
        return (input_tokens * input_rate) + (output_tokens * output_rate)

    @classmethod
    def get_rates(cls, provider: str, model: str) -> tuple[float, float]:
        rate = cls.RATES.get((provider, model))
        if rate is not None:
            return rate
        for (p, m), r in cls.RATES.items():
            if p == provider and (m in model or m == "default"):
                return r
        return cls.DEFAULT_RATES

    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        return max(1, len(text) // 4)
