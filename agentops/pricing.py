"""Прайс-лист моделей ($ за 1M токенів) і розрахунок вартості LLM-виклику.

llama3.1:8b крутиться локально і реально коштує $0, тому для SLA/dashboard
використовуємо "shadow pricing": рахуємо, скільки коштував би той самий трафік
на хмарній моделі. Ціни збігаються з вбудованим прайсом Arize Phoenix.
"""
from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float


PRICE_PROFILES: dict[str, Price] = {
    "gpt-4o-mini": Price(0.15, 0.60),
    "gpt-4.1-mini": Price(0.40, 1.60),
    "gpt-4.1": Price(2.00, 8.00),
    "o1": Price(15.00, 60.00),
    "o1-pro": Price(150.00, 600.00),  # для демо hard-stop бюджету
    "local": Price(0.0, 0.0),
}


def get_price(profile: str | None = None) -> Price:
    profile = profile or config.PRICE_PROFILE
    if profile not in PRICE_PROFILES:
        raise ValueError(f"Unknown PRICE_PROFILE '{profile}'. Known: {sorted(PRICE_PROFILES)}")
    return PRICE_PROFILES[profile]


def cost_usd(tokens_in: int, tokens_out: int, profile: str | None = None) -> float:
    p = get_price(profile)
    return tokens_in * p.input_per_m / 1_000_000 + tokens_out * p.output_per_m / 1_000_000


def estimate_tokens(text: str) -> int:
    """Груба оцінка кількості токенів (~4 символи на токен) для прогнозу до виклику моделі."""
    return max(1, len(text) // 4)
