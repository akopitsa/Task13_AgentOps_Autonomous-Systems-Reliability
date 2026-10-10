"""Єдине місце конфігурації (12-factor: усе через env)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


# --- LLM (Ollama через OpenAI-сумісний API, як у Lesson 12) ---
LLM_MODEL = _env("LLM_MODEL", "llama3.1:8b")
LLM_BASE_URL = _env("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
LLM_API_KEY = _env("LLM_API_KEY", "ollama")
LLM_TIMEOUT_S = float(_env("LLM_TIMEOUT_S", "180"))

# --- Tracing: phoenix | langsmith | both | none ---
TRACING_BACKEND = _env("TRACING_BACKEND", "phoenix")
PHOENIX_URL = _env("PHOENIX_URL", "http://127.0.0.1:6006")
PHOENIX_PROJECT = _env("PHOENIX_PROJECT", "acme-support-agent")
DEPLOY_ENV = _env("DEPLOY_ENV", "dev")
PROMPT_VERSION = _env("PROMPT_VERSION", "v1.1")

# --- SLA (Lab 3) ---
# Модель локальна (Ollama → $0), тому рахуємо "тіньову" вартість: скільки коштував би run
# на хмарній моделі з профілю PRICE_PROFILE. Див. agentops/pricing.py
PRICE_PROFILE = _env("PRICE_PROFILE", "gpt-4.1-mini")
RUN_BUDGET_USD = float(_env("RUN_BUDGET_USD", "0.15"))
SOFT_BUDGET_RATIO = float(_env("SOFT_BUDGET_RATIO", "0.8"))
MAX_MODEL_CALLS = int(_env("MAX_MODEL_CALLS", "6"))
CB_FAILURE_THRESHOLD = int(_env("CB_FAILURE_THRESHOLD", "3"))
CB_RECOVERY_TIMEOUT_S = float(_env("CB_RECOVERY_TIMEOUT_S", "30"))
SLACK_WEBHOOK_URL = _env("SLACK_WEBHOOK_URL", "")

# --- Guardrails (Lab 2) ---
ALLOWED_DOMAINS = [d.strip() for d in _env("ALLOWED_DOMAINS", "docs.acme.test,help.acme.test").split(",") if d.strip()]
