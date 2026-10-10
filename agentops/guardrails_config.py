"""Guardrails AI конфігурація (Lab 2).

Кастомні валідатори зареєстровані через @register_validator — це той самий API, що й
у Guardrails Hub, але без hub-токена та важких NER-моделей (Presidio/spaCy).

  acme/no_customer_pii   — email, телефон, картка (Luhn), IBAN (mod-97), паспорт UA,
                           + "taint tracking": значення з get_customer (ім'я, адреса),
                           передані через metadata["sensitive_values"]
  acme/allowed_domains   — URL/домени тільки з allowlist (аналог ValidURL(allowed_domains=...))
  acme/prompt_injection  — евристики direct/indirect prompt injection (EN/UA/RU)

Гарди:
  INPUT_GUARD        — вхід користувача: injection → EXCEPTION (run блокується)
  TOOL_ARGS_GUARD    — аргументи search/fetch: чужий домен → EXCEPTION (tool call не виконується)
  TOOL_OUTPUT_GUARD  — відповідь інструмента: injection → FIX (вирізаємо інструкцію)
  OUTPUT_GUARD       — фінальна відповідь: PII → FIX ([REDACTED_*]), чужі URL → FIX
"""
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable
from urllib.parse import urlparse

# Валідатори легкі (regex), тож синхронний режим: без спроб захопити event loop агента
os.environ.setdefault("GUARDRAILS_RUN_SYNC", "true")

from guardrails import Guard, OnFailAction
from guardrails.validators import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

from . import config

AUDIT_LOG = config.ARTIFACTS / "guardrails_audit.jsonl"


# =====================================================================
# PII
# =====================================================================
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<![\d+])(?:\+\d{1,3}[\s-]?)?\(?0?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"
)
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){14,15}\d(?!\d)")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
PASSPORT_UA_RE = re.compile(r"(?<![A-ZА-ЯІЇЄ])[A-ZА-ЯІЇЄ]{2}\d{6}(?!\d)")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if len(digits) not in (15, 16) or digits[0] not in (2, 3, 4, 5, 6):
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(iban: str) -> bool:
    s = iban[4:] + iban[:4]
    num = "".join(str(int(c, 36)) for c in s)
    return int(num) % 97 == 1


def _phone_ok(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    # Номер має містити хоча б один роздільник/плюс або починатися з 0 — щоб не ловити ID замовлень
    return 10 <= len(digits) <= 13 and (match.startswith(("+", "0", "(")) or bool(re.search(r"[\s-]", match)))


def find_pii(text: str, sensitive_values: Iterable[str] = ()) -> list[tuple[str, str]]:
    """Повертає список (тип, знайдене значення)."""
    found: list[tuple[str, str]] = []
    found += [("EMAIL", m) for m in EMAIL_RE.findall(text)]
    found += [("IBAN", m) for m in IBAN_RE.findall(text) if iban_ok(m)]
    found += [("CARD", m.strip()) for m in CARD_RE.findall(text) if luhn_ok(m)]
    found += [("PHONE", m.strip()) for m in PHONE_RE.findall(text) if _phone_ok(m.strip())]
    found += [("PASSPORT", m) for m in PASSPORT_UA_RE.findall(text)]
    for v in sensitive_values:
        if v and len(v) >= 4 and v.lower() in text.lower():
            found.append(("CUSTOMER_DATA", v))
    return found


def redact_pii(text: str, sensitive_values: Iterable[str] = ()) -> str:
    # Довші значення спершу, щоб не залишати "хвостів"
    for kind, value in sorted(find_pii(text, sensitive_values), key=lambda kv: -len(kv[1])):
        text = re.sub(re.escape(value), f"[REDACTED_{kind}]", text, flags=re.IGNORECASE)
    return text


@register_validator(name="acme/no_customer_pii", data_type="string")
class NoCustomerPII(Validator):
    """Блокує/маскує персональні дані клієнтів у відповіді агента."""

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        sensitive = metadata.get("sensitive_values", [])
        found = find_pii(value, sensitive)
        if not found:
            return PassResult()
        kinds = sorted({k for k, _ in found})
        return FailResult(
            error_message=f"PII detected in output: {', '.join(kinds)}",
            fix_value=redact_pii(value, sensitive),
            metadata={"pii_types": kinds, "pii_count": len(found)},
        )


# =====================================================================
# ALLOWED DOMAINS
# =====================================================================
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
BARE_DOMAIN_RE = re.compile(r"^(?:[a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)


def extract_domains(text: str) -> list[str]:
    domains = [urlparse(u).hostname or "" for u in URL_RE.findall(text)]
    stripped = text.strip().lower()
    if not domains and BARE_DOMAIN_RE.match(stripped):
        domains.append(stripped)
    return [d.lower() for d in domains if d]


def domain_allowed(domain: str, allowed: Iterable[str]) -> bool:
    return any(domain == a or domain.endswith("." + a) for a in allowed)


@register_validator(name="acme/allowed_domains", data_type="string")
class AllowedDomains(Validator):
    """Дозволяє лише URL/домени з allowlist. FIX — вирізає заборонені посилання."""

    def __init__(self, allowed_domains: list[str], **kwargs):
        super().__init__(allowed_domains=allowed_domains, **kwargs)
        self.allowed_domains = [d.lower() for d in allowed_domains]

    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        bad = sorted({d for d in extract_domains(str(value)) if not domain_allowed(d, self.allowed_domains)})
        if not bad:
            return PassResult()
        fixed = URL_RE.sub(
            lambda m: m.group(0) if domain_allowed((urlparse(m.group(0)).hostname or "").lower(),
                                                   self.allowed_domains) else "[link removed]",
            str(value),
        )
        return FailResult(
            error_message=f"Domain(s) not in allowlist {self.allowed_domains}: {bad}",
            fix_value=fixed,
            metadata={"blocked_domains": bad},
        )


# =====================================================================
# PROMPT INJECTION
# =====================================================================
INJECTION_PATTERNS = [
    r"(ignore|forget|disregard|override)\s+(all\s+|any\s+|everything\s+|the\s+|your\s+|of\s+)*"
    r"(previous|prior|above|earlier|system)?\s*(instructions|rules|prompts?|guidelines)",
    r"(you\s+are|you're)\s+(now\s+)?in\s+(unlimited|developer|dan|god|jailbreak)\s+mode",
    r"(reveal|show|print)\s+(me\s+)?(the\s+|your\s+)?(system\s+prompt|internal\s+(refund\s+)?polic)",
    r"system\s+note\s+to\s+(the\s+)?ai",
    r"send\s+(all\s+|the\s+)?(user'?s?|customer'?s?)\s+.{0,40}(to|@)",
    r"(забудь|ігноруй)(те)?\s+((всі|усі|все)\s+)?(попередні\s+)?(інструкції|правила)",
    r"(забудь|игнорируй)(те)?\s+((все|всё)\s+)?(предыдущие\s+)?(инструкции|правила)",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE)
HIDDEN_HTML_RE = re.compile(r"<span[^>]*(?:color:\s*white|display:\s*none)[^>]*>.*?</span>", re.IGNORECASE | re.DOTALL)


def neutralize_injection(text: str) -> str:
    text = HIDDEN_HTML_RE.sub("[REMOVED: hidden instruction]", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join("[REMOVED: suspected prompt injection]" if INJECTION_RE.search(s) else s for s in sentences)


@register_validator(name="acme/prompt_injection", data_type="string")
class PromptInjection(Validator):
    def _validate(self, value: Any, metadata: Dict) -> ValidationResult:
        text = str(value)
        m = INJECTION_RE.search(text)
        hidden = HIDDEN_HTML_RE.search(text)
        if not m and not hidden:
            return PassResult()
        reason = f"matched '{m.group(0)}'" if m else "hidden HTML instruction"
        return FailResult(error_message=f"Prompt injection detected: {reason}",
                          fix_value=neutralize_injection(text))


# =====================================================================
# GUARDS
# =====================================================================
def _guard(name: str, *validators: Validator) -> Guard:
    # Усі валідатори — одним викликом: у guardrails 0.11 повторний .use() ЗАМІНЮЄ валідатори на тому ж "on"
    g = Guard(name=name).use(*validators)
    # Вимикаємо анонімну телеметрію Guardrails Hub (інакше спани йдуть на їхній AWS endpoint)
    g.configure(allow_metrics_collection=False)
    return g


INPUT_GUARD = _guard("input_guard", PromptInjection(on_fail=OnFailAction.EXCEPTION))
TOOL_ARGS_GUARD = _guard("tool_args_guard",
                         AllowedDomains(allowed_domains=config.ALLOWED_DOMAINS, on_fail=OnFailAction.EXCEPTION))
TOOL_OUTPUT_GUARD = _guard("tool_output_guard", PromptInjection(on_fail=OnFailAction.FIX))
OUTPUT_GUARD = _guard(
    "output_guard",
    NoCustomerPII(on_fail=OnFailAction.FIX),
    AllowedDomains(allowed_domains=config.ALLOWED_DOMAINS, on_fail=OnFailAction.FIX),
)


def audit(event: str, **fields) -> None:
    """Аудит-лог інцидентів guardrails (без сирих PII — лише типи й кількість)."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def failed_summaries(outcome) -> list[dict]:
    return [
        {"validator": s.validator_name, "reason": s.failure_reason}
        for s in (outcome.validation_summaries or [])
        if s.validator_status == "fail"
    ]
