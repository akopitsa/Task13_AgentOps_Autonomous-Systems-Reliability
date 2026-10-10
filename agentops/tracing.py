"""Observability (Lab 1): Arize Phoenix (OpenInference/OTEL) та/або LangSmith.

TRACING_BACKEND:
  phoenix   — OTLP → self-hosted Phoenix (docker), OpenInference LangChain instrumentation
  langsmith — нативний трейсинг LangChain (потрібні LANGSMITH_API_KEY, LANGSMITH_PROJECT)
  both      — обидва одночасно
  none      — вимкнено (CI unit-тести)
"""
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace as trace_api

from . import config

_tracer_provider = None
_tracer: trace_api.Tracer | None = None


def setup_tracing(backend: str | None = None):
    global _tracer_provider, _tracer
    backend = (backend or config.TRACING_BACKEND).lower()
    if _tracer_provider is not None:
        return _tracer_provider

    if backend in ("langsmith", "both"):
        # LangChain підхоплює ці змінні автоматично — код агента не змінюється
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", config.PHOENIX_PROJECT)
        if not os.getenv("LANGSMITH_API_KEY"):
            print("[TRACING] LANGSMITH_API_KEY не задано — LangSmith трейсинг не працюватиме.")

    if backend in ("phoenix", "both"):
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        _tracer_provider = register(
            project_name=config.PHOENIX_PROJECT,
            endpoint=f"{config.PHOENIX_URL}/v1/traces",
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        _tracer = _tracer_provider.get_tracer("acme.agentops")
        print(f"[TRACING] Phoenix → {config.PHOENIX_URL} (project: {config.PHOENIX_PROJECT})")
    return _tracer_provider


def shutdown_tracing():
    if _tracer_provider is not None:
        _tracer_provider.force_flush()


def _current_langchain_span():
    try:
        from openinference.instrumentation.langchain import get_current_span
        return get_current_span()
    except Exception:
        return None


def annotate_current_span(**attributes: Any) -> None:
    """Додає атрибути до поточного LangChain-спану (напр. cost/budget у middleware-вузлі)."""
    span = _current_langchain_span()
    if span is not None and span.is_recording():
        for k, v in attributes.items():
            span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))


@contextmanager
def guardrail_span(name: str, input_text: str, **attributes: Any) -> Iterator[dict]:
    """Дочірній спан типу GUARDRAIL у дереві трейсу агента.

    Використання:
        with guardrail_span("output_guard", text) as s:
            s["output"] = fixed_text; s["passed"] = False
    """
    result: dict = {}
    parent = _current_langchain_span()
    if _tracer is None or parent is None:
        yield result
        return
    ctx = trace_api.set_span_in_context(parent)
    with _tracer.start_as_current_span(f"guardrail.{name}", context=ctx) as span:
        span.set_attribute("openinference.span.kind", "GUARDRAIL")
        span.set_attribute("input.value", input_text[:4000])
        for k, v in attributes.items():
            span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else json.dumps(v, default=str))
        yield result
        span.set_attribute("output.value", str(result.get("output", ""))[:4000])
        span.set_attribute("guardrail.passed", bool(result.get("passed", True)))
        if result.get("reason"):
            span.set_attribute("guardrail.reason", str(result["reason"]))
