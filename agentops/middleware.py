"""AgentOps middleware для langchain.agents.create_agent (Lab 2 + Lab 3).

Порядок у create_agent(middleware=[Guardrails, Budget, CircuitBreaker]):
  before_*  → виконуються зліва направо
  after_*   → справа наліво
  wrap_*    → перший у списку = зовнішній шар (guardrails бачить tool call першим
              і блокує заборонений домен ще ДО circuit breaker і самого MCP-виклику)
"""
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from guardrails.errors import ValidationError
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from typing_extensions import NotRequired

from . import config, pricing
from .alerts import send_alert
from .guardrails_config import (
    INPUT_GUARD,
    OUTPUT_GUARD,
    TOOL_ARGS_GUARD,
    TOOL_OUTPUT_GUARD,
    audit,
    failed_summaries,
)
from .tracing import annotate_current_span, guardrail_span


class AgentOpsState(AgentState):
    run_cost_usd: NotRequired[float]
    run_tokens_in: NotRequired[int]
    run_tokens_out: NotRequired[int]
    model_calls: NotRequired[int]
    budget_warned: NotRequired[bool]
    stop_reason: NotRequired[str]


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


# =====================================================================
# LAB 2 · GUARDRAILS
# =====================================================================
REFUSAL_INJECTION = (
    "I can't help with that request: it tries to override my operating rules. "
    "If you need a refund or policy exception, I can open a ticket for a human supervisor."
)
DOMAIN_TOOLS = {"search_docs": "site", "fetch_page": "url"}


class GuardrailsMiddleware(AgentMiddleware):
    state_schema = AgentOpsState

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AgentOpsState, runtime) -> dict | None:
        user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not user_msgs:
            return None
        text = _text(user_msgs[-1].content)
        with guardrail_span("input_guard", text) as s:
            try:
                INPUT_GUARD.validate(text)
                s["output"], s["passed"] = text, True
                return None
            except ValidationError as e:
                reason = str(e)
                s["output"], s["passed"], s["reason"] = REFUSAL_INJECTION, False, reason
        audit("input_blocked", guard="input_guard", reason=reason, input_preview=text[:120])
        await send_alert("Guardrail: prompt injection blocked", "warning",
                         guard="input_guard", reason=reason[:200], action="run stopped, needs review")
        return {"messages": [AIMessage(content=REFUSAL_INJECTION)], "stop_reason": "prompt_injection", "jump_to": "end"}

    async def awrap_tool_call(self, request, handler: Callable[[Any], Awaitable[ToolMessage]]):
        call = request.tool_call
        name, args = call["name"], call.get("args", {})

        # 1) allowlist доменів для search/fetch ДО виконання tool call
        if name in DOMAIN_TOOLS:
            target = str(args.get(DOMAIN_TOOLS[name], ""))
            with guardrail_span("tool_args_guard", target, tool=name) as s:
                try:
                    TOOL_ARGS_GUARD.validate(target)
                    s["output"], s["passed"] = target, True
                except ValidationError as e:
                    s["output"], s["passed"], s["reason"] = "BLOCKED", False, str(e)
                    audit("tool_call_blocked", guard="tool_args_guard", tool=name, target=target, reason=str(e))
                    return ToolMessage(
                        content=f"BLOCKED by guardrail: '{target}' is not an allowed domain. "
                                f"Allowed domains: {', '.join(config.ALLOWED_DOMAINS)}. "
                                f"Tell the user you cannot access this site.",
                        tool_call_id=call["id"], name=name, status="error",
                    )

        result = await handler(request)

        # 2) indirect prompt injection у відповіді інструмента
        if isinstance(result, ToolMessage) and result.status != "error":
            content = _text(result.content)
            with guardrail_span("tool_output_guard", content, tool=name) as s:
                outcome = TOOL_OUTPUT_GUARD.validate(content)
                fixed = outcome.validated_output
                s["output"], s["passed"] = fixed, fixed == content
                if fixed != content:
                    s["reason"] = "indirect prompt injection neutralized"
                    audit("tool_output_sanitized", guard="tool_output_guard", tool=name,
                          failures=failed_summaries(outcome))
                    result = ToolMessage(content=fixed, tool_call_id=result.tool_call_id, name=result.name)
        return result

    async def aafter_agent(self, state: AgentOpsState, runtime) -> dict | None:
        final = state["messages"][-1]
        if not isinstance(final, AIMessage):
            return None
        text = _text(final.content)
        sensitive = _sensitive_values_from_tools(state["messages"])
        with guardrail_span("output_guard", text) as s:
            outcome = OUTPUT_GUARD.validate(text, metadata={"sensitive_values": sensitive})
            fixed = outcome.validated_output
            s["output"], s["passed"] = fixed, fixed == text
            if fixed == text:
                return None
            failures = failed_summaries(outcome)
            s["reason"] = "; ".join(f["reason"] for f in failures)
        audit("output_redacted", guard="output_guard", failures=failures)
        return {"messages": [AIMessage(content=fixed, id=final.id)]}


def _sensitive_values_from_tools(messages) -> list[str]:
    """Taint tracking: значення полів профілю клієнта з get_customer вважаються PII."""
    values: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.name == "get_customer":
            for line in _text(m.content).splitlines():
                key, _, val = line.partition(":")
                if key.strip() in {"name", "email", "phone", "card", "iban", "address"} and val.strip():
                    values.append(val.strip())
    return values


# =====================================================================
# LAB 3 · TOKEN BUDGET (hard stop $0.15 / run, soft warning 80%)
# =====================================================================
TOOLS_SCHEMA_OVERHEAD_TOKENS = 450  # ~ стільки займають описи 4 MCP-інструментів у промпті


class BudgetMiddleware(AgentMiddleware):
    state_schema = AgentOpsState

    def __init__(self, budget_usd: float | None = None, price_profile: str | None = None,
                 soft_ratio: float | None = None, max_model_calls: int | None = None):
        super().__init__()
        self.budget = budget_usd if budget_usd is not None else config.RUN_BUDGET_USD
        self.profile = price_profile or config.PRICE_PROFILE
        self.soft_ratio = soft_ratio if soft_ratio is not None else config.SOFT_BUDGET_RATIO
        self.max_calls = max_model_calls or config.MAX_MODEL_CALLS
        self.price = pricing.get_price(self.profile)

    async def abefore_agent(self, state, runtime) -> dict:
        # Бюджет рахується на один run, навіть якщо thread (пам'ять) спільний
        return {"run_cost_usd": 0.0, "run_tokens_in": 0, "run_tokens_out": 0,
                "model_calls": 0, "budget_warned": False, "stop_reason": ""}

    @hook_config(can_jump_to=["end"])
    async def abefore_model(self, state: AgentOpsState, runtime) -> dict | None:
        spent = state.get("run_cost_usd", 0.0)
        calls = state.get("model_calls", 0)
        if calls >= self.max_calls:
            return await self._stop(state, "step_limit",
                                    f"Step limit reached ({self.max_calls} model calls).")
        # Прогноз: вартість наступного виклику лише за вхідними токенами (мінімум)
        prompt_tokens = sum(pricing.estimate_tokens(_text(m.content)) for m in state["messages"])
        projected = spent + (prompt_tokens + TOOLS_SCHEMA_OVERHEAD_TOKENS) * self.price.input_per_m / 1e6
        annotate_current_span(**{"budget.spent_usd": spent, "budget.projected_usd": projected,
                                 "budget.limit_usd": self.budget, "budget.price_profile": self.profile})
        if projected > self.budget:
            return await self._stop(state, "budget_exceeded",
                                    f"Next LLM call would exceed the run budget "
                                    f"(projected ${projected:.4f} > ${self.budget:.2f}).")
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_model(self, state: AgentOpsState, runtime) -> dict | None:
        msg = state["messages"][-1]
        usage = getattr(msg, "usage_metadata", None) or {}
        t_in, t_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        step_cost = pricing.cost_usd(t_in, t_out, self.profile)
        update: dict = {
            "run_cost_usd": state.get("run_cost_usd", 0.0) + step_cost,
            "run_tokens_in": state.get("run_tokens_in", 0) + t_in,
            "run_tokens_out": state.get("run_tokens_out", 0) + t_out,
            "model_calls": state.get("model_calls", 0) + 1,
        }
        total = update["run_cost_usd"]
        annotate_current_span(**{"budget.step_cost_usd": step_cost, "budget.run_cost_usd": total,
                                 "budget.limit_usd": self.budget})

        if total > self.budget:
            # HARD STOP: замінюємо відповідь моделі (разом з її tool_calls) і завершуємо run
            stop = await self._stop({**state, **update}, "budget_exceeded",
                                    f"Run cost ${total:.4f} exceeded budget ${self.budget:.2f}.",
                                    replace_id=msg.id)
            return {**update, **stop}
        if total >= self.soft_ratio * self.budget and not state.get("budget_warned"):
            update["budget_warned"] = True
            await send_alert("Agent run budget at 80%", "warning",
                             run_cost_usd=round(total, 4), budget_usd=self.budget, price_profile=self.profile)
        return update

    async def _stop(self, state, reason: str, detail: str, replace_id: str | None = None) -> dict:
        await send_alert(f"Agent hard-stop: {reason}", "critical", detail=detail,
                         run_cost_usd=round(state.get("run_cost_usd", 0.0), 4), budget_usd=self.budget,
                         model_calls=state.get("model_calls", 0), price_profile=self.profile)
        text = ("I had to stop processing this request because it exceeded the allowed "
                "resource budget. A support specialist will follow up. "
                f"[stop_reason={reason}]")
        msg = AIMessage(content=text, id=replace_id) if replace_id else AIMessage(content=text)
        return {"messages": [msg], "stop_reason": reason, "jump_to": "end"}


# =====================================================================
# LAB 3 · CIRCUIT BREAKER (CLOSED → OPEN → HALF_OPEN)
# =====================================================================
class CBState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


INFRA_ERROR_RE = re.compile(r"\b5\d\d\b|timeout|timed out|unavailable|connection|rate.?limit|\b429\b", re.IGNORECASE)


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int
    recovery_timeout_s: float
    state: CBState = CBState.CLOSED
    consecutive_failures: int = 0
    opened_at: float = 0.0
    transitions: list = field(default_factory=list)

    def _move(self, new: CBState) -> tuple[CBState, CBState]:
        old, self.state = self.state, new
        self.transitions.append((time.time(), old.value, new.value))
        return old, new

    def before_call(self) -> tuple[bool, tuple | None]:
        """(дозволено?, перехід стану або None)."""
        if self.state == CBState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_timeout_s:
                return True, self._move(CBState.HALF_OPEN)  # пробний запит
            return False, None
        return True, None

    def on_success(self) -> tuple | None:
        self.consecutive_failures = 0
        if self.state == CBState.HALF_OPEN:
            return self._move(CBState.CLOSED)
        return None

    def on_failure(self) -> tuple | None:
        self.consecutive_failures += 1
        if self.state == CBState.HALF_OPEN or (
            self.state == CBState.CLOSED and self.consecutive_failures >= self.failure_threshold
        ):
            self.opened_at = time.monotonic()
            return self._move(CBState.OPEN)
        return None


class CircuitBreakerMiddleware(AgentMiddleware):
    """Per-tool circuit breaker. Стан живе в процесі і спільний для всіх run-ів,
    бо захищає саме downstream-сервіс, а не окремий запит."""

    def __init__(self, failure_threshold: int | None = None, recovery_timeout_s: float | None = None):
        super().__init__()
        self.failure_threshold = failure_threshold or config.CB_FAILURE_THRESHOLD
        self.recovery_timeout_s = recovery_timeout_s if recovery_timeout_s is not None else config.CB_RECOVERY_TIMEOUT_S
        self.breakers: dict[str, CircuitBreaker] = {}

    def breaker(self, tool: str) -> CircuitBreaker:
        if tool not in self.breakers:
            self.breakers[tool] = CircuitBreaker(tool, self.failure_threshold, self.recovery_timeout_s)
        return self.breakers[tool]

    async def _on_transition(self, cb: CircuitBreaker, transition, error: str = "") -> None:
        if not transition:
            return
        old, new = transition
        annotate_current_span(**{"circuit_breaker.tool": cb.name, "circuit_breaker.state": new.value})
        if new == CBState.OPEN:
            await send_alert(f"Circuit breaker OPEN: {cb.name}", "critical",
                             tool=cb.name, transition=f"{old.value} → {new.value}",
                             consecutive_failures=cb.consecutive_failures, last_error=error[:200],
                             retry_after_s=self.recovery_timeout_s)
        elif new == CBState.CLOSED:
            await send_alert(f"Circuit breaker recovered: {cb.name}", "info",
                             tool=cb.name, transition=f"{old.value} → {new.value}")

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        cb = self.breaker(call["name"])
        allowed, transition = cb.before_call()
        await self._on_transition(cb, transition)
        if not allowed:
            return ToolMessage(
                content=f"CIRCUIT OPEN: tool '{cb.name}' is temporarily disabled after "
                        f"{cb.consecutive_failures} consecutive failures. Do not retry. Apologize to the user, "
                        f"say the service is degraded and offer to open a support ticket.",
                tool_call_id=call["id"], name=cb.name, status="error",
            )
        try:
            result = await handler(request)
        except Exception as e:  # транспортна помилка MCP
            await self._on_transition(cb, cb.on_failure(), str(e))
            raise
        content = _text(result.content) if isinstance(result, ToolMessage) else ""
        if isinstance(result, ToolMessage) and result.status == "error" and INFRA_ERROR_RE.search(content):
            await self._on_transition(cb, cb.on_failure(), content)
        else:
            # бізнес-помилки (404 / not found) не означають, що сервіс лежить
            await self._on_transition(cb, cb.on_success())
        return result
