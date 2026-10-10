"""Lab 3: hard-stop бюджету ($0.15/run) і circuit breaker зі Slack-алертами.

    python demo_sla.py                       # алерти → SLACK_WEBHOOK_URL (якщо задано)
    python demo_sla.py --mock-slack          # алерти → локальний mock webhook (artifacts/slack_mock_inbox.jsonl)
    python demo_sla.py --only budget|breaker

A) Budget: той самий запит, але з тіньовою ціною o1-pro ($150/$600 за 1M) → run
   перевищує $0.15 і зупиняється middleware (critical алерт).
B) Circuit breaker (chaos engineering): Order API падає на 100% → після 3 збоїв
   поспіль breaker OPEN (critical алерт у Slack) → наступні виклики не доходять до API →
   chaos вимкнено → після recovery timeout HALF_OPEN → успіх → CLOSED (info алерт).
"""
import argparse
import asyncio
import json
import time

from agentops import config
from agentops.agent import run_agent, support_agent
from agentops.middleware import BudgetMiddleware, CBState, CircuitBreakerMiddleware
from agentops.tracing import setup_tracing, shutdown_tracing

ORDER_Q = "Where is my order A-1042?"


def log_run(r: dict):
    print(f"→ {r['answer']}\n  tools={r['tool_calls']} stop={r['stop_reason']} cost=${r['cost_usd']:.4f} "
          f"calls={r['model_calls']} latency={r['latency_s']}s")
    with open(config.ARTIFACTS / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in r.items() if k != "messages"}, ensure_ascii=False) + "\n")


async def demo_budget():
    print("\n================ A) BUDGET HARD-STOP ($0.15/run, shadow price o1-pro) ================")
    budget = BudgetMiddleware(budget_usd=0.15, price_profile="o1-pro")
    async with support_agent(budget=budget) as agent:
        r = await run_agent(agent, "Can I return headphones 20 days after delivery?", scenario="sla_budget_o1pro")
        log_run(r)
        assert r["stop_reason"] == "budget_exceeded", "budget middleware did not stop the run!"


async def demo_breaker(recovery_s: float):
    print(f"\n================ B) CIRCUIT BREAKER (threshold={config.CB_FAILURE_THRESHOLD}, "
          f"recovery={recovery_s}s) ================")
    breaker = CircuitBreakerMiddleware(recovery_timeout_s=recovery_s)
    cb = breaker.breaker("get_order_status")

    print("\n--- B1: Order API down (CHAOS_ORDER_API_FAIL_RATE=1.0) ---")
    async with support_agent(breaker=breaker, extra_env={"CHAOS_ORDER_API_FAIL_RATE": "1.0"}) as agent:
        for i in range(5):
            r = await run_agent(agent, ORDER_Q, scenario="sla_chaos_order_api")
            log_run(r)
            print(f"  breaker: {cb.state.value}, consecutive_failures={cb.consecutive_failures}")
            if cb.state == CBState.OPEN:
                break
        print("\n--- B2: breaker OPEN → запит не доходить до Order API ---")
        r = await run_agent(agent, ORDER_Q, scenario="sla_breaker_open")
        log_run(r)

    print("\n--- B3: Order API відновлено, чекаємо recovery timeout → HALF_OPEN → CLOSED ---")
    wait = max(0.0, recovery_s - (time.monotonic() - cb.opened_at))
    await asyncio.sleep(wait)
    async with support_agent(breaker=breaker, extra_env={"CHAOS_ORDER_API_FAIL_RATE": "0"}) as agent:
        r = await run_agent(agent, ORDER_Q, scenario="sla_breaker_recovery")
        log_run(r)
    print("\nTransitions:", [(t[1], t[2]) for t in cb.transitions])


async def main(only: str | None, mock_slack: bool, recovery_s: float):
    if mock_slack:
        from tools.mock_slack import start_mock_slack
        config.SLACK_WEBHOOK_URL = start_mock_slack()
        print(f"[SLACK] mock webhook: {config.SLACK_WEBHOOK_URL}")
    print(f"[SLACK] alerts → {config.SLACK_WEBHOOK_URL or 'console + artifacts/alerts.jsonl only'}")
    setup_tracing()
    if only in (None, "budget"):
        await demo_budget()
    if only in (None, "breaker"):
        await demo_breaker(recovery_s)
    shutdown_tracing()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["budget", "breaker"])
    p.add_argument("--mock-slack", action="store_true")
    p.add_argument("--recovery-s", type=float, default=20.0)
    a = p.parse_args()
    asyncio.run(main(a.only, a.mock_slack, a.recovery_s))
