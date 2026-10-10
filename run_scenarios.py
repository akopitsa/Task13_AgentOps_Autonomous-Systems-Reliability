"""Lab 1: проганяє набір сценаріїв через агента з трейсингом у Phoenix/LangSmith.

    python run_scenarios.py                     # усі сценарії, 1 раз
    python run_scenarios.py --repeat 2          # кожен сценарій двічі
    python run_scenarios.py --only faq_returns order_status

Результати run-ів → artifacts/runs.jsonl (вхід для dashboard.py поряд з даними Phoenix).
"""
import argparse
import asyncio
import json

from agentops import config
from agentops.agent import run_agent, support_agent
from agentops.tracing import setup_tracing, shutdown_tracing
from scenarios import SCENARIOS


async def main(repeat: int, only: list[str] | None):
    setup_tracing()
    selected = [s for s in SCENARIOS if not only or s[0] in only]
    async with support_agent() as agent:
        for i in range(repeat):
            for scenario, question in selected:
                print(f"\n=== [{i + 1}/{repeat}] {scenario}: {question}")
                r = await run_agent(agent, question, scenario=scenario, user_id=f"user-{scenario}")
                print(f"→ {r['answer']}")
                print(f"  tools={r['tool_calls']} stop={r['stop_reason']} cost=${r['cost_usd']:.5f} "
                      f"tokens={r['tokens_in']}/{r['tokens_out']} latency={r['latency_s']}s")
                with open(config.ARTIFACTS / "runs.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "messages"}, ensure_ascii=False) + "\n")
    shutdown_tracing()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    asyncio.run(main(a.repeat, a.only))
