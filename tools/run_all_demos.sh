#!/usr/bin/env bash
# Проганяє всі демо Lab 2–4 послідовно (Ollama на CPU обробляє один запит за раз).
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONUNBUFFERED=1

echo "##### Lab 2: guardrails attacks"
$PY demo_guardrails.py > artifacts/demo_guardrails.log 2>&1; echo "exit=$?"

echo "##### Lab 3: budget + circuit breaker (mock Slack)"
rm -f artifacts/slack_mock_inbox.jsonl
$PY demo_sla.py --mock-slack > artifacts/demo_sla.log 2>&1; echo "exit=$?"

echo "##### Lab 4: eval gate, prompt v1.1 (expected PASS)"
PHOENIX_PROJECT=acme-support-agent-evals $PY -m evals.run_eval > artifacts/eval_v1.1.log 2>&1; echo "exit=$?"
cp artifacts/eval_report.md artifacts/eval_report_v1.1.md; cp artifacts/eval_report.json artifacts/eval_report_v1.1.json

echo "##### Lab 4: eval gate, prompt v2-regressed (expected FAIL)"
PROMPT_VERSION=v2-regressed PHOENIX_PROJECT=acme-support-agent-evals $PY -m evals.run_eval > artifacts/eval_v2-regressed.log 2>&1; echo "exit=$?"
cp artifacts/eval_report.md artifacts/eval_report_v2-regressed.md; cp artifacts/eval_report.json artifacts/eval_report_v2-regressed.json
