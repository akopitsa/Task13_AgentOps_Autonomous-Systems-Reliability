# Lesson 13 · AgentOps & Autonomous Systems Reliability — домашнє завдання

**Власний агент:** `acme-support-agent` — агент клієнтської підтримки інтернет-магазину Acme Electronics.
Runtime з Lesson 12 (LangChain `create_agent` + MCP-сервер через stdio + Ollama `llama3.1:8b`),
до якого додано повний AgentOps-контур з Lesson 13:

| Lab | Що зроблено | Де в коді | Deliverable |
|---|---|---|---|
| **1 · Observability** | Агент підключено до **Arize Phoenix** (self-hosted, OpenInference/OTEL); LangSmith — тим самим перемикачем `TRACING_BACKEND`. Cost dashboard: **$/run** і **latency кожного спану** | `agentops/tracing.py`, `phoenix_setup.py`, `dashboard.py` | Phoenix project `acme-support-agent` + `artifacts/cost_dashboard.html` |
| **2 · Guardrails** | **Guardrails AI**: anti-PII на виході агента, allowlist доменів для search/fetch tool, захист від direct/indirect prompt injection | `agentops/guardrails_config.py`, `GuardrailsMiddleware` | `artifacts/guardrails_attack_log.md` |
| **3 · SLA** | Middleware: **hard-stop при $0.15/run**, soft-alert на 80%, step limit; **circuit breaker** + **Slack alert при переході в OPEN** | `BudgetMiddleware`, `CircuitBreakerMiddleware`, `agentops/alerts.py` | `demo_sla.py`, `artifacts/alerts.jsonl`, `artifacts/slack_mock_inbox.jsonl` |
| **4 · Regression** | **Golden dataset (7 кейсів)**, accuracy + LLM-judge relevance, **threshold gate у GitHub Actions** | `evals/`, `.github/workflows/agent-eval.yml` | `artifacts/eval_report.md` (PASS для `v1.1`, FAIL для `v2-regressed`) |
| + | Post-mortem симульованого інциденту | | [`POSTMORTEM.md`](POSTMORTEM.md) |

<!-- RESULTS -->

---

## Архітектура

```text
 user ──► create_agent (LangGraph)  ──────────────── OTLP ──►  Arize Phoenix (docker :6006)
            │  middleware (порядок = шари цибулі)                 ├─ trace tree / latency
            │   1. GuardrailsMiddleware                          ├─ cost (тіньова ціна моделі)
            │        before_agent : INPUT_GUARD (injection)       └─ annotations: accuracy/relevance
            │        wrap_tool    : TOOL_ARGS_GUARD (allowlist) + TOOL_OUTPUT_GUARD (indirect injection)
            │        after_agent  : OUTPUT_GUARD (PII → [REDACTED_*], чужі URL)
            │   2. BudgetMiddleware   before/after_model : $/run, 80% warn, hard-stop $0.15, max 6 calls
            │   3. CircuitBreakerMiddleware wrap_tool    : CLOSED → OPEN → HALF_OPEN ──► Slack webhook
            ▼
       ChatOpenAI ──► Ollama llama3.1:8b (OpenAI-compatible /v1)
            │
       MCP stdio ──► support_mcp_server.py: search_docs · fetch_page · get_customer (PII) · get_order_status (chaos)
```

Чому саме так:

* **Middleware, а не обгортки навколо агента.** `create_agent` у LangChain 1.x має хуки
  `before_agent / before_model / after_model / wrap_tool_call / after_agent`. Кожен хук — окремий спан
  у трейсі, тож видно, *який саме* шар зупинив run і скільки він коштував за часом.
* **Guardrail раніше за circuit breaker.** `wrap_tool_call` першого middleware — зовнішній шар:
  заборонений домен блокується ще до того, як CB порахує виклик і до самого MCP-запиту.
* **Тіньова ціна (shadow pricing).** `llama3.1:8b` локальна і реально коштує $0, тому вартість рахується
  так, ніби трафік ішов на `gpt-4.1-mini` ($0.40 / $1.60 за 1M токенів, як у вбудованому прайсі Phoenix).
  Ця сама ціна зареєстрована в Phoenix як custom-модель, тому **Phoenix сам рахує cost** кожного спану
  й trace, а dashboard бере ці числа з Phoenix (GraphQL `Trace.costSummary`).
* **Кастомні валідатори Guardrails AI** (`@register_validator`) замість Hub-валідаторів: Hub потребує
  токена і тягне Presidio/spaCy (~500 MB). API той самий (`Guard().use(...)`, `OnFailAction.FIX/EXCEPTION`),
  тож заміна на `hub://guardrails/detect_pii` — одна строчка. Імена/адреси клієнта ловляться через
  *taint tracking*: значення з відповіді `get_customer` передаються валідатору через `metadata`.

---

## Запуск

```bash
cd homework
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
docker compose up -d                      # Phoenix UI → http://localhost:6006
ollama serve & ollama pull llama3.1:8b
cp .env.example .env                      # за потреби: SLACK_WEBHOOK_URL, LANGSMITH_API_KEY
```

| Крок | Команда | Що отримуємо |
|---|---|---|
| 0 | `.venv/bin/python phoenix_setup.py` | тіньова ціна `llama3.1:8b` у Phoenix (Settings → Models) |
| Lab 1 | `.venv/bin/python run_scenarios.py` | 8 сценаріїв → трейси в Phoenix |
| Lab 1 | `.venv/bin/python dashboard.py` | `artifacts/cost_dashboard.html` |
| Lab 2 | `.venv/bin/python demo_guardrails.py` | `artifacts/guardrails_attack_log.md` |
| Lab 3 | `.venv/bin/python demo_sla.py --mock-slack` | hard-stop + circuit breaker, алерти в mock Slack |
| Lab 3 | `SLACK_WEBHOOK_URL=https://hooks.slack.com/... .venv/bin/python demo_sla.py` | ті самі алерти в реальний Slack-канал |
| Lab 4 | `TRACING_BACKEND=none .venv/bin/pytest -q tests` | 42 unit-тести без LLM |
| Lab 4 | `.venv/bin/python -m evals.run_eval` | golden dataset + gate (exit 1 при регресії) |
| Lab 4 | `PROMPT_VERSION=v2-regressed .venv/bin/python -m evals.run_eval` | демонстрація заблокованої регресії |

LangSmith замість/разом з Phoenix: `export LANGSMITH_API_KEY=... TRACING_BACKEND=langsmith` (або `both`) —
LangChain трейсить автоматично, код агента не змінюється.

---

## Структура

```text
homework/
├── support_mcp_server.py        MCP-сервер інструментів (mock KB, PII, chaos для Order API)
├── agentops/
│   ├── agent.py                 create_agent + MCP + system prompts (v1.1, v2-regressed) + run metadata
│   ├── tracing.py               Phoenix / LangSmith, GUARDRAIL-спани, атрибути на поточному спані
│   ├── guardrails_config.py     валідатори Guardrails AI + 4 гарди + аудит-лог
│   ├── middleware.py            Guardrails / Budget / CircuitBreaker middleware
│   ├── pricing.py               прайс-профілі ($/1M токенів)
│   ├── alerts.py                Slack Incoming Webhook (Block Kit)
│   └── config.py                усі налаштування через env
├── phoenix_setup.py             реєстрація ціни моделі в Phoenix
├── run_scenarios.py · scenarios.py · dashboard.py          Lab 1
├── demo_guardrails.py                                       Lab 2
├── demo_sla.py · tools/mock_slack.py                        Lab 3
├── evals/golden_dataset.jsonl · evals/run_eval.py           Lab 4
├── tests/                       unit-тести guardrails, middleware, eval-перевірок
├── .github/workflows/agent-eval.yml
├── docker-compose.yml · requirements.txt · .env.example
├── POSTMORTEM.md
└── artifacts/                   логи, звіти, dashboard, скріншоти
```
