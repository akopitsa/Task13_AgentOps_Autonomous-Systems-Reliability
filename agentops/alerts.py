"""Slack-алерти через Incoming Webhook (Lab 3).

Якщо SLACK_WEBHOOK_URL не задано — алерт пишеться в консоль і в artifacts/alerts.jsonl,
щоб пайплайн працював і без Slack (dev / CI).
"""
import json
import sys
from datetime import datetime, timezone

import httpx

from . import config

_EMOJI = {"critical": ":rotating_light:", "warning": ":warning:", "info": ":white_check_mark:"}


def _payload(title: str, severity: str, fields: dict) -> dict:
    lines = "\n".join(f"*{k}:* {v}" for k, v in fields.items())
    text = f"{_EMOJI.get(severity, '')} [{severity.upper()}] {title}"
    return {
        "text": text,  # fallback для нотифікацій
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"{title}"[:150]}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"{_EMOJI.get(severity, '')} *Severity:* {severity}\n{lines}"}},
            {"type": "context", "elements": [{"type": "mrkdwn",
                                              "text": f"acme-support-agent · env={config.DEPLOY_ENV} · "
                                                      f"Phoenix: {config.PHOENIX_URL}/projects"}]},
        ],
    }


async def send_alert(title: str, severity: str = "warning", **fields) -> bool:
    """Надсилає алерт. Повертає True, якщо Slack прийняв повідомлення."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), "title": title, "severity": severity, **fields}
    with open(config.ARTIFACTS / "alerts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    print(f"[ALERT:{severity.upper()}] {title} {fields}", file=sys.stderr)

    if not config.SLACK_WEBHOOK_URL:
        return False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.post(config.SLACK_WEBHOOK_URL, json=_payload(title, severity, fields))
        if resp.status_code != 200:
            print(f"[ALERT] Slack відповів {resp.status_code}: {resp.text}", file=sys.stderr)
        return resp.status_code == 200
    except httpx.HTTPError as e:
        # Алертинг не повинен валити агента
        print(f"[ALERT] Не вдалося надіслати в Slack: {e}", file=sys.stderr)
        return False
