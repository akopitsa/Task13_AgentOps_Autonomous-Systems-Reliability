# Корінь проєкту в sys.path для тестів (pytest rootdir conftest)
import pytest


@pytest.fixture(autouse=True)
def _isolate_artifacts(monkeypatch, tmp_path):
    """Тести не повинні писати в справжні artifacts/ (аудит-лог, алерти)."""
    from agentops import config, guardrails_config
    monkeypatch.setattr(config, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(config, "SLACK_WEBHOOK_URL", "")
    monkeypatch.setattr(guardrails_config, "AUDIT_LOG", tmp_path / "guardrails_audit.jsonl")
