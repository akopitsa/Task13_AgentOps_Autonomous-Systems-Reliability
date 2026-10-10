"""Локальний mock Slack Incoming Webhook — для перевірки алертів без реального Slack.

    python tools/mock_slack.py --port 8099
    export SLACK_WEBHOOK_URL=http://127.0.0.1:8099/services/T000/B000/XXXX

Кожен POST зберігається в artifacts/slack_mock_inbox.jsonl і друкується в консоль.
Можна також запустити у фоні з коду: start_mock_slack().
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

INBOX = Path(__file__).resolve().parent.parent / "artifacts" / "slack_mock_inbox.jsonl"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.loads(body or b"{}")
        INBOX.parent.mkdir(exist_ok=True)
        with open(INBOX, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(f"[MOCK SLACK] {payload.get('text')}")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")  # так само відповідає справжній Slack webhook

    def log_message(self, *args):
        pass


def start_mock_slack(port: int = 8099) -> str:
    srv = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}/services/T000/B000/mock"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    port = p.parse_args().port
    print(f"Mock Slack webhook: http://127.0.0.1:{port}/services/T000/B000/mock")
    HTTPServer(("127.0.0.1", port), _Handler).serve_forever()
