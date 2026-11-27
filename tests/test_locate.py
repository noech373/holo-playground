"""holo-locate against a stub OpenAI-compatible server: what is sent, and answers without timings."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from holo_playground import locate

COMPLETION = {
    "id": "x", "object": "chat.completion", "created": 0, "model": "stub",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": '{"x": 500, "y": 250}'}}],
}


@pytest.fixture
def api():
    """A hosted API stand-in (no llama-server timings) that records each request."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.headers["Authorization"], body))
            payload = json.dumps(COMPLETION).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1", requests
    server.shutdown()
    server.server_close()


def run(monkeypatch, tmp_path, *options):
    image = tmp_path / "shot.png"
    Image.new("RGB", (200, 100), "white").save(image)
    monkeypatch.setattr(sys, "argv", ["holo-locate", str(image), "the button", *options])
    locate.main()


def test_key_and_model_are_sent_and_thinking_is_disabled(api, monkeypatch, tmp_path, capsys):
    url, requests = api
    monkeypatch.delenv("HAI_API_KEY", raising=False)
    run(monkeypatch, tmp_path, "--base-url", url, "--api-key", "secret", "--model", "some-model")
    [(authorization, body)] = requests
    assert authorization == "Bearer secret"
    assert body["model"] == "some-model"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    out = capsys.readouterr().out
    assert "pixels (100, 25)" in out
    assert "total" in out and "prefill" not in out
    assert (tmp_path / "shot_click.png").exists()


@pytest.mark.parametrize(("env", "expected"), [("from-env", "Bearer from-env"), (None, "Bearer none")])
def test_default_key_comes_from_the_environment(api, monkeypatch, tmp_path, env, expected):
    url, requests = api
    if env:
        monkeypatch.setenv("HAI_API_KEY", env)
    else:
        monkeypatch.delenv("HAI_API_KEY", raising=False)
    run(monkeypatch, tmp_path, "--base-url", url)
    [(authorization, body)] = requests
    assert authorization == expected
    assert body["model"] == "holo"
