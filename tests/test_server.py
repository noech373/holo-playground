"""start_server must never hand back a process it does not control."""

import subprocess

import pytest

from holo_playground import server


def test_refuses_to_start_when_the_port_is_taken(monkeypatch, tmp_path):
    # Regression: a second start in a notebook returned the new, already-dead process while the
    # first server kept answering /health; stop_server() then stopped nothing.
    monkeypatch.setattr(server, "server_up", lambda port: True)

    def no_spawn(*args, **kwargs):
        raise AssertionError("must not launch llama-server")

    monkeypatch.setattr(subprocess, "Popen", no_spawn)
    with pytest.raises(RuntimeError, match="already running on port 8080"):
        server.start_server("9b", "512", 8080, tmp_path / "server.log")
