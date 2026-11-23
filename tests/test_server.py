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


class HangingProcess:
    """A llama-server stuck in its shutdown: wait() is cut by a Ctrl-C until it is killed."""

    def __init__(self):
        self.killed = False

    def terminate(self):
        pass

    def wait(self, timeout=None):
        if not self.killed:
            raise KeyboardInterrupt
        return -9

    def kill(self):
        self.killed = True


def test_stop_kills_a_hanging_server_even_when_ctrl_c_cuts_the_wait():
    # Regression: two orphan llama-servers found on 2026-11-23, one holding 2.8 GB of RAM.
    proc = HangingProcess()
    server.stop_server(proc)  # type: ignore[arg-type]
    assert proc.killed
