"""Start and stop a local llama-server for a Holo model."""

import subprocess
import time
import urllib.request
from pathlib import Path

LLAMA_SERVER = Path.home() / "llama.cpp/build/bin/llama-server"
MODELS_DIR = Path.home() / "models"
MODELS = {
    "4b": (
        MODELS_DIR / "holo-3.1-4b/Holo-3.1-4B.Q4_K_M.gguf",
        MODELS_DIR / "holo-3.1-4b/Holo-3.1-4B.mmproj-Q8_0.gguf",
    ),
    "9b": (
        MODELS_DIR / "holo-3.1-9b/Holo-3.1-9B.Q4_K_M.gguf",
        MODELS_DIR / "holo-3.1-9b/Holo-3.1-9B.mmproj-Q8_0.gguf",
    ),
}
# Tuned for a 4-core CPU-only machine with ~8 GB of RAM.
SERVER_FLAGS = [
    "-c", "8192", "-np", "1", "-t", "4", "-tb", "6",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
]
DEFAULT_BUDGET = "default"
STARTUP_TIMEOUT_S = 300


def server_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start_server(
    model: str,
    budget: str,
    port: int,
    log: Path,
    *,
    cache_prompt: bool = False,
    reasoning_budget: int = 0,
) -> subprocess.Popen:
    """Start llama-server and wait until /health answers.

    budget: --image-max-tokens value, or DEFAULT_BUDGET to leave it unset.
    cache_prompt: keep the KV cache of the previous request (off by default so that every
        request re-processes the whole prompt and timings stay comparable).
    reasoning_budget: 0 disables thinking, -1 leaves it unlimited.
    """
    # Otherwise the new server fails to bind and exits, the old one answers /health, and we
    # return a dead process: the caller believes it owns a server it can never stop.
    if server_up(port):
        raise RuntimeError(f"a server is already running on port {port}, stop it first")
    gguf, mmproj = MODELS[model]
    cmd = [str(LLAMA_SERVER), "-m", str(gguf), "-mm", str(mmproj), *SERVER_FLAGS,
            "--reasoning-budget", str(reasoning_budget),
            "--alias", f"holo-3.1-{model}", "--port", str(port)]
    if not cache_prompt:
        cmd.append("--no-cache-prompt")
    if budget != DEFAULT_BUDGET:
        cmd += ["--image-max-tokens", budget]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server_up(port):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"llama-server failed to start, see {log}")
        time.sleep(2)
    if proc.poll() is not None:  # /health answered but our process is gone: someone else's server
        raise RuntimeError(f"llama-server exited right after start, see {log}")
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
