"""Start and stop a local llama-server for a Holo model."""

import argparse
import json
import subprocess
import time
import urllib.request
from datetime import datetime
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
STARTUP_TIMEOUT_S = 600  # a cold load of the 9B (weights not in the page cache) took >300 s
STOP_TIMEOUT_S = 10


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
        # Own session: a Ctrl-C in the terminal reaches only us, and we stop the server once.
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT,
                                start_new_session=True)
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
    """SIGTERM, then SIGKILL if llama-server hangs in its shutdown or a Ctrl-C cuts the wait.

    Without the KeyboardInterrupt case, a second Ctrl-C skipped kill() and left an orphan
    llama-server holding 2.8 GB of RAM, unseen, during later measurements (2026-11-23).
    """
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        proc.kill()
        proc.wait()


def warm_up(port: int) -> float:
    """Send a tiny text-only request and return its duration in seconds.

    llama-server mmaps the weights: they are read from disk on first use, and the kernel may
    drop them again under memory pressure. Every token goes through every layer, so one token
    pages the whole LLM in. Text-only: it shares no prefix with a real request, so it leaves
    nothing useful in the KV cache. (It does not touch the vision encoder, which is small.)
    """
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    request = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"}
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=STARTUP_TIMEOUT_S) as r:
        r.read()
    return time.perf_counter() - start


def main() -> None:
    """holo-serve: run llama-server for the agent until Ctrl-C."""
    parser = argparse.ArgumentParser(description="Serve a Holo model for the agent until Ctrl-C.")
    parser.add_argument("--model", choices=sorted(MODELS), default="9b")
    parser.add_argument("--budget", default="512", help="--image-max-tokens, or 'default'")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no-cache", action="store_true",
                        help="re-process the whole prompt at every request (to measure without the KV cache)")
    parser.add_argument("--log", type=Path, help="server log (default: runs/server-<stamp>.log)")
    args = parser.parse_args()

    log = args.log or Path("runs") / f"server-{datetime.now():%Y%m%d-%H%M%S}.log"
    cache = "off" if args.no_cache else "on"
    print(f"starting {args.model} (image budget {args.budget}, KV cache {cache}), log in {log} ...",
            flush=True)
    # KV cache on by default: the agent appends messages step after step and re-uses it.
    proc = start_server(args.model, args.budget, args.port, log, cache_prompt=not args.no_cache)
    try:
        print(f"warm-up: {warm_up(args.port):.1f} s", flush=True)
        print(f"ready: http://localhost:{args.port}/v1  model alias holo-3.1-{args.model}  (Ctrl-C to stop)",
                flush=True)
        proc.wait()
        print(f"llama-server exited with code {proc.returncode}, see {log}")
    except KeyboardInterrupt:
        print("stopping llama-server ...")
    finally:
        stop_server(proc)
