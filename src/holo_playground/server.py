"""Start and stop a local llama-server for a Holo model."""

import argparse
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from openai import OpenAI

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
# Tuned for a 4-core CPU-only machine: WSL2 capped at 10 GB of RAM, ~7 GB free for the model.
# fmt: off
SERVER_FLAGS = [
    "-c", "8192", "-np", "1", "-t", "4", "-tb", "6",
    "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
]
# fmt: on
DEFAULT_BUDGET = "default"
STARTUP_TIMEOUT_S = 600  # a cold load of the 9B (weights not in the page cache) took >300 s
STOP_TIMEOUT_S = 10
# extra_body of every chat request: llama-server and H's API both read it. H's API thinks by default.
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}


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
    # fmt: off
    cmd = [
        str(LLAMA_SERVER), "-m", str(gguf), "-mm", str(mmproj), *SERVER_FLAGS,
        "--reasoning-budget", str(reasoning_budget),
        "--alias", f"holo-3.1-{model}", "--port", str(port),
    ]
    # fmt: on
    if not cache_prompt:
        cmd.append("--no-cache-prompt")
    if budget != DEFAULT_BUDGET:
        cmd += ["--image-max-tokens", budget]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as log_file:
        # Own session: a Ctrl-C in the terminal reaches only us, and we stop the server once.
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
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

    A second Ctrl-C must not skip kill(): an orphan server would hold GBs of RAM unseen.
    """
    proc.terminate()
    try:
        proc.wait(timeout=STOP_TIMEOUT_S)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        proc.kill()
        proc.wait()


def warm_up(client: OpenAI, model: str) -> float:
    """Send a tiny text-only request and return its duration in seconds.

    llama-server mmaps the weights: they are read from disk on first use, and the kernel may
    drop them again under memory pressure. Every token goes through every layer, so one token
    pages the whole LLM in. Text-only: it shares no prefix with a real request, so it leaves
    nothing useful in the KV cache. (It does not touch the vision encoder, which is small.)
    """
    start = time.perf_counter()
    client.with_options(timeout=STARTUP_TIMEOUT_S).chat.completions.create(
        model=model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
    )
    return time.perf_counter() - start


def main() -> None:
    """holo-serve: run llama-server for the agent until Ctrl-C."""
    parser = argparse.ArgumentParser(description="Serve a Holo model for the agent until Ctrl-C.")
    parser.add_argument("--model", choices=sorted(MODELS), default="9b")
    parser.add_argument("--budget", default="512", help="--image-max-tokens, or 'default'")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="re-process the whole prompt at every request (to measure without the KV cache)",
    )
    parser.add_argument("--log", type=Path, help="server log (default: runs/server-<stamp>.log)")
    args = parser.parse_args()

    log = args.log or Path("runs") / f"server-{datetime.now():%Y%m%d-%H%M%S}.log"
    cache = "off" if args.no_cache else "on"
    print(
        f"starting {args.model} (image budget {args.budget}, KV cache {cache}), log in {log} ...", flush=True
    )
    # KV cache on by default: the agent appends messages step after step and re-uses it.
    proc = start_server(args.model, args.budget, args.port, log, cache_prompt=not args.no_cache)
    try:
        client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="none")
        print(f"warm-up: {warm_up(client, f'holo-3.1-{args.model}'):.1f} s", flush=True)
        print(
            f"ready: http://localhost:{args.port}/v1  model alias holo-3.1-{args.model}  (Ctrl-C to stop)",
            flush=True,
        )
        proc.wait()
        print(f"llama-server exited with code {proc.returncode}, see {log}")
    except KeyboardInterrupt:
        print("stopping llama-server ...")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
