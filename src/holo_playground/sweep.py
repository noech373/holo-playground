"""Measure accuracy and latency of a Holo model across --image-max-tokens budgets.

For each budget, start a dedicated llama-server, ask the model to click every target
of a labelled screenshot, then stop the server. Results go to
<out-dir>/<timestamp>-<model>/: results.csv, clicks-<budget>.png, server-<budget>.log.
"""

import argparse
import csv
import json
import math
import subprocess
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI

from holo_playground.locate import Screenshot, draw_clicks, locate

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
    "--reasoning-budget", "0",
    "--no-cache-prompt",
]
DEFAULT_BUDGET = "default"
STARTUP_TIMEOUT_S = 300

FIELDS = [
    "model", "image_max_tokens", "target", "target_w_px", "target_h_px",
    "x", "y", "x_px", "y_px", "hit", "center_dist_px", "edge_dist_px",
    "prefill_tokens", "prefill_s", "prefill_tok_per_s",
    "gen_tokens", "gen_s", "gen_tok_per_s", "total_s",
]


def server_up(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def start_server(model: str, budget: str, port: int, log: Path) -> subprocess.Popen:
    gguf, mmproj = MODELS[model]
    cmd = [str(LLAMA_SERVER), "-m", str(gguf), "-mm", str(mmproj), *SERVER_FLAGS,
            "--alias", f"holo-3.1-{model}", "--port", str(port)]
    if budget != DEFAULT_BUDGET:
        cmd += ["--image-max-tokens", budget]
    with log.open("w") as log_file:
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not server_up(port):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError(f"llama-server failed to start, see {log}")
        time.sleep(2)
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def score(x: float, y: float, box: list[int]) -> dict:
    """Compare a click (in pixels) with a ground-truth box [x0, y0, x1, y1]."""
    x0, y0, x1, y1 = box
    return {
        "target_w_px": x1 - x0,
        "target_h_px": y1 - y0,
        "hit": x0 <= x <= x1 and y0 <= y <= y1,
        "center_dist_px": round(math.hypot(x - (x0 + x1) / 2, y - (y0 + y1) / 2), 1),
        "edge_dist_px": round(math.hypot(max(x0 - x, 0, x - x1), max(y0 - y, 0, y - y1)), 1),
    }


def run_budget(
    client: OpenAI, screenshot: Screenshot, targets: list[dict], model: str, budget: str
) -> Iterator[dict]:
    for target in targets:
        pred = locate(client, screenshot, target["element"])
        x, y = screenshot.to_pixels(pred.click)
        t = pred.timings
        row = {
            "model": model,
            "image_max_tokens": budget,
            "target": target["name"],
            "x": pred.click.x,
            "y": pred.click.y,
            "x_px": round(x),
            "y_px": round(y),
            **score(x, y, target["box"]),
            "prefill_tokens": t.get("prompt_n"),
            "prefill_s": round(t.get("prompt_ms", 0) / 1000, 2),
            "prefill_tok_per_s": round(t.get("prompt_per_second", 0), 1),
            "gen_tokens": t.get("predicted_n"),
            "gen_s": round(t.get("predicted_ms", 0) / 1000, 2),
            "gen_tok_per_s": round(t.get("predicted_per_second", 0), 1),
            "total_s": round(pred.wall_s, 2),
        }
        print(
            f"   {target['name']:<14} {'HIT ' if row['hit'] else 'MISS'} ({row['x_px']}, {row['y_px']})"
            f" {row['center_dist_px']} px from center"
            f" | prefill {row['prefill_tokens']} tok in {row['prefill_s']} s ({row['prefill_tok_per_s']} tok/s)"
            f" | total {row['total_s']} s"
        )
        yield row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="screenshot")
    parser.add_argument("targets", type=Path, help="JSON list of {name, element, box: [x0, y0, x1, y1]}")
    parser.add_argument("--model", choices=MODELS, default="9b")
    parser.add_argument("--budgets", default="default,1024,512,256,128",
                        help="comma-separated --image-max-tokens values ('default' = not set)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if server_up(args.port):
        raise SystemExit(f"a server is already running on port {args.port}, stop it first")

    targets = json.loads(args.targets.read_text())
    screenshot = Screenshot.load(args.image)
    run_dir = args.out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.model}"
    run_dir.mkdir(parents=True)
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="none")

    with (run_dir / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for budget in args.budgets.split(","):
            print(f"== {args.model} | image-max-tokens {budget}")
            proc = start_server(args.model, budget, args.port, run_dir / f"server-{budget}.log")
            rows = []
            try:
                for row in run_budget(client, screenshot, targets, args.model, budget):
                    writer.writerow(row)
                    f.flush()
                    rows.append(row)
            finally:
                stop_server(proc)
            draw_clicks(
                screenshot,
                [(r["x_px"], r["y_px"]) for r in rows],
                run_dir / f"clicks-{budget}.png",
                boxes=[t["box"] for t in targets],
            )

    print(f"saved {run_dir}")


if __name__ == "__main__":
    main()
