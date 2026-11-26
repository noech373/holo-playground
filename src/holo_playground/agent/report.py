"""Summarize an agent trace: a Markdown table on stdout, a trajectory strip and a per-step cost chart."""

import argparse
import json
import textwrap
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch
from PIL import Image

from holo_playground.report import GRID, INK, MUTED

PREFILL, GEN = "#2a78d6", "#eb6834"  # categorical slots 1 and 2, as in docs/sweep.png
SURFACE = "#fcfcfb"


def load(trace: Path) -> tuple[list[dict], dict]:
    lines = [json.loads(line) for line in trace.read_text().splitlines()]
    steps = [line for line in lines if line["type"] == "step"]
    end = next(line for line in reversed(lines) if line["type"] == "end")
    return steps, end


def action(step: dict) -> str:
    """Short human label of what the agent did at this step."""
    call = step["tool_call"]
    match call["tool_name"]:
        case "click":
            return f"click: {call['element']}"
        case "write":
            return f"type: {call['content']!r}"
        case "scroll":
            return f"scroll {call['direction']}"
        case "answer":
            return f"answer: {call['content']!r}"
    return call["tool_name"]


def markdown_table(steps: list[dict], end: dict) -> str:
    rows = ["| step | action | computed tokens (prompt_n) | cached tokens (cache_n) "
            "| prefill | generation | total |",
            "|---|---|---|---|---|---|---|"]
    for s in steps:
        rows.append(f"| {s['step']} | {action(s)} | {s['prompt_n']} | {s['cache_n']} | "
                    f"{s['prompt_ms'] / 1000:.0f} s | {s['gen_ms'] / 1000:.0f} s ({s['gen_tokens']} tok) | "
                    f"{s['wall_s']:.0f} s |")
    rows.append(f"\n{end['stop_reason']} after {end['n_steps']} steps in {end['total_s']:.0f} s: "
                f"{end['answer']!r}, success={end['success']}, reached_page={end.get('reached_page')}")
    return "\n".join(rows)


def plot_trajectory(steps: list[dict], run_dir: Path, out: Path) -> None:
    """One thumbnail per step, the click drawn where the model aimed, the action below."""
    fig = Figure(figsize=(3.2 * len(steps), 2.9), dpi=150)
    axes = fig.subplots(1, len(steps), squeeze=False)[0]
    for ax, s in zip(axes, steps, strict=True):
        with Image.open(run_dir / s["screenshot"]) as img:
            ax.imshow(img)
        if s["x_px"] is not None:
            ax.scatter([s["x_px"]], [s["y_px"]], s=260, facecolors="none", edgecolors=SURFACE,
                       linewidths=4.5)
            ax.scatter([s["x_px"]], [s["y_px"]], s=260, facecolors="none", edgecolors=GEN,
                       linewidths=2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_title(f"Step {s['step']}", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("\n".join(textwrap.wrap(action(s), 38)), fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


def plot_step_cost(steps: list[dict], out: Path) -> None:
    """Seconds per step, split into prefill and generation; computed vs. cached tokens on top."""
    xs = range(len(steps))
    prefill = [s["prompt_ms"] / 1000 for s in steps]
    gen = [s["gen_ms"] / 1000 for s in steps]
    fig = Figure(figsize=(7.5, 3.8), dpi=150)
    ax = fig.subplots()
    # edgecolor=SURFACE: a thin gap between the two stacked segments
    ax.bar(xs, prefill, width=0.55, color=PREFILL, edgecolor=SURFACE, linewidth=1)
    ax.bar(xs, gen, width=0.55, bottom=prefill, color=GEN, edgecolor=SURFACE, linewidth=1)
    for x, s, p, g in zip(xs, steps, prefill, gen, strict=True):
        ax.annotate(f"{s['prompt_n']:,} computed\n{s['cache_n']:,} cached", (x, p + g),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
    labels = []
    for prev, s in zip([None, *steps], steps, strict=False):  # (previous, current) pairs
        # The cache only shrinks when the prompt's prefix changed: here, a screenshot evicted.
        reset = prev is not None and s["cache_n"] < prev["cache_n"] + prev["prompt_n"] - 50
        label = f"{s['step']}\n{s['tool_call']['tool_name']}"
        labels.append(label + ("\n(screenshot evicted)" if reset else ""))
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("seconds", color=MUTED)
    ax.set_ylim(0, max(p + g for p, g in zip(prefill, gen, strict=True)) * 1.28)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Time per agent step (Holo 3.1 9B, CPU), with prompt tokens computed vs. reused "
                 "from the KV cache", loc="left", fontsize=9.5, color=INK)
    ax.legend(handles=[Patch(color=PREFILL, label="prefill (reading the prompt)"),
                       Patch(color=GEN, label="generation (writing the step)")],
              frameon=False, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="trace.jsonl written by holo-agent (screenshots next to it)")
    parser.add_argument("--out", type=Path, default=Path("docs"), help="folder for the two PNG files")
    args = parser.parse_args()

    steps, end = load(args.trace)
    print(markdown_table(steps, end))
    args.out.mkdir(parents=True, exist_ok=True)
    plot_trajectory(steps, args.trace.parent, args.out / "agent-trajectory.png")
    plot_step_cost(steps, args.out / "agent-step-cost.png")


if __name__ == "__main__":
    main()
