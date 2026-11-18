"""Summarize sweep results: a Markdown table on stdout and an accuracy/latency chart."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from PIL import Image

BUDGET_ORDER = ["128", "256", "512", "1024", "default"]
# Qwen-VL: 16 px patches merged 2x2, so one image token covers 32x32 px of the resized image.
TOKEN_PX = 32
DEFAULT_MAX_TOKENS = 4096  # llama.cpp's cap for Qwen-VL when --image-max-tokens is not set
COLORS = {"4b": "#2a78d6", "9b": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(csv_paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
    groups = defaultdict(list)
    for path in csv_paths:
        with path.open() as f:
            for row in csv.DictReader(f):
                groups[(row["model"], row["image_max_tokens"])].append(row)
    return groups


def image_tokens(width: int, height: int, budget: str) -> int:
    """Image tokens llama.cpp produces for a Qwen-VL model (calc_size_preserved_ratio in mtmd-image.cpp).

    Each side is rounded to a multiple of TOKEN_PX. Above the budget, both sides are scaled by the same
    factor and rounded down, so the actual count lands a few percent under the budget.
    """
    max_tokens = DEFAULT_MAX_TOKENS if budget == "default" else int(budget)
    max_pixels = max_tokens * TOKEN_PX**2
    w = max(TOKEN_PX, round(width / TOKEN_PX) * TOKEN_PX)  # Python round() is half-to-even, like llama.cpp
    h = max(TOKEN_PX, round(height / TOKEN_PX) * TOKEN_PX)
    if w * h > max_pixels:
        beta = math.sqrt(width * height / max_pixels)
        w = max(TOKEN_PX, math.floor(width / beta / TOKEN_PX) * TOKEN_PX)
        h = max(TOKEN_PX, math.floor(height / beta / TOKEN_PX) * TOKEN_PX)
    return (w // TOKEN_PX) * (h // TOKEN_PX)


def summarize(rows: list[dict]) -> dict:
    return {
        "hits": sum(r["hit"] == "True" for r in rows),
        "n": len(rows),
        "median_dist": statistics.median(float(r["center_dist_px"]) for r in rows),
        "dists": [float(r["center_dist_px"]) for r in rows],
        "prefill_s": statistics.mean(float(r["prefill_s"]) for r in rows),
        "gen_tok_s": statistics.mean(float(r["gen_tok_per_s"]) for r in rows),
        "total_s": statistics.mean(float(r["total_s"]) for r in rows),
    }


def budget_label(budget: str) -> str:
    return f"default ({DEFAULT_MAX_TOKENS})" if budget == "default" else budget


def markdown_table(stats: dict, models: list[str], budgets: list[str], n_image: dict[str, int]) -> str:
    lines = [
        "| `--image-max-tokens` | image tokens | model | hits | median dist. to center | prefill | gen tok/s | latency |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for budget in reversed(budgets):
        for model in models:
            s = stats[(model, budget)]
            lines.append(
                f"| {budget_label(budget)} | {n_image[budget]} | {model.upper()} | {s['hits']}/{s['n']}"
                f" | {s['median_dist']:.1f} px | {s['prefill_s']:.0f} s | {s['gen_tok_s']:.1f} | {s['total_s']:.0f} s |"
            )
    return "\n".join(lines)


def plot(stats: dict, models: list[str], budgets: list[str], n_image: dict[str, int], out: Path,
         min_target_px: int | None = None) -> None:
    labels = [f"{budget_label(b)}\n({n_image[b]})" for b in budgets]
    xs = range(len(budgets))
    fig = Figure(figsize=(10, 4), dpi=150)  # no pyplot: no GUI backend needed
    ax_acc, ax_lat = fig.subplots(1, 2)
    for ax in (ax_acc, ax_lat):
        ax.set_xticks(list(xs), labels)
        ax.set_xlabel("--image-max-tokens (actual image tokens)", color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    for i, model in enumerate(models):
        color = COLORS.get(model, INK)
        series = [stats[(model, b)] for b in budgets]
        offset = (i - (len(models) - 1) / 2) * 0.12  # keep the two models' dots apart
        for x, s in zip(xs, series):  # individual targets, faint
            ax_acc.scatter([x + offset] * len(s["dists"]), s["dists"], s=14, color=color, alpha=0.35,
                            linewidths=0)
        ax_acc.plot(xs, [s["median_dist"] for s in series], color=color, linewidth=2, marker="o",
                    markersize=6, label=model.upper())
        ax_lat.plot(xs, [s["total_s"] for s in series], color=color, linewidth=2, marker="o",
                    markersize=6, label=model.upper())
        last = series[-1]
        ax_lat.annotate(f"{model.upper()}  {last['total_s']:.0f} s", (len(budgets) - 1, last["total_s"]),
                        xytext=(8, 0), textcoords="offset points", va="center", color=INK, fontsize=9)

    ax_acc.set_yscale("log")
    ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    if min_target_px:
        ax_acc.axhline(min_target_px / 2, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax_acc.text(0.99, min_target_px / 2 * 1.08, f"half the smallest target ({min_target_px / 2:g} px)",
                    transform=ax_acc.get_yaxis_transform(), ha="right", va="bottom", color=MUTED, fontsize=8)
    ax_acc.set_title("Distance from click to target center (px, log)\nline = median, dots = each target",
                    loc="left", fontsize=10, color=INK)
    ax_lat.set_title("Latency per request (s), CPU only", loc="left", fontsize=10, color=INK)
    ax_lat.set_ylim(bottom=0)
    ax_lat.set_xlim(right=len(budgets) - 0.4)
    ax_acc.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor="#fcfcfb")
    print(f"saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", type=Path, nargs="+", help="results.csv files written by holo-sweep")
    parser.add_argument("--screenshot", type=Path, required=True,
                        help="screenshot the sweeps ran on, to count its image tokens")
    parser.add_argument("--chart", type=Path, default=Path("docs/sweep.png"))
    args = parser.parse_args()

    groups = load(args.csvs)
    stats = {key: summarize(rows) for key, rows in groups.items()}
    models = sorted({m for m, _ in groups})
    budgets = [b for b in BUDGET_ORDER if all((m, b) in stats for m in models)]
    with Image.open(args.screenshot) as img:
        n_image = {b: image_tokens(*img.size, b) for b in budgets}

    print(markdown_table(stats, models, budgets, n_image))
    args.chart.parent.mkdir(parents=True, exist_ok=True)
    min_target_px = min(
        min(int(r["target_w_px"]), int(r["target_h_px"])) for rows in groups.values() for r in rows
    )
    plot(stats, models, budgets, n_image, args.chart, min_target_px)


if __name__ == "__main__":
    main()
