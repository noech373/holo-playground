"""Agent report: traces with llama-server timings, and traces from a hosted API without them."""

import pytest
from PIL import Image

from holo_playground.agent.report import markdown_table, plot_step_cost, plot_trajectory

TIMINGS = {"prompt_n": 1500, "cache_n": 0, "prompt_ms": 100_000, "gen_tokens": 120, "gen_ms": 30_000}
END = {"stop_reason": "answer", "n_steps": 2, "total_s": 5.0, "answer": "mit", "success": True}


def steps(**timings):
    base = {"x_px": None, "y_px": None, "wall_s": 1.7, "screenshot": "shot.png", **timings}
    return [
        {**base, "step": 1, "tool_call": {"tool_name": "scroll", "direction": "down"}},
        {**base, "step": 2, "tool_call": {"tool_name": "answer", "content": "mit"}},
    ]


def test_table_with_timings_splits_prefill_and_generation():
    table = markdown_table(steps(**TIMINGS), END)
    assert "| 1 | scroll down | 1500 | 0 | 100 s | 30 s (120 tok) | 2 s |" in table


def test_table_without_timings_shows_the_wall_time():
    table = markdown_table(steps(**dict.fromkeys(TIMINGS)), END)
    assert "| step | action | model call |" in table
    assert "| 1 | scroll down | 1.7 s |" in table
    assert "None |" not in table


@pytest.mark.parametrize("timings", [TIMINGS, dict.fromkeys(TIMINGS)], ids=["local", "hosted"])
def test_both_figures_are_written(tmp_path, timings):
    Image.new("RGB", (128, 80), "white").save(tmp_path / "shot.png")
    plot_trajectory(steps(**timings), tmp_path, tmp_path / "trajectory.png")
    plot_step_cost(steps(**timings), tmp_path / "cost.png")
    assert (tmp_path / "trajectory.png").stat().st_size > 0
    assert (tmp_path / "cost.png").stat().st_size > 0
