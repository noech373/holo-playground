"""run_trial end to end, without Chromium or a model: a scripted policy and a fake browser."""

import json
from types import SimpleNamespace

import pytest
from PIL import Image

from holo_playground.agent.evaluate import Regex
from holo_playground.agent.run import Task, run_trial
from holo_playground.locate import Screenshot

TASK = Task(
    id="holo-license",
    start_url="https://huggingface.co/Hcompany",
    instruction="Open the page of the Holo 3.1 9B model and give its license.",
    allowed_domains={"huggingface.co"},
    check=Regex(regex=r"\bapache[- ]?2\.0\b"),
    must_visit="/Hcompany/Holo-3.1-9B",
)
CLICK = {"note": None, "thought": "open the model",
         "tool_call": {"tool_name": "click", "element": "Holo-3.1-9B", "x": 500, "y": 250}}
SCROLL = {"note": None, "thought": "look further down",
          "tool_call": {"tool_name": "scroll", "direction": "down"}}
ANSWER = {"note": "License: apache-2.0", "thought": "done",
          "tool_call": {"tool_name": "answer", "content": "apache-2.0"}}


class FakeBrowser:
    """Same surface as Browser for the loop: a click opens the model page."""

    def __init__(self, allowed_domains):
        self.page = SimpleNamespace(url="")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def reset(self, url):
        self.page.url = url

    def screenshot(self, path):
        Image.new("RGB", (1280, 800)).save(path)
        return Screenshot.load(path)

    def execute(self, tool_call):
        if tool_call.tool_name == "click":
            self.page.url = "https://huggingface.co/Hcompany/Holo-3.1-9B"
            return "clicked; url changed to /Hcompany/Holo-3.1-9B"
        return f"{tool_call.tool_name}: page unchanged"

    def to_pixels(self, x, y):
        return x / 1000 * 1280, y / 1000 * 800


class ScriptedClient:
    """Answers each chat.completions.create call with the next step of the script."""

    def __init__(self, steps):
        self._steps = iter(steps)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        step = next(self._steps)
        if isinstance(step, Exception):
            raise step
        message = SimpleNamespace(content=json.dumps(step))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
            model_extra={"timings": {"prompt_n": 600, "cache_n": 1500, "prompt_ms": 50_000.0,
                                     "predicted_n": 120, "predicted_ms": 33_000.0}},
        )


def trial(tmp_path, steps, **kwargs):
    client = ScriptedClient(steps)
    result = run_trial(client, "holo", TASK, tmp_path, run_id="r1",  # type: ignore[arg-type]
                       make_browser=FakeBrowser, **kwargs)
    lines = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    return result, lines


def test_success_writes_one_line_per_step_then_a_judged_end_line(tmp_path):
    result, lines = trial(tmp_path, [CLICK, ANSWER])

    assert (result.stop_reason, result.answer, result.n_steps) == ("answer", "apache-2.0", 2)
    assert [line["type"] for line in lines] == ["step", "step", "end"]
    click, answer, end = lines
    assert (click["x_px"], click["y_px"]) == (640, 200)
    assert click["url_after"].endswith("/Hcompany/Holo-3.1-9B")
    assert (click["prompt_n"], click["cache_n"], click["screenshot"]) == (600, 1500, "step-01.png")
    assert answer["result"] is None and answer["x_px"] is None
    assert end["success"] is True and end["reached_page"] is True and end["n_steps"] == 2
    assert (tmp_path / "step-02.png").exists()


def test_running_out_of_steps_is_a_failure_not_a_null(tmp_path):
    result, lines = trial(tmp_path, [SCROLL, SCROLL], max_steps=2)

    assert result.stop_reason == "max_steps"
    assert lines[-1]["success"] is False
    assert lines[-1]["reached_page"] is False


def test_a_crash_still_leaves_the_steps_and_an_unjudged_end_line(tmp_path):
    with pytest.raises(ConnectionError):
        trial(tmp_path, [CLICK, ConnectionError("llama-server died")])

    lines = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [line["type"] for line in lines] == ["step", "end"]
    assert lines[-1]["stop_reason"] == "error" and lines[-1]["success"] is None
    assert "llama-server died" in lines[-1]["error"]
