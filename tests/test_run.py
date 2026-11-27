"""run_trial end to end, without Chromium or a model: a scripted policy and a fake browser."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from PIL import Image

from holo_playground.agent.evaluate import Regex
from holo_playground.agent.run import run_trial
from holo_playground.agent.tasks import Task
from holo_playground.screen import Screenshot

TASK = Task(
    id="holo-license",
    start_url="https://huggingface.co/Hcompany",
    instruction="Open the page of the Holo 3.1 9B model and give its license.",
    allowed_domains={"huggingface.co"},
    check=Regex(regex=r"\bapache[- ]?2\.0\b"),
    must_visit="/Hcompany/Holo-3.1-9B",
)
CLICK = {
    "note": None,
    "thought": "open the model",
    "tool_call": {"tool_name": "click", "element": "Holo-3.1-9B", "x": 500, "y": 250},
}
SCROLL = {
    "note": None,
    "thought": "look further down",
    "tool_call": {"tool_name": "scroll", "direction": "down"},
}
ANSWER = {
    "note": "License: apache-2.0",
    "thought": "done",
    "tool_call": {"tool_name": "answer", "content": "apache-2.0"},
}


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


@dataclass
class Truncated:
    """A scripted answer cut by max_tokens (finish_reason "length")."""

    text: str


class ScriptedClient:
    """Answers each chat.completions.with_raw_response.create call with the next step of the script."""

    def __init__(self, steps, retries=0):
        self._steps = iter(steps)
        self._retries = retries
        self.max_tokens = []  # max_tokens of each request
        raw = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=SimpleNamespace(with_raw_response=raw))

    def _create(self, **kwargs):
        self.max_tokens.append(kwargs["max_tokens"])
        step = next(self._steps)
        if isinstance(step, Exception):
            raise step
        finish_reason = "stop"
        if isinstance(step, Truncated):
            step, finish_reason = step.text, "length"
        content = step if isinstance(step, str) else json.dumps(step)  # str: sent as is
        message = SimpleNamespace(content=content)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            model_extra={
                "timings": {
                    "prompt_n": 600,
                    "cache_n": 1500,
                    "prompt_ms": 50_000.0,
                    "predicted_n": 120,
                    "predicted_ms": 33_000.0,
                }
            },
        )
        return SimpleNamespace(parse=lambda: response, retries_taken=self._retries)


def trial(tmp_path, steps, retries=0, client=None, **kwargs):
    client = client or ScriptedClient(steps, retries)
    result = run_trial(
        client,  # type: ignore[arg-type]
        "holo",
        TASK,
        tmp_path,
        run_id="r1",
        make_browser=FakeBrowser,
        **kwargs,
    )
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
    assert click["retries"] == 0


def test_retries_of_the_client_are_traced(tmp_path):
    # A rate-limited request is re-sent by the OpenAI client: its wall_s includes the wait.
    _, lines = trial(tmp_path, [ANSWER], retries=2)

    assert lines[0]["retries"] == 2


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


def test_an_output_that_is_not_a_step_is_an_agent_failure_not_a_crash(tmp_path):
    # An endpoint that ignores response_format can answer anything: a format failure.
    result, lines = trial(tmp_path, [CLICK, 'Sure! I will click on the "Models" link.'])

    assert (result.stop_reason, result.n_steps) == ("invalid_step", 2)
    assert lines[-1]["stop_reason"] == "invalid_step" and lines[-1]["success"] is False
    assert lines[-1]["raw_output"] == 'Sure! I will click on the "Models" link.'


def test_a_truncated_step_keeps_the_cut_output_in_the_end_line(tmp_path):
    cut = '{"note": null, "thought": "The page shows the H company organization and'
    result, lines = trial(tmp_path, [Truncated(cut)])

    assert (result.stop_reason, result.success) == ("truncated", False)
    assert lines[-1]["raw_output"] == cut


def test_an_answer_leaves_no_raw_output(tmp_path):
    _, lines = trial(tmp_path, [ANSWER])

    assert lines[-1]["raw_output"] is None


def test_max_tokens_reaches_every_request(tmp_path):
    client = ScriptedClient([CLICK, ANSWER])
    trial(tmp_path, None, client=client, max_tokens=600)

    assert client.max_tokens == [600, 600]
