"""Memory: message layout, screenshot eviction, and the append-only prefix the KV cache relies on."""

import json
from pathlib import Path

import pytest

from holo_playground.agent.actions import Step
from holo_playground.agent.memory import EVICTED, Memory
from holo_playground.agent.policy import PolicyOutput
from holo_playground.screen import Screenshot

TASK = "Open the Models page of Hugging Face."


def shot(i: int) -> Screenshot:
    return Screenshot(Path(f"step-{i:02d}.png"), 1280, 800, f"data:image/png;base64,IMG{i}")


def output(i: int) -> PolicyOutput:
    # Spaces after ':' and ',' as the model writes them: Memory must re-send this text verbatim.
    raw = json.dumps({"note": None, "thought": f"step {i}",
                      "tool_call": {"tool_name": "scroll", "direction": "down"}})
    return PolicyOutput(Step.model_validate_json(raw), raw, {}, 0.0)


def run(n_steps: int, keep_last: int = 3) -> Memory:
    """Memory after n_steps steps, ready for the next policy call."""
    memory = Memory("SYSTEM", TASK, keep_last)
    memory.add_observation(shot(1))
    for i in range(1, n_steps + 1):
        memory.add_step(output(i), f"scrolled down #{i}")
        memory.add_observation(shot(i + 1))
    return memory


def images(messages: list[dict]) -> list[str]:
    return [part["image_url"]["url"] for m in messages if isinstance(m["content"], list)
            for part in m["content"] if part["type"] == "image_url"]


def text(message: dict) -> str:
    return "".join(p.get("text", "") for p in message["content"])


def test_first_request_is_system_then_task_and_screenshot():
    messages = run(0).messages()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert text(messages[1]) == f"Task: {TASK}\n<observation></observation>"
    assert images(messages) == ["data:image/png;base64,IMG1"]


def test_roles_alternate_and_raw_step_is_resent_verbatim():
    messages = run(2).messages()
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == output(1).raw
    assert text(messages[3]).startswith('<tool_output tool="scroll">scrolled down #1</tool_output>\n')


@pytest.mark.parametrize("keep_last", [1, 3])
def test_only_the_last_screenshots_are_sent(keep_last):
    messages = run(5, keep_last).messages()
    assert images(messages) == [f"data:image/png;base64,IMG{i}" for i in range(7 - keep_last, 7)]
    evicted = [m for m in messages if m["role"] == "user" and EVICTED in text(m)]
    assert len(evicted) == 6 - keep_last


def test_task_is_never_evicted():
    first_user = run(5, keep_last=1).messages()[1]
    assert text(first_user) == f"Task: {TASK}\n<observation>{EVICTED}</observation>"


@pytest.mark.parametrize("keep_last", [1, 3])
def test_request_is_a_prefix_of_the_next_until_an_eviction(keep_last):
    """What the KV cache needs: appending steps never rewrites earlier messages..."""
    for n in range(keep_last - 1):
        before, after = run(n, keep_last).messages(), run(n + 1, keep_last).messages()
        assert after[: len(before)] == before
    # ...until the first eviction, which rewrites the message of the oldest screenshot.
    before, after = run(keep_last - 1, keep_last).messages(), run(keep_last, keep_last).messages()
    assert after[: len(before)] != before


def test_order_is_enforced():
    memory = Memory("SYSTEM", TASK)
    with pytest.raises(RuntimeError):
        memory.messages()
    memory.add_observation(shot(1))
    with pytest.raises(RuntimeError):
        memory.add_observation(shot(2))
    memory.add_step(output(1), "scrolled down")
    with pytest.raises(RuntimeError):
        memory.add_step(output(2), "scrolled down")
    with pytest.raises(RuntimeError):
        memory.messages()
