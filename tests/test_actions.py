"""The Step schema: what the model is allowed to answer, and what we refuse."""

import json

import pytest
from pydantic import ValidationError

from holo_playground.agent.actions import (
    AnswerArgs,
    ClickArgs,
    ScrollArgs,
    Step,
    WriteArgs,
)


def parse(tool_call: dict, **fields) -> Step:
    return Step.model_validate_json(
        json.dumps({"note": None, "thought": "t", **fields, "tool_call": tool_call})
    )


@pytest.mark.parametrize(
    ("tool_call", "cls"),
    [
        ({"tool_name": "click", "element": "Models link", "x": 473, "y": 40}, ClickArgs),
        ({"tool_name": "write", "content": "Holo", "press_enter": True}, WriteArgs),
        ({"tool_name": "scroll", "direction": "down"}, ScrollArgs),
        ({"tool_name": "answer", "content": "Apache-2.0"}, AnswerArgs),
    ],
)
def test_tool_name_selects_the_action_class(tool_call, cls):
    assert isinstance(parse(tool_call).tool_call, cls)


def test_note_can_be_null():
    assert parse({"tool_name": "answer", "content": "x"}).note is None


def test_note_is_required():
    # H's format expects a note at every step (null when nothing new). If it were optional, the
    # grammar would let the model skip it, and it does: that is how we found out.
    assert "note" in Step.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        Step.model_validate_json('{"thought": "t", "tool_call": {"tool_name": "answer", "content": "x"}}')


def test_write_does_not_press_enter_by_default():
    assert parse({"tool_name": "write", "content": "Holo"}).tool_call.press_enter is False


@pytest.mark.parametrize(
    "tool_call",
    [
        {"tool_name": "click", "element": "e", "x": 1001, "y": 40},  # outside [0, 1000]
        {"tool_name": "click", "element": "e", "x": -1, "y": 40},
        {"tool_name": "click", "x": 473, "y": 40},  # element missing
        {"tool_name": "scroll", "direction": "left"},
        {"tool_name": "double_click", "x": 1, "y": 1},  # unknown action
    ],
)
def test_invalid_tool_calls_are_rejected(tool_call):
    with pytest.raises(ValidationError):
        parse(tool_call)


def test_thought_is_required():
    with pytest.raises(ValidationError):
        Step.model_validate_json('{"note": null, "tool_call": {"tool_name": "answer", "content": "x"}}')


def test_schema_generates_note_then_thought_then_tool_call():
    # Constrained decoding follows the schema's property order: the reasoning must come
    # before the action, or it becomes an after-the-fact justification.
    assert list(Step.model_json_schema()["properties"]) == ["note", "thought", "tool_call"]


def test_history_round_trip():
    # The agent puts step.model_dump_json() back in the history: it must parse again unchanged.
    step = parse({"tool_name": "click", "element": "Models link", "x": 473, "y": 40}, note="n")
    assert Step.model_validate_json(step.model_dump_json()) == step
