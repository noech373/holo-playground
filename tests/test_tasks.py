"""Task files: everything wrong in them must fail at load time, before a night of trials."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from holo_playground.agent.evaluate import Contains, Regex
from holo_playground.agent.tasks import load_tasks

GOOD = {
    "id": "holo-license",
    "start_url": "https://huggingface.co/Hcompany",
    "instruction": "Open the page of the Holo 3.1 9B model and give its license.",
    "allowed_domains": ["huggingface.co"],
    "check": {"regex": r"\bapache[- ]?2\.0\b"},
    "must_visit": "/Hcompany/Holo-3.1-9B",
}


def write(tmp_path: Path, tasks: list[dict]) -> Path:
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(tasks))
    return path


def test_the_repo_task_file_loads():
    tasks = load_tasks(Path(__file__).parent.parent / "tasks" / "hf.json")
    assert tasks and all(t.check is not None for t in tasks)


def test_check_type_is_picked_from_its_key(tmp_path):
    contains = {**GOOD, "id": "other", "check": {"contains": "apache-2.0"}}
    regex_task, contains_task = load_tasks(write(tmp_path, [GOOD, contains]))
    assert isinstance(regex_task.check, Regex)
    assert isinstance(contains_task.check, Contains)
    assert regex_task.allowed_domains == frozenset({"huggingface.co"})


@pytest.mark.parametrize(
    "change",
    [
        {"must_vist": "/Hcompany"},  # typo in a field name
        {"check": {"contains": "a", "regex": "b"}},  # two checks in one
        {"check": {"regx": "apache"}},  # typo in the check
        {"check": {"regex": "apache(2.0"}},  # pattern does not compile
        {"id": "Holo License"},  # not usable as a directory name
        {"start_url": "https://github.com/hcompany"},  # blocked by its own guardrail
    ],
)
def test_invalid_task_fails_at_load(tmp_path, change):
    with pytest.raises(ValidationError):
        load_tasks(write(tmp_path, [{**GOOD, **change}]))


def test_duplicate_ids_fail_at_load(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        load_tasks(write(tmp_path, [GOOD, GOOD]))
