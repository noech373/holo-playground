import json

from holo_playground.agent.trace import Trace


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_lines_carry_ids_type_and_step_count(tmp_path):
    path = tmp_path / "trace.jsonl"
    with Trace(path, run_id="r1", task_id="t1", trial=2) as trace:
        trace.step(step=1, thought="click the search box")
        trace.step(step=2, thought="answer", note="licence: apache-2.0")
        trace.end(stop_reason="answer", answer="apache-2.0")

    lines = read_lines(path)
    assert [line["type"] for line in lines] == ["step", "step", "end"]
    assert all(line["run_id"] == "r1" and line["task_id"] == "t1" and line["trial"] == 2
               for line in lines)
    assert lines[-1]["n_steps"] == 2
    assert lines[1]["note"] == "licence: apache-2.0"


def test_each_line_is_on_disk_before_close(tmp_path):
    """A crash mid-trial must not lose the steps already taken."""
    path = tmp_path / "trace.jsonl"
    trace = Trace(path, run_id="r1", task_id="t1", trial=1)
    trace.step(step=1, thought="é non-ASCII kept as is")
    assert read_lines(path)[0]["thought"] == "é non-ASCII kept as is"
    trace.__exit__(None, None, None)


def test_appends_to_an_existing_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    for trial in (1, 2):
        with Trace(path, run_id="r1", task_id="t1", trial=trial) as trace:
            trace.end(stop_reason="max_steps")
    assert [line["trial"] for line in read_lines(path)] == [1, 2]
