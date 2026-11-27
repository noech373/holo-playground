"""Run the Holo web agent on tasks: screenshot -> policy -> browser action, until answer."""

import argparse
import json
import os
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from holo_playground.agent.actions import AnswerArgs, ClickArgs
from holo_playground.agent.browser import Browser
from holo_playground.agent.evaluate import check_answer
from holo_playground.agent.memory import Memory
from holo_playground.agent.policy import (
    MAX_TOKENS,
    InvalidStep,
    TruncatedStep,
    build_system_prompt,
    policy,
)
from holo_playground.agent.tasks import Task, load_tasks
from holo_playground.agent.trace import Trace
from holo_playground.server import warm_up

MAX_STEPS = 10


@dataclass
class TrialResult:
    """Outcome of one trial.

    success: False for any agent failure (wrong answer, max_steps, truncated, invalid_step). None
    only when it cannot be judged: the task has no check, or the trial crashed (an infra failure,
    recorded in the trace's end line). Whether the agent reached the page is judged apart
    (reached_page in the trace): a right answer without reaching the page is a guess.
    """

    answer: str | None
    n_steps: int
    stop_reason: str  # "answer" | "max_steps" | "truncated" | "invalid_step"
    visited: list[str] = field(default_factory=list)
    success: bool | None = None  # set by run_trial
    raw_output: str | None = None  # the model's text when it was not a usable step (truncated, invalid)


def run_trial(
    client: OpenAI,
    model: str,
    task: Task,
    run_dir: Path,
    *,
    keep_last: int = 3,
    temperature: float = 0.8,
    max_steps: int = MAX_STEPS,
    max_tokens: int = MAX_TOKENS,
    run_id: str = "",
    trial: int = 1,
    make_browser: Callable[[frozenset[str]], Browser] = Browser,
) -> TrialResult:
    """Run one trial and trace it to run_dir/trace.jsonl. The end line is written even on a crash."""
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with Trace(run_dir / "trace.jsonl", run_id=run_id, task_id=task.id, trial=trial) as trace:
        try:
            result = _loop(
                client,
                model,
                task,
                run_dir,
                trace,
                make_browser,
                keep_last=keep_last,
                temperature=temperature,
                max_steps=max_steps,
                max_tokens=max_tokens,
            )
        except Exception as e:
            trace.end(
                stop_reason="error",
                answer=None,
                success=None,
                total_s=round(time.perf_counter() - start, 2),
                error=repr(e),
            )
            raise
        result.success = check_answer(result.answer, task.check) if task.check else None
        trace.end(
            stop_reason=result.stop_reason,
            answer=result.answer,
            success=result.success,
            reached_page=(any(task.must_visit in url for url in result.visited) if task.must_visit else None),
            total_s=round(time.perf_counter() - start, 2),
            raw_output=result.raw_output,
        )
    return result


def _loop(
    client: OpenAI,
    model: str,
    task: Task,
    run_dir: Path,
    trace: Trace,
    make_browser: Callable[[frozenset[str]], Browser],
    *,
    keep_last: int,
    temperature: float,
    max_steps: int,
    max_tokens: int,
) -> TrialResult:
    with make_browser(task.allowed_domains) as browser:
        state = Memory(system=build_system_prompt(), task=task.instruction, keep_last=keep_last)
        browser.reset(task.start_url)
        visited = [browser.page.url]

        for i in range(1, max_steps + 1):
            shot_name = f"step-{i:02d}.png"
            screenshot = browser.screenshot(path=run_dir / shot_name)
            state.add_observation(screenshot)

            try:
                output = policy(
                    client, state.messages(), model=model, temperature=temperature, max_tokens=max_tokens
                )
            except TruncatedStep as e:
                print(f"step {i}: truncated: {e}", flush=True)
                return TrialResult(None, i, "truncated", visited, raw_output=e.raw)
            except InvalidStep as e:
                print(f"step {i}: invalid step: {e}", flush=True)
                return TrialResult(None, i, "invalid_step", visited, raw_output=e.raw)

            tool_call = output.step.tool_call
            t = output.timings
            print(
                f"step {i}: {tool_call.tool_name} | prompt_n {t.get('prompt_n')} "
                f"cache_n {t.get('cache_n')} | prefill {t.get('prompt_ms', 0) / 1000:.0f} s, "
                f"gen {t.get('predicted_n')} tok {t.get('predicted_ms', 0) / 1000:.0f} s | "
                f"wall {output.wall_s:.0f} s",
                flush=True,
            )
            print(f"  thought: {output.step.thought}", flush=True)
            if output.step.note:
                print(f"  note: {output.step.note}", flush=True)
            url_before = browser.page.url
            if isinstance(tool_call, AnswerArgs):
                tool_output = None
            else:
                tool_output = browser.execute(tool_call)
                state.add_step(output, tool_output)
                print(f"  -> {tool_output}", flush=True)
            visited.append(browser.page.url)

            x_px = y_px = None
            if isinstance(tool_call, ClickArgs):
                x_px, y_px = (round(v) for v in browser.to_pixels(tool_call.x, tool_call.y))
            trace.step(
                step=i,
                thought=output.step.thought,
                note=output.step.note,
                tool_call=tool_call.model_dump(),
                x_px=x_px,
                y_px=y_px,
                result=tool_output,
                url_before=url_before,
                url_after=browser.page.url,
                prompt_n=t.get("prompt_n"),
                cache_n=t.get("cache_n"),
                prompt_ms=t.get("prompt_ms"),
                gen_tokens=t.get("predicted_n"),
                gen_ms=t.get("predicted_ms"),
                wall_s=round(output.wall_s, 2),
                retries=output.retries,
                screenshot=shot_name,
            )
            if isinstance(tool_call, AnswerArgs):
                return TrialResult(tool_call.content, i, "answer", visited)

        return TrialResult(None, max_steps, "max_steps", visited)


def save_server_props(base_url: str, path: Path) -> None:
    """Record the server's model and settings next to the run (llama-server's /props only).

    With the server started separately, this is how a run keeps track of the model and image
    budget it used. A hosted API has no /props: nothing is saved.
    """
    root = base_url.removesuffix("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=5) as r:
            path.write_text(json.dumps(json.load(r), indent=2))
    except (OSError, ValueError):
        print(f"no /props at {root}: server settings not recorded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://localhost:8080/v1",
        help="any OpenAI-compatible endpoint: holo-serve, or a hosted API",
    )
    parser.add_argument(
        "--model",
        default="holo-3.1-9b",
        help="model name (llama-server ignores it, a hosted API needs it)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HAI_API_KEY", "none"),
        help="default: $HAI_API_KEY",
    )
    parser.add_argument("--tasks", type=Path, default=Path("tasks/hf.json"), help="JSON task file")
    parser.add_argument("--task-id", action="append", help="run only this task (repeatable); default: all")
    parser.add_argument("--trials", type=int, default=1, help="trials per task")
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help=f"cap on the tokens of one step (default {MAX_TOKENS}, sized for the local 9B; 600 for H's API)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="skip the unmeasured warm-up request")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if args.task_id:
        known = {t.id for t in tasks}
        if unknown := [i for i in args.task_id if i not in known]:
            parser.error(f"unknown task id(s) {unknown}; {args.tasks} has {sorted(known)}")
        tasks = [t for t in tasks if t.id in args.task_id]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.no_warmup:
        warm_up(client, args.model)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path("runs") / run_id
    root.mkdir(parents=True, exist_ok=True)
    # What was asked, next to what happened: the settings and the exact tasks, as run.
    save_server_props(args.base_url, root / "server.json")
    settings = {k: v for k, v in vars(args).items() if k not in ("api_key", "tasks", "task_id")}
    (root / "run.json").write_text(json.dumps({**settings, "tasks_file": str(args.tasks)}, indent=2))
    (root / "tasks.json").write_text(json.dumps([t.model_dump(mode="json") for t in tasks], indent=2))

    summary = []
    for trial in range(1, args.trials + 1):
        for task in tasks:
            print(f"\n=== {task.id}, trial {trial}/{args.trials} ===", flush=True)
            run_dir = root / task.id / f"trial-{trial}"
            start = time.perf_counter()
            try:
                result = run_trial(
                    client,
                    args.model,
                    task,
                    run_dir,
                    keep_last=args.keep_last,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    run_id=run_id,
                    trial=trial,
                )
                outcome = (result.stop_reason, result.n_steps, result.answer, result.success)
            except Exception as e:
                # One crashed trial must not cost the rest of the night. Its trace already
                # holds an end line with success: null and the error.
                print(f"trial crashed: {e!r}", flush=True)
                outcome = ("error", None, None, None)
            summary.append((task.id, trial, *outcome, time.perf_counter() - start))
            print(f"{outcome[0]} after {outcome[1]} steps: {outcome[2]!r}, success={outcome[3]}", flush=True)

    print("\n| task | trial | stop | steps | success | time |\n|---|---|---|---|---|---|")
    for task_id, trial, stop, n_steps, _answer, success, seconds in summary:
        print(f"| {task_id} | {trial} | {stop} | {n_steps} | {success} | {seconds / 60:.1f} min |")
    print(f"traces and screenshots in {root}")


if __name__ == "__main__":
    main()
