"""Run the Holo web agent on a task: screenshot -> policy -> browser action, until answer."""

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
from holo_playground.agent.evaluate import Check, Regex, check_answer
from holo_playground.agent.memory import Memory
from holo_playground.agent.policy import TruncatedStep, build_system_prompt, policy
from holo_playground.agent.trace import Trace

MAX_STEPS = 10


@dataclass
class Task:
    id: str
    start_url: str
    instruction: str
    allowed_domains: set[str]
    check: Check | None = None  # None: the answer is not checked, success stays null
    must_visit: str | None = None  # url fragment the agent must reach (not just guess the answer)


@dataclass
class TrialResult:
    answer: str | None
    n_steps: int
    stop_reason: str  # "answer" | "max_steps" | "truncated"
    visited: list[str] = field(default_factory=list)  # url after reset, then after each step


def run_trial(
    client: OpenAI,
    model: str,
    task: Task,
    run_dir: Path,
    *,
    keep_last: int = 3,
    temperature: float = 0.8,
    max_steps: int = MAX_STEPS,
    run_id: str = "",
    trial: int = 1,
    make_browser: Callable[[set[str]], Browser] = Browser,
) -> TrialResult:
    """Run one trial and trace it to run_dir/trace.jsonl. The end line is written even on a crash."""
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with Trace(run_dir / "trace.jsonl", run_id=run_id, task_id=task.id, trial=trial) as trace:
        try:
            result = _loop(client, model, task, run_dir, trace, make_browser,
                           keep_last=keep_last, temperature=temperature, max_steps=max_steps)
        except Exception as e:
            trace.end(stop_reason="error", answer=None, success=None,
                      total_s=round(time.perf_counter() - start, 2), error=repr(e))
            raise
        # success: false for an agent failure (wrong answer, max_steps, truncated), null only
        # when it cannot be judged (no check, or the error branch above: an infra failure).
        # reached_page is kept apart: a right answer without reaching the page is a guess.
        trace.end(
            stop_reason=result.stop_reason,
            answer=result.answer,
            success=check_answer(result.answer, task.check) if task.check else None,
            reached_page=(any(task.must_visit in url for url in result.visited)
                          if task.must_visit else None),
            total_s=round(time.perf_counter() - start, 2),
        )
    return result


def _loop(client: OpenAI, model: str, task: Task, run_dir: Path, trace: Trace,
          make_browser: Callable[[set[str]], Browser], *,
          keep_last: int, temperature: float, max_steps: int) -> TrialResult:
    with make_browser(task.allowed_domains) as browser:
        state = Memory(
            system=build_system_prompt(), task=task.instruction, keep_last=keep_last
        )
        browser.reset(task.start_url)
        visited = [browser.page.url]

        for i in range(1, max_steps + 1):
            shot_name = f"step-{i:02d}.png"
            screenshot = browser.screenshot(path=run_dir / shot_name)
            state.add_observation(screenshot)

            try:
                output = policy(
                    client, state.messages(), model=model, temperature=temperature
                )
            except TruncatedStep:
                return TrialResult(None, i, "truncated", visited)

            tool_call = output.step.tool_call
            # Printed before the answer check: the last step has the longest history.
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
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--no-warmup", action="store_true", help="skip the unmeasured warm-up request"
    )
    args = parser.parse_args()

    task = Task(
        id="holo-license",
        start_url="https://huggingface.co/Hcompany",
        instruction="Open the page of the Holo 3.1 9B model and give its license.",
        allowed_domains={"huggingface.co"},
        check=Regex(regex=r"\bapache[- ]?2\.0\b"),
        must_visit="/Hcompany/Holo-3.1-9B",
    )
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / run_id / task.id
    run_dir.mkdir(parents=True, exist_ok=True)
    save_server_props(args.base_url, run_dir / "server.json")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.no_warmup:
        # Not measured: pages the mmapped weights back in if the server sat idle. Text-only,
        # so it shares no prefix with step 1 and step 1 still measures a full prefill.
        client.chat.completions.create(
            model=args.model, max_tokens=1, messages=[{"role": "user", "content": "hi"}]
        )
    start = time.perf_counter()
    result = run_trial(
        client,
        args.model,
        task,
        run_dir,
        keep_last=args.keep_last,
        temperature=args.temperature,
        run_id=run_id,
    )
    total_s = time.perf_counter() - start
    print(
        f"{result.stop_reason} after {result.n_steps} steps in {total_s:.0f} s: {result.answer!r}"
    )
    if task.check:
        print(f"success: {check_answer(result.answer, task.check)}")
    print(f"trace and screenshots in {run_dir}")


if __name__ == "__main__":
    main()
