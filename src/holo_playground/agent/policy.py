"""Policy: ask a Holo model, served by llama-server, for the agent's next step."""

import json
import time
from dataclasses import dataclass

from openai import OpenAI
from pydantic import ValidationError

from holo_playground.agent.actions import Step

STEP_SCHEMA = Step.model_json_schema()
# A step is 125-150 tokens with the short-thought rule: 300 leaves room
# while capping one step at ~2 min of generation on this CPU.
MAX_TOKENS = 300


class TruncatedStep(Exception):
    """The model hit max_tokens: the JSON is cut, the grammar cannot make it valid."""


class InvalidStep(Exception):
    """The output does not parse as a Step: an endpoint that ignores response_format."""


@dataclass
class PolicyOutput:
    step: Step
    raw: str            # exact generated text: goes back into the history as is (KV cache)
    timings: dict       # llama-server: cache_n, prompt_n, prompt_ms, predicted_n, predicted_ms
    wall_s: float

# Structure from H Company's structured-outputs guide: role, actions in prose, rules, then the
# JSON schema in <output_format>. The model never sees `response_format`: llama-server turns it
# into a grammar that only masks invalid tokens, so the meaning of each field must be in here.
SYSTEM_PROMPT = """You are a web agent. You complete the user's task in a web browser, one action per step.

## What you see
- <observation> holds a screenshot of the current browser viewport.
- <tool_output> describes the effect of your previous action, e.g. "url changed to /models" or "page unchanged".
Older screenshots may be replaced by [screenshot evicted]; your notes from those steps are still there.

## Actions
- click: click on an element. Give a short description of the element and its x, y position.
- write: type text into the focused text field. Click the field first, then write. Set press_enter to submit.
- scroll: scroll the page up or down by about three quarters of the screen, to see more content.
- answer: give the final answer to the task and stop. Use it as soon as the task is done.

## Rules
- Exactly one action per step.
- Coordinates are integers in [0, 1000] relative to the screenshot: (0, 0) is the top-left corner, (1000, 1000) the bottom-right corner.
- note: write down any information from the current screenshot that the task needs, since screenshots are evicted later. Use null if there is nothing new.
- If an action had no effect, do not repeat it: try something else.{thought_rule}

## Output format
<output_format>
```json
{schema}
```
</output_format>"""

SHORT_THOUGHT_RULE = "\n- thought: one short sentence."


def build_system_prompt(short_thought: bool = True) -> str:
    return SYSTEM_PROMPT.format(
        thought_rule=SHORT_THOUGHT_RULE if short_thought else "",
        schema=json.dumps(STEP_SCHEMA),
    )


def policy(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.8,
    max_tokens: int = MAX_TOKENS,
) -> PolicyOutput:
    """Ask the model for the next step, constrained to the Step schema by the server.

    temperature: 0.8 is H's recommendation. Sampling happens after the grammar mask,
        so it also applies to the digits of the click coordinates.
    model: llama-server serves a single model and ignores this name (we pass its --alias);
        a hosted API (e.g. H's) needs the real model name.
    """
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_schema", "json_schema": {"name": "step", "schema": STEP_SCHEMA}},
    )
    wall_s = time.perf_counter() - start

    choice = response.choices[0]
    raw = choice.message.content or ""
    if choice.finish_reason == "length":
        raise TruncatedStep(f"no complete step within {max_tokens} tokens: {raw[-80:]!r}")
    try:
        step = Step.model_validate_json(raw)
    except ValidationError as e:
        raise InvalidStep(f"{e.error_count()} error(s) in {raw[:80]!r}") from e
    return PolicyOutput(
        step=step,
        raw=raw,
        timings=(response.model_extra or {}).get("timings", {}),
        wall_s=wall_s,
    )
