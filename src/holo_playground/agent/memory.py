"""Memory: the agent's state, rendered as the chat messages sent to the policy at each step."""

from dataclasses import dataclass

from holo_playground.agent.policy import PolicyOutput
from holo_playground.screen import Screenshot

EVICTED = "[screenshot evicted]"


@dataclass
class _Action:
    raw: str          # the model's exact output, re-sent as is so the KV cache keeps matching
    tool_name: str
    tool_output: str  # what Browser.execute reported


class Memory:
    """Task + every screenshot and step so far; only the last `keep_last` screenshots are sent.

    Layout (one user message per observation, so user and assistant always alternate):
        system     SYSTEM_PROMPT
        user       Task: ... <observation>[img 1]</observation>
        assistant  {"note": ..., "thought": ..., "tool_call": ...}      (raw step 1)
        user       <tool_output tool="click">...</tool_output> <observation>[img 2]</observation>
        assistant  ...
    The prefix only changes when a screenshot is evicted: until then each request re-uses the
    KV cache of the previous one and only the new step, tool output and screenshot are computed.
    """

    def __init__(self, system: str, task: str, keep_last: int = 3) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be >= 1: the model needs the current screenshot")
        self.system, self.task, self.keep_last = system, task, keep_last
        self._observations: list[Screenshot] = []
        self._actions: list[_Action] = []

    def add_observation(self, screenshot: Screenshot) -> None:
        if len(self._observations) != len(self._actions):
            raise RuntimeError("add_observation twice in a row: record the step in between")
        self._observations.append(screenshot)

    def add_step(self, out: PolicyOutput, tool_output: str) -> None:
        if len(self._actions) != len(self._observations) - 1:
            raise RuntimeError("add_step without a new observation")
        self._actions.append(_Action(out.raw, out.step.tool_call.tool_name, tool_output))


    def messages(self) -> list[dict]:
        """Chat messages for the next policy call: must end with an observation."""
        if len(self._observations) != len(self._actions) + 1:
            raise RuntimeError("messages() needs a new observation after the last step")
        first_kept = len(self._observations) - self.keep_last
        messages: list[dict] = [{"role": "system", "content": self.system}]
        for i, screenshot in enumerate(self._observations):
            # The task sits before the first screenshot and is never evicted: it stays in the
            # stable prefix. The previous tool output opens the next user message.
            prefix = f"Task: {self.task}\n" if i == 0 else self._tool_output(self._actions[i - 1])
            if i >= first_kept:
                content = [
                    {"type": "text", "text": f"{prefix}<observation>"},
                    {"type": "image_url", "image_url": {"url": screenshot.data_uri}},
                    {"type": "text", "text": "</observation>"},
                ]
            else:
                content = [{"type": "text", "text": f"{prefix}<observation>{EVICTED}</observation>"}]
            messages.append({"role": "user", "content": content})
            if i < len(self._actions):
                messages.append({"role": "assistant", "content": self._actions[i].raw})
        return messages

    @staticmethod
    def _tool_output(action: _Action) -> str:
        return f'<tool_output tool="{action.tool_name}">{action.tool_output}</tool_output>\n'
