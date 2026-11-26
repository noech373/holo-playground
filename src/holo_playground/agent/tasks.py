"""Task definitions, loaded from a JSON file: a campaign is data, not code."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from holo_playground.agent.browser import domain_allowed
from holo_playground.agent.evaluate import Check


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    start_url: str
    instruction: str
    allowed_domains: frozenset[str]
    check: Check | None = None  # None: the answer is not checked, success stays null
    must_visit: str | None = None  # url fragment the agent must reach (not just guess the answer)

    @model_validator(mode="after")
    def _start_url_allowed(self) -> "Task":
        if not domain_allowed(self.start_url, self.allowed_domains):
            raise ValueError(f"start_url {self.start_url} is outside {set(self.allowed_domains)}")
        return self


def load_tasks(path: Path) -> list[Task]:
    """Parse and validate a JSON list of tasks. Raises on any invalid task or duplicate id."""
    tasks = TypeAdapter(list[Task]).validate_json(path.read_bytes())
    ids = [t.id for t in tasks]
    if duplicates := {i for i in ids if ids.count(i) > 1}:
        raise ValueError(f"duplicate task ids in {path}: {sorted(duplicates)}")
    return tasks
