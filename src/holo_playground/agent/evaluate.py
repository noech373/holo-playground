"""Answer checks: does a trial's final answer contain what the task expects?"""

import re

from pydantic import BaseModel, field_validator


class Contains(BaseModel):
    """The normalized answer contains this text (normalized the same way)."""
    contains: str


class Regex(BaseModel):
    """re.search on the normalized (lowercase) answer: write the pattern in lowercase."""
    regex: str

    @field_validator("regex")
    @classmethod
    def _compiles(cls, pattern: str) -> str:
        # A broken pattern fails when the task is loaded, not after a trial. re.error is not a
        # ValueError: pydantic only turns ValueError/AssertionError into a ValidationError.
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from e
        return pattern


Check = Contains | Regex


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, nothing more.

    Stripping punctuation would merge distinct values ("v1.10" and "v11.0" both -> "v110"):
    tolerate formatting variants with a Regex check instead.
    """
    return " ".join(text.lower().split())


def check_answer(answer: str | None, check: Check) -> bool:
    """True if the answer passes the check. No answer (max_steps, truncated) is a failure."""
    if answer is None:
        return False
    a = normalize(answer)
    match check:
        case Contains(contains=expected):
            return normalize(expected) in a
        case Regex(regex=pattern):
            return re.search(pattern, a) is not None
