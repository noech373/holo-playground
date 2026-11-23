"""JSONL trace of a trial: one line per step, then one end line. The source of truth for analysis."""

import json
from pathlib import Path
from typing import Self


class Trace:
    """Append-only trace file; every line carries the ids (run_id, task_id, trial).

    Each line is written and flushed as soon as it is known: a trial that crashes or is
    interrupted at step 7 after 20 minutes still leaves its first 6 steps on disk.
    """

    def __init__(self, path: Path, **ids: str | int) -> None:
        self.path, self.ids = path, ids
        self.n_steps = 0
        self._file = path.open("a", encoding="utf-8")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self._file.close()

    def step(self, **fields) -> None:
        self.n_steps += 1
        self._write({"type": "step", **fields})

    def end(self, **fields) -> None:
        self._write({"type": "end", "n_steps": self.n_steps, **fields})

    def _write(self, record: dict) -> None:
        self._file.write(json.dumps({**self.ids, **record}, ensure_ascii=False) + "\n")
        self._file.flush()
