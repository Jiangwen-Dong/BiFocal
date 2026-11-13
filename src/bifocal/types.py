"""Small public types shared by the package boundary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MemoryAction(str, Enum):
    """Actions visible to the orchestrator in the checked memory protocol."""

    USE = "use"
    DEFER = "defer"
    OPEN = "open"
    READ = "read"
    REVIEW = "review"
    SUBMIT = "submit"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RunPaths:
    """External resources required by a run; none are bundled with the code."""

    model: Path
    data: Path | None = None
    runs: Path | None = None
