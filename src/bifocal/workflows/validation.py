"""Offline validation for a BiFocal run directory."""

from __future__ import annotations

import json
from pathlib import Path


def validate_run(run_dir):
    """Parse every JSON/JSONL artifact and require a run record.

    This is an archive-level integrity check, not a scientific audit.
    """
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("run directory does not exist: %s" % root)
    records = [root / "RUN.json", root / "run.json"]
    run_record = next((path for path in records if path.is_file()), None)
    if run_record is None:
        raise ValueError("run directory has no RUN.json or run.json")
    json.loads(run_record.read_text(encoding="utf-8"))
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            checked += 1
        elif path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    json.loads(line)
            checked += 1
    return {"ok": True, "run": str(root), "json_files_checked": checked}
