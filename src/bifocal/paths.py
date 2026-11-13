"""Resolve user-supplied model, data, and run paths.

BiFocal deliberately ships no model or benchmark payload. Callers provide paths
through command-line arguments or the environment variables below.
"""

from __future__ import annotations

import os
from pathlib import Path


def project_root():
    """Return the release or working tree root."""
    override = os.environ.get("BIFOCAL_ROOT")
    if override:
        return str(Path(override).expanduser().resolve())
    return str(Path.cwd().resolve())


def project_path(*parts):
    return os.path.join(project_root(), *parts)


def model_root():
    return os.environ.get("BIFOCAL_MODELS") or project_path("models")


def model_path(name):
    """Return a named model directory under ``BIFOCAL_MODELS`` or ``models/``."""
    configured = os.environ.get("BIFOCAL_MODEL_DIR")
    if configured and (not name or name in {"Qwen3-VL-8B", "Qwen/Qwen3-VL-8B"}):
        return str(Path(configured).expanduser().resolve())
    return os.path.join(model_root(), name)


def data_root():
    return os.environ.get("BIFOCAL_DATA")


def run_root():
    return os.environ.get("BIFOCAL_RUNS") or project_path("runs")
