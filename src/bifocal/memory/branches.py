"""Matched ADD/IGNORE branches for atomic event retention."""

from __future__ import annotations

import copy

from ..harness import actions as A


def tree_from_dict(data, root=None):
    return A.AtomicTimeTree(
        data["video"], float(data["duration"]), root=(root or ""),
        revisions=copy.deepcopy(data["revisions"]),
        events=copy.deepcopy(data["events"]),
        reviews=copy.deepcopy(data["reviews"]),
        memory_uses=copy.deepcopy(data.get("memory_uses") or {}),
        questions={int(key): value for key, value in
                   copy.deepcopy(data["questions"]).items()},
        branches={int(key): value for key, value in
                  copy.deepcopy(data["branches"]).items()},
        global_claims=copy.deepcopy(data.get("global_claims") or {}),
        global_operations=copy.deepcopy(data.get("global_operations") or {}),
        next_revision=int(data["next_revision"]),
        next_event=int(data["next_event"]), next_review=int(data["next_review"]),
        next_memory_use=int(data.get("next_memory_use") or 1),
        next_global_claim=int(data.get("next_global_claim") or 1),
        next_global_operation=int(data.get("next_global_operation") or 1))

