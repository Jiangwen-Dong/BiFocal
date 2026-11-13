"""Hash-bound isolation and audit artifacts for atomic memory decisions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re


FORBIDDEN_STATE_KEYS = frozenset((
    "gold", "gold_window", "answer_key", "correct", "training_only", "reward",
    "memory_positive", "citation_tier", "gold_answer", "correct_answer",
))
READER_ALLOWED_TEXT_PATHS = (
    "question", "options", "draft",
    "branches/ADD/questions", "branches/IGNORE/questions",
)
ORCHESTRATOR_ALLOWED_TEXT_PATHS = (
    "question", "options", "draft", "candidate_preview", "use_prompt",
    "memory_uses/*/option", "memory_uses/*/quote", "memory_uses/*/preview",
    "*memory_uses/*/option", "*memory_uses/*/quote", "*memory_uses/*/preview",
    "revisions/*/branch_summary", "revisions/*/quote", "events/*/text",
    "*revisions/*/branch_summary", "*revisions/*/quote", "*events/*/text",
    "revisions/*/raw_reply", "*revisions/*/raw_reply",
    "*reviews/*/reader_quote", "*reviews/*/review_quote",
    "global_claims/*/event_text", "global_claims/*/quote",
    "*global_claims/*/event_text", "*global_claims/*/quote",
    "branches/USE/questions", "branches/DEFER/questions",
)
ALLOWED_TEXT_PATHS = READER_ALLOWED_TEXT_PATHS
SAFE_OBJECT_PATH = re.compile(
    r"^(?:(?:tree/)?(?:events|revisions|branches|memory_uses|"
    r"global_claims|global_operations)/[A-Za-z0-9_-]+|"
    r"memory/[A-Za-z0-9_-]+)$")


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_precommit_snapshot(directory, payload):
    """Write one atomic READ state before its revision/event is committed."""
    required = ("tree", "reader_reply", "parsed_report", "frame_handle",
                "branch", "span", "revision_id", "question_number", "question")
    if not isinstance(payload, dict):
        raise ValueError("precommit payload must be an object")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError("precommit payload missing %s" % ", ".join(missing))
    # Hash the JSON representation that will be reloaded later. Tree question
    # keys are integer in memory and strings on disk, so hashing the live object
    # would make a valid anchor fail its own reload check.
    serialized_payload = json.loads(
        json.dumps(payload, ensure_ascii=False, sort_keys=True))
    state_sha = canonical_sha256(serialized_payload)
    stem = "%04d-%s-b%d-%s" % (
        int(payload["question_number"]), payload["revision_id"],
        int(payload["branch"]), state_sha[:16])
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, stem + ".json")
    record = {
        "schema_version": "atomic-read-precommit-v1",
        "snapshot_sha256": state_sha,
        "payload": copy.deepcopy(serialized_payload),
    }
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)
    return path

