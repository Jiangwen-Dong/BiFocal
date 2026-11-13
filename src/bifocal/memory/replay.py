"""The versioned decision record for the learned memory gate.

The record has two sides. The pre-outcome side is what the policy may see:
the gold-blind state, the displayed candidate actions, the deployed policy's
candidate, the branch that ran, branch probabilities and the structured result.
The training-only side holds delayed labels and realized costs. A test keeps
the two sides a strict partition of the schema.

One record is one branch. Paired branches share `decision_id` and use the
roles `observed` and `alternative`. A single unpaired decision uses `single`.
`policy_candidate_id` is what the frozen policy chose; `branch_action_id` is
what this record actually ran. Keeping those separate is what lets the scorer
learn from off-policy branches.
"""

import hashlib
import json


SCHEMA_VERSION = "replay-2"

STATE_REQUIRED = (
    "question", "options", "draft", "belief", "budget_calls_left",
    "budget_frames_left", "budget_stored_frames_left", "budget_minutes_left",
    "n_chunks", "read_coverage", "shown_notes", "review_status", "can_continue", "can_stop",
    "prior_actions", "refused_actions", "spent_actions", "question_time",
    "named_span",
)
STATE_FIELDS = STATE_REQUIRED
NOTE_FIELDS = (
    "handle", "entry_id", "video", "video_group", "layer", "source", "status", "question",
    "description", "update", "span", "assigned_span", "sighting_span",
    "chunk", "question_number", "answer_side", "reviewed", "stored_frames",
    "source_reader", "answer_support", "citation_allowed", "review_status",
)
REVIEW_STATUS_FIELDS = ("unit", "status", "actor", "frames_from_store")
REVIEW_STATUSES = ("pending", "accepted", "rejected", "unsure", "retracted", "no_frames")
REVIEW_ACTORS = ("reader", "harness", "policy")

CANDIDATE_FIELDS = (
    "candidate_id", "candidate_index", "operation", "display_label",
    "target_kind", "target_id", "target_layer", "target_chunk", "target_chunks",
    "target_span",
    "projected_calls", "projected_new_frames", "projected_stored_frames",
    "features",
)
CANDIDATE_FEATURE_FIELDS = (
    "note_age_questions", "same_chunk", "same_video", "question_term_overlap",
    "note_question_overlap", "has_span", "span_seconds", "sighting_seconds",
    "broad_leaf", "asked_before", "review_taken", "local_note", "global_note",
    "video_note", "prior_note", "note_accepted", "note_rejected", "source_reader",
    "source_harness", "retrieval_score", "question_names_window",
    "read_chunks_left", "calls_left", "frames_left", "minutes_left",
    "stored_frames_left", "memory_available", "continue_available",
    "stop_available", "operation_is_memory", "operation_is_continue",
    "operation_is_stop", "projected_calls", "projected_new_frames",
    "projected_stored_frames",
)
CANDIDATE_FEATURE_TYPES = {
    "note_age_questions": "integer", "same_chunk": "boolean",
    "same_video": "boolean", "question_term_overlap": "integer",
    "note_question_overlap": "integer", "has_span": "boolean",
    "span_seconds": "number", "sighting_seconds": "number",
    "broad_leaf": "boolean", "asked_before": "boolean",
    "review_taken": "boolean", "local_note": "boolean",
    "global_note": "boolean", "video_note": "boolean", "prior_note": "boolean",
    "note_accepted": "boolean", "note_rejected": "boolean",
    "source_reader": "boolean", "source_harness": "boolean",
    "retrieval_score": "number", "question_names_window": "boolean",
    "read_chunks_left": "integer", "calls_left": "integer",
    "frames_left": "integer", "minutes_left": "number",
    "stored_frames_left": "integer", "memory_available": "boolean",
    "continue_available": "boolean", "stop_available": "boolean",
    "operation_is_memory": "boolean", "operation_is_continue": "boolean",
    "operation_is_stop": "boolean", "projected_calls": "integer",
    "projected_new_frames": "integer", "projected_stored_frames": "integer",
}
OPERATIONS = ("retrieve", "check", "cite", "continue", "stop")
BRANCH_ROLES = ("single", "observed", "alternative")
BEHAVIOR_POLICIES = ("deterministic", "softmax", "epsilon")
BRANCH_SCHEMES = ("single", "uniform_alternative")
COLLECTION_MODES = ("deployed", "branch_collection", "offline_replay")
SPLITS = ("train", "dev", "test")
RETRIEVAL_SCORE_SOURCES = ("lexical", "external")
TARGET_KINDS = ("entry", "chunk", "option", "answer", "none")
TARGET_LAYERS = ("local", "video", "global", "prior", "none")
NOTE_LAYERS = ("local", "video", "global", "prior")
NOTE_SOURCES = ("reader", "harness")
NOTE_STATUSES = ("accepted", "rejected", "retracted", "unreviewed")

OBSERVATION_FIELDS = (
    "summary", "refused", "refusal_reason", "terminal", "next_decision_id",
    "next_state_hash", "terminal_scope", "written_entry_id", "opened_entry_id", "review_outcome",
    "read_window", "action_calls_used", "action_new_frames_used",
    "action_stored_frames_used",
)
REVIEW_OUTCOMES = ("none", "accepted", "rejected", "unsure", "no_frames")
SNAPSHOT_FIELDS = ("path", "call_log_index", "state_hash", "session_hash")
DECODING_FIELDS = ("temperature", "top_p", "max_tokens", "seed")

REWARD_COMPONENT_FIELDS = (
    "terminal", "call_cost", "new_frame_cost", "stored_frame_cost", "read_overlap",
)
TERMINAL_KINDS = ("pending", "submit", "abstain", "forced", "empty", "error")
MEMORY_TERMINALS = ("incomplete", "supported", "uncited", "abstained")
MEMORY_LABEL_OBJECTIVE = "supported_correct_cost_v1"
LABEL_OBJECTIVES = ("", MEMORY_LABEL_OBJECTIVE)
LABEL_STATUSES = ("pending", "labeled", "answer_only", "excluded")
EXCLUSION_REASONS = (
    "forced", "capped", "cut", "faulted", "refused", "harness", "empty",
    "error", "other",
)

TRAINING_DEFAULTS = {
    "terminal_answer": "",
    "terminal_kind": "pending",
    "terminal_correct": None,
    "memory_terminal": "incomplete",
    "memory_positive": False,
    "label_objective": "",
    "memory_success": None,
    "forced": False,
    "capped": False,
    "cut": False,
    "faulted": False,
    "refused": False,
    "harness": False,
    "empty": False,
    "label_status": "pending",
    "label_source": "",
    "realized_calls": 0,
    "realized_new_frames": 0,
    "realized_stored_frames": 0,
    "realized_wall_seconds": 0.0,
    "reward_components": {},
    "gold_window": None,
    "candidate_overlaps": {},
    "baseline_value": None,
    "advantage": None,
    "sample_weight": None,
    "cited_entry_id": None,
    "citation_path": "",
    "excluded_reason": "",
}

REQUIRED = (
    "campaign_id", "run_id", "trajectory_id", "branch_trajectory_id",
    "parent_branch_trajectory_id", "round_id", "collection_mode",
    "policy_version", "adapter_hash", "decoding", "video", "video_group", "uid",
    "question_number", "turn", "decision_id", "branch_root_decision_id",
    "branch_id", "branch_role", "origin_decision_id", "branch_scheme",
    "code_hash", "config_hash", "prompt_schema", "action_schema",
    "candidate_generator_version", "retrieval_score_source", "retrieval_model",
    "retrieval_score_map_sha256", "cost_model_version", "split", "fold",
    "call_cap", "frame_cap", "state", "candidates", "policy_candidate_id",
    "branch_action_id", "alternative_candidate_id", "behavior_policy",
    "behavior_probability", "behavior_temperature", "epsilon",
    "branch_probability", "pair_sampling_probability",
    "candidate_probabilities", "selection_seed", "snapshot_ref", "observation",
)

DEFAULTS = {"state_hash": "", "candidates_hash": "", "record_hash": "",
            "training_only": dict(TRAINING_DEFAULTS)}
FIELDS = ("schema_version",) + REQUIRED + tuple(DEFAULTS)
PRE_OUTCOME_FIELDS = ("schema_version",) + REQUIRED + (
    "state_hash", "candidates_hash", "record_hash",
)
GOLD_FIELDS = ("training_only",)
NULLABLE_REQUIRED = frozenset((
    "alternative_candidate_id", "origin_decision_id", "selection_seed",
    "parent_branch_trajectory_id", "adapter_hash",
    "behavior_temperature", "epsilon", "retrieval_model",
    "retrieval_score_map_sha256",
))


def canonical_sha256(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def cited_entry_id(unit, offered=None, reports=None, evidence=None):
    """Resolve a cited handle to the stable memory entry it names."""
    text = str(unit or "").strip().upper()
    if not text:
        return None
    if text.startswith("M"):
        for item in offered or ():
            if str(item.get("handle", "")).strip().upper() == text:
                return item.get("entry_id")
    elif text.startswith("E"):
        for item in evidence or ():
            if str(item.get("handle", "")).strip().upper() == text:
                return item.get("entry_id")
    elif text.startswith("W"):
        for item in reports or ():
            if "W%s" % item.get("chunk") == text:
                return item.get("entry_id") or item.get("id")
    elif text.startswith("G"):
        for report in reports or ():
            for item in report.get("global_notes_named") or []:
                if str(item.get("handle", "")).strip().upper() == text:
                    return item.get("entry_id")
    return None
