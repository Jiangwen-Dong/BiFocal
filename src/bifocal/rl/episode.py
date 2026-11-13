#!/usr/bin/env python3
"""Online episode records and the base-policy supply and variance report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys

from ..harness import actions as A


SCHEMA_VERSION = "online-memory-episode-v1"
LEGAL_ACTIONS = {"open", "read", "review", "check", "use", "defer",
                 "evolve_global", "submit", "abstain"}
TERMINALS = {"agent_submit", "agent_abstain", "protocol_abstain"}
INFRA_FAULTS = ("backend_cut", "frames_missing", "gold_leak",
                "capacity", "short_cache")
MIXED_GROUP_FLOOR = 16
EVIDENCE_WEIGHT = 0.20
GROUNDING_WEIGHT = 0.10
MEMORY_REUSE_WEIGHT = 0.10
PROGRESS_WEIGHT = 0.05
COST_WEIGHT = 0.10
PROTOCOL_WEIGHT = 0.20


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _sampling_ok(step):
    prompt = step.get("prompt_token_ids")
    tokens = step.get("completion_token_ids")
    logprobs = step.get("completion_logprobs")
    return bool(
        isinstance(prompt, list) and prompt
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in prompt)
        and isinstance(tokens, list) and tokens
        and all(isinstance(value, int) and not isinstance(value, bool)
                for value in tokens)
        and isinstance(logprobs, list)
        and len(logprobs) == len(tokens)
        and all(isinstance(value, (int, float))
                and not isinstance(value, bool) for value in logprobs)
        and step.get("finish_reason") == "stop")


def _budget(settings):
    return {
        "calls": int(settings.get("max_calls") or 0),
        "frames": int(settings.get("frame_cap") or 0),
        "minutes": float(settings.get("max_minutes") or 0.0),
        "tokens": int(settings.get("max_tokens") or 0),
        "new_tokens": int(settings.get("max_new_tokens") or 0),
    }


def _policy_versions(settings):
    model = settings.get("model")
    if isinstance(model, dict):
        model = model.get("name")
    serving = settings.get("serving") or {}
    base = serving.get("model_revision") or model or "unknown"
    adapter = settings.get("adapter") or ""
    o_adapter = settings.get("o_adapter") or adapter
    reader_adapter = settings.get("reader_adapter") or ""
    roles = set(settings.get("adapter_roles") or ())
    return {
        "base_model": base,
        "model_sha256": str(settings.get("model_sha256")
                            or serving.get("model_sha256") or ""),
        "quantization": str(settings.get("model_quantization") or "bf16"),
        "o": o_adapter if "turn" in roles and o_adapter else base,
        "o_sha256": str(settings.get("o_adapter_sha256")
                         or settings.get("adapter_sha256") or ""),
        "reader": reader_adapter if any(
            str(role).startswith(("loopread", "selective", "all", "verify",
                                  "recall", "zoom"))
            for role in roles) and reader_adapter else base,
        "reader_sha256": str(settings.get("reader_adapter_sha256") or ""),
        "reviewer": adapter if any(
            str(role).startswith("review") for role in roles) and adapter
        else base,
    }


def _norm_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _normalize_gold_windows(value):
    if not isinstance(value, (list, tuple)) or not value:
        return []
    if (len(value) == 2
            and all(isinstance(item, (int, float)) for item in value)):
        candidates = [value]
    else:
        candidates = value
    windows = []
    for item in candidates:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            first, last = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if last > first:
            windows.append((min(first, last), max(first, last)))
    windows.sort()
    merged = []
    for first, last in windows:
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def _gold_windows(row, settings):
    value = ((settings or {}).get("training_gold_windows")
             or (settings or {}).get("training_gold_window")
             or row.get("clue_intervals")
             or row.get("window"))
    return _normalize_gold_windows(value)


def _step_span(step):
    action = step.get("action") or {}
    if action.get("window"):
        value = action["window"]
        try:
            return float(value[0]), float(value[1])
        except (IndexError, TypeError, ValueError):
            pass
    for op in step.get("memory_ops") or ():
        if op.get("op") == "append" and op.get("time_span"):
            value = op["time_span"]
            try:
                return float(value[0]), float(value[1])
            except (IndexError, TypeError, ValueError):
                continue
    return None


def _covered_seconds(intervals, gold):
    gold = _normalize_gold_windows(gold)
    if not gold:
        return 0.0
    clipped = []
    for span in intervals:
        if not span:
            continue
        for first, last in gold:
            start = max(first, float(span[0]))
            stop = min(last, float(span[1]))
            if stop > start:
                clipped.append((start, stop))
    covered = _normalize_gold_windows(clipped)
    return sum(last - first for first, last in covered)


def _read_evidence_gains(steps, gold):
    gold_windows = _normalize_gold_windows(gold)
    seen, previous, gains = [], 0.0, {}
    for index, step in enumerate(steps):
        if step.get("actor") != "o" or step.get("action", {}).get("verb") != "read":
            continue
        if not any(op.get("op") == "append" and op.get("committed")
                   for op in step.get("memory_ops") or ()):
            continue
        span = _step_span(step)
        if span:
            seen.append(span)
        covered = _covered_seconds(seen, gold_windows)
        if gold_windows:
            width = max(1e-9, sum(
                last - first for first, last in gold_windows))
        else:
            width = 1.0
        gain = max(0.0, (covered - previous) / width)
        previous = covered
        gains[index] = gain
    return gains


def _accepted_evidence(row):
    accepted = []
    for value in row.get("reviews") or ():
        if value.get("outcome") == "accept":
            accepted.append(value)
    for value in row.get("memory_uses") or ():
        if value.get("action") in ("use", "check", "cite") \
                and value.get("outcome") == "accept":
            accepted.append(value)
    return accepted


def _submitted_revision(row):
    commit = row.get("commit") or {}
    revision = commit.get("revision_id") or row.get("submitted_revision") or ""
    if not revision:
        unit = str(commit.get("unit") or "")
        if re.fullmatch(r"R\d+", unit):
            revision = unit
    if not revision:
        entry = str(row.get("submitted_entry_id") or "")
        if re.fullmatch(r"R\d+", entry):
            revision = entry
    return str(revision)


def _grounding_reward(row):
    if not row.get("commit_ok") or float(row.get("credit") or 0.0) <= 0.0:
        return 0.0
    quote = _norm_text(row.get("submitted_quote")
                       or (row.get("commit") or {}).get("quote"))
    if not quote:
        return 0.0
    revision_id = _submitted_revision(row)
    memory = row.get("atomic_memory") or {}
    revision = (memory.get("revisions") or {}).get(revision_id) or {}
    events = memory.get("events") or {}
    event_text = " ".join(
        str((events.get(event_id) or {}).get("text") or "")
        for event_id in revision.get("events") or ()
        if isinstance(events.get(event_id), dict))
    observed = _norm_text(" ".join((
        *(str(revision.get(name) or "") for name in (
            "branch_summary", "quote", "raw_reply")),
        event_text)))
    if not observed or quote not in observed:
        return 0.0
    for accepted in _accepted_evidence(row):
        same_revision = str(accepted.get("revision_id") or "") == revision_id
        accepted_quote = _norm_text(accepted.get("quote")
                                    or accepted.get("review_quote"))
        if same_revision and accepted_quote == quote:
            return GROUNDING_WEIGHT
    return 0.0


def _memory_reuse_reward(row):
    if float(row.get("credit") or 0.0) <= 0.0:
        return 0.0
    revision_id = _submitted_revision(row)
    if not revision_id:
        return 0.0
    accepted_use = any(
        str(value.get("revision_id") or "") == revision_id
        for value in row.get("memory_uses") or ()
        if value.get("action") in ("use", "check", "cite")
        and value.get("outcome") == "accept")
    return MEMORY_REUSE_WEIGHT if accepted_use else 0.0


def _reader_provenance_valid(reader_steps, records):
    """Check the Reader-owned event records without trusting model text."""
    if not reader_steps or len(records) != len(reader_steps):
        return False
    for record in records:
        if not isinstance(record, dict) or not record.get(
                "memory_operation_valid", False):
            return False
        operation = str(record.get("memory_operation") or "").upper()
        events = record.get("events") or []
        if operation in ("ADD", "SUPERSEDE"):
            if len(events) != 1:
                return False
            event = events[0]
            if str(event.get("span_source") or "") != "read_window":
                return False
            try:
                start, end = float(event.get("start")), float(event.get("end"))
            except (TypeError, ValueError):
                return False
            if not (math.isfinite(start) and math.isfinite(end)
                    and 0.0 <= start < end
                    and str(event.get("text") or "").strip()):
                return False
            if operation == "SUPERSEDE" and not str(
                    event.get("supersedes_event_id") or "").strip():
                return False
        elif operation == "KEEP":
            if events:
                return False
        else:
            return False
    return True


def _reader_role_reward(row, steps, budget, reward, reader_clean,
                        reader_memory_valid, reader_event_records):
    """Compute the Reader return from Reader-owned terms plus final O result."""
    reader_steps = [step for step in steps if step.get("actor") == "reader"]
    reader_ids = {str(record.get("revision_id") or "")
                  for record in reader_event_records}
    accepted_ids = {str(value.get("revision_id") or "")
                    for value in _accepted_evidence(row)}
    submitted = _submitted_revision(row)
    reader_reuse = MEMORY_REUSE_WEIGHT if accepted_ids.intersection(
        reader_ids) else 0.0
    reader_citation = 0.05 if (
        submitted and submitted in reader_ids and row.get("commit_ok")
        and float(row.get("credit") or 0.0) > 0.0) else 0.0
    frame_cap = max(1, int(budget.get("frames") or 1))
    call_cap = max(1, int(budget.get("calls") or 1))
    reader_frames = sum(int(step.get("frames") or 0) for step in reader_steps)
    reader_calls = sum(int(step.get("calls") or 0) for step in reader_steps)
    reader_cost = (-0.05 * min(1.0, reader_frames / float(frame_cap))
                   - 0.05 * min(1.0, reader_calls / float(call_cap)))
    provenance = _reader_provenance_valid(
        reader_steps, reader_event_records)
    grounding = GROUNDING_WEIGHT if accepted_ids.intersection(
        reader_ids) else 0.0
    return {
        "task": float(reward.get("task") or 0.0),
        "schema": 0.02 if reader_clean else 0.0,
        "operation": 0.02 if reader_clean and reader_memory_valid else 0.0,
        "provenance": 0.03 if reader_clean and provenance else 0.0,
        "grounding": float(grounding),
        "memory_reuse": float(reader_reuse),
        "citation": float(reader_citation),
        "invalid_operation": -0.20 if (
            reader_steps and not reader_memory_valid) else 0.0,
        "cost": round(reader_cost, 6),
    }


def _reward(row, steps, budget, gold_window=None, gold_windows=None):
    accepted = _accepted_evidence(row)
    ended = str(row.get("ended_by") or "")
    if ended == "agent_submit":
        task = 1.0 if float(row.get("credit") or 0.0) > 0 else -1.0
    elif ended == "agent_abstain":
        task = -1.0 if accepted else -0.5
    elif ended == "protocol_abstain":
        task = -1.0
    else:
        task = -1.0
    call_cap = max(1, int(budget.get("calls") or 1))
    frame_cap = max(1, int(budget.get("frames") or 1))
    evidence_by_step = _read_evidence_gains(
        steps, gold_windows if gold_windows is not None else gold_window)
    progress = (PROGRESS_WEIGHT if accepted and any(
        step.get("action", {}).get("verb") == "read" for step in steps) else 0.0)
    reuse = _memory_reuse_reward(row)
    grounding = _grounding_reward(row)
    o_steps = [index for index, step in enumerate(steps)
               if step.get("actor") == "o"]
    terminal_index = o_steps[-1] if o_steps else None
    step_returns, totals = [], {
        "task": task, "evidence_gain": 0.0, "grounding": grounding,
        "memory_reuse": reuse, "progress": progress,
        "global_memory": 0.0, "cost": 0.0, "protocol": 0.0}
    global_rewarded = False
    for index in o_steps:
        step = steps[index]
        call_cost = min(1.0, float(step.get("calls") or 0) / call_cap)
        frame_cost = min(1.0, float(step.get("frames") or 0) / frame_cap)
        global_memory = 0.0
        if not global_rewarded and step.get("valid") and any(
                op.get("op") == "evolve_global" and op.get("committed")
                for op in step.get("memory_ops") or ()):
            global_memory = A.GLOBAL_REWARD
            global_rewarded = True
        components = {
            "answer": task if index == terminal_index else 0.0,
            "evidence_gain": EVIDENCE_WEIGHT * evidence_by_step.get(index, 0.0),
            "grounding": grounding if index == terminal_index else 0.0,
            "memory_reuse": reuse if index == terminal_index else 0.0,
            "progress": progress if index == terminal_index else 0.0,
            "global_memory": global_memory,
            "cost": -0.05 * call_cost - 0.05 * frame_cost,
            "protocol": (-PROTOCOL_WEIGHT if not step.get("valid") else 0.0),
        }
        step_returns.append({
            "step_index": index, "action": step.get("action"), **components,
            "reward": sum(components.values())})
        for name in ("evidence_gain", "global_memory", "cost", "protocol"):
            totals[name] += components[name]
    if totals["cost"] < -COST_WEIGHT:
        scale = COST_WEIGHT / abs(totals["cost"])
        for value in step_returns:
            value["cost"] *= scale
            value["reward"] = sum(
                item for name, item in value.items()
                if name in ("answer", "evidence_gain", "grounding",
                            "memory_reuse", "progress", "global_memory",
                            "cost", "protocol"))
        totals["cost"] = -COST_WEIGHT
    total = sum(totals.values())
    return {
        "task": round(task, 6),
        "evidence_gain": round(totals["evidence_gain"], 6),
        "grounding": round(grounding, 6),
        "memory_reuse": round(reuse, 6),
        "progress": round(progress, 6),
        "global_memory": round(totals["global_memory"], 6),
        "cost": round(totals["cost"], 6),
        "protocol": round(totals["protocol"], 6),
        "total": round(total, 6),
        "step_returns": step_returns,
    }


def _action_for_call(call):
    action = call.get("parsed_action")
    if action not in LEGAL_ACTIONS:
        return {"verb": action or "unparsed"}
    verb = action
    if verb in ("open", "read"):
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "branch": parsed.branch,
                    "window": list(parsed.window) if parsed.window else None}
        except A.ActionError:
            return {"verb": verb}
    if verb == "review":
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "revision": parsed.revision}
        except A.ActionError:
            return {"verb": verb}
    if verb == "check":
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "memory_handle": parsed.memory_handle}
        except A.ActionError:
            return {"verb": verb}
    if verb in ("use", "defer"):
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "revision": parsed.revision}
        except A.ActionError:
            return {"verb": verb}
    if verb == "evolve_global":
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb,
                    "global_operation": parsed.global_operation,
                    "global_target": parsed.global_target,
                    "revision": parsed.revision}
        except A.ActionError:
            return {"verb": verb}
    if verb == "submit":
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "option": parsed.option,
                    "revision": parsed.revision,
                    "memory_handle": parsed.memory_handle,
                    "quote": parsed.quote}
        except A.ActionError:
            return {"verb": verb}
    if verb == "abstain":
        try:
            parsed = A.parse_action(call.get("reply") or "")
            return {"verb": verb, "reason": parsed.reason}
        except A.ActionError:
            return {"verb": verb}
    return {"verb": verb}


def from_row(row, calls, settings, group_id=None, rollout_index=0,
             source_ref=""):
    """Build one episode from actual harness rows and calls."""
    uid = str(row.get("uid") or "")
    row_run = str(row.get("run_id") or "")
    row_calls = [call for call in calls
                 if str(call.get("uid") or "") == uid
                 and (not row_run or str(call.get("run_id") or "") == row_run)]
    turns = [turn for turn in row.get("turns") or ()
             if turn.get("actor", "o") == "o"]
    turn_requests = [str(turn.get("request_id") or "") for turn in turns]
    turn_by_request = {request_id: turn for request_id, turn
                       in zip(turn_requests, turns) if request_id}
    turn_calls = [call for call in row_calls if call.get("role") == "turn"]
    pairing_errors = []
    if len(turns) != len(turn_calls):
        pairing_errors.append("count")
    if len(turn_requests) != len(set(turn_requests)):
        pairing_errors.append("duplicate_turn_request_id")
    used_turns = set()
    steps, o_index = [], 0
    last_turn_step = None
    for index, call in enumerate(row_calls):
        role = str(call.get("role") or "")
        turn = {}
        if role == "turn":
            request_id = str(call.get("request_id") or "")
            turn = turn_by_request.get(request_id)
            if turn is None or not request_id or request_id in used_turns:
                turn = turns[o_index] if o_index < len(turns) else {}
                pairing_errors.append(request_id or "missing_request_id")
            else:
                used_turns.add(request_id)
            o_index += 1
            actor = "o"
            action = _action_for_call(call)
            legal = (call.get("parsed_action") in LEGAL_ACTIONS
                     and int(call.get("action_line_count") or 0) == 1)
            valid = legal and not (turn.get("refusals") or [])
            memory_ops = list(turn.get("memory_ops") or [])
            memory_sha = turn.get("memory_sha256")
        elif role.startswith("loopread-"):
            actor, action = "reader", {"verb": "read_result"}
            strict_reader = bool(settings.get("reader_policy_training"))
            reader_report = None
            if strict_reader:
                reader_span = call.get("reader_span") or []
                reader_report = A.parse_reader_report(
                    call.get("reply") or "", reader_policy=True,
                    reader_schema_version=int(settings.get(
                        "reader_schema_version", 1)),
                    first=(float(reader_span[0])
                           if len(reader_span) == 2 else None),
                    last=(float(reader_span[1])
                          if len(reader_span) == 2 else None))
                legal = reader_report.get("parse_status") == "ok"
                valid = bool(
                    legal and call.get("finish_reason") == "stop"
                    and not call.get("recovered_action")
                    and not call.get("reader_retry_cut")
                    and not str(role).endswith("-retry"))
            else:
                legal, valid = True, True
            memory_ops, memory_sha = [], None
        elif role.startswith("review-"):
            try:
                A.parse_review(call.get("reply") or "")
                format_valid = True
            except A.ActionError:
                format_valid = False
            refused = bool(last_turn_step is not None
                           and not last_turn_step.get("valid"))
            actor, action = "reviewer", {"verb": "review_result"}
            legal = format_valid
            valid = format_valid and last_turn_step is not None and not refused
            memory_ops, memory_sha = [], None
        else:
            actor, action, legal, valid = role, {"verb": role or "unknown"}, True, True
            memory_ops, memory_sha = [], None
        adapter_active = bool(call.get("adapter_active"))
        turn_state = turn.get("state") or {}
        charged_calls = (turn.get("calls") if actor == "o"
                         else call.get("calls"))
        charged_frames = (turn.get("frames") if actor == "o"
                          else call.get("frames"))
        adapter_name = str(call.get("adapter_name") or "")
        if not adapter_name:
            adapter_name = ("o" if adapter_active and actor in {
                "o", "reviewer"} else "base")
        step = {
            "t": index,
            "actor": actor,
            "adapter": adapter_name,
            "parent_step": index - 1 if index else None,
            "state_ref": "calls.jsonl#%d" % index,
            "state_sha256": _sha256(call.get("prompt")),
            "prompt_token_ids": call.get("prompt_token_ids"),
            "completion_token_ids": call.get("completion_token_ids"),
            "completion_logprobs": call.get("completion_logprobs"),
            "completion_raw_logprobs": call.get("completion_raw_logprobs"),
            "generation_completion_raw_logprobs": call.get(
                "generation_completion_raw_logprobs"),
            "allowed_token_ids": copy.deepcopy(call.get("allowed_token_ids")),
            "generation_constraint": call.get("generation_constraint"),
            "allowed_action_completions": copy.deepcopy(
                call.get("allowed_action_completions")),
            "memory_decision_alternative_text": call.get(
                "memory_decision_alternative_text"),
            "memory_decision_revision_id": call.get(
                "memory_decision_revision_id"),
            "memory_decision_token_mask": copy.deepcopy(
                call.get("memory_decision_token_mask")),
            "memory_decision_chosen_token_ids": copy.deepcopy(
                call.get("memory_decision_chosen_token_ids")),
            "memory_decision_alternative_token_ids": copy.deepcopy(
                call.get("memory_decision_alternative_token_ids")),
            "memory_decision_eos_token_ids": copy.deepcopy(
                call.get("memory_decision_eos_token_ids")),
            "logprob_kind": call.get("logprob_kind"),
            "raw_logprob_kind": call.get("raw_logprob_kind"),
            "no_repeat_ngram_size": call.get("no_repeat_ngram_size"),
            "sampling_logprobs_status": "pending",
            "action": action,
            "action_text": (call.get("reply") or "").strip(),
            "prompt": str(call.get("prompt") or ""),
            "video_path": str(call.get("video_path") or ""),
            "frame_dir": str(call.get("frame_dir") or ""),
            "frame_manifest": copy.deepcopy(call.get("manifest")),
            "tensor_dump": str(call.get("tensor_dump") or ""),
            "legal": bool(legal),
            "valid": bool(valid),
            "memory_sha256": memory_sha,
            "memory_ops": memory_ops,
            "seed": call.get("seed"),
            "temperature": call.get("temperature"),
            "top_p": call.get("top_p"),
            "top_k": call.get("top_k"),
            "use_cache": call.get("use_cache"),
            "finish_reason": call.get("finish_reason"),
            "calls": int(charged_calls or 0),
            "frames": int(charged_frames or 0),
            "max_seconds": call.get("max_seconds"),
            "repair": bool(call.get("repair")),
            "normalized": bool(call.get("normalized_action")),
            "recovered_action": bool(call.get("recovered_action")),
            "recovered_turn_cut": bool(call.get("recovered_turn_cut")),
            "reader_retry": bool(str(role).endswith("-retry")),
            "reader_retry_cut": bool(call.get("reader_retry_cut")),
            "reader_parse_status": (
                (reader_report or {}).get("parse_status")
                if role.startswith("loopread-") else None),
            "reader_memory_operation": (
                (reader_report or {}).get("memory_operation")
                if role.startswith("loopread-") else None),
            "reader_memory_target": (
                (reader_report or {}).get("memory_target")
                if role.startswith("loopread-") else None),
            "reader_policy_training": bool(
                settings.get("reader_policy_training")),
            "scripted_turn": bool(call.get("scripted_turn")),
            "stage": turn_state.get("stage"),
            "current_question_read": bool(
                turn_state.get("current_question_read")),
            "recommended_actions": turn_state.get("recommended_actions"),
            "recommended_action_verbs": turn_state.get(
                "recommended_action_verbs"),
            "legal_action_verbs": turn_state.get("legal_action_verbs"),
            "other_legal_action_verbs": turn_state.get(
                "other_legal_action_verbs"),
            "menu_class": turn_state.get("selected_menu_class"),
            "preferred_target_match": turn_state.get(
                "selected_preferred_target"),
            "menu_decision_mode": turn_state.get("menu_decision_mode"),
            "menu_exploration_probability": turn_state.get(
                "off_menu_probability"),
        }
        step["sampling_logprobs_status"] = (
            "ok" if _sampling_ok(step) else "invalid")
        steps.append(step)
        if actor == "o" and action.get("verb") in ("review", "use"):
            last_turn_step = step
    if set(turn_requests) - used_turns:
        pairing_errors.append("unpaired_turn")
    first = turn_calls[0] if turn_calls else {}
    budget = _budget(settings)
    reward = _reward(row, steps, budget, _gold_windows(row, settings))
    ended = str(row.get("ended_by") or "")
    fault = str(row.get("fault") or "")
    infrastructure_valid = not (
        row.get("capped") or row.get("truncated")
        or any(str(fault).startswith(mark) for mark in INFRA_FAULTS))
    o_steps = [step for step in steps if step.get("actor") == "o"]
    scripted_o_steps = [step for step in o_steps if step.get("scripted_turn")]
    generated_o_steps = [step for step in o_steps
                         if not step.get("scripted_turn")]
    unmasked = [step for step in generated_o_steps
                if step.get("valid") and not step.get("repair")
                and not step.get("normalized")
                and not step.get("recovered_action")
                and not step.get("recovered_turn_cut")]
    masked = [step for step in o_steps if step not in unmasked]
    explicit_masks = all(
        step.get("scripted_turn") or step.get("repair")
        or step.get("normalized") or step.get("recovered_action")
        or step.get("recovered_turn_cut") for step in masked)
    terminal_step = generated_o_steps[-1] if generated_o_steps else None
    terminal_clean = bool(
        terminal_step is not None and terminal_step.get("valid")
        and not terminal_step.get("recovered_action")
        and not terminal_step.get("recovered_turn_cut"))
    all_o_clean = bool(generated_o_steps) and len(unmasked) == len(
        generated_o_steps)
    all_o_constrained = all(
        step.get("generation_constraint") == "action_menu_prefix_v1"
        and isinstance(step.get("allowed_token_ids"), list)
        and len(step["allowed_token_ids"]) == len(
            step.get("completion_token_ids") or ())
        and all(bool(row) for row in step["allowed_token_ids"])
        for step in generated_o_steps)
    protocol_clean = bool(
        infrastructure_valid and not pairing_errors and turn_calls
        and not row.get("protocol_fallback")
        and not row.get("harness_recovery")
        and ended in TERMINALS and ended != "protocol_abstain"
        and terminal_clean and unmasked)
    unmasked_sampling = all(
        _sampling_ok(step) for step in steps
        if not step.get("scripted_turn")
        and (step.get("actor") != "o" or step in unmasked))
    hybrid_ready = bool(
        protocol_clean and unmasked_sampling and explicit_masks
        and all_o_constrained)
    scripted_anchor_clean = bool(
        settings.get("reader_anchor") is True
        and len(scripted_o_steps) == 1
        and scripted_o_steps[0].get("valid")
        and scripted_o_steps[0].get("legal")
        and (scripted_o_steps[0].get("action") or {}).get("verb") == "read"
        and not scripted_o_steps[0].get("repair")
        and not scripted_o_steps[0].get("normalized")
        and not scripted_o_steps[0].get("recovered_action")
        and not scripted_o_steps[0].get("recovered_turn_cut"))
    training_ready = bool(
        hybrid_ready and all_o_clean and all_o_constrained
        and not scripted_o_steps
        and all(_sampling_ok(step) for step in steps))
    if settings.get("diagnostic_only"):
        hybrid_ready = False
        training_ready = False
    if str(row.get("answer_source") or "") == "budget_submit":
        hybrid_ready = False
        training_ready = False
    reader_steps = [step for step in steps if step.get("actor") == "reader"]
    reader_clean = bool(
        reader_steps and all(
            step.get("valid")
            and step.get("finish_reason") == "stop"
            and not step.get("reader_retry")
            and not step.get("reader_retry_cut")
            and not step.get("repair")
            and not step.get("normalized")
            and not step.get("recovered_action")
            and not step.get("recovered_turn_cut")
            and _sampling_ok(step)
            for step in reader_steps))
    question_number = int(row.get("question_number") or 0)
    revision_rows = [
        value for value in
        ((row.get("atomic_memory") or {}).get("revisions") or {}).values()
        if isinstance(value, dict)
        and int(value.get("question_number") or question_number)
        == question_number]
    anchor_revision_ids = {
        str(value) for value in settings.get("anchor_revision_ids") or ()}
    if anchor_revision_ids:
        revision_rows = [
            value for value in revision_rows
            if str(value.get("revision_id") or "") not in anchor_revision_ids]
    reader_memory_valid = bool(revision_rows) and all(
        bool(value.get("memory_operation_valid", True))
        for value in revision_rows)
    reader_event_records = []
    for value in revision_rows:
        if not isinstance(value, dict):
            continue
        events = []
        for event_id in value.get("events") or ():
            event = ((row.get("atomic_memory") or {}).get("events") or {}).get(
                event_id) or {}
            events.append({
                "event_id": str(event_id),
                "start": event.get("start"),
                "end": event.get("end"),
                "text": str(event.get("text") or ""),
                "span_source": str(event.get("span_source") or ""),
                "supersedes_event_id": str(
                    event.get("supersedes_event_id") or ""),
                "supersession_error": str(
                    event.get("supersession_error") or ""),
            })
        reader_event_records.append({
            "revision_id": str(value.get("revision_id") or ""),
            "question_number": int(value.get("question_number") or question_number),
            "branch_id": str(value.get("branch_id") or ""),
            "memory_operation": str(value.get("memory_operation") or ""),
            "memory_operation_valid": bool(
                value.get("memory_operation_valid", True)),
            "memory_operation_error": str(
                value.get("memory_operation_error") or ""),
            "events": events,
        })
    reader_provenance_valid = _reader_provenance_valid(
        reader_steps, reader_event_records)
    reader_execution_clean = bool(
        settings.get("reader_policy_training")
        and infrastructure_valid and reader_clean and reader_memory_valid
        and reader_provenance_valid)
    reader_reward_clean = bool(
        reader_execution_clean
        and (training_ready
             or (scripted_anchor_clean and hybrid_ready and all_o_clean)))
    reader_training_ready = reader_reward_clean
    if settings.get("diagnostic_only"):
        reader_training_ready = False
    reader_components = _reader_role_reward(
        row, steps, budget, reward, reader_clean, reader_memory_valid,
        reader_event_records)
    optimization = {
        "task": float(reward.get("task") or 0.0),
        "cost": float(reward.get("cost") or 0.0),
    }
    optimization_zeroed = [
        "evidence_gain", "grounding", "memory_reuse", "progress",
        "global_memory", "protocol"]
    optimization["total"] = sum(optimization.values())
    reader_optimization_components = {
        "task": float(reader_components.get("task") or 0.0),
        "cost": float(reader_components.get("cost") or 0.0),
    }
    reader_optimization_reward = sum(reader_optimization_components.values())
    cited_cards = [
        value for value in row.get("memory_uses") or ()
        if value.get("action") == "cite" and value.get("outcome") == "accept"]
    memory_carrier = None
    if cited_cards:
        cited = max(cited_cards, key=lambda value: int(value.get("sequence") or 0))
        memory_carrier = {
            "card_id": str(cited.get("card_id") or ""),
            "scope": str(cited.get("scope") or ""),
            "revision_id": str(cited.get("revision_id") or ""),
            "source_question": int(cited.get("source_question_number") or 0),
        }
    episode = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": "%s:%s:%d" % (
            row.get("run_id") or source_ref or "run", uid, int(rollout_index)),
        "group_id": group_id or uid,
        "rollout_index": int(rollout_index),
        "split": settings.get("data_split"),
        "uid": uid,
        "video_id": row.get("video"),
        "question_number": int(row.get("question_number") or 0),
        "budget": _budget(settings),
        "sampling": {
            "temperature": first.get("temperature"),
            "top_p": first.get("top_p"),
            "top_k": first.get("top_k"),
            "seed": first.get("seed"),
        },
        "policy_versions": _policy_versions(settings),
        "memory_root_ref": source_ref or ("rows.jsonl#%s" % uid),
        "memory_fork_from": None,
        "memory_budget": {
            "entry_tokens": 512, "branch_tokens": 2048,
            "global_tokens": 2048, "events_per_read": 1,
        },
        "memory_uses": copy.deepcopy(row.get("memory_uses") or []),
        "memory_protocol": {
            "global_memory_enabled": bool(settings.get("global_memory")),
            "memory_visible": bool(settings.get("memory_visible", True)),
            "local_memory_visible": bool(
                settings.get("local_memory_visible",
                             settings.get("memory_visible", True))),
            "memory_scope": str(settings.get("memory_scope")
                                or "prior_question"),
            "reader_policy_training": bool(
                settings.get("reader_policy_training")),
            "reader_schema_version": int(
                settings.get("reader_schema_version", 1)),
            "reader_anchor": bool(settings.get("reader_anchor")),
            "menu_layout": str(settings.get("menu_layout") or "grouped"),
            "memory_menu_order": str(
                settings.get("memory_menu_order") or "use_first"),
            "memory_decision_timing": str(
                settings.get("memory_decision_timing") or "pre_read"),
            "card_list_visible": bool(
                settings.get("card_list_visible", False)),
            "card_local_visible": bool(
                settings.get("card_local_visible", True)),
            "card_global_visible": bool(
                settings.get("card_global_visible", True)),
            "cards_per_scope": int(settings.get("cards_per_scope", 4)),
            "force_card_check": bool(
                settings.get("force_card_check", False)),
            "global_source": str(settings.get("global_source") or "timeline"),
            "read_order": str(settings.get("read_order") or "positional"),
            "facts_block": bool(settings.get("facts_block", False)),
            "diagnostic_only": bool(settings.get("diagnostic_only")),
            "budget_answer": bool(settings.get("budget_answer")),
            "fresh_memory_each_question": bool(
                settings.get("fresh_memory_each_question", False)),
            "reader_min_runtime_seconds": float(settings.get(
                "reader_min_runtime_seconds", A.READER_MIN_RUNTIME_SECONDS)),
            "policy_turn_reserve_seconds": float(settings.get(
                "policy_turn_reserve_seconds", A.POLICY_TURN_RESERVE_SECONDS)),
            "final_answer_reserve_seconds": float(settings.get(
                "final_answer_reserve_seconds", A.FINAL_ANSWER_RESERVE_SECONDS)),
            "min_check_to_final_seconds": float(settings.get(
                "min_check_to_final_seconds", A.MIN_CHECK_TO_FINAL_SECONDS)),
            "max_reads_per_question": (
                int(settings["max_reads_per_question"])
                if settings.get("max_reads_per_question") else None),
        },
        "steps": steps,
        "outcome": {
            "terminal": ended,
            "answer": row.get("answer") or "",
            "answered": ended == "agent_submit",
            "answer_source": str(row.get("answer_source") or ""),
            "complete": bool(row.get("complete")),
            "correct": bool(float(row.get("credit") or 0.0) > 0.0),
            "calls": int(row.get("calls") or 0),
            "frames": int(row.get("frames") or 0),
            "fault": fault,
            "protocol_fallback": bool(row.get("protocol_fallback")),
            "harness_recovery": bool(row.get("harness_recovery")),
            "infrastructure_valid": infrastructure_valid,
        },
        "training_only": {
            "gold": row.get("gold"),
            "num_options": int(row.get("num_options") or 0) or None,
            "correct": bool(float(row.get("credit") or 0.0) > 0.0),
            "reward_components": reward,
            "reward": reward["total"],
            "step_returns": reward.get("step_returns") or [],
            "memory_rewards": {
                "memory_reuse": reward.get("memory_reuse", 0.0),
                "grounding": reward.get("grounding", 0.0),
                "global_memory": reward.get("global_memory", 0.0),
            },
            "memory_carrier": memory_carrier,
            "quality": ("clean" if training_ready else
                        "repair_masked" if hybrid_ready else "excluded"),
            "hybrid_ready": hybrid_ready,
            "optimization_reward_components": optimization,
            "optimization_zeroed_components": optimization_zeroed,
            "optimization_reward": round(optimization["total"], 6),
            "training_ready": training_ready,
            "reader_execution_clean": reader_execution_clean,
            "reader_reward_clean": reader_reward_clean,
            "reader_training_ready": reader_training_ready,
            "reader_memory_valid": reader_memory_valid,
            "reader_provenance_valid": reader_provenance_valid,
            "anchor_revision_ids": sorted(anchor_revision_ids),
            "reader_reward_components": reader_components,
            "reader_reward": round(sum(reader_components.values()), 6),
            "reader_optimization_reward_components": (
                reader_optimization_components),
            "reader_optimization_zeroed_components": [
                "schema", "operation", "provenance", "grounding",
                "memory_reuse", "citation", "invalid_operation"],
            "reader_optimization_reward": round(
                reader_optimization_reward, 6),
            "reader_event_records": reader_event_records,
            "pairing_errors": sorted(set(pairing_errors)),
            "missing": ([] if not pairing_errors else ["turn_call_pairing"])
                       + [name for name in
                          ("prompt_token_ids", "completion_token_ids",
                           "completion_logprobs")
                          if any(not _sampling_ok(step) for step in steps
                                 if not step.get("scripted_turn")
                                 and not (step.get("actor") == "o"
                                          and (step.get("repair")
                                               or step.get("recovered_action")
                                               or step.get("recovered_turn_cut"))))],
        },
    }
    return episode


def validate_episode(episode):
    problems = []
    required = ("schema_version", "episode_id", "group_id", "rollout_index",
                "uid", "budget", "sampling", "steps", "outcome",
                "training_only")
    for name in required:
        if name not in episode:
            problems.append(name)
    if episode.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version")
    if not isinstance(episode.get("steps"), list):
        problems.append("steps")
        return problems
    for index, step in enumerate(episode["steps"]):
        for name in ("t", "actor", "action", "legal", "valid", "seed",
                     "temperature", "finish_reason", "prompt_token_ids",
                     "completion_token_ids", "completion_logprobs"):
            if name not in step:
                problems.append("steps[%d].%s" % (index, name))
    outcome = episode.get("outcome") or {}
    for name in ("terminal", "answer", "answered", "complete", "correct",
                 "calls", "frames", "infrastructure_valid"):
        if name not in outcome:
            problems.append("outcome.%s" % name)
    training = episode.get("training_only") or {}
    for name in ("gold", "correct", "reward", "reward_components",
                 "training_ready"):
        if name not in training:
            problems.append("training_only.%s" % name)
    if (episode.get("memory_protocol") or {}).get("reader_policy_training"):
        for name in ("reader_execution_clean", "reader_reward_clean",
                     "reader_memory_valid", "reader_provenance_valid",
                     "reader_reward_components", "reader_optimization_reward"):
            if name not in training:
                problems.append("training_only.%s" % name)
    return sorted(set(problems))


def supply_report(episodes, expected_groups=20, group_size=3,
                  required_mixed_groups=None):
    groups = {}
    for episode in episodes:
        groups.setdefault(str(episode["group_id"]), []).append(episode)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row["rollout_index"]))
    o_steps = [step for episode in episodes for step in episode["steps"]
               if step.get("actor") == "o"]
    legal = sum(1 for step in o_steps if step.get("valid"))
    terminal = sum(1 for episode in episodes
                   if episode["outcome"]["terminal"] in TERMINALS)
    sampling_complete = all(
        _sampling_ok(step)
        for episode in episodes for step in episode["steps"])
    infrastructure = all(
        episode["outcome"].get("infrastructure_valid") for episode in episodes)
    adapter_roles = True
    for episode in episodes:
        versions = episode["policy_versions"]
        expected = {
            "o": versions["o"] != versions["base_model"],
            "reader": versions["reader"] != versions["base_model"],
            "reviewer": versions["reviewer"] != versions["base_model"],
        }
        for step in episode["steps"]:
            actor = str(step.get("actor") or "")
            wanted = ("o" if expected.get(actor) and actor in {"o", "reviewer"}
                      else actor if expected.get(actor) else "base")
            if step.get("adapter") != wanted:
                adapter_roles = False
    turn_pairing = all(
        "turn_call_pairing" not in
        (episode.get("training_only", {}).get("missing") or [])
        for episode in episodes)
    reader_steps = [step for episode in episodes for step in episode["steps"]
                    if step.get("actor") == "reader"]
    frame_accounting = bool(reader_steps) and all(
        int(step.get("frames") or 0) > 0 for step in reader_steps)
    mixed = 0
    for rows in groups.values():
        outcomes = {bool(row["outcome"]["correct"]) for row in rows
                    if row["outcome"]["terminal"] in TERMINALS
                    and row["outcome"]["terminal"] != "protocol_abstain"
                    and row["outcome"].get("infrastructure_valid")
                    and not row["outcome"].get("protocol_fallback")
                    and not row["outcome"].get("harness_recovery")
                    and row["training_only"].get("training_ready")}
        if outcomes == {True, False}:
            mixed += 1
    correct = sum(1 for episode in episodes if episode["outcome"]["correct"])
    legal_rate = legal / float(len(o_steps) or 1)
    terminal_rate = terminal / float(len(episodes) or 1)
    mixed_rate = mixed / float(len(groups) or 1)
    correct_rate = correct / float(len(episodes) or 1)
    required_mixed = min(
        int(required_mixed_groups)
        if required_mixed_groups is not None else MIXED_GROUP_FLOOR,
        len(groups))
    checks = {
        "sample_complete": len(groups) == int(expected_groups)
        and all(len(rows) == int(group_size)
                and sorted(int(row["rollout_index"]) for row in rows)
                == list(range(int(group_size)))
                for rows in groups.values()),
        "legal_actions": legal_rate >= 0.90,
        "terminals": terminal_rate >= 0.80,
        "task_variance": mixed >= required_mixed,
        "correct_mix": 0.05 <= correct_rate <= 0.95,
        "sampling_complete": sampling_complete,
        "infrastructure": infrastructure,
        "adapter_roles": adapter_roles,
        "turn_pairing": turn_pairing,
        "frame_accounting": frame_accounting,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "groups": len(groups),
        "episodes": len(episodes),
        "group_size": int(group_size),
        "legal_action_rate": round(legal_rate, 6),
        "terminal_rate": round(terminal_rate, 6),
        "mixed_task_group_rate": round(mixed_rate, 6),
        "required_mixed_groups": required_mixed,
        "correct_rate": round(correct_rate, 6),
        "counts": {
            "o_steps": len(o_steps), "legal_o_steps": legal,
            "terminals": terminal, "mixed_task_groups": mixed,
            "correct_episodes": correct,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", action="append", required=True)
    parser.add_argument("--expected-groups", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--required-mixed-groups", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args(argv)
    episodes = []
    for path in args.episodes:
        episodes.extend(_jsonl(path))
    problems = []
    for episode in episodes:
        for problem in validate_episode(episode):
            problems.append("%s:%s" % (episode.get("episode_id"), problem))
    report = supply_report(
        episodes, args.expected_groups, args.group_size,
        required_mixed_groups=args.required_mixed_groups)
    report["schema_problems"] = sorted(set(problems))
    if problems:
        report["passes"] = False
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_pass and not report["passes"]:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
