#!/usr/bin/env python3
"""Validate and credit one USE/DEFER decision token per rollout."""

from __future__ import annotations

import re


MEMORY_VERBS = ("use", "defer")
ACTION_LINE = re.compile(r"^ACTION:\s+(USE|DEFER)\s+(R\d+)$", re.IGNORECASE)
DECISION_LINE = re.compile(r"^\s*ACTION:\s+(USE|DEFER)\s+(R\d+)\s*$",
                           re.IGNORECASE)
INVALID_FLAGS = ("repair", "normalized", "recovered_action",
                 "recovered_turn_cut")


def _verb(value):
    if not isinstance(value, dict):
        return ""
    return str(value.get("verb") or value.get("action") or "").lower()


def parse_action_text(text):
    """Return the canonical USE/DEFER action or raise."""
    match = ACTION_LINE.fullmatch(str(text or "").strip())
    if not match:
        raise ValueError("memory decision is not one canonical USE/DEFER line")
    return {
        "verb": match.group(1).lower(),
        "revision_id": match.group(2).upper(),
        "text": "ACTION: %s %s" % (
            match.group(1).upper(), match.group(2).upper()),
    }


def flip_action_text(text):
    parsed = parse_action_text(text)
    other = "DEFER" if parsed["verb"] == "use" else "USE"
    return "ACTION: %s %s" % (other, parsed["revision_id"])


def _recommended(step):
    rows = []
    for value in step.get("recommended_actions") or ():
        verb = _verb(value)
        if verb in MEMORY_VERBS:
            rows.append(value)
    if len(rows) != 2 or {_verb(value) for value in rows} != set(MEMORY_VERBS):
        raise ValueError("memory turn does not offer exactly USE and DEFER")
    revisions = {
        (("R" + str(value.get("revision"))) if str(
            value.get("revision") or "").isdigit() else str(
                value.get("revision") or "").upper())
        for value in rows}
    if len(revisions) != 1 or not next(iter(revisions)):
        raise ValueError("USE and DEFER do not target the same revision")
    return {
        _verb(value): value for value in rows
    }, next(iter(revisions))


def decision_token_mask(step):
    """Validate a stored mask against the recorded USE and DEFER token ids."""
    mask = step.get("memory_decision_token_mask")
    chosen = step.get("memory_decision_chosen_token_ids")
    alternative = step.get("memory_decision_alternative_token_ids")
    eos_ids = step.get("memory_decision_eos_token_ids") or []
    tokens = list(step.get("completion_token_ids") or ())
    if (not isinstance(mask, list) or len(mask) != len(tokens)
            or any(not isinstance(value, bool) for value in mask)):
        raise ValueError("memory decision mask does not match completion tokens")
    if (not isinstance(chosen, list) or not chosen
            or not isinstance(alternative, list) or not alternative):
        raise ValueError("memory decision token ids are missing")
    if tokens[:len(chosen)] != chosen:
        raise ValueError("memory decision chosen token ids differ from completion")
    stop = {int(value) for value in eos_ids}
    if any(int(value) not in stop for value in tokens[len(chosen):]):
        raise ValueError("memory decision completion has an unexplained suffix")
    limit = min(len(chosen), len(alternative))
    prefix = 0
    while prefix < limit and chosen[prefix] == alternative[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < limit - prefix
           and chosen[-1 - suffix] == alternative[-1 - suffix]):
        suffix += 1
    end = len(chosen) - suffix
    expected = [False] * len(tokens)
    for index in range(prefix, end):
        expected[index] = True
    if mask != expected:
        raise ValueError("memory decision mask differs from the USE/DEFER token span")
    return mask


def annotate_decision_tokens(text, completion_token_ids, tokenizer, eos_ids=()):
    """Record the exact completion-token span where USE differs from DEFER."""
    canonical = parse_action_text(text)
    alternative = flip_action_text(canonical["text"])
    try:
        chosen_ids = list(tokenizer.encode(
            canonical["text"], add_special_tokens=False))
        alternative_ids = list(tokenizer.encode(
            alternative, add_special_tokens=False))
    except (TypeError, ValueError) as error:
        raise ValueError("cannot tokenize memory decision: %s" % error) from error
    completion = list(completion_token_ids)
    stop = {int(value) for value in eos_ids}
    content = list(completion)
    while content and int(content[-1]) in stop:
        content.pop()
    if content != chosen_ids:
        raise ValueError("canonical memory action does not match completion tokens")
    limit = min(len(chosen_ids), len(alternative_ids))
    prefix = 0
    while prefix < limit and chosen_ids[prefix] == alternative_ids[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < limit - prefix
           and chosen_ids[-1 - suffix] == alternative_ids[-1 - suffix]):
        suffix += 1
    end = len(chosen_ids) - suffix
    if prefix >= end:
        raise ValueError("USE/DEFER completions have no differing token")
    mask = [False] * len(completion)
    for index in range(prefix, end):
        mask[index] = True
    return {
        "memory_decision_alternative_text": alternative,
        "memory_decision_revision_id": canonical["revision_id"],
        "memory_decision_token_mask": mask,
        "memory_decision_chosen_token_ids": chosen_ids,
        "memory_decision_alternative_token_ids": alternative_ids,
        "memory_decision_eos_token_ids": sorted(stop),
    }


def _is_invalid_step(step):
    return any(bool(step.get(name)) for name in INVALID_FLAGS)


def decision_step(episode):
    """Locate and validate the one clean memory decision in an episode."""
    generated = [
        step for step in episode.get("steps") or ()
        if step.get("actor") == "o" and not step.get("scripted_turn")]
    candidates = []
    for index, step in enumerate(generated):
        verbs = [_verb(value) for value in
                 (step.get("recommended_actions") or ())]
        if not any(verb in MEMORY_VERBS for verb in verbs):
            continue
        _recommended(step)
        candidates.append((index, step))
    if len(candidates) != 1:
        raise ValueError("episode has %d memory decision turns, need one"
                         % len(candidates))
    index, step = candidates[0]
    if index != 0:
        raise ValueError("memory decision is not the first O turn")
    if any(_is_invalid_step(value) for value in generated[:index + 1]):
        raise ValueError("memory decision follows a repaired or recovered step")
    if not step.get("legal") or not step.get("valid"):
        raise ValueError("memory decision step is not legal and valid")
    if step.get("generation_constraint") != "action_menu_prefix_v1":
        raise ValueError("memory decision lacks the action constraint")
    if step.get("menu_decision_mode") != "guided":
        raise ValueError("memory decision was not made in guided menu mode")
    exploration = float(step.get("menu_exploration_probability") or 0.0)
    if exploration != 0.0:
        raise ValueError("memory decision had nonzero menu exploration")
    recommendations, revision_id = _recommended(step)
    action = step.get("action") or {}
    verb = _verb(action)
    if verb not in MEMORY_VERBS:
        raise ValueError("memory turn chose %s, not USE or DEFER" % verb)
    action_revision = str(action.get("revision") or "")
    action_revision = ("R" + action_revision) if action_revision.isdigit() \
        else action_revision.upper()
    if action_revision != revision_id:
        raise ValueError("chosen memory revision differs from the menu")
    parsed = parse_action_text(step.get("action_text"))
    if parsed["verb"] != verb or parsed["revision_id"] != revision_id:
        raise ValueError("memory action text differs from the parsed action")
    if str(parsed["text"]) != str(step.get("action_text") or "").strip():
        raise ValueError("memory action text is not canonical")
    mask = decision_token_mask(step)
    return {
        "step": step,
        "verb": verb,
        "revision_id": revision_id,
        "recommended": recommendations,
        "mask": mask,
        "decision_tokens": sum(mask),
        "completion_tokens": len(mask),
    }

