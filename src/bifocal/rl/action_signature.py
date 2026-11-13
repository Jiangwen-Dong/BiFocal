"""Canonical O action signatures for C1 mixed-action auditing."""

from __future__ import annotations

import collections


def _memory_op(step, operation):
    rows = [
        op for op in step.get("memory_ops") or ()
        if str(op.get("op") or "") == operation]
    return rows[0] if len(rows) == 1 else None


def canonical_o_action_signature(step):
    """Return a semantic action key that ignores rendered handles and quotes."""
    if str(step.get("actor") or "") != "o":
        raise ValueError("canonical action requires an O step")
    action = step.get("action") or {}
    verb = str(action.get("verb") or "").lower()
    if verb in ("open", "read"):
        branch = action.get("branch")
        if branch is None:
            raise ValueError("%s step has no branch" % verb)
        return (verb, int(branch))
    if verb in ("review", "use", "defer"):
        revision = action.get("revision")
        if revision is None:
            raise ValueError("%s step has no revision" % verb)
        return (verb, int(revision))
    if verb == "check":
        operation = _memory_op(step, "check")
        if operation is None:
            raise ValueError("CHECK step has no unique memory operation")
        return ("check", str(operation.get("scope") or ""),
                str(operation.get("target_revision_id") or ""))
    if verb == "submit":
        operation = _memory_op(step, "cite")
        if operation is not None:
            return ("submit", str(action.get("option") or ""),
                    str(operation.get("scope") or ""),
                    str(operation.get("target_revision_id") or ""))
        return ("submit", str(action.get("option") or ""), "revision",
                str(action.get("revision") or ""))
    if verb == "evolve_global":
        return (
            "evolve_global",
            str(action.get("global_operation") or ""),
            str(action.get("revision") or ""),
            str(action.get("global_target") or ""),
        )
    if verb == "abstain":
        return ("abstain",)
    raise ValueError("unknown O action verb: %s" % (verb or "missing"))


def group_has_mixed_actions(rows):
    """Whether rollouts diverge semantically at any shared question."""
    traces = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for row in rows or ():
        rollout = int(row.get("rollout_index") or 0)
        for step in row.get("steps") or ():
            if str(step.get("actor") or "") != "o":
                continue
            question = int(step.get("question_number") or 0)
            traces[question][rollout].append(
                canonical_o_action_signature(step))
    for rollout_traces in traces.values():
        sequences = {
            tuple(trace) for trace in rollout_traces.values()}
        if len(sequences) > 1:
            return True
    return False
