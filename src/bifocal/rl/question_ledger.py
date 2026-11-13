#!/usr/bin/env python3
"""One row per question, per rollout, per run, written while the run is live.

Files written under the run output directory:

* question-ledger.jsonl   identity, outcome, cost, allocation summary, memory counts
* reader-allocation.jsonl one row per Reader per question
* memory-items.jsonl      one row per memory item shown to O per question
* eval-only.jsonl         gold answer, gold windows and evidence overlap
* ledger-errors.jsonl     any row the ledger failed to write

The gold answer and every evidence-window number stay in eval-only.jsonl.
Aggregates are never computed here; every table is derived from these rows.
"""

import hashlib
import json
import os
import re

from . import episode as E

SCHEMA = "question-ledger-v1"
STAGES = ("exposed", "mentioned", "opened", "checked", "accepted",
          "rejected", "cited")
NUMBER = r"\d+(?:\.\d+)?"
CARD_LINE = re.compile(
    r"^(M\d+(?:\.\d+)?) \[(local|global)\] Q(\d+) "
    r"(?:(%s)-(%s)s|time unknown) (\S+)" % (NUMBER, NUMBER))
ENTRY_LINE = re.compile(r"^\s+(?:M\d+\.\d+|-) (%s)-(%s)s \S+ \| " % (
    NUMBER, NUMBER))
HANDLE = re.compile(r"\bM\d+(?:\.\d+)?\b")
LETTER = re.compile(r"^\(?([A-H])\)?(?:[\s.:)]|$)")
FILES = {"question": "question-ledger.jsonl",
         "reader": "reader-allocation.jsonl",
         "memory": "memory-items.jsonl",
         "eval": "eval-only.jsonl",
         "error": "ledger-errors.jsonl"}


def _letter(value):
    match = LETTER.match(str(value or "").strip().upper())
    return match.group(1) if match else ""


def _role_class(role):
    role = str(role or "")
    if role == "turn":
        return "o"
    if role.startswith("loopread"):
        return "reader"
    if role.startswith("review"):
        return "review"
    if role.startswith("check-card"):
        return "check"
    return "other"


def _parent(handle):
    return str(handle).split(".")[0]


def prompt_revision(paths, schema):
    digest = hashlib.sha256(str(schema).encode("utf-8"))
    for path in sorted(paths):
        with open(path, "rb") as handle:
            digest.update(hashlib.sha256(handle.read()).digest())
    return digest.hexdigest()[:16]


def cost_block(episode, calls, budget, refusals=None, wall_seconds=None):
    """Counts and tokens for one question. `calls` holds only this run's calls."""
    by_class = {"o": 0, "reader": 0, "review": 0, "check": 0, "other": 0}
    tokens_in = tokens_out = frames = retries = 0
    call_seconds = gpu_seconds = 0.0
    length_cut = recovered = 0
    for call in calls:
        by_class[_role_class(call.get("role"))] += 1
        tokens_in += int(call.get("prompt_tokens") or 0)
        tokens_out += int(call.get("completion_tokens") or 0)
        frames += int(call.get("frames") or 0)
        seconds = float(call.get("seconds") or 0.0)
        call_seconds += seconds
        gpu_seconds += seconds * max(1, len(call.get("card_pair") or [0]))
        retries += 1 if int(call.get("retry_number") or 0) else 0
        length_cut += 1 if call.get("finish_reason") == "length" else 0
        recovered += 1 if call.get("recovered_action") else 0
    steps = episode.get("steps") or []
    o_steps = [step for step in steps if step.get("actor") == "o"]
    if refusals is None:
        invalid = sum(1 for step in o_steps if step.get("legal") is False)
        invalid_checks = 0
    else:
        invalid = sum(1 for item in refusals
                      if item.get("kind") not in ("check_invalid",))
        invalid_checks = sum(1 for item in refusals
                             if item.get("kind") == "check_invalid")
    outcome = episode.get("outcome") or {}
    recovered += 1 if outcome.get("harness_recovery") else 0
    total_calls = len(calls)
    return {
        "calls_total": total_calls, "calls_o": by_class["o"],
        "calls_reader": by_class["reader"],
        "calls_review": by_class["review"], "calls_check": by_class["check"],
        "calls_other": by_class["other"],
        "reads": sum(1 for step in o_steps
                     if (step.get("action") or {}).get("verb") == "read"),
        "reviews": sum(1 for step in o_steps
                       if (step.get("action") or {}).get("verb") == "review"),
        "frames_selected": frames, "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_seconds": (None if wall_seconds is None
                         else round(float(wall_seconds), 3)),
        "call_seconds": round(call_seconds, 3),
        "gpu_seconds_occupancy": round(gpu_seconds, 3),
        "gpu_seconds_basis": "call_seconds x cards in lane; shared lane "
                             "overlaps concurrent calls",
        "turns": len(o_steps), "invalid_actions": invalid,
        "invalid_checks": invalid_checks, "retry_calls": retries,
        "recovered_calls": recovered, "length_cut_calls": length_cut,
        "cap_hit": {
            "calls": total_calls >= int(budget.get("calls") or 10 ** 9),
            "frames": frames >= int(budget.get("frames") or 10 ** 9),
            "time": (wall_seconds is not None and float(wall_seconds)
                     >= 60.0 * float(budget.get("minutes") or 10 ** 9)),
            "tokens": length_cut > 0},
    }


FACT_LINE = re.compile(r"^B\d+ \S+ \| ", re.M)


def facts_shown(prompt):
    text = str(prompt or "")
    start = text.find("Facts read so far in this question:")
    if start < 0:
        return 0
    return len(FACT_LINE.findall(text[start:].split("\n\n")[0]))


RECOMMENDED_READ = re.compile(r"Recommended now:\nACTION: READ B(\d+)")
LISTED_SECTION = re.compile(r"(?:Recommended|Available) now:\n(.*?)(?:\n\n|$)",
                            re.S)
LISTED_READ = re.compile(r"^ACTION: READ B(\d+)", re.M)


def first_listed_read(prompt):
    section = LISTED_SECTION.search(str(prompt or ""))
    match = LISTED_READ.search(section.group(1)) if section else None
    return int(match.group(1)) if match else None


def read_choices(episode):
    """Per READ by O: the branch the menu recommended, the one chosen."""
    rows = []
    for index, step in enumerate(episode.get("steps") or []):
        action = step.get("action") or {}
        if step.get("actor") != "o" or action.get("verb") != "read":
            continue
        match = RECOMMENDED_READ.search(str(step.get("prompt") or ""))
        recommended = int(match.group(1)) if match else None
        chosen = action.get("branch")
        listed = first_listed_read(step.get("prompt"))
        rows.append({"step": index, "recommended": recommended,
                     "first_listed_read": listed,
                     "followed_first_listed": (
                         listed == chosen if listed is not None else None),
                     "chosen": chosen,
                     "followed": (recommended == chosen
                                  if recommended is not None else None),
                     "facts_shown": facts_shown(step.get("prompt"))})
    return rows


def reader_rows(tree, question_number, calls, chosen):
    """One row per Reader (branch) that read in this question."""
    revisions = [row for row in tree.get("revisions", {}).values()
                 if int(row.get("question_number") or 0) == question_number
                 and row.get("reader")]
    earlier = {tuple(row.get("window") or ())
               for row in tree.get("revisions", {}).values()
               if int(row.get("question_number") or 0) < question_number
               and row.get("window")}
    per_call = {}
    for call in calls:
        match = re.fullmatch(r"loopread-(\d+)(?:-retry)?", str(call.get("role")))
        if match:
            slot = per_call.setdefault(int(match.group(1)),
                                       {"calls": 0, "frames": 0})
            slot["calls"] += 1
            slot["frames"] += int(call.get("frames") or 0)
    grouped = {}
    for row in revisions:
        grouped.setdefault(str(row.get("reader")), []).append(row)
    rows = []
    for reader, items in sorted(grouped.items()):
        number = int(re.sub(r"\D", "", reader) or 0)
        windows = [tuple(item.get("window") or ()) for item in items]
        answers = [_letter(item.get("answer")) for item in items]
        used = per_call.get(number, {"calls": 0, "frames": 0})
        rows.append({
            "reader": reader, "branches": sorted({i.get("branch_id")
                                                  for i in items}),
            "calls": used["calls"], "frames": used["frames"],
            "reads": len(items), "chunk_spans": [list(w) for w in windows],
            "reread_in_question": len(windows) - len(set(windows)),
            "reread_from_earlier_question": sum(
                1 for w in set(windows) if w in earlier),
            "read_ok": sum(1 for i in items
                           if i.get("read_status") == "success"),
            "branch_answers": answers,
            "agrees_with_final": [bool(a and a == chosen) for a in answers],
        })
    return rows


def _exposed_cards(prompt):
    text = str(prompt or "")
    start = text.find("Prior memory cards:")
    cards = []
    if start < 0:
        return cards
    for line in text[start:].splitlines()[1:]:
        if not line.strip():
            break
        match = CARD_LINE.match(line)
        if match:
            handle, scope, origin, first, last, state = match.groups()
            span = (float(first), float(last)) if first else ()
            cards.append((handle, scope, int(origin), span, state, []))
            continue
        entry = ENTRY_LINE.match(line)
        if entry and cards:
            cards[-1][5].append([float(entry.group(1)), float(entry.group(2))])
    return cards


def _new_item(handle, scope, origin, span, question_number, protocol):
    visible = protocol.get("card_%s_visible" % scope)
    return {"handle": handle, "scope": scope, "origin_question": origin,
            "origin": ("this_question" if origin == question_number
                       else "earlier_question"),
            "span": list(span), "stages": {name: False for name in STAGES},
            "actions": [], "turns_exposed": 0, "states_seen": [],
            "entries_checked": [], "outcomes": [], "cited_quote_valid": None,
            "entry_spans": [],
            "stale": False,
            "out_of_scope": visible is False}


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().strip("\"'").lower()


def _record_check(item, use, index):
    outcome = str(use.get("outcome") or "")
    item["entries_checked"].append(str(use.get("use_id") or ""))
    item.setdefault("revisions_checked", []).append(
        str(use.get("revision_id") or ""))
    item["outcomes"].append(outcome)
    if outcome in ("accept", "reject", "unsure"):
        _mark(item, "checked", "harness", index, outcome=outcome)
    if outcome == "accept":
        _mark(item, "accepted", "harness", index)
        item.setdefault("accepted_quotes", []).append(use.get("quote"))
    elif outcome == "reject":
        _mark(item, "rejected", "harness", index)


def _record_cite(item, action, step, index, uses):
    _mark(item, "cited", "o", index)
    quote = _norm(action.get("quote"))
    accepted = [_norm(q) for q in item.get("accepted_quotes") or ()]
    item["cited_quote_valid"] = bool(quote and quote in accepted)
    revisions = item.get("revisions_checked") or []
    item["cited_revision"] = (
        "R%s" % action["revision"] if action.get("revision")
        else (revisions[-1] if revisions else None))


def _mark(item, stage, actor, step, **extra):
    item["stages"][stage] = True
    item["actions"].append(dict({"stage": stage, "actor": actor,
                                 "step": step}, **extra))


def memory_item_rows(episode, tree, calls, question_number, protocol):
    """Lifecycle of every memory item O saw, from prompts and actions."""
    items, latest = {}, {}
    uses = tree.get("memory_uses", {})
    replies = " ".join(str(c.get("reply") or "") for c in calls
                       if c.get("role") == "turn")
    mentioned = {_parent(h) for h in HANDLE.findall(replies)}
    mentioned_full = set(HANDLE.findall(replies))
    for index, step in enumerate(episode.get("steps") or []):
        if step.get("actor") != "o":
            continue
        for handle, scope, origin, span, state, entries in _exposed_cards(
                step.get("prompt")):
            key = (handle, scope, origin, span)
            item = items.get(key)
            if item is None:
                item = items[key] = _new_item(
                    handle, scope, origin, span, question_number, protocol)
                _mark(item, "exposed", "harness", index)
            item["turns_exposed"] += 1
            item["states_seen"].append(state)
            for entry_span in entries:
                if entry_span not in item["entry_spans"]:
                    item["entry_spans"].append(entry_span)
            item["stale"] = item["stale"] or "stale" in state
            latest[handle] = key
        action = step.get("action") or {}
        target = str(action.get("memory_handle") or "")
        key = latest.get(_parent(target)) if target else None
        if key is None and target:
            key = (target, "unknown", -1, ())
            items.setdefault(key, dict(_new_item(
                target, "unknown", -1, (), question_number, protocol),
                out_of_scope=True))
        if action.get("verb") == "check" and key:
            _mark(items[key], "opened", "o", index, handle=target)
            for op in step.get("memory_ops") or ():
                use = uses.get(str(op.get("entry_id")), {})
                _record_check(items[key], use, index)
        if action.get("verb") == "submit" and key:
            _record_cite(items[key], action, step, index, uses)
    for key, item in items.items():
        if item["handle"] in mentioned or item["handle"] in mentioned_full:
            _mark(item, "mentioned", "o", -1)
    return list(items.values())


def memory_counts(tree, question_number, items):
    """Global writes and entry counts for one question."""
    counts = {"add": 0, "supersede": 0, "keep": 0, "refused": 0,
              "duplicate_write": 0}
    for op in tree.get("global_operations", {}).values():
        if int(op.get("question_number") or 0) != question_number:
            continue
        name = str(op.get("operation") or "")
        if op.get("accepted"):
            counts[name] = counts.get(name, 0) + 1
        else:
            counts["refused"] += 1
            reason = str(op.get("reason") or "")
            counts["duplicate_write"] += int(
                "duplicate" in reason or " matches " in reason)
    local = sum(1 for row in tree.get("revisions", {}).values()
                if int(row.get("question_number") or 0) == question_number
                and row.get("reader"))
    shown = set()
    for i in items:
        if i["scope"] != "global":
            continue
        if i["entry_spans"]:
            shown.update(("entry", i["origin_question"], tuple(e))
                         for e in i["entry_spans"])
        else:
            shown.add(("card", i["handle"], i["origin_question"],
                       tuple(i["span"])))
    counts.update({"local_entries_written": local,
                   "global_entries_displayed": len(shown),
                   "active_global_claims": sum(
                       1 for c in tree.get("global_claims", {}).values()
                       if c.get("state") == "active")})
    return counts


def arm_label(protocol):
    if not protocol.get("card_list_visible"):
        return "off"
    local = bool(protocol.get("card_local_visible"))
    shown = bool(protocol.get("card_global_visible"))
    return "both" if local and shown else "local" if local else \
        "global" if shown else "off"


def _check_overlaps(tree, items, windows):
    """Spans of the stored frames each memory CHECK looked at."""
    rows = []
    for item in items or ():
        for revision_id in item.get("revisions_checked") or ():
            span = ((tree.get("revisions") or {}).get(revision_id)
                    or {}).get("window") or ()
            if len(span) == 2:
                rows.append({
                    "handle": item["handle"], "revision": revision_id,
                    "span": list(span),
                    "overlaps_evidence": E._covered_seconds(
                        [span], windows) >= 1.0})
    return rows


def eval_row(row_id, episode, tree, question_number, readers, chosen, gold,
             windows, items=None):
    """Gold answer and evidence overlap. Never merged into the main ledger."""
    windows = E._normalize_gold_windows(windows)
    per_reader = []
    for reader in readers:
        spans = reader["chunk_spans"]
        hits = [E._covered_seconds([span], windows) >= 1.0 for span in spans]
        per_reader.append({"reader": reader["reader"],
                           "reads_overlapping_evidence": sum(hits),
                           "reads": len(spans)})
    reads = sum(r["reads"] for r in per_reader)
    hit = sum(r["reads_overlapping_evidence"] for r in per_reader)
    checks = _check_overlaps(tree, items, windows)
    return {"schema": SCHEMA, "row_id": row_id, "gold": gold,
            "checks": checks,
            "fraction_checks_overlapping_evidence": (
                round(sum(c["overlaps_evidence"] for c in checks)
                      / len(checks), 4) if checks else None),
            "chosen_option": chosen, "correct": bool(gold and gold == chosen),
            "gold_windows": [list(w) for w in windows],
            "fraction_reads_overlapping_evidence": (
                round(hit / reads, 4) if reads else None),
            "per_reader": per_reader}


def build_bundle(ctx, key, episode, calls, tree, refusals=None,
                 wall_seconds=None, forced=False, gold=None, windows=None):
    """All ledger rows for one question and rollout, before disposition."""
    number = int(key["question_number"])
    outcome = episode.get("outcome") or {}
    chosen = _letter(outcome.get("answer"))
    protocol = episode.get("memory_protocol") or ctx.get("protocol") or {}
    row_id = "%s:%s:q%d:r%d" % (ctx["run_id"], key["video"], number,
                                int(key["rollout_index"]))
    readers = reader_rows(tree, number, calls, chosen)
    items = memory_item_rows(episode, tree, calls, number, protocol)
    kind = terminal_type(outcome, forced)
    clean = bool(outcome.get("complete") and outcome.get("infrastructure_valid")
                 and not outcome.get("protocol_fallback")
                 and not outcome.get("harness_recovery") and not forced)
    reasons = [name for name, bad in (
        ("infrastructure_failure", kind == "infrastructure_failure"),
        ("forced_failure", forced),
        ("protocol_fallback", bool(outcome.get("protocol_fallback"))),
        ("harness_recovery", bool(outcome.get("harness_recovery"))),
        ("incomplete", not outcome.get("complete"))) if bad]
    question = {
        "schema": SCHEMA, "row_id": row_id,
        **{name: ctx.get(name) for name in (
            "run_id", "code_commit", "prompt_revision", "adapter_id",
            "adapter_sha256", "round", "benchmark", "split", "model_sha256",
            "global_source", "read_order", "facts_block")},
        "arm": arm_label(protocol), "protocol": protocol,
        "video_id": key["video"], "question_id": key.get("uid"),
        "question_number": number, "question_position": key.get("position"),
        "rollout_index": int(key["rollout_index"]), "seed": key.get("seed"),
        "budget": ctx.get("budget"), "pair_key": "%s:%s:%s" % (
            ctx.get("benchmark"), key["video"], key.get("uid")),
        "video_cluster": key["video"],
        "terminal_type": kind, "chosen_option": chosen or None,
        "answered": bool(outcome.get("answered")), "clean": clean,
        "episode_clean": clean,
        "valid_terminal": bool(outcome.get("complete") and not forced),
        "exclusion_reasons": reasons,
        "cost": cost_block(episode, calls, ctx.get("budget") or {},
                           refusals, wall_seconds),
        "allocation": {"readers": len(readers),
                       "chunks_read": sum(r["reads"] for r in readers),
                       "rereads": sum(r["reread_in_question"]
                                      + r["reread_from_earlier_question"]
                                      for r in readers)},
        "memory": memory_counts(tree, number, items),
        "read_choices": read_choices(episode),
        "refusals": [{"kind": item.get("kind"), "why": item.get("why"),
                      "first_why": item.get("first_why")}
                     for item in refusals or ()]}
    return {"question": question,
            "readers": [dict(r, row_id=row_id) for r in readers],
            "memory": [dict(i, row_id=row_id) for i in items],
            "eval": eval_row(row_id, episode, tree, number, readers, chosen,
                             gold, windows, items)}


class Ledger:
    def __init__(self, out_dir, ctx):
        self.dir, self.ctx = out_dir, ctx
        self.pending, self.errors, self.rows = {}, 0, 0
        os.makedirs(out_dir, exist_ok=True)

    def _write(self, kind, row):
        try:
            with open(os.path.join(self.dir, FILES[kind]), "a",
                      encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        except (OSError, TypeError, ValueError) as error:
            self.errors += 1
            try:
                with open(os.path.join(self.dir, FILES["error"]), "a",
                          encoding="utf-8") as handle:
                    handle.write(json.dumps(
                        {"kind": kind, "error": str(error)}) + "\n")
            except OSError:
                pass

    def stage(self, key, bundle):
        self.pending[(key["video"], int(key["question_number"]),
                      int(key["rollout_index"]))] = bundle

    def failure(self, key, error, kind="infrastructure_failure"):
        row_id = "%s:%s:q%d:r%d" % (
            self.ctx["run_id"], key["video"], int(key["question_number"]),
            int(key["rollout_index"]))
        question = {"schema": SCHEMA, "row_id": row_id,
                    "run_id": self.ctx["run_id"],
                    "video_id": key["video"], "question_id": key.get("uid"),
                    "question_number": int(key["question_number"]),
                    "rollout_index": int(key["rollout_index"]),
                    "terminal_type": kind,
                    "clean": False, "episode_clean": False,
                    "valid_terminal": False,
                    "exclusion_reasons": [kind],
                    "error": str(error)[:300]}
        self.stage(key, {"question": question, "readers": [], "memory": [],
                         "eval": None})

    def finalize(self, video, question_number, kept, blockers):
        for pend_key in [k for k in self.pending
                         if k[0] == video and k[1] == question_number]:
            bundle = self.pending.pop(pend_key)
            row = bundle["question"]
            row["kept"] = bool(kept)
            row["disposition"] = "kept" if kept else "rolled_back"
            # terminal_type and episode_clean describe the episode itself.
            # clean and valid_terminal mean "usable in a comparison".
            row["clean"] = bool(row.get("episode_clean") and kept)
            row["valid_terminal"] = bool(row.get("valid_terminal") and kept)
            row["rollback_blockers"] = [] if kept else blockers
            if not kept and "rolled_back" not in row["exclusion_reasons"]:
                row["exclusion_reasons"] = row["exclusion_reasons"] + [
                    "rolled_back"]
            self._write("question", row)
            self.rows += 1
            for kind in ("readers", "memory"):
                for item in bundle[kind]:
                    self._write("reader" if kind == "readers" else "memory",
                                item)
            if bundle["eval"]:
                self._write("eval", bundle["eval"])


def terminal_type(outcome, forced=False):
    if not outcome or outcome.get("infrastructure_valid") is False:
        return "infrastructure_failure"
    if forced:
        return "forced"
    if outcome.get("protocol_fallback") or \
            outcome.get("terminal") == "protocol_abstain":
        return "protocol_failure"
    if outcome.get("fault") == "turn_limit" or not outcome.get("complete"):
        return "cut"
    if outcome.get("terminal") == "agent_abstain":
        return "abstain"
    return "correct" if outcome.get("correct") else "wrong"
