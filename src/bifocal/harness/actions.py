"""Atomic, step-by-step time-tree memory for one Qwen and two agent roles.

The reader branch owns detailed, append-only observations. The global tree owns a
short time-ordered view backed by those revisions. Every model turn selects exactly
one action, and the harness executes it before the next decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from . import protocol as C
from ..memory import claim as MEMORY_CLAIM
from ..memory import global_store as TREE


SCHEMA_VERSION = "atomic-time-tree-v1"
READ_FPS = 0.5
BRANCH_SUMMARY_CHARS = 6000
READER_SUMMARY_MAX_WORDS = 120
BRANCH_HISTORY_CHARS = 2000
GLOBAL_TIMELINE_CHARS = 6000
OPEN_BRANCH_CHARS = 8000
MAX_EVENTS = 1
REVIEW_FRAMES = 12
TURN_HANDLES = 20
MIN_TURNS = 8
MAX_POLICY_REFUSALS = 4
MAX_POLICY_REPAIRS = 3
MAX_NON_ACCEPT_REVIEWS_PER_BRANCH = 2
MAX_MEMORY_ACTION_REFUSALS = 4
# Invalid checker evidence is an observation failure, not a policy action
# failure. It stays recorded but does not consume the policy repair allowance.
NON_POLICY_REFUSAL_KINDS = frozenset(("check_invalid",))
MEMORY_DECISION_TIMINGS = ("pre_read", "after_read", "off")
MAX_READS_PER_BRANCH = 2
MEMORY_USE_FRAMES = 12
MENU_BRANCH_LIMIT = 12
READER_MIN_RUNTIME_SECONDS = 390.0
POLICY_TURN_RESERVE_SECONDS = 15.0
FINAL_ANSWER_RESERVE_SECONDS = 45.0
MIN_CHECK_TO_FINAL_SECONDS = 60.0
# UTF-8 bytes are a conservative upper bound on tokenizer tokens because every
# token occupies at least one byte. These keep the declared token ceilings.
GLOBAL_CLAIM_BYTES = 512
GLOBAL_VIEW_BYTES = 2048
GLOBAL_REWARD = 0.10
LIFECYCLE_RULE = (
    "Lifecycle rule: if any current-question accepted review or frame-backed "
    "card CHECK exists, submit the latest accepted evidence; otherwise, if 2 or "
    "fewer calls remain, abstain; otherwise, "
    "if a displayed relevant memory CHECK fits, CHECK it before another READ; "
    "otherwise, "
    "if any current-question unreviewed answer-bearing revision exists, review "
    "the latest one; "
    "otherwise, if calls and frames remain and an unread branch exists, read any "
    "unread branch; otherwise abstain with reason \"no evidence\".")
GLOBAL_LIFECYCLE_RULE = (
    "Lifecycle rule: if accepted evidence exists and the global-memory stage is "
    "available, choose one EVOLVE_GLOBAL operation before SUBMIT; otherwise, if "
    "the global stage is complete and accepted evidence exists, submit the latest "
    "accepted review or frame-backed card CHECK; otherwise, if 2 or fewer calls "
    "remain, abstain; otherwise, "
    "if a displayed relevant memory CHECK fits, CHECK it before another READ; "
    "otherwise, "
    "if an unreviewed answer-bearing revision exists, review the latest one; "
    "otherwise, if calls and frames remain and an unread branch exists, read one "
    "unread branch; otherwise abstain with reason \"no evidence\".")
BUDGET_ANSWER_RULE = (
    "Lifecycle rule: if current-question accepted evidence exists, submit it; "
    "otherwise, if an unreviewed answer-bearing revision exists and at least 3 "
    "calls remain, review it; otherwise, if READ or CHECK can still fit, inspect "
    "more evidence; otherwise, if a current-question READ exists and either one "
    "call or no more than 4 minutes remain, submit the best option directly "
    "without claiming a citation; "
    "otherwise abstain only when no current-question READ or answer call remains.")

_VERBS = ("OPEN", "READ", "REVIEW", "CHECK", "USE", "DEFER", "EVOLVE_GLOBAL",
          "SUBMIT", "SUBMAIL", "SUBMAL", "SUBMAILED", "ABSTAIN", "ABSTAINS")
_ACTION_SHAPED = re.compile(
    r"(?i)\b(?:OPEN\s+B\d+|READ\s+B\d+|REVIEW\s+R\d+|"
    r"CHECK\s+M\d+(?:\.\d+)?|USE\s+R\d+|DEFER\s+R\d+|"
    r"EVOLVE_GLOBAL\s+(?:ADD|SUPERSEDE|KEEP)\b|"
    r"SUBMIT\s+\(?[A-H]\)?(?:\s*@\s*(?:R\d+|M\d+(?:\.\d+)?))?)\b")
_ACTION_ABSTAIN = re.compile(r"(?i)\bABSTAINS?\s+(?!because\b)\S+")
_ACTION_ABSTAIN_ANY = re.compile(r"(?i)\bABSTAINS?\b")


class ActionError(ValueError):
    """A policy reply is not exactly one valid atomic action."""


@dataclass(frozen=True)
class Action:
    verb: str
    branch: Optional[int] = None
    revision: Optional[int] = None
    option: str = ""
    quote: str = ""
    window: Optional[tuple] = None
    reason: str = ""
    global_operation: str = ""
    global_target: Optional[int] = None
    memory_handle: str = ""

    def as_dict(self):
        return {"action": self.verb, "branch": self.branch,
                "revision": self.revision, "option": self.option,
                "quote": self.quote,
                "window": list(self.window) if self.window else None,
                "reason": self.reason,
                "global_operation": self.global_operation,
                "global_target": self.global_target,
                "memory_handle": self.memory_handle}


def _lines(text):
    return [line.strip() for line in str(text or "").splitlines()
            if line.strip() and line.strip() not in ("```", "```text")]


def action_line_count(text):
    count = 0
    seen_body = False
    for line in _lines(text):
        if re.match(r"(?i)^WHY\s*:", line):
            continue
        line = re.sub(r"(?i)^ACTION\s*:\s*", "", line)
        words = line.upper().split()
        if not words:
            continue
        first = words[0].rstrip(":")
        if not seen_body:
            seen_body = True
            count += int(first in _VERBS)
        elif _ACTION_SHAPED.search(line) or _ACTION_ABSTAIN.search(line):
            count += 1
    return count


def parse_action(text):
    """Parse exactly one anchored action, with an optional preceding WHY line."""
    lines = _lines(text)
    why = [line for line in lines if re.match(r"(?i)^WHY\s*:", line)]
    body = [line for line in lines if not re.match(r"(?i)^WHY\s*:", line)]
    if len(why) > 1 or not body:
        raise ActionError("reply must contain one action line")
    if why and lines.index(why[0]) > lines.index(body[0]):
        raise ActionError("WHY must precede the action line")
    body[0] = re.sub(r"(?i)^ACTION\s*:\s*", "", body[0])
    if len(body) > 1:
        trailing_actions = [line for line in body[1:]
                            if (_ACTION_SHAPED.search(line)
                                or _ACTION_ABSTAIN.search(line))]
        if not re.match(r"(?i)^ABSTAIN\s+", body[0]) or trailing_actions:
            raise ActionError("reply must contain one action line")
        body = [" ".join(body)]
    line = body[0]
    found = re.fullmatch(
        r"(?i)(OPEN|READ|REVIEW|CHECK|USE|DEFER|EVOLVE_GLOBAL|SUBMIT|SUBMAIL|"
        r"SUBMAL|SUBMAILED|ABSTAIN|ABSTAINS)(?:\s+(.*))?", line)
    if not found:
        raise ActionError(
            "unknown or malformed action; use one displayed form beginning ACTION:")
    verb, rest = found.group(1).upper(), (found.group(2) or "").strip()
    if verb in ("SUBMAIL", "SUBMAL", "SUBMAILED"):
        verb = "SUBMIT"
    if verb == "ABSTAINS":
        verb = "ABSTAIN"
        rest = rest or "no valid action"
    if verb == "OPEN":
        match = re.fullmatch(r"B(\d+)", rest, re.I)
        if not match:
            raise ActionError("OPEN needs exactly one branch")
        return Action("open", branch=int(match.group(1)))
    if verb == "READ":
        match = re.fullmatch(
            r"B(\d+)(?:\s+(?:\[(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s?\]"
            r"|(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s?))?"
            r"(?:\s+(.*))?",
            rest, re.I)
        if not match:
            raise ActionError("READ needs one branch and an optional window")
        trailing = match.group(6) or ""
        if (trailing and (re.match(
                r"(?i)^(?:B\d+|R\d+|\[|\d+(?:\.\d+)?\s*-\s*\d|,)", trailing)
                          or _ACTION_SHAPED.search(trailing)
                          or _ACTION_ABSTAIN_ANY.search(trailing)
                          or re.search(r"(?i)\bWHY\s*:", trailing))):
            raise ActionError("READ has trailing action or window")
        window = None
        if match.group(2) is not None or match.group(4) is not None:
            first = float(match.group(2) or match.group(4))
            last = float(match.group(3) or match.group(5))
            if first >= last:
                raise ActionError("READ window must increase")
            window = (first, last)
        return Action("read", branch=int(match.group(1)), window=window)
    if verb == "REVIEW":
        match = re.fullmatch(r"R(\d+)", rest, re.I)
        if not match:
            raise ActionError("REVIEW needs exactly one revision")
        return Action("review", revision=int(match.group(1)))
    if verb == "CHECK":
        match = re.fullmatch(r"M(\d+(?:\.\d+)?)", rest, re.I)
        if not match:
            raise ActionError("CHECK needs exactly one memory handle")
        return Action("check", memory_handle="M%s" % match.group(1))
    if verb in ("USE", "DEFER"):
        match = re.fullmatch(r"R(\d+)", rest, re.I)
        if not match:
            raise ActionError("%s needs exactly one revision" % verb)
        return Action(verb.lower(), revision=int(match.group(1)))
    if verb == "EVOLVE_GLOBAL":
        keep = re.fullmatch(r"KEEP", rest, re.I)
        if keep:
            return Action("evolve_global", global_operation="keep")
        add = re.fullmatch(r"ADD\s+R(\d+)", rest, re.I)
        if add:
            return Action("evolve_global", revision=int(add.group(1)),
                          global_operation="add")
        supersede = re.fullmatch(r"SUPERSEDE\s+G(\d+)\s+R(\d+)", rest, re.I)
        if supersede:
            return Action("evolve_global", global_target=int(supersede.group(1)),
                          revision=int(supersede.group(2)),
                          global_operation="supersede")
        raise ActionError(
            "EVOLVE_GLOBAL needs ADD R<number>, SUPERSEDE G<number> R<number>, or KEEP")
    if verb == "SUBMIT":
        bare = re.fullmatch(r"([A-H])", rest, re.I)
        if bare:
            return Action("submit", option=bare.group(1).upper())
        match = re.fullmatch(
            r'(?:([A-H])|\(([A-H])\))\s*@\s*([RM])(\d+(?:\.\d+)?)'
            r'(?:\s*\|\s*|\s+)(.+)', rest, re.I)
        if not match:
            raise ActionError("SUBMIT needs one option, target and quote")
        raw_quote = match.group(5).strip()
        try:
            quote = json.loads(raw_quote)
        except (TypeError, ValueError):
            quote = None
        if not isinstance(quote, str) or not quote:
            raise ActionError("SUBMIT quote must be one double-quoted string")
        target_kind = match.group(3).upper()
        target_text = match.group(4)
        return Action(
            "submit", option=(match.group(1) or match.group(2)).upper(),
            revision=(int(target_text) if target_kind == "R" else None),
            memory_handle=("M%s" % target_text if target_kind == "M" else ""),
            quote=quote)
    reason = rest.strip()
    if (not reason or _ACTION_SHAPED.search(reason)
            or _ACTION_ABSTAIN.search(reason)):
        raise ActionError("ABSTAIN needs a reason")
    return Action("abstain", reason=reason)


def action_log(text):
    """`(parsed action name, action-line count)` for the call log."""
    try:
        return parse_action(text).verb, action_line_count(text)
    except ActionError:
        return "unparsed", action_line_count(text)


def _field(text, name):
    match = re.search(r"(?im)^\s*%s\s*:\s*(.*?)\s*$" % re.escape(name), text or "")
    return match.group(1).strip() if match else ""


def _line_values(text, name):
    pattern = re.compile(
        r"(?im)^[ \t]*%s[ \t]*:[ \t]*([^\r\n]*?)[ \t]*$"
        % re.escape(name))
    return [match.group(1).strip() for match in pattern.finditer(text or "")]


def _summary_event_text(summary, limit=15):
    """Choose the first grounded clause from a Reader branch summary."""
    text = re.sub(r"\s+", " ", str(summary or "")).strip()
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+|;\s+", text, maxsplit=1)[0].strip()
    words = sentence.split()
    return " ".join(words[:int(limit)]).rstrip(" ,;:-")


def _reader_time_seconds(value, accept_clock=False):
    """Parse seconds or an optional HH:MM:SS/MM:SS clock value."""
    text = str(value or "").strip().lower().removesuffix("s")
    try:
        return float(text)
    except ValueError:
        pass
    if not accept_clock:
        raise ValueError("not a numeric event time")
    parts = text.split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise ValueError("not a numeric or clock event time")
    numbers = [int(part) for part in parts]
    if any(number < 0 for number in numbers):
        raise ValueError("negative event time")
    if len(numbers) == 2:
        minutes, seconds = numbers
        if seconds >= 60:
            raise ValueError("clock seconds must be below 60")
        return minutes * 60.0 + seconds
    hours, minutes, seconds = numbers
    if minutes >= 60 or seconds >= 60:
        raise ValueError("clock minutes and seconds must be below 60")
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_reader_report(text, first=None, last=None, reader_policy=False,
                        reader_schema_version=1, allow_keep=True,
                        max_options=None):
    """Parse one branch report and, optionally, one explicit memory operation."""
    summary = _field(text, "BRANCH_SUMMARY")
    can = _field(text, "CAN_ANSWER").upper()
    answer = _field(text, "ANSWER").upper()
    answer = answer if answer in tuple("ABCDEFGH") + ("S", "SAME") else ""
    quote = _field(text, "QUOTE").strip().strip('"')
    quote = "" if quote.upper() in ("NONE", "N/A") else quote
    if int(reader_schema_version) not in (1, 2, 3):
        raise ValueError("unsupported Reader schema version")
    operation_lines = re.findall(
        r"(?im)^\s*MEMORY_OP\s*:\s*(.*?)\s*$", text or "")
    operation_line = (operation_lines[0] if operation_lines else "").upper()
    operation, target = "ADD", ""
    events, errors = [], []
    if reader_policy:
        if can not in ("YES", "NO", "S"):
            errors.append("CAN_ANSWER must be YES, NO or S")
        if can == "YES":
            if answer not in tuple("ABCDEFGH"):
                errors.append("CAN_ANSWER YES needs an A-H ANSWER")
            elif max_options is not None and answer not in "ABCDEFGH"[:int(
                    max_options)]:
                errors.append("CAN_ANSWER YES names an option outside this question")
            if not quote:
                errors.append("CAN_ANSWER YES needs a QUOTE")
        elif can == "NO" and answer and answer != "NONE":
            errors.append("CAN_ANSWER NO needs ANSWER: NONE")
        elif can == "S" and answer != "S":
            errors.append("CAN_ANSWER S needs ANSWER: S")
        if len(operation_lines) > 1:
            errors.append("multiple MEMORY_OP lines")
        if not operation_line:
            errors.append("missing MEMORY_OP")
        elif operation_line == "ADD":
            operation = "ADD"
        elif operation_line == "KEEP":
            operation = "KEEP"
        elif re.fullmatch(r"SUPERSEDE\s+E[1-9][0-9]*", operation_line):
            operation = "SUPERSEDE"
            target = operation_line.split()[-1]
        else:
            errors.append("malformed MEMORY_OP: %s" % operation_line[:120])
        if operation == "KEEP" and not allow_keep:
            errors.append("KEEP is disabled for this anchored read")
    if summary.strip().lower().strip("().") in ("none", "no previous summary"):
        summary = ""
        errors.append("empty BRANCH_SUMMARY")
    event_text = ""
    claim = ""
    claim_start = None
    claim_end = None
    if reader_policy and int(reader_schema_version) == 3:
        claims = _line_values(text, "CLAIM")
        starts = _line_values(text, "START")
        ends = _line_values(text, "END")
        if len(claims) != 1:
            errors.append("schema v3 needs exactly one CLAIM line")
        if len(starts) != 1:
            errors.append("schema v3 needs exactly one START line")
        if len(ends) != 1:
            errors.append("schema v3 needs exactly one END line")
        claim = claims[0].strip().strip('"') if claims else ""
        start_text = starts[0].strip() if starts else ""
        end_text = ends[0].strip() if ends else ""
        if operation == "KEEP":
            if claim.upper() not in ("NONE", "N/A", "S"):
                errors.append("KEEP CLAIM must be NONE")
            if start_text.upper() not in ("NONE", "N/A", "S") \
                    or end_text.upper() not in ("NONE", "N/A", "S"):
                errors.append("KEEP START and END must be NONE")
        elif operation in ("ADD", "SUPERSEDE"):
            if MEMORY_CLAIM.is_generic_claim(claim):
                errors.append("ADD or SUPERSEDE needs one concrete CLAIM")
            try:
                claim_start = float(start_text)
                claim_end = float(end_text)
            except (TypeError, ValueError):
                errors.append("CLAIM START and END must be numeric seconds")
            else:
                if not claim_start < claim_end:
                    errors.append("CLAIM START must be before END")
                if first is not None and claim_start < float(first):
                    errors.append("CLAIM START is outside the read window")
                if last is not None and claim_end > float(last):
                    errors.append("CLAIM END is outside the read window")
                if not errors:
                    events = [{
                        "start": claim_start,
                        "end": claim_end,
                        "text": claim,
                        "supersedes_event_id": (
                            target if operation == "SUPERSEDE" else ""),
                        "span_source": "reader_claim",
                    }]
                    event_text = claim
    elif int(reader_schema_version) == 2:
        supplied_event_text = _field(text, "EVENT_TEXT").strip().strip('"')
        raw_events = _line_values(text, "EVENT")
        if len(raw_events) > 1:
            errors.append("more than 1 events")
        raw_event = raw_events[0] if raw_events else ""
        if reader_policy:
            if first is None or last is None:
                errors.append("coarse Reader EVENT_TEXT needs the read window")
            elif operation in ("ADD", "SUPERSEDE"):
                event_text = (
                    supplied_event_text
                    if supplied_event_text
                    and supplied_event_text.upper() not in ("NONE", "N/A", "S")
                    else _summary_event_text(summary))
                if not event_text:
                    errors.append("ADD or SUPERSEDE needs a non-empty summary")
                else:
                    events = [{
                        "start": float(first), "end": float(last),
                        "text": event_text,
                        "supersedes_event_id": (
                            target if operation == "SUPERSEDE" else ""),
                        "span_source": "read_window",
                    }]
            elif operation == "KEEP":
                event_text = ""
        elif first is None or last is None:
            errors.append("coarse Reader EVENT_TEXT needs the read window")
        else:
            supplied_sentinel = supplied_event_text.upper() in (
                "NONE", "N/A", "S")
            raw_sentinel = raw_event.strip().upper() in ("NONE", "N/A", "S")
            candidate = (
                raw_event if supplied_sentinel and raw_event and not raw_sentinel
                else supplied_event_text or raw_event)
            if "|" in candidate:
                candidate = candidate.split("|", 1)[1].strip()
            candidate = candidate.strip().strip('"')
            if candidate.upper() in ("NONE", "N/A", "S"):
                operation = "KEEP"
                event_text = ""
            else:
                event_text = candidate or _summary_event_text(summary)
                if not event_text:
                    errors.append("ADD or SUPERSEDE needs a non-empty summary")
                else:
                    events = [{
                        "start": float(first), "end": float(last),
                        "text": event_text,
                        "supersedes_event_id": "",
                        "span_source": "read_window",
                    }]
    event_rows = ([] if int(reader_schema_version) == 2
                  else re.findall(r"(?im)^\s*EVENT\s*:\s*(.*?)\s*$", text or ""))
    for raw in event_rows:
        if reader_policy and operation == "KEEP" and raw.strip().upper() in (
                "NONE", "N/A"):
            continue
        parts = [part.strip() for part in raw.split("|")]
        span_pattern = (
            r"((?:\d+:)?\d{1,2}:\d{2}s?|\d+(?:\.\d+)?s?)\s*-\s*"
            r"((?:\d+:)?\d{1,2}:\d{2}s?|\d+(?:\.\d+)?s?)\s*s?"
            if reader_policy else
            r"(\d+(?:\.\d+)?s?)\s*-\s*(\d+(?:\.\d+)?s?)\s*s?")
        compact = re.fullmatch(
            r"(?:(?:B\d+|E\d+)\s+)*" + span_pattern + r"\s+(.+)",
            parts[0], re.I) if parts else None
        if len(parts) >= 3:
            span = re.fullmatch(span_pattern, parts[0], re.I)
            event_text = parts[1]
            supersedes_raw = parts[2]
        elif compact is not None:
            span = compact
            event_text = compact.group(3)
            supersedes_raw = parts[1] if len(parts) == 2 else ""
        elif len(parts) == 2:
            span = re.fullmatch(span_pattern, parts[0], re.I)
            event_text = parts[1]
            supersedes_raw = ""
        else:
            span, event_text, supersedes_raw = None, "", ""
        if not span or not event_text:
            errors.append("malformed EVENT: %s" % raw[:120])
            continue
        try:
            event_first = _reader_time_seconds(span.group(1), reader_policy)
            event_last = _reader_time_seconds(span.group(2), reader_policy)
        except ValueError:
            errors.append("malformed EVENT time: %s" % raw[:120])
            continue
        if event_first >= event_last:
            errors.append("EVENT window must increase: %s" % raw[:120])
            continue
        if ((first is not None and event_first < float(first))
                or (last is not None and event_last > float(last))):
            errors.append("EVENT is outside the read window: %s" % raw[:120])
            continue
        supersedes = ""
        if supersedes_raw:
            edge = re.fullmatch(
                r"SUPERSEDES\s+(?:B\d+\s+)?(E\d+)", supersedes_raw, re.I)
            if not edge:
                errors.append("malformed EVENT supersession: %s" % supersedes_raw)
                continue
            supersedes = edge.group(1).upper()
        events.append({"start": event_first, "end": event_last, "text": event_text,
                       "supersedes_event_id": supersedes})
    if len(events) > MAX_EVENTS:
        errors.append("more than %d events" % MAX_EVENTS)
        events = events[:MAX_EVENTS]
    if not summary:
        if "empty BRANCH_SUMMARY" not in errors:
            errors.append("missing BRANCH_SUMMARY")
    if reader_policy:
        if operation == "ADD" and not events:
            errors.append("ADD requires one EVENT")
        if operation == "SUPERSEDE":
            if not target:
                errors.append("SUPERSEDE requires an event target")
            if len(events) != 1:
                errors.append("SUPERSEDE requires exactly one EVENT")
            elif events[0].get("supersedes_event_id") and (
                    events[0]["supersedes_event_id"] != target):
                errors.append("MEMORY_OP and EVENT supersession targets differ")
        if operation == "KEEP" and events:
            errors.append("KEEP must not add an EVENT")
    return {"parse_status": "ok" if not errors else "failed",
            "branch_summary": summary, "can_answer": can,
            "answer": answer, "quote": quote, "events": events,
            "event_text": event_text, "claim": claim,
            "claim_start": claim_start, "claim_end": claim_end,
            "memory_operation": operation,
            "memory_target": target,
            "reader_policy": bool(reader_policy),
            "reader_schema_version": int(reader_schema_version),
            "span_source": ("read_window" if int(reader_schema_version) == 2
                            else ""),
            "parse_errors": errors}

def answer_bearing(revision):
    return bool(revision.get("read_status") == "success"
                and not revision.get("keep_noop")
                and revision.get("parse_status") == "ok"
                and revision.get("frame_status") == "ok"
                and revision.get("frame_handle")
                and revision.get("can_answer") == "YES"
                and str(revision.get("answer") or "") in list("ABCDEFGH"))


def parse_review(text):
    lines = _lines(text)
    why = [line for line in lines if re.match(r"(?i)^WHY\s*:", line)]
    body_lines = [line for line in lines if not re.match(r"(?i)^WHY\s*:", line)]
    found = [line for line in body_lines if re.match(r"(?i)^REVIEW\s*:", line)]
    if len(why) > 1 or len(body_lines) != 1 or len(found) != 1:
        raise ActionError("review needs exactly one REVIEW line")
    body = re.sub(r"(?i)^REVIEW\s*:\s*", "", found[0]).strip()
    parts = [part.strip() for part in body.split("|")]
    outcome = (parts[0] if parts else "").upper()
    if outcome not in ("ACCEPT", "REJECT", "UNSURE"):
        raise ActionError("review outcome must be ACCEPT, REJECT or UNSURE")
    revision = re.fullmatch(r"R(\d+)", parts[1], re.I) if len(parts) > 1 else None
    if not revision:
        raise ActionError("review needs one revision")
    parsed = {"outcome": outcome.lower(), "revision": int(revision.group(1)),
              "option": "", "quote": "",
              "why": _field(text, "WHY") or "no reason given"}
    if outcome == "ACCEPT":
        if len(parts) < 4 or parts[2].upper() not in list("ABCDEFGH"):
            raise ActionError("accepted review needs one A-H option and one quote")
        quote = parts[3].strip().strip('"')
        if not quote:
            raise ActionError("accepted review needs one quote")
        parsed.update(option=parts[2].upper(), quote=quote)
    elif len(parts) < 3 or not " | ".join(parts[2:]).strip():
        raise ActionError("rejected or unsure review needs a reason")
    return parsed


def option_is_valid(option, options):
    return str(option or "").upper() in "ABCDEFGH"[:len(options)]


def _norm(value):
    return C._normalise(value)


@dataclass
class AtomicTimeTree:
    video: str
    duration: float
    root: str = ""
    revisions: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    reviews: dict = field(default_factory=dict)
    memory_uses: dict = field(default_factory=dict)
    questions: dict = field(default_factory=dict)
    branches: dict = field(default_factory=dict)
    global_claims: dict = field(default_factory=dict)
    global_operations: dict = field(default_factory=dict)
    next_revision: int = 1
    next_event: int = 1
    next_review: int = 1
    next_memory_use: int = 1
    next_global_claim: int = 1
    next_global_operation: int = 1

    @property
    def path(self):
        return os.path.join(str(self.root).rstrip("/"), str(self.video),
                            "atomic-tree.json") if self.root else ""

    def allocate_revision_id(self):
        revision_id = "R%d" % self.next_revision
        self.next_revision += 1
        return revision_id

    def allocate_global_claim_id(self):
        claim_id = "G%d" % self.next_global_claim
        self.next_global_claim += 1
        return claim_id

    def allocate_global_operation_id(self):
        operation_id = "GO%d" % self.next_global_operation
        self.next_global_operation += 1
        return operation_id

    def tip(self, branch):
        rows = self.branches.get(int(branch)) or []
        return rows[-1] if rows else ""

    def latest_summary(self, branch, question_number=None):
        for revision_id in reversed(self.branches.get(int(branch)) or []):
            revision = self.revisions[revision_id]
            if (question_number is not None
                    and int(revision.get("question_number") or 0)
                    != int(question_number)):
                continue
            if revision["read_status"] == "success" and revision.get("branch_summary"):
                return {"revision_id": revision_id,
                        "text": revision["branch_summary"]}
        return {"revision_id": "", "text": ""}

    def active_global_claims(self):
        rows = [row for row in self.global_claims.values()
                if row.get("state") == "active"]
        return sorted(rows, key=lambda row: int(row.get("creation_order") or 0))

    def accepted_local_revisions(self, question_number):
        """Accepted, grounded local revisions created for this question."""
        rows = []
        for revision_id, revision in self.revisions.items():
            if int(revision.get("question_number") or 0) != int(question_number):
                continue
            if not answer_bearing(revision) or self.review_stale(revision_id):
                continue
            review = self.effective_review(revision_id, revision.get("answer"))
            if (review is None or review.get("outcome") != "accept"
                    or int(review.get("question_number") or 0)
                    != int(question_number)
                    or review.get("frames_from_store") is not True
                    or int(review.get("stored_frames") or 0) < 1):
                continue
            if not revision.get("frame_handle") or not revision.get("events"):
                continue
            rows.append(copy.deepcopy(revision))
        return sorted(rows, key=lambda row: int(row["revision_id"][1:]))

    def promotable_local_revisions(self, question_number):
        """Accepted local revisions that are not already in global memory."""
        promotable = []
        for revision in self.accepted_local_revisions(question_number):
            event_text = " ".join(
                str(self.events[event_id].get("text") or "")
                for event_id in revision.get("events") or ()
                if event_id in self.events)
            duplicate, _reason = self._global_duplicate(revision, event_text)
            if duplicate is None:
                promotable.append(revision)
        return promotable

    def _global_source(self, revision_id, question_number):
        revision_id = _revision_key(revision_id)
        revision = self.revisions.get(revision_id)
        if revision is None:
            return None, "source revision does not exist"
        if self.review_stale(revision_id):
            return None, "source revision is stale"
        if int(revision.get("question_number") or 0) != int(question_number):
            return None, "source revision belongs to another question"
        if revision_id not in {
                row["revision_id"]
                for row in self.accepted_local_revisions(question_number)}:
            return None, "source revision is not accepted and grounded"
        return revision, ""

    def _global_duplicate(self, revision, event_text):
        quote = _norm(revision.get("quote"))
        for claim in self.active_global_claims():
            if quote and quote == _norm(claim.get("quote")):
                return claim, "duplicate quote"
            old_span = claim.get("time_span") or ()
            new_span = revision.get("window") or ()
            if (not event_text or _norm(event_text) != _norm(claim.get("event_text"))
                    or len(old_span) != 2 or len(new_span) != 2):
                continue
            left = max(float(old_span[0]), float(new_span[0]))
            right = min(float(old_span[1]), float(new_span[1]))
            intersection = max(0.0, right - left)
            union = (max(float(old_span[1]), float(new_span[1]))
                     - min(float(old_span[0]), float(new_span[0])))
            if union > 0.0 and intersection / union >= 0.50:
                return claim, "duplicate event span"
        return None, ""

    def _global_claim_bytes(self, claim):
        text = "%s %s %s %s-%s %s %s" % (
            claim.get("claim_id") or "", claim.get("source_revision_id") or "",
            claim.get("source_branch") or "", claim.get("time_span", [0, 0])[0],
            claim.get("time_span", [0, 0])[1], claim.get("event_text") or "",
            claim.get("quote") or "")
        return len(text.encode("utf-8"))

    def record_global_operation(self, operation, question_number, claim_id="",
                                target_claim_id="", source_revision_id="",
                                accepted=False, reason="", provenance=None):
        operation_id = self.allocate_global_operation_id()
        record = {
            "operation_id": operation_id,
            "operation": str(operation),
            "claim_id": str(claim_id or ""),
            "target_claim_id": str(target_claim_id or ""),
            "source_revision_id": _revision_key(source_revision_id)
            if source_revision_id else "",
            "question_number": int(question_number),
            "accepted": bool(accepted),
            "reason": str(reason or ""),
            "provenance": copy.deepcopy(provenance or {}),
            "creation_order": self.next_global_operation - 1,
        }
        self.global_operations[operation_id] = record
        self.save()
        return record

    def _append_global_claim(self, operation, revision, question_number,
                             supersedes_claim_id="", provenance=None):
        events = [self.events[event_id] for event_id in revision.get("events") or ()
                  if event_id in self.events]
        event_text = " ".join(str(row.get("text") or "").strip()
                              for row in events if row.get("text"))
        claim_id = self.allocate_global_claim_id()
        claim = {
            "claim_id": claim_id,
            "video": self.video,
            "source_revision_id": revision["revision_id"],
            "source_event_ids": [row["event_id"] for row in events],
            "source_branch": str(revision.get("branch_id") or ""),
            "source_question": int(question_number),
            "time_span": [float(revision["window"][0]),
                          float(revision["window"][1])],
            "event_text": event_text,
            "quote": str(revision.get("quote") or ""),
            "option": str(revision.get("answer") or ""),
            "state": "active",
            "supersedes_claim_id": str(supersedes_claim_id or ""),
            "superseded_by_claim_id": "",
            "creation_order": self.next_global_claim - 1,
            "policy_provenance": copy.deepcopy(provenance or {}),
            "operation": str(operation),
        }
        if self._global_claim_bytes(claim) > GLOBAL_CLAIM_BYTES:
            self.next_global_claim -= 1
            raise ActionError("global claim exceeds its 512-token-safe byte budget")
        active_bytes = sum(self._global_claim_bytes(row)
                           for row in self.active_global_claims())
        if active_bytes + self._global_claim_bytes(claim) > GLOBAL_VIEW_BYTES:
            self.next_global_claim -= 1
            raise ActionError("global memory view budget is full")
        self.global_claims[claim_id] = claim
        self.save()
        return claim

    def evolve_global(self, operation, question_number, source_revision_id="",
                      target_claim_id="", provenance=None):
        """Validate and append one O-owned global-memory operation."""
        operation = str(operation or "").lower()
        source = None
        source_error = ""
        if source_revision_id:
            source, source_error = self._global_source(
                source_revision_id, question_number)
        if operation == "keep":
            record = self.record_global_operation(
                operation, question_number, accepted=True,
                provenance=provenance)
            return True, "global memory kept", None, record
        if operation not in ("add", "supersede"):
            raise ValueError("unknown global operation: %s" % operation)
        if source_error:
            record = self.record_global_operation(
                operation, question_number,
                target_claim_id=target_claim_id,
                source_revision_id=source_revision_id, accepted=False,
                reason=source_error, provenance=provenance)
            return False, source_error, None, record
        event_text = " ".join(
            str(self.events[event_id].get("text") or "")
            for event_id in source.get("events") or ()
            if event_id in self.events)
        target = None
        if operation == "supersede":
            target_key = str(target_claim_id or "").upper()
            if not target_key.startswith("G"):
                target_key = "G%d" % int(target_claim_id or 0)
            target = self.global_claims.get(target_key)
            if target is None or str(target.get("video") or "") != str(self.video):
                reason = "global target does not exist in this video"
            elif target.get("state") != "active":
                reason = "global target is stale or already superseded"
            else:
                reason = ""
            if reason:
                record = self.record_global_operation(
                    operation, question_number,
                    target_claim_id=target_key,
                    source_revision_id=source_revision_id,
                    accepted=False, reason=reason, provenance=provenance)
                return False, reason, None, record
        duplicate, duplicate_reason = self._global_duplicate(source, event_text)
        if duplicate is not None:
            reason = "%s matches %s" % (duplicate_reason, duplicate["claim_id"])
            record = self.record_global_operation(
                operation, question_number,
                target_claim_id="G%d" % int(target_claim_id or 0)
                if target_claim_id else "",
                source_revision_id=source_revision_id,
                accepted=False, reason=reason, provenance=provenance)
            return False, reason, None, record
        try:
            claim = self._append_global_claim(
                operation, source, question_number,
                supersedes_claim_id=(target or {}).get("claim_id", ""),
                provenance=provenance)
        except ActionError as error:
            record = self.record_global_operation(
                operation, question_number,
                target_claim_id=(target or {}).get("claim_id", ""),
                source_revision_id=source_revision_id,
                accepted=False, reason=str(error), provenance=provenance)
            return False, str(error), None, record
        if target is not None:
            target["state"] = "superseded"
            target["superseded_by_claim_id"] = claim["claim_id"]
        record = self.record_global_operation(
            operation, question_number, claim_id=claim["claim_id"],
            target_claim_id=(target or {}).get("claim_id", ""),
            source_revision_id=source_revision_id, accepted=True,
            provenance=provenance)
        self.save()
        return True, "global %s accepted" % operation, claim, record

    def append_read(self, branch, first, last, fps, raw_reply, report,
                    frame_handle, *, frame_status="ok", requested_frames=0,
                    actual_frames=0, rendered_frames=0, effective_fps=0.0,
                    frame_error="", revision_id=None, question_number=0):
        branch = int(branch)
        revision_id = revision_id or self.allocate_revision_id()
        summary = str((report or {}).get("branch_summary") or "")
        status = "success" if ((report or {}).get("parse_status") == "ok"
                               and frame_status == "ok" and summary) else "failed"
        revision = {
            "revision_id": revision_id,
            "branch_id": "B%d" % branch,
            "parent_revision_id": self.tip(branch),
            "question_number": int(question_number),
            "reader": "W%d" % branch,
            "window": [float(first), float(last)],
            "fps": float(fps),
            "read_status": status,
            "parse_status": str((report or {}).get("parse_status") or "failed"),
            "frame_status": str(frame_status),
            "frame_handle": copy.deepcopy(frame_handle),
            "keep_noop": False,
            "frame_error": str(frame_error),
            "requested_frames": int(requested_frames),
            "actual_frames": int(actual_frames),
            "rendered_frames": int(rendered_frames),
            "effective_fps": float(effective_fps),
            "branch_summary": summary if status == "success" else "",
            "can_answer": str((report or {}).get("can_answer") or ""),
            "answer": str((report or {}).get("answer") or ""),
            "quote": str((report or {}).get("quote") or ""),
            "raw_reply": str(raw_reply or ""),
            "events": [],
        }
        previous_summary = self.latest_summary(branch)["text"]
        previous_tip = self.revisions.get(self.tip(branch)) or {}
        self.revisions[revision_id] = revision
        self.branches.setdefault(branch, []).append(revision_id)
        operation = str((report or {}).get("memory_operation") or "ADD").upper()
        target = str((report or {}).get("memory_target") or "").upper()
        reader_policy = bool((report or {}).get("reader_policy"))
        memory_error = ""
        specs = list((report or {}).get("events") or [])
        if status == "failed":
            specs = [{"start": float(first), "end": float(last),
                      "text": "read attempt; no usable structured event",
                      "supersedes_event_id": "", "fallback": True}]
            memory_error = "report parse failed"
        elif reader_policy and operation == "KEEP":
            specs = []
            memory_error = self._keep_error(
                branch, previous_summary, previous_tip, report)
            if not memory_error:
                revision["branch_summary"] = previous_summary
                revision["can_answer"] = str(
                    previous_tip.get("can_answer") or "")
                revision["answer"] = str(previous_tip.get("answer") or "")
                revision["quote"] = ""
                revision["keep_noop"] = True
                branch_rows = self.branches.get(branch) or []
                if branch_rows and branch_rows[-1] == revision_id:
                    branch_rows.pop()
        elif reader_policy and operation == "SUPERSEDE":
            if not specs or not target:
                memory_error = "SUPERSEDE needs one event and a target"
            else:
                specs = [dict(specs[0], supersedes_event_id=target)]
        elif reader_policy and operation == "ADD":
            for spec in specs:
                duplicate = self._duplicate_event_error(
                    branch, str((report or {}).get("quote") or ""), spec)
                if duplicate:
                    memory_error = duplicate
                    break
        elif reader_policy and operation not in ("ADD", "KEEP", "SUPERSEDE"):
            memory_error = "unknown memory operation"
        if reader_policy and memory_error:
            specs = []
        elif not specs and operation != "KEEP" and not memory_error:
            specs = [{"start": float(first), "end": float(last),
                      "text": "read attempt; no usable structured event",
                      "supersedes_event_id": "", "fallback": True}]
            memory_error = "operation produced no event"
        revision["memory_operation"] = operation
        revision["memory_target"] = target
        for spec in specs:
            event_id = "E%d" % self.next_event
            self.next_event += 1
            supersedes = str(spec.get("supersedes_event_id") or "").upper()
            error = self._supersession_error(branch, revision_id, supersedes)
            if error and reader_policy:
                memory_error = error
                continue
            event = {
                "event_id": event_id, "branch_id": "B%d" % branch,
                "revision_id": revision_id,
                "start": float(spec["start"]), "end": float(spec["end"]),
                "text": str(spec["text"]),
                "frame_handle": copy.deepcopy(frame_handle),
                "supersedes_event_id": "" if error else supersedes,
                "supersession_error": error,
                "span_source": str(spec.get("span_source") or "model_event"),
                "fallback": bool(spec.get("fallback")),
            }
            self.events[event_id] = event
            revision["events"].append(event_id)
        revision["memory_operation_error"] = memory_error
        revision["memory_operation_valid"] = not bool(memory_error)
        superseded_events = {
            event["supersedes_event_id"] for event in self.events.values()
            if event.get("supersedes_event_id")}
        for claim in self.global_claims.values():
            if claim.get("state") != "active":
                continue
            matched = superseded_events.intersection(
                claim.get("source_event_ids") or ())
            if matched:
                # Global state is owned by global operations. A local correction
                # marks provenance risk but does not silently retire the global
                # card; O must supersede it explicitly.
                claim["source_local_stale"] = True
                claim["source_local_superseded_event_ids"] = sorted(matched)
        self.save()
        return revision

    def _keep_error(self, branch, previous_summary, previous_tip, report):
        report = report or {}
        if not previous_summary:
            return "KEEP has no prior branch memory"
        return ""

    def _duplicate_event_error(self, branch, quote, spec):
        quote_norm = re.sub(r"\s+", " ", str(quote or "")).strip().lower()
        text_norm = re.sub(r"\s+", " ", str(spec.get("text") or "")).strip().lower()
        first, last = float(spec["start"]), float(spec["end"])
        for event in self.current_events(branch):
            event_text = re.sub(
                r"\s+", " ", str(event.get("text") or "")).strip().lower()
            event_first = float(event.get("start") or 0.0)
            event_last = float(event.get("end") or 0.0)
            overlap = max(0.0, min(last, event_last) - max(first, event_first))
            union = max(last, event_last) - min(first, event_first)
            iou = overlap / union if union > 0 else 0.0
            revision = self.revisions.get(str(event.get("revision_id") or "")) or {}
            event_quote = re.sub(
                r"\s+", " ", str(revision.get("quote") or "")).strip().lower()
            if quote_norm and event_quote and quote_norm == event_quote:
                return "duplicate ADD quote"
            if text_norm and event_text and text_norm == event_text and iou >= 0.5:
                return "duplicate ADD event"
        return ""

    def _supersession_error(self, branch, revision_id, target):
        if not target:
            return ""
        old = self.events.get(target)
        if old is None:
            return "unknown event %s" % target
        if old["branch_id"] != "B%d" % branch:
            return "event %s belongs to another branch" % target
        if old["revision_id"] == revision_id:
            return "an event cannot supersede one from the same revision"
        if any(row.get("supersedes_event_id") == target for row in self.events.values()):
            return "event %s is already superseded" % target
        return ""

    def superseded_ids(self):
        return {row["supersedes_event_id"] for row in self.events.values()
                if row.get("supersedes_event_id")}

    def current_events(self, branch=None, question_number=None):
        hidden = self.superseded_ids()
        rows = [row for row in self.events.values()
                if row["event_id"] not in hidden
                and (branch is None or row["branch_id"] == "B%d" % int(branch))]
        if question_number is not None:
            rows = [
                row for row in rows
                if int((self.revisions.get(row.get("revision_id")) or {}).get(
                    "question_number") or 0) == int(question_number)]
        return sorted(rows, key=lambda row: (row["start"], row["end"],
                                             int(row["event_id"][1:])))

    def append_review(self, revision_id, option, outcome, quote,
                      question_number=0, reason="", frames=0,
                      frames_from_store=True, call_id="", stored_frames=0,
                      prompt_sha256="", reply_sha256=""):
        revision = self.revisions[_revision_key(revision_id)]
        if not answer_bearing(revision):
            raise ActionError("revision is not answer-bearing")
        if (revision.get("read_status") != "success"
                or revision.get("frame_status") != "ok"
                or not revision.get("frame_handle")):
            raise ActionError("revision has no stored frames")
        if self.review_stale(revision_id):
            raise ActionError("revision has been superseded")
        if outcome == "accept":
            if option != revision["answer"]:
                raise ActionError("reviewed option differs from the reader answer")
            source = revision["branch_summary"] + " " + revision["quote"]
            if not C._quote_grounded(quote, source):
                raise ActionError("review quote is not in the revision")
            question = str(self.questions.get(int(question_number)) or "")
            options = C.options_of(question) if question else []
            if question and not MEMORY_CLAIM.covers_question_attribute(
                    question, quote, options, option=option):
                raise ActionError(
                    "review quote does not cover a question attribute")
        review_id = "V%d" % self.next_review
        self.next_review += 1
        review = {"review_id": review_id, "revision_id": revision["revision_id"],
                  "question_number": int(question_number), "option": option,
                  "outcome": outcome, "review_quote": quote, "reason": reason,
                  "reader_quote": revision["quote"],
                  "frames": int(frames), "frames_from_store": bool(frames_from_store),
                  "stored_frames": int(stored_frames),
                  "prompt_sha256": str(prompt_sha256),
                  "reply_sha256": str(reply_sha256),
                  "call_id": str(call_id or ""), "sequence": self.next_review - 1}
        self.reviews[review_id] = review
        self.save()
        return review

    def effective_review(self, revision_id, option):
        rows = [row for row in self.reviews.values()
                if row["revision_id"] == _revision_key(revision_id)
                and row["option"] == str(option or "").upper()]
        return max(rows, key=lambda row: row["sequence"]) if rows else None

    def prior_memory_candidate(self, question_number, active_global_only=False,
                               allow_local=True):
        """Return the one newest eligible prior memory for this question."""
        if any(int(row.get("question_number") or 0) == int(question_number)
               for row in self.memory_uses.values()):
            return None
        active_sources = {
            row["source_revision_id"] for row in self.active_global_claims()}
        all_candidates = []
        for revision_id, revision in self.revisions.items():
            source_question = int(revision.get("question_number") or 0)
            if source_question < 1 or source_question >= int(question_number):
                continue
            if not answer_bearing(revision) or self.review_stale(revision_id):
                continue
            if int((revision.get("frame_handle") or {}).get("n") or 0) < 1:
                continue
            event_ids = list(revision.get("events") or ())
            if len(event_ids) != 1 or event_ids[0] not in self.events:
                continue
            own_reviews = [row for row in self.reviews.values()
                           if row.get("revision_id") == revision_id
                           and int(row.get("question_number") or 0)
                           == source_question]
            if not own_reviews:
                continue
            latest = max(own_reviews, key=lambda row: row["sequence"])
            if latest.get("outcome") != "accept":
                continue
            event = self.events[event_ids[0]]
            raw = unicodedata.normalize("NFKC", str(event.get("text") or ""))
            words = raw.split()[:16]
            preview = " ".join(words)
            if len(preview) > 160:
                prefix = preview[:160]
                cut = prefix.rfind(" ")
                preview = prefix[:cut] if cut > 0 else prefix
            preview = preview.strip()
            all_candidates.append({
                "revision_id": revision_id,
                "revision_number": int(revision_id[1:]),
                "source_question_number": source_question,
                "branch_id": str(revision.get("branch_id") or ""),
                "window": list(revision.get("window") or ()),
                "stored_frames": int(
                    (revision.get("frame_handle") or {}).get("n") or 0),
                "event_id": event_ids[0],
                "preview": preview,
                "preview_sha256": hashlib.sha256(
                    preview.encode("utf-8")).hexdigest(),
            })
        candidates = ([row for row in all_candidates
                       if row["revision_id"] in active_sources]
                      if active_global_only else all_candidates)
        if active_global_only and allow_local and not candidates:
            candidates = all_candidates
        if not candidates:
            return None
        candidates.sort(key=lambda row: (
            row["source_question_number"], row["revision_number"]))
        return candidates[-1]

    def record_memory_use(self, candidate, question_number, action, outcome,
                          option="", quote="", reason="", frames=0,
                          frames_from_store=False, prompt_sha256="",
                          reply_sha256="", call_id="", card_id="",
                          scope="local", global_claim_id="", agreement=None):
        use_id = "U%d" % self.next_memory_use
        self.next_memory_use += 1
        record = {
            "use_id": use_id,
            "revision_id": str(candidate["revision_id"]),
            "card_id": str(card_id),
            "scope": str(scope),
            "global_claim_id": str(global_claim_id),
            "source_question_number": int(candidate["source_question_number"]),
            "question_number": int(question_number),
            "action": str(action),
            "outcome": str(outcome),
            "option": str(option).upper(),
            "quote": str(quote),
            "reason": str(reason),
            "frames": int(frames),
            "frames_from_store": bool(frames_from_store),
            "stored_frames": int(candidate["stored_frames"]),
            "preview": str(candidate["preview"]),
            "preview_sha256": str(candidate["preview_sha256"]),
            "prompt_sha256": str(prompt_sha256),
            "reply_sha256": str(reply_sha256),
            "call_id": str(call_id or ""),
            "agreement": copy.deepcopy(agreement or {}),
            "sequence": self.next_memory_use - 1,
        }
        self.memory_uses[use_id] = record
        self.save()
        return record

    def memory_use_supports(self, revision_id, option, quote, question_number):
        revision_id = _revision_key(revision_id)
        if self.review_stale(revision_id):
            return False, "prior memory has been superseded", None
        rows = [row for row in self.memory_uses.values()
                if row.get("revision_id") == revision_id
                and int(row.get("question_number") or 0) == int(question_number)
                and row.get("action") == "use"
                and row.get("outcome") == "accept"]
        if not rows:
            return False, "prior revision has no accepted USE for this question", None
        record = max(rows, key=lambda row: int(row.get("sequence") or 0))
        if record.get("option") != str(option or "").upper():
            return False, "submitted option differs from the USE option", record
        if _norm(quote) != _norm(record.get("quote")):
            return False, "submitted quote differs from the USE quote", record
        revision = self.revisions.get(revision_id) or {}
        if (int(record.get("frames") or 0) <= 0
                or record.get("frames_from_store") is not True
                or int(record.get("stored_frames") or 0) <= 0
                or revision.get("frame_status") != "ok"
                or not revision.get("frame_handle")):
            return False, "USE is not backed by the target stored frames", record
        return True, "accepted current-question USE supports the option", record

    def memory_card_supports(self, card, option, quote, question_number,
                             options=None, require_check=False):
        card_id = str(card.get("card_id") or "")
        option = str(option or "").upper()
        quote = str(quote or "")
        if not card_id:
            return False, "memory card has no id", None
        if options is not None and not option_is_valid(option, options):
            return False, "memory card option is outside this question", None
        if (not require_check
                and card.get("check_status") == "accepted_review"
                and str(card.get("option") or "").upper() == option
                and _norm(card.get("quote")) == _norm(quote)):
            return True, "accepted review supports the card option", None
        revision_id = str(card.get("source_revision_id") or "")
        rows = [
            row for row in self.memory_uses.values()
            if str(row.get("card_id") or "") == card_id
            and str(row.get("revision_id") or "") == revision_id
            and int(row.get("question_number") or 0) == int(question_number)
            and row.get("action") == "check"
            and row.get("outcome") == "accept"]
        if not rows:
            return False, "memory card has no accepted CHECK for this question", None
        record = max(rows, key=lambda row: int(row.get("sequence") or 0))
        if str(record.get("option") or "").upper() != option:
            return False, "submitted option differs from the checked card option", record
        if _norm(record.get("quote")) != _norm(quote):
            return False, "submitted quote differs from the checked card quote", record
        if (int(record.get("frames") or 0) <= 0
                or record.get("frames_from_store") is not True):
            return False, "memory card CHECK is not frame-backed", record
        return True, "accepted CHECK supports the card option", record

    def review_stale(self, revision_id):
        revision = self.revisions[_revision_key(revision_id)]
        hidden = self.superseded_ids()
        return any(event_id in hidden for event_id in revision.get("events") or ())

    def can_submit(self, revision_id, option, quote, question_number=None):
        revision = self.revisions.get(_revision_key(revision_id))
        if revision is None:
            return False, "revision does not exist"
        if self.review_stale(revision_id):
            return False, "revision has been superseded"
        revision_question = int(revision.get("question_number") or 0)
        if question_number is not None and revision_question != int(question_number):
            ok, why, _record = self.memory_use_supports(
                revision_id, option, quote, question_number)
            return ok, why
        review = self.effective_review(revision_id, option)
        if review is None or review["outcome"] != "accept":
            return False, "revision has no accepted review for the option"
        if (revision.get("frame_status") != "ok"
                or not revision.get("frame_handle")
                or int(review.get("stored_frames") or 0) <= 0
                or review.get("frames_from_store") is not True):
            return False, "review is not frame-backed"
        if _norm(quote) != _norm(review["review_quote"]):
            return False, "submitted quote differs from the reviewed quote"
        return True, "accepted review supports the option"

    def current_lines(self, question_time=None, branch=None, limit=GLOBAL_TIMELINE_CHARS):
        lines, omitted, _chars = self.current_view(
            question_time=question_time, branch=branch, limit=limit)
        return lines, omitted

    def current_view(self, question_time=None, branch=None,
                     limit=GLOBAL_TIMELINE_CHARS, question_number=None):
        rows = self.current_events(branch, question_number=question_number)
        if question_time:
            qmid = (float(question_time[0]) + float(question_time[1])) / 2.0
            rows = sorted(rows, key=lambda row: (
                0 if min(row["end"], question_time[1]) - max(
                    row["start"], question_time[0]) >= 1.0 else 1,
                abs((row["start"] + row["end"]) / 2.0 - qmid),
                row["start"], row["end"], int(row["event_id"][1:])))
        lines, used = [], 0
        all_lines = []
        for row in rows:
            line = "%s %s %g-%gs %s" % (
                row["branch_id"], row["event_id"], row["start"], row["end"],
                row["text"])
            all_lines.append(line)
            if used + len(line) + 1 > limit:
                continue
            lines.append(line)
            used += len(line) + 1
        omitted = len(rows) - len(lines)
        omitted_chars = sum(len(line) + 1 for line in all_lines) - used
        lines = sorted(lines, key=lambda line: (
            self.events[line.split()[1]]["start"],
            self.events[line.split()[1]]["end"],
            int(line.split()[1][1:])))
        return lines, omitted, max(0, omitted_chars)

    def render_global(self, question_time=None, include_local_evidence=True):
        lines, omitted, omitted_chars = self.current_view(
            question_time=question_time, limit=GLOBAL_TIMELINE_CHARS - 80)
        claims = self.active_global_claims()
        if not claims:
            evidence = (("\n".join(lines) or "(empty)")
                        if include_local_evidence else "(empty)")
            text = "Global time tree:\n" + evidence
            if omitted and include_local_evidence:
                text += "\n[%d events, %d characters omitted]" % (
                    omitted, omitted_chars)
            return text
        claim_lines = []
        used = 0
        for claim in claims:
            line = "%s R%s %s %g-%gs %s" % (
                claim["claim_id"], claim["source_revision_id"][1:],
                claim["source_branch"], claim["time_span"][0],
                claim["time_span"][1], claim["event_text"])
            line_bytes = len(line.encode("utf-8")) + 1
            if used + line_bytes > GLOBAL_VIEW_BYTES:
                continue
            claim_lines.append(line)
            used += line_bytes
        text = "Global claims:\n" + ("\n".join(claim_lines) or "(empty)")
        if include_local_evidence:
            text += ("\nLocal evidence timeline (not global claims):\n" +
                     ("\n".join(lines) or "(empty)"))
        if len(claim_lines) < len(claims):
            text += "\n[%d global claims omitted by view budget]" % (
                len(claims) - len(claim_lines))
        if omitted and include_local_evidence:
            text += "\n[%d events, %d characters omitted]" % (omitted, omitted_chars)
        return text

    def render_branch(self, branch, question_time=None, question_number=None):
        summary = self.latest_summary(
            branch, question_number=question_number)["text"]
        shown = summary[:BRANCH_SUMMARY_CHARS]
        if len(summary) > BRANCH_SUMMARY_CHARS:
            shown += "\n[%d summary characters omitted]" % (
                len(summary) - BRANCH_SUMMARY_CHARS)
        event_budget = max(0, min(BRANCH_HISTORY_CHARS,
                                  OPEN_BRANCH_CHARS - len(shown) - 80))
        lines, omitted, omitted_chars = self.current_view(
            question_time=question_time, branch=branch,
            limit=event_budget, question_number=question_number)
        text = "Branch B%d:\n%s\n%s" % (
            int(branch), shown or "(no successful summary yet)",
            "\n".join(lines) or "(no current events)")
        if omitted:
            text += "\n[%d events, %d characters omitted]" % (omitted, omitted_chars)
        return text

    def to_dict(self):
        return {"schema_version": SCHEMA_VERSION, "video": self.video,
                "duration": float(self.duration),
                "revisions": copy.deepcopy(self.revisions),
                "events": copy.deepcopy(self.events),
                "reviews": copy.deepcopy(self.reviews),
                "memory_uses": copy.deepcopy(self.memory_uses),
                "questions": copy.deepcopy(self.questions),
                "branches": copy.deepcopy(self.branches),
                "global_claims": copy.deepcopy(self.global_claims),
                "global_operations": copy.deepcopy(self.global_operations),
                "next_revision": self.next_revision,
                "next_event": self.next_event, "next_review": self.next_review,
                "next_memory_use": self.next_memory_use,
                "next_global_claim": self.next_global_claim,
                "next_global_operation": self.next_global_operation}

    def save(self):
        if self.path:
            TREE.write_json(self.path, self.to_dict())


def _reader_prompt(tree, branch, first, last, fps, n, question,
                   question_number=0, memory_visible=True,
                   local_memory_visible=None,
                   reader_policy=False, reader_schema_version=1,
                   reader_allow_keep=True):
    if local_memory_visible is None:
        local_memory_visible = memory_visible
    if not memory_visible:
        history, omitted, omitted_chars = [], 0, 0
        full_summary = ""
        visible_events = []
    else:
        visible_question = None if local_memory_visible else question_number
        history, omitted, omitted_chars = tree.current_view(
            question_time=(first, last), branch=branch,
            limit=BRANCH_HISTORY_CHARS,
            question_number=visible_question)
        full_summary = tree.latest_summary(
            branch, question_number=visible_question)["text"]
        visible_events = tree.current_events(
            branch, question_number=visible_question)
    summary = full_summary[:BRANCH_SUMMARY_CHARS]
    if len(full_summary) > BRANCH_SUMMARY_CHARS:
        summary += "\n[%d summary characters omitted]" % (
            len(full_summary) - BRANCH_SUMMARY_CHARS)
    if omitted:
        history.append("[%d older events, %d characters omitted]"
                       % (omitted, omitted_chars))
    current_event_ids = [str(event.get("event_id") or "")
                         for event in visible_events]
    matching_events = [
        event for event in visible_events
        if abs(float(event.get("start") or 0.0) - float(first)) <= 1e-6
        and abs(float(event.get("end") or 0.0) - float(last)) <= 1e-6]
    matching_event_id = (
        str(matching_events[-1].get("event_id") or "")
        if matching_events else "")
    if not reader_policy and int(reader_schema_version) == 2:
        return """You are reader W{branch}, responsible only for branch B{branch}.

Current branch memory:
{summary}

Current events from this branch:
{history}

Question: {question}
Read seconds {first:g} to {last:g} at {fps:g} frames per second ({n} frames).
Always describe the visible people, objects and actions, even when they do not answer
the question. Do not write none or no previous summary when the frames show content.
Use collective descriptions. Do not enumerate people as first, second, third, and so on.
Return exactly this field sequence, with no angle brackets:
BRANCH_SUMMARY: A short summary of the visible people, objects and actions.
CAN_ANSWER: NO
ANSWER: NONE
QUOTE: NONE
EVENT_TEXT: one concrete fact from these frames

Replace the example EVENT_TEXT with one fact you actually saw. Write EVENT_TEXT: NONE
only when the frames add no new concrete fact. Do not output timestamps. The harness
records one event at the full read window [{first:g}, {last:g}].
Keep BRANCH_SUMMARY at most {summary_words} words. Stop after EVENT_TEXT. Do not list
frames, repeat fields, or add closing text. Do not batch another action.""" .format(
            branch=int(branch), summary=summary or "(no previous summary)",
            history="\n".join(history) or "(none)", question=question,
            first=first, last=last, fps=fps, n=n,
            summary_words=READER_SUMMARY_MAX_WORDS)
    if not reader_policy:
        return """You are reader W{branch}, responsible only for branch B{branch}.

Current branch memory:
{summary}

Current events from this branch:
{history}

Question: {question}
Read seconds {first} to {last} at {fps:g} frames per second ({n} frames).
Always describe the visible people, objects and actions, even when they do not answer
the question. Do not write none or no previous summary when the frames show content.
Use collective descriptions. Do not enumerate people as first, second, third, and so on.
Return exactly this shape, in order:
BRANCH_SUMMARY: <updated branch memory; at most {summary_words} words; do not list every frame>
CAN_ANSWER: YES | NO | POINTS
ANSWER: A-H or NONE
QUOTE: 3-8 words from your summary, or NONE
EVENT: {example_start:g}-{example_end:g}s | one concrete fact from these frames
Replace "one concrete fact from these frames" with what you saw.

Write exactly one EVENT line. Use SUPERSEDES only for an old event in this branch
that your new observation corrects. Keep the complete reply compact enough to
finish within the reply-token budget: put the summary first, then every required
field. The event text after | is required; do not output time only or use t=.
Stop immediately after that line. Do not list frames, repeat event lines, or add
closing text. Do not batch another action.""" .format(
            branch=int(branch), summary=summary or "(no previous summary)",
            history="\n".join(history) or "(none)",
            question=question, first=first, last=last, fps=fps, n=n,
            example_start=first,
            example_end=min(float(last), float(first) + 4.0),
            summary_words=READER_SUMMARY_MAX_WORDS)
    if reader_policy and int(reader_schema_version) == 2:
        if matching_event_id and not reader_allow_keep:
            operation_rule = (
                "The current window already has event %s. Do not use KEEP in "
                "this anchored diagnostic. Use ADD only if this read produces "
                "a distinct concrete event, or SUPERSEDE %s only if correcting "
                "it." % (matching_event_id, matching_event_id))
            default_operation = "ADD"
            sample_summary = (
                "A short summary of the visible people, objects and actions.")
            sample_can_answer = "NO"
            sample_answer = "NONE"
        elif matching_event_id:
            current_event_text = next((
                str(event.get("text") or "") for event in reversed(matching_events)
                if str(event.get("event_id") or "") == matching_event_id), "")
            operation_rule = (
                "The current read window is already covered by %s: %s. If this "
                "read confirms the same evidence, use KEEP. For KEEP, write "
                "BRANCH_SUMMARY: S, CAN_ANSWER: S, ANSWER: S, "
                "QUOTE: NONE, and EVENT_TEXT: NONE. If it corrects %s, use "
                "SUPERSEDE %s. If it adds distinct evidence, use ADD with a new "
                "concrete EVENT_TEXT that is different from the existing event "
                "text."
                % (matching_event_id, current_event_text or "(empty event text)",
                   matching_event_id, matching_event_id))
            default_operation = "KEEP"
            sample_summary = "S"
            sample_can_answer = "S"
            sample_answer = "S"
            sample_event_text = "NONE"
        else:
            operation_rule = (
                "No current event covers this read window. Use ADD for new "
                "grounded evidence.")
            default_operation = "ADD"
            sample_summary = (
                "A short summary of the visible people, objects and actions.")
            sample_can_answer = "NO"
            sample_answer = "NONE"
        keep_instruction = (
            "Use KEEP only when no new grounded event should be written; KEEP "
            "requires BRANCH_SUMMARY: S, CAN_ANSWER: S, ANSWER: S, QUOTE: NONE "
            "and EVENT_TEXT: NONE. Use SUPERSEDE and one visible event id only "
            "when correcting a visible event. If MEMORY_OP is KEEP, "
            "BRANCH_SUMMARY, CAN_ANSWER and ANSWER MUST be S."
            if reader_allow_keep else
            "Do not output KEEP in this anchored diagnostic.")
        return """You are reader W{branch}, responsible only for branch B{branch}.

Current branch memory:
{summary}

Current events from this branch:
{history}

Question: {question}
Read seconds {first} to {last} at {fps:g} frames per second ({n} frames).
Always describe the visible people, objects and actions, even when they do not answer
the question. Do not write none or no previous summary when the frames show content.
Use collective descriptions. Do not enumerate people as first, second, third, and so on.
If these frames support one answer option, use CAN_ANSWER: YES, give its A-H letter,
and quote a short supporting phrase from your BRANCH_SUMMARY. If they do not support an
option, use CAN_ANSWER: NO with ANSWER: NONE. Do not default to NO without checking.
Return exactly this plain field sequence, with no angle brackets:
MEMORY_OP: {default_operation}
BRANCH_SUMMARY: {sample_summary}
CAN_ANSWER: {sample_can_answer}
ANSWER: {sample_answer}
QUOTE: NONE

The harness will derive the event text from BRANCH_SUMMARY and record it at the
exact read window [{first:g}, {last:g}]. Do not output EVENT_TEXT.
Do not output timestamps. {operation_rule}
{keep_instruction}
Visible event ids now: {current_event_ids}.
Write exactly one MEMORY_OP line. Keep BRANCH_SUMMARY at most {summary_words}
words. Stop after EVENT_TEXT. Do not repeat or add closing text.""" .format(
            branch=int(branch), summary=summary or "(no previous summary)",
            history="\n".join(history) or "(none)",
            question=question, first=first, last=last, fps=fps, n=n,
            summary_words=READER_SUMMARY_MAX_WORDS,
            current_event_ids=", ".join(current_event_ids) or "(none)",
            default_operation=default_operation,
            sample_summary=sample_summary,
            sample_can_answer=sample_can_answer,
            sample_answer=sample_answer,
            operation_rule=operation_rule,
            keep_instruction=keep_instruction)

    if reader_policy and int(reader_schema_version) == 3:
        if matching_event_id and not reader_allow_keep:
            operation_rule = (
                "The current window already has event %s. Do not use KEEP in "
                "this anchored diagnostic. Use ADD for a distinct concrete "
                "claim or SUPERSEDE %s when correcting it."
                % (matching_event_id, matching_event_id))
            default_operation = "ADD"
            sample_summary = "A short summary of the visible people and actions."
            sample_claim = "FACT"
            sample_start = "START_SECONDS"
            sample_end = "END_SECONDS"
            sample_can_answer = "NO"
            sample_answer = "NONE"
        elif matching_event_id:
            current_event_text = next((
                str(event.get("text") or "") for event in reversed(matching_events)
                if str(event.get("event_id") or "") == matching_event_id), "")
            operation_rule = (
                "The current window is already covered by %s: %s. Use KEEP "
                "when it adds no durable fact; use SUPERSEDE %s with one "
                "corrected concrete claim; use ADD only for distinct evidence."
                % (matching_event_id, current_event_text or "(empty)",
                   matching_event_id))
            default_operation = "KEEP"
            sample_summary = "S"
            sample_claim = "NONE"
            sample_start = "NONE"
            sample_end = "NONE"
            sample_can_answer = "S"
            sample_answer = "S"
        else:
            operation_rule = (
                "No current event covers this read window. Use ADD for one "
                "grounded fact.")
            default_operation = "ADD"
            sample_summary = "A short summary of the visible people and actions."
            sample_claim = "FACT"
            sample_start = "START_SECONDS"
            sample_end = "END_SECONDS"
            sample_can_answer = "NO"
            sample_answer = "NONE"
        return """You are reader W{branch}, responsible only for branch B{branch}.

Current branch memory:
{summary}

Current events from this branch:
{history}

Question: {question}
Read seconds {first:g} to {last:g} at {fps:g} frames per second ({n} frames).
Describe only what the frames show. Do not use world knowledge.
Return exactly this field sequence, with no angle brackets:
MEMORY_OP: {default_operation}
BRANCH_SUMMARY: {sample_summary}
CLAIM: {sample_claim}
START: {sample_start}
END: {sample_end}
CAN_ANSWER: {sample_can_answer}
ANSWER: {sample_answer}
QUOTE: NONE

For ADD or SUPERSEDE, CLAIM must be one concrete entity-attribute-value fact,
START and END must be numeric seconds inside [{first:g}, {last:g}], and START
must be before END. For KEEP, CLAIM, START and END must all be NONE.
{operation_rule}
Visible event ids now: {current_event_ids}.
Write exactly one MEMORY_OP line and stop after QUOTE. Keep BRANCH_SUMMARY at
most {summary_words} words.""" .format(
            branch=int(branch), summary=summary or "(no previous summary)",
            history="\n".join(history) or "(none)", question=question,
            first=first, last=last, fps=fps, n=n,
            summary_words=READER_SUMMARY_MAX_WORDS,
            current_event_ids=", ".join(current_event_ids) or "(none)",
            default_operation=default_operation,
            sample_summary=sample_summary, sample_claim=sample_claim,
            sample_start=sample_start, sample_end=sample_end,
            sample_can_answer=sample_can_answer,
            sample_answer=sample_answer, operation_rule=operation_rule)

    return """You are reader W{branch}, responsible only for branch B{branch}.

Current branch memory:
{summary}

Current events from this branch:
{history}

Question: {question}
Read seconds {first} to {last} at {fps:g} frames per second ({n} frames).
Always describe the visible people, objects and actions, even when they do not answer
the question. Do not write none or no previous summary when the frames show content.
Use collective descriptions. Do not enumerate people as first, second, third, and so on.
Return exactly this plain field sequence, with no angle brackets:
{operation_help}
BRANCH_SUMMARY: A short summary of the visible people, objects and actions.
CAN_ANSWER: NO
ANSWER: NONE
QUOTE: NONE
EVENT: {example_start:g}-{example_end:g}s | one concrete fact from these frames
Replace "one concrete fact from these frames" with what you saw.

The example above uses ADD. Keep the field names exactly as written. Put the
actual updated summary after BRANCH_SUMMARY and use at most {summary_words} words.
Replace ADD with KEEP only when no new grounded event should be written. Replace
ADD with SUPERSEDE, a space, and one visible event id such as E1, only when
correcting a visible event in this branch.
Write exactly one MEMORY_OP line and do not list operation alternatives.
- ADD requires one EVENT.
- KEEP requires EVENT: NONE.
- SUPERSEDE requires one EVENT and a visible event id.
Visible event ids now: {current_event_ids}.
The event text after | is required; do not output time only or use t=. Stop
immediately after that line. Do not list frames, repeat event lines, or add
closing text. Do not batch another action.""" .format(
        branch=int(branch), summary=summary or "(no previous summary)",
        history="\n".join(history) or "(none)",
        question=question, first=first, last=last, fps=fps, n=n,
        example_start=first,
        example_end=min(float(last), float(first) + 4.0),
        summary_words=READER_SUMMARY_MAX_WORDS,
        current_event_ids=", ".join(current_event_ids) or "(none)",
        operation_help="MEMORY_OP: ADD")


def _review_prompt(tree, revision, question, question_time=None,
                   memory_visible=True, local_memory_visible=None,
                   question_number=None):
    if local_memory_visible is None:
        local_memory_visible = memory_visible
    branch = int(revision["branch_id"][1:])
    if question_number is None:
        question_number = int(revision.get("question_number") or 0)
    if not memory_visible:
        branch_text = "(memory ablation: hidden)"
    else:
        branch_text = tree.render_branch(
            branch, question_time=question_time,
            question_number=(None if local_memory_visible else question_number))
    summary = revision["branch_summary"]
    shown = summary[:BRANCH_SUMMARY_CHARS]
    if len(summary) > BRANCH_SUMMARY_CHARS:
        shown += "\n[%d summary characters omitted]" % (
            len(summary) - BRANCH_SUMMARY_CHARS)
    return """You are O. Review one answer-bearing report before it may support an answer.

Branch memory:
{branch}

Read revision {revision_id}:
Summary: {summary}
CAN_ANSWER: {can}
ANSWER: {answer}
QUOTE: {quote}

Question: {question}
Reply with exactly one line beginning REVIEW:. Do not output WHY, analysis,
explanation, or any other line. Stop after the REVIEW line.
Decide from the frames, not from the report text alone. On ACCEPT, the harness
requires one A-H option and one quote visible in the report or frames.
REVIEW: ACCEPT | {revision_id} | <A-H option letter> | <quote>
REVIEW: REJECT | {revision_id} | <reason>
REVIEW: UNSURE | {revision_id} | <reason>""".format(
        branch=branch_text,
        revision_id=revision["revision_id"],
        summary=shown, can=revision["can_answer"],
        answer=revision["answer"], quote=revision["quote"], question=question)


def _review_repair_prompt(tree, revision, question, error, question_time=None,
                          memory_visible=True, local_memory_visible=None,
                          question_number=None):
    base = _review_prompt(
        tree, revision, question, question_time=question_time,
        memory_visible=memory_visible,
        local_memory_visible=local_memory_visible,
        question_number=question_number)
    return ("""Your previous REVIEW reply was unusable: {error}
Return exactly one complete line and nothing else. Keep the required revision id.
Use: REVIEW: ACCEPT | R{revision} | <A-H option letter> | <quote>
Or: REVIEW: REJECT | R{revision} | <reason>
Or: REVIEW: UNSURE | R{revision} | <reason>

{base}""".format(error=str(error), revision=revision["revision_id"][1:],
                  base=base))


def _memory_use_prompt(tree, candidate, question, local_memory_visible=True):
    """Frozen checker prompt after O chooses USE for one prior memory."""
    revision = tree.revisions[candidate["revision_id"]]
    event = tree.events[candidate["event_id"]]
    summary = str(revision.get("branch_summary") or "")
    can_answer = str(revision.get("can_answer") or "")
    answer = str(revision.get("answer") or "")
    quote = str(revision.get("quote") or "")
    if not local_memory_visible:
        summary = ""
        can_answer = "hidden (global claim only)"
        answer = "hidden (global claim only)"
        quote = "hidden (global claim only)"
    shown = summary[:BRANCH_SUMMARY_CHARS]
    if len(summary) > BRANCH_SUMMARY_CHARS:
        shown += "\n[%d summary characters omitted]" % (
            len(summary) - BRANCH_SUMMARY_CHARS)
    options = C.options_of(question)
    option_text = "\n".join("(%s) %s" % (letter, value)
                            for letter, value in zip("ABCDEFGH", options))
    return """You are the frozen memory checker. Check whether one prior memory can
support one option for the current question. Use the stored frames. Do not answer
from the prior option by assumption.

Current question:
{question}

Current options:
{options}

Prior memory {revision_id} from question {source_question}:
Branch: {branch}
Time: {start}-{end}s
Summary: {summary}
CAN_ANSWER: {can_answer}
ANSWER: {answer}
QUOTE: {quote}
Event: {event}

You may choose an option different from the prior question's option. A quoted
statement must appear in the summary, reader quote or event shown above.
Reply with exactly one line and no explanation. Use one of these exact forms:
REVIEW: ACCEPT | {revision_id} | <A-H option letter> | <quote>
REVIEW: REJECT | {revision_id} | <reason>
REVIEW: UNSURE | {revision_id} | <reason>""".format(
        question=question, options=option_text,
        revision_id=revision["revision_id"],
        source_question=candidate["source_question_number"],
        branch=revision.get("branch_id") or "", start=_seconds(event["start"]),
        end=_seconds(event["end"]), summary=shown,
        can_answer=can_answer, answer=answer, quote=quote,
        event=event.get("text") or "")


def _card_check_prompt(card, question):
    options = C.options_of(question)
    option_text = "\n".join(
        "(%s) %s" % (letter, value)
        for letter, value in zip("ABCDEFGH", options))
    revision_id = str(card.get("source_revision_id") or "")
    return """You are the frozen memory checker. Inspect the stored frames and answer
only from visible evidence. The remembered claim is intentionally hidden.

Current question:
{question}

Current options:
{options}

Return exactly one line and no explanation:
REVIEW: ACCEPT | {revision_id} | <A-H option letter> | <visible quote>
REVIEW: REJECT | {revision_id} | <reason>
REVIEW: UNSURE | {revision_id} | <reason>

An ACCEPT must name one current option and quote a short span of concrete
visible evidence. Use visible words such as on-screen text, a named object,
clothing, or an action. Prefer concrete visible wording. Do not
merely repeat the option without card evidence, and do not explain why. If no
card evidence is visible, return REJECT or UNSURE.

Example form only:
REVIEW: ACCEPT | R1 | B | "goalkeeper extends right glove"
""".format(
        question=question, options=option_text, revision_id=revision_id)


def _memory_use_repair_prompt(prompt, error, revision=None):
    named = ("Name revision R%d in the REVIEW line; do not name another.\n"
             % int(revision) if revision is not None else "")
    return ("""Your previous memory-check reply was unusable: %s
%sReturn exactly one complete REVIEW line. For ACCEPT, use an A-H option letter
from Current options and quote concrete visible words from the frames. Do not
merely repeat the option without card evidence. If no concrete evidence is
visible, return REJECT or UNSURE.

%s""" % (str(error), named, prompt))


def _empty_branch(tree, branch):
    if (branch is not None and not tree.latest_summary(branch)["text"]
            and not tree.current_events(branch)):
        return branch
    return None


def _seconds(value):
    text = repr(float(value))
    return text[:-2] if text.endswith(".0") else text


def turn_cap(branches):
    """OPEN, one retry and one review per branch, then a terminal action."""
    return max(MIN_TURNS, 3 * int(branches) + 2)


def _review_applies(tree, row, question_number):
    review_question = int(row.get("question_number") or 0)
    return review_question == int(question_number)


def _review_pair_applies(tree, review, question_number):
    revision = tree.revisions.get(str(review.get("revision_id") or ""))
    return bool(revision is not None
                and _review_applies(tree, review, question_number)
                and _review_applies(tree, revision, question_number))


def _blocked_branches(tree, question_number):
    """Branches permanently closed for this question after two non-ACCEPTs."""
    counts = {}
    for review in tree.reviews.values():
        if int(review.get("question_number") or 0) != int(question_number):
            continue
        if review.get("outcome") not in ("reject", "unsure"):
            continue
        revision = tree.revisions.get(str(review.get("revision_id") or ""))
        if revision is None:
            continue
        branch = int(str(revision.get("branch_id") or "B0")[1:])
        counts[branch] = counts.get(branch, 0) + 1
    return sorted(branch for branch, count in counts.items()
                  if count >= MAX_NON_ACCEPT_REVIEWS_PER_BRANCH)


def _turn_context(tree, branches, question_number, open_branch, frames_left=None,
                  calls_left=None, memory_visible=True, allow_read_window=True,
                  local_memory_visible=None,
                  memory_use=False, allow_reread=False, global_memory=False,
                  force_answer_stage=False, question_time=None,
                  read_steer=None,
                  memory_decision_timing="pre_read",
                  max_reads_per_question=None, budget_answer=False,
                  force_budget_answer=False, time_left_seconds=None,
                  reader_min_runtime_seconds=None,
                  policy_turn_reserve_seconds=POLICY_TURN_RESERVE_SECONDS,
                  final_answer_reserve_seconds=FINAL_ANSWER_RESERVE_SECONDS):
    if memory_decision_timing not in MEMORY_DECISION_TIMINGS:
        raise ValueError("unknown memory decision timing: %s"
                         % memory_decision_timing)
    if local_memory_visible is None:
        local_memory_visible = memory_visible

    ever_read = {int(row["branch_id"][1:]) for row in tree.revisions.values()
                 if row.get("read_status") == "success"}

    def branch_priority(branch):
        start, end = branches[branch]
        if read_steer is not None:
            # Steered: a chunk the question's clock cue points at first, then
            # chunks never read in this video, then chunk order.
            cued = [cue for cue in read_steer
                    if min(end, float(cue[1])) - max(start, float(cue[0]))
                    >= 1.0]
            near = min((abs((start + end) / 2.0
                            - (float(cue[0]) + float(cue[1])) / 2.0)
                        for cue in read_steer), default=0.0)
            return (0 if cued else 1, int(branch in ever_read), near, branch)
        if not question_time:
            return (1, branch)
        midpoint = (float(question_time[0]) + float(question_time[1])) / 2.0
        overlap = min(end, float(question_time[1])) - max(
            start, float(question_time[0]))
        distance = abs((start + end) / 2.0 - midpoint)
        return (0 if overlap >= 1.0 else 1, distance, branch)

    effective = [row for row in tree.reviews.values()
                 if tree.effective_review(row["revision_id"], row["option"]) is row
                 and (local_memory_visible or int(row.get("question_number") or 0)
                      == int(question_number))]
    accepted = [row for row in effective if row["outcome"] == "accept"
                and not tree.review_stale(row["revision_id"])
                and _review_pair_applies(tree, row, question_number)]
    accepted_uses = []
    for row in tree.memory_uses.values():
        if row.get("action") != "use" or row.get("outcome") != "accept" \
                or int(row.get("question_number") or 0) != int(question_number) \
                or tree.review_stale(row.get("revision_id")):
            continue
        accepted_uses.append({
            "review_id": row["use_id"],
            "revision_id": row["revision_id"],
            "option": row.get("option") or "",
            "outcome": "accept",
            "review_quote": row.get("quote") or "",
            "sequence": int(row.get("sequence") or 0),
            "memory_use": True,
        })
    accepted.extend(accepted_uses)
    accepted_card_checks = []
    for row in tree.memory_uses.values():
        scope = str(row.get("scope") or "")
        revision_id = str(row.get("revision_id") or "")
        revision = tree.revisions.get(revision_id) or {}
        if (row.get("action") != "check" or row.get("outcome") != "accept"
                or int(row.get("question_number") or 0) != int(question_number)
                or int(row.get("frames") or 0) <= 0
                or row.get("frames_from_store") is not True
                or int(row.get("stored_frames") or 0) <= 0
                or not row.get("option") or not row.get("quote")
                or scope not in ("local", "global")
                or not revision
                or revision.get("read_status") != "success"
                or revision.get("parse_status") != "ok"
                or revision.get("frame_status") != "ok"
                or (scope == "local" and tree.review_stale(revision_id))):
            continue
        accepted_card_checks.append(dict(row))
    accepted_card_checks.sort(key=lambda row: int(row.get("sequence") or 0))
    terminal_evidence = bool(accepted or accepted_card_checks)
    current = [row for row in tree.revisions.values()
               if int(row.get("question_number") or 0) == int(question_number)]
    current_answers = sorted(
        (row["revision_id"] for row in current
         if answer_bearing(row) and not tree.review_stale(row["revision_id"])),
        key=lambda value: int(value[1:]))
    reviewed = {row["revision_id"] for row in effective}
    latest_answer = current_answers[-1] if current_answers else ""
    unreviewed_answers = ([latest_answer]
                          if latest_answer and latest_answer not in reviewed else [])
    review_frames = 0
    if unreviewed_answers:
        revision = tree.revisions.get(unreviewed_answers[-1]) or {}
        review_frames = min(int((revision.get("frame_handle") or {}).get("n") or 0),
                            REVIEW_FRAMES)
    review_fits = frames_left is None or review_frames <= int(frames_left)
    read_branches = {int(row["branch_id"][1:]) for row in current
                     if row.get("read_status") == "success"}
    current_question_read = bool(read_branches)
    read_attempt_counts = {}
    for row in current:
        branch = int(row["branch_id"][1:])
        read_attempt_counts[branch] = read_attempt_counts.get(branch, 0) + 1
    blocked_branches = _blocked_branches(tree, question_number)
    exhausted_branches = sorted(
        branch for branch, count in read_attempt_counts.items()
        if count >= MAX_READS_PER_BRANCH)
    ranked_openable = sorted(
        [branch for branch in branches
         if branch not in read_branches
         and branch not in blocked_branches],
        key=branch_priority)
    unread_branches = [branch for branch in ranked_openable
                       if branch not in read_branches
                       and branch not in blocked_branches]
    open_needs_read = (open_branch is not None and open_branch not in read_branches)
    empty_branch = _empty_branch(tree, open_branch)
    read_time_fits = bool(
        time_left_seconds is None
        or reader_min_runtime_seconds is None
        or float(time_left_seconds) >= (
            float(reader_min_runtime_seconds)
            + float(policy_turn_reserve_seconds or 0.0)
            + float(final_answer_reserve_seconds or 0.0)))
    read_call_fits = bool(calls_left is None or int(calls_left) >= 3)
    open_call_fits = bool(calls_left is None or int(calls_left) >= 4)

    def read_fits(branch):
        if not read_time_fits or not read_call_fits:
            return False
        if frames_left is None:
            return True
        first, last = branches[branch]
        return int(frames_left) >= max(1, round((last - first) * READ_FPS))

    read_limit_reached = bool(
        max_reads_per_question is not None
        and sum(read_attempt_counts.values()) >= int(max_reads_per_question))
    readable_branches = [] if read_limit_reached else [
        branch for branch in unread_branches
        if branch not in blocked_branches
        and int(read_attempt_counts.get(branch, 0)) < MAX_READS_PER_BRANCH
        and read_fits(branch)]
    if allow_reread and not read_limit_reached:
        readable_branches.extend(
            branch for branch in sorted(branches, key=branch_priority)
            if branch not in readable_branches
            and branch not in blocked_branches
            and int(read_attempt_counts.get(branch, 0)) > 0
            and int(read_attempt_counts.get(branch, 0)) < MAX_READS_PER_BRANCH
            and read_fits(branch))
    open_readable = bool(
        not read_limit_reached
        and open_needs_read
        and open_branch not in blocked_branches
        and int(read_attempt_counts.get(open_branch, 0)) < MAX_READS_PER_BRANCH
        and read_fits(open_branch))
    can_read = open_readable or bool(readable_branches)
    call_budget_exhausted = bool(calls_left is not None
                                 and 0 < int(calls_left) <= 2)
    terminal_required = bool(
        (current or terminal_evidence) and call_budget_exhausted)
    call_abstain = bool(call_budget_exhausted and not terminal_evidence)
    frame_abstain = bool(current and not terminal_evidence and (
        (unreviewed_answers and not review_fits)
        or (unread_branches and not can_read and not read_limit_reached)))
    budget_abstain = call_abstain or frame_abstain
    empty_blocks = empty_branch is not None and can_read and not budget_abstain
    review_required = (unreviewed_answers[-1]
                       if unreviewed_answers and review_fits and not empty_blocks
                       and not budget_abstain
                       and not terminal_evidence
                       and not (budget_answer and terminal_required) else "")
    terminal_available = bool(current or terminal_evidence) and not empty_blocks
    current_read_required = bool(
        accepted and not current_question_read and can_read)
    submit_available = bool(
        ((accepted and current_question_read) or accepted_card_checks)
        and not review_required and not empty_blocks)
    abstain_available = (bool(terminal_evidence)
                         or bool(current and (not unread_branches or not can_read))
                         or budget_abstain)
    abstain_available = (abstain_available and not review_required
                         and not empty_blocks and not current_read_required)
    scan_required = bool(can_read and (readable_branches or open_readable)
                         and (read_branches or open_needs_read or current)
                         and not review_required and not terminal_evidence
                         and not empty_blocks and not budget_abstain)
    exhausted_scan = bool(budget_abstain and read_branches and not readable_branches)
    readable_branches = sorted(readable_branches, key=branch_priority)
    unread_readable = [branch for branch in readable_branches
                       if branch in unread_branches]
    next_read_branch = (open_branch if open_readable
                        else unread_readable[0] if unread_readable
                        else readable_branches[0] if readable_branches else None)
    openable_branches = [
        branch for branch in sorted(branches, key=branch_priority)
        if branch not in exhausted_branches
        and branch not in blocked_branches
        and read_time_fits and read_call_fits and open_call_fits
        and (can_read or _empty_branch(tree, branch) is None)]
    timing_ready = (
        memory_decision_timing == "pre_read"
        or (memory_decision_timing == "after_read" and current_question_read))
    candidate = (tree.prior_memory_candidate(
                     question_number, active_global_only=global_memory,
                     allow_local=local_memory_visible)
                 if (memory_visible and memory_use
                     and memory_decision_timing != "off" and timing_ready)
                 else None)
    use_frames = (min(int(candidate["stored_frames"]), MEMORY_USE_FRAMES)
                  if candidate else 0)
    use_fits = (candidate is not None
                and (frames_left is None
                     or 2 * use_frames <= int(frames_left))
                and (calls_left is None or int(calls_left) >= 4))
    if force_answer_stage:
        candidate = None
        can_read = False
        openable_branches = []
        review_required = ""
        scan_required = False
        budget_abstain = False
        empty_blocks = False
        terminal_required = True
        abstain_available = False
        submit_available = bool(
            ((accepted and current_question_read) or accepted_card_checks)
            and not empty_blocks)
        terminal_available = submit_available
    budget_answer_available = bool(force_budget_answer and budget_answer)
    if budget_answer_available:
        candidate = None
        review_required = ""
        scan_required = False
        budget_abstain = False
        terminal_required = True
        terminal_available = False
        submit_available = False
        abstain_available = False
        can_read = False
        openable_branches = []
    return {
        "effective": effective, "accepted": accepted,
        "accepted_card_checks": accepted_card_checks,
        "current_answer_revisions": current_answers,
        "unreviewed_answer_revisions": unreviewed_answers,
        "read_branches": read_branches, "unread_branches": unread_branches,
        "current_question_read": current_question_read,
        "current_read_required": current_read_required,
        "local_memory_visible": bool(local_memory_visible),
        "readable_branches": readable_branches,
        "blocked_branches": blocked_branches,
        "exhausted_branches": exhausted_branches,
        "next_read_branch": next_read_branch,
        "empty_branch": empty_branch, "review_required": review_required,
        "empty_blocks": empty_blocks,
        "terminal_available": terminal_available,
        "submit_available": submit_available,
        "abstain_available": abstain_available,
        "scan_required": scan_required, "can_read": can_read,
        "exhausted_scan": exhausted_scan,
        "terminal_required": terminal_required,
        "budget_abstain": budget_abstain,
        "openable_branches": openable_branches,
        "memory_candidate": candidate,
        "memory_use_frames": use_frames,
        "memory_use_fits": use_fits,
        "accepted_uses": accepted_uses,
        "calls_left": calls_left,
        "answer_stage": bool(force_answer_stage),
        "budget_answer_enabled": bool(budget_answer),
        "budget_answer_available": budget_answer_available,
        "read_time_fits": read_time_fits,
    }


def _required_action_matches(action, required, branch_spans=None):
    if not required:
        return True
    if required.startswith("SUBMIT ") and "@" not in required:
        return action.verb == "submit" and action.option == required.split()[1]
    if required == "ABSTAIN":
        return action.verb == "abstain"
    verb, handle = required.split()
    if verb == "READ" and action.verb == "read":
        if "B%d" % action.branch != handle:
            return False
        span = (branch_spans or {}).get(action.branch)
        return action.window is None or (
            span is not None and tuple(action.window) == tuple(span))
    if verb == "REVIEW" and action.verb == "review":
        return "R%d" % action.revision == handle
    if verb == "EVOLVE_GLOBAL" and action.verb == "evolve_global":
        return True
    return False


def _recommended_action_menu(context):
    """Return the stage name and the small preferred verb set."""
    if context.get("budget_answer_available"):
        return "answer", {"submit"}
    if context["memory_candidate"] is not None:
        verbs = {"defer"}
        if context["memory_use_fits"]:
            verbs.add("use")
        return "memory", verbs
    if context["budget_abstain"]:
        return "answer", {"abstain"}
    if context.get("accepted_card_checks"):
        return "answer", {"submit"} if context["submit_available"] else {"abstain"}
    if (context["accepted"] and not context.get("current_question_read")
            and context["can_read"]):
        return "inspect", {"read"}
    if context["review_required"]:
        return "verify", {"review"}
    if context["accepted"]:
        return "answer", {"submit"} if context["submit_available"] else {"abstain"}
    if context["can_read"] and context["read_branches"]:
        return "inspect", {"read"}
    if context["can_read"]:
        return "locate", {"read"}
    if context["openable_branches"]:
        return "locate", {"open"}
    if context["abstain_available"]:
        return "locate", {"abstain"}
    return "locate", set()


def _recommended_action_objects(context, stage, memory_menu_order="use_first"):
    """Concrete preferred targets, not just preferred verbs."""
    if memory_menu_order not in ("use_first", "defer_first"):
        raise ValueError("unknown memory menu order: %s" % memory_menu_order)
    candidate = context["memory_candidate"]
    if context.get("budget_answer_available"):
        return [Action("submit", option=option)
                for option in context.get("answer_options") or ()]
    if stage == "memory" and candidate is not None:
        use_action = (Action("use", revision=int(candidate["revision_id"][1:]))
                      if context["memory_use_fits"] else None)
        defer_action = Action(
            "defer", revision=int(candidate["revision_id"][1:]))
        if use_action is None:
            return [defer_action]
        return ([use_action, defer_action] if memory_menu_order == "use_first"
                else [defer_action, use_action])
    if context["budget_abstain"]:
        return [Action("abstain", reason="budget nearly spent")]
    if context["review_required"]:
        return [Action(
            "review", revision=int(context["review_required"][1:]))]
    if context.get("accepted_card_checks"):
        if context["submit_available"]:
            return []
        if context["abstain_available"]:
            return [Action("abstain", reason="no usable answer")]
    if context["accepted"]:
        if context["submit_available"]:
            row = sorted(context["accepted"],
                         key=lambda value: value["sequence"])[-1]
            return [Action("submit", option=row["option"],
                           revision=int(row["revision_id"][1:]),
                           quote=row["review_quote"])]
        if context["abstain_available"]:
            return [Action("abstain", reason="no usable answer")]
    if context["can_read"] and context["next_read_branch"] is not None:
        return [Action("read", branch=int(context["next_read_branch"]))]
    if context["openable_branches"]:
        return [Action("open", branch=int(context["openable_branches"][0]))]
    if context["abstain_available"]:
        return [Action("abstain", reason="no usable evidence")]
    return []


def _legal_action_verbs(context):
    """The legal verb families at this state; concrete targets stay in the prompt."""
    if context.get("budget_answer_available"):
        return {"submit"}
    if context.get("review_required"):
        return {"review"}
    legal = set()
    if context["memory_candidate"] is not None:
        legal.add("defer")
        if context["memory_use_fits"]:
            legal.add("use")
    if context["openable_branches"]:
        legal.add("open")
    if context["can_read"]:
        legal.add("read")
    if context["review_required"]:
        legal.add("review")
    if context["submit_available"]:
        legal.add("submit")
    if context["abstain_available"]:
        legal.add("abstain")
    return legal


def _action_signature(action):
    """Exact action identity used by the recommended/off-menu audit."""
    verb = str(getattr(action, "verb", "") or "")
    if verb in ("read", "open"):
        window = tuple(action.window) if action.window else None
        return verb, action.branch, window
    if verb in ("review", "use", "defer"):
        return verb, action.revision
    if verb == "check":
        return verb, action.memory_handle
    if verb == "evolve_global":
        return (verb, action.global_operation, action.global_target,
                action.revision)
    if verb == "submit":
        return verb, action.option, action.revision, action.memory_handle, action.quote
    return (verb,)


def _action_line_signature(line):
    try:
        return _action_signature(parse_action(line))
    except ActionError:
        return None


def _action_text(action):
    """Canonical exact text for one parsed action."""
    verb = action.verb
    if verb in ("read", "open"):
        text = "ACTION: %s B%d" % (verb.upper(), int(action.branch))
        if action.window:
            text += " [%g-%gs]" % (float(action.window[0]),
                                   float(action.window[1]))
        return text
    if verb in ("review", "use", "defer"):
        return "ACTION: %s R%d" % (verb.upper(), int(action.revision))
    if verb == "check":
        return "ACTION: CHECK %s" % action.memory_handle
    if verb == "evolve_global":
        operation = str(action.global_operation or "").lower()
        if operation == "keep":
            return "ACTION: EVOLVE_GLOBAL KEEP"
        if operation == "add":
            return "ACTION: EVOLVE_GLOBAL ADD R%d" % int(action.revision)
        if operation == "supersede":
            return "ACTION: EVOLVE_GLOBAL SUPERSEDE G%d R%d" % (
                int(action.global_target), int(action.revision))
    if verb == "submit":
        if not action.memory_handle and action.revision is None and not action.quote:
            return "ACTION: SUBMIT %s" % action.option
        target = (action.memory_handle if action.memory_handle else
                  "R%d" % int(action.revision))
        return "ACTION: SUBMIT %s @ %s | %s" % (
            action.option, target, json.dumps(action.quote))
    if verb == "abstain":
        return "ACTION: ABSTAIN no valid action"
    raise ActionError("action cannot be rendered: %s" % verb)


def _allowed_action_completions(prompt):
    """Return the exact legal action texts displayed in a turn prompt."""
    completions = []
    dropped = []
    for line in str(prompt or "").splitlines():
        line = line.strip()
        if not line.startswith("ACTION:"):
            continue
        command = line.split("  ", 1)[0].strip()
        try:
            action = parse_action(command)
            text = _action_text(action)
            if _action_signature(parse_action(text)) != _action_signature(action):
                raise ActionError("action text does not round-trip")
        except ActionError:
            dropped.append(line)
            continue
        if text not in completions:
            completions.append(text)
    if dropped:
        raise ActionError(
            "displayed actions cannot be constrained: %s" % "; ".join(dropped))
    return completions


def _displayed_action_signatures(prompt):
    return {
        _action_line_signature(line.strip())
        for line in str(prompt or "").splitlines()
        if line.strip().startswith("ACTION:")}


def _menu_class(prompt, action):
    return ("in_menu" if _action_signature(action)
            in _displayed_action_signatures(prompt) else "off_menu")


def _normalize_policy_reply(reply):
    """Repair only unambiguous action-format variants before strict parsing."""
    text = str(reply or "")
    text = re.sub(r"(?im)^(\s*(?:ACTION:\s*)?)ABSTAINS\s*$",
                  r"\1ABSTAIN no valid action", text)
    text = re.sub(r"(?i)\b(?:SUBMAIL|SUBMAL|SUBMAILED)\b", "SUBMIT", text)
    text = re.sub(r"(?i)(R\d+)[^\x00-\x7f]+(?=\s*$)", r"\1", text,
                  flags=re.MULTILINE)
    return text


def _complete_submit_quote(reply, context):
    """Fill only an unambiguous missing quote from an accepted review."""
    text = str(reply or "")
    match = re.fullmatch(
        r"(?is)(?:(WHY\s*:[^\n]*)\n+)?\s*ACTION:\s*(?:SUBMIT|SUBMAIL|"
        r"SUBMAL|SUBMAILED)\s+"
        r"\(?([A-H])\)?\s*@\s*R([1-9][0-9]*)\s*", text)
    if match is None:
        return text
    option = match.group(2).upper()
    revision_id = "R%d" % int(match.group(3))
    matches = [
        row for row in context.get("accepted") or ()
        if str(row.get("option") or "").upper() == option
        and str(row.get("revision_id") or "") == revision_id
        and str(row.get("review_quote") or "")]
    if len(matches) != 1:
        return text
    action = "ACTION: SUBMIT %s @ %s | %s" % (
        option, revision_id, json.dumps(matches[0]["review_quote"]))
    why = (match.group(1).strip() + "\n") if match.group(1) else ""
    return why + action


def _menu_explore_turn(seed, turn_index, probability):
    """Choose guided or exploratory menu framing before decoding one turn."""
    value = float(probability)
    if not 0.0 <= value <= 1.0:
        raise ValueError("off-menu probability must be in [0, 1]")
    if value <= 0.0:
        return False
    if value >= 1.0:
        return True
    return random.Random("menu:%s:%d" % (seed, int(turn_index))).random() < value


def _memory_card_view(tree, question, question_number, local_visible,
                      global_visible, per_scope=4, scope_rotation=0,
                      global_source="timeline"):
    from ..memory import cards as MEMORY_CARDS
    cards = MEMORY_CARDS.build_cards(
        tree, question_number, local_visible=local_visible,
        global_visible=global_visible, global_source=global_source)
    for card in cards:
        card_id = str(card.get("card_id") or "")
        uses = [
            row for row in tree.memory_uses.values()
            if str(row.get("card_id") or "") == card_id
            and int(row.get("question_number") or 0) == int(question_number)]
        if uses:
            latest = max(uses, key=lambda row: int(row.get("sequence") or 0))
            card["check_status"] = "%s_%s" % (
                latest.get("action") or "use", latest.get("outcome") or "unknown")
            card["checked_option"] = str(latest.get("option") or "")
            card["checked_quote"] = str(latest.get("quote") or "")
        for entry in card.get("timeline_entries") or ():
            revision_id = str(entry.get("source_revision_id") or "")
            entry_uses = [
                row for row in uses
                if str(row.get("card_id") or "") == card_id
                and str(row.get("scope") or "") == str(card.get("scope") or "")
                and str(row.get("revision_id") or "") == revision_id]
            if entry_uses:
                latest = max(
                    entry_uses, key=lambda row: int(row.get("sequence") or 0))
                entry["check_status"] = "%s_%s" % (
                    latest.get("action") or "use",
                    latest.get("outcome") or "unknown")
                entry["checked_option"] = str(latest.get("option") or "")
                entry["checked_quote"] = str(latest.get("quote") or "")
            else:
                entry["check_status"] = "not_checked"
                entry.pop("checked_option", None)
                entry.pop("checked_quote", None)
    both_visible = bool(local_visible and global_visible)
    scope_order = (("local", "global")
                   if (int(question_number) + int(scope_rotation)) % 2 == 0
                   else ("global", "local"))
    cards = MEMORY_CARDS.rank_cards(
        cards, question, duration=tree.duration, per_scope=per_scope,
        interleave_scopes=both_visible, scope_order=scope_order)
    text, handles = MEMORY_CARDS.render_cards(
        cards, question=question, duration=tree.duration)
    return cards, text, handles


def _card_action_lines(card_handles, question, force_card_check,
                       frames_left=None, calls_left=None, filter_budget=False,
                       time_left_seconds=None, min_action_seconds=None,
                       min_submit_seconds=None, min_calls_for_check=3):
    """Return card actions; optionally drop ones that cannot fit."""
    options = C.options_of(question)
    lines = []
    for handle, card in card_handles.items():
        status = str(card.get("check_status") or "")
        option = (str(card.get("checked_option") or "").upper()
                  if status in ("check_accept", "cite_accept")
                  else str(card.get("option") or "").upper())
        quote = (str(card.get("checked_quote") or "")
                 if status in ("check_accept", "cite_accept")
                 else str(card.get("quote") or ""))
        check_time_fits = bool(
            time_left_seconds is None or min_action_seconds is None
            or float(time_left_seconds) >= float(min_action_seconds))
        submit_time_fits = bool(
            time_left_seconds is None or min_submit_seconds is None
            or float(time_left_seconds) >= float(min_submit_seconds))
        if (status in ("accepted_review", "check_accept", "cite_accept")
                and not (force_card_check and status == "accepted_review")
                and option in "ABCDEFGH"[:len(options)] and quote
                and (calls_left is None or int(calls_left) >= 1)
                and submit_time_fits):
            lines.append("ACTION: SUBMIT %s @ %s | %s"
                         % (option, handle, json.dumps(quote)))
            continue
        blocked = status in (
            "check_reject", "check_unsure", "check_invalid", "cite_reject",
            "cite_unsure", "cite_invalid")
        frame_handle = card.get("frame_handle") or {}
        stored_frames = min(int(card.get("stored_frames") or 0), MEMORY_USE_FRAMES)
        check_fits = bool(
            frame_handle and stored_frames > 0
            and (frames_left is None or stored_frames <= int(frames_left))
            and (calls_left is None
                 or int(calls_left) >= int(min_calls_for_check))
            and check_time_fits)
        if not blocked and check_fits:
            lines.append("ACTION: CHECK %s" % handle)
    return lines


def _current_read_evidence(tree, question_number, limit=8):
    """Render current-question observations for the budget-final answer."""
    rows = sorted(
        (row for row in tree.revisions.values()
         if int(row.get("question_number") or 0) == int(question_number)
         and row.get("read_status") == "success"),
        key=lambda row: int(str(row.get("revision_id") or "R0")[1:]))
    lines = []
    for row in rows[-int(limit):]:
        span = row.get("window") or []
        span_text = ("%.0f-%.0fs" % (float(span[0]), float(span[1]))
                     if len(span) == 2 else "unknown time")
        summary = " ".join(str(row.get("branch_summary") or "").split())
        quote = " ".join(str(row.get("quote") or "").split())
        line = "%s %s %s: %s" % (
            row.get("revision_id"), row.get("branch_id"), span_text,
            summary or "(no summary)")
        if quote:
            line += ' Quote: "%s"' % quote
        lines.append(line)
    if len(rows) > int(limit):
        lines.insert(0, "[%d older READ revisions omitted]" % (len(rows) - int(limit)))
    return "\n".join(lines) or "(none)"


def _budget_answer_stage(context, card_actions, budget_answer, calls_left,
                         time_left_seconds=None,
                         final_answer_reserve_seconds=FINAL_ANSWER_RESERVE_SECONDS):
    """True only when one final answer call fits and a current READ exists."""
    if (not budget_answer or context["accepted"]
            or context.get("accepted_card_checks")):
        return False
    if calls_left is not None and int(calls_left) <= 0:
        return False
    if not context["current_question_read"]:
        return False
    if (time_left_seconds is not None
            and float(time_left_seconds) <= float(final_answer_reserve_seconds)):
        return not card_actions
    if context["terminal_required"] and not context["review_required"]:
        return not card_actions
    return not (context["can_read"] or context["review_required"] or card_actions)


def _time_terminal_stage(context, budget_answer, time_left_seconds, calls_left,
                         final_answer_reserve_seconds=FINAL_ANSWER_RESERVE_SECONDS):
    """Force an existing accepted answer when the wall-clock reserve is reached."""
    return bool(
        budget_answer and time_left_seconds is not None
        and float(time_left_seconds) <= float(final_answer_reserve_seconds)
        and ((context["accepted"] and context["current_question_read"])
             or context.get("accepted_card_checks"))
        and (calls_left is None or int(calls_left) > 0))


def _facts_lines(tree, question_number, limit=8, words=16):
    """Compact per-chunk facts read in this question. No option letters."""
    rows = sorted(
        (row for row in tree.revisions.values()
         if int(row.get("question_number") or 0) == int(question_number)
         and row.get("read_status") == "success" and row.get("branch_summary")),
        key=lambda row: int(row["revision_id"][1:]))
    lines = []
    for row in rows[-int(limit):]:
        window = row.get("window") or ()
        text = MEMORY_CLAIM.strip_option_markers(
            row["branch_summary"]).split()
        short = " ".join(text[:words]) + ("..." if len(text) > words else "")
        lines.append("%s %s | %s" % (
            row["branch_id"],
            "%g-%gs" % (window[0], window[1]) if len(window) == 2 else "?",
            short))
    return lines


def _turn_prompt(tree, question, branches, open_branch, question_time=None,
                 question_number=0, last_refusal=None, opened_branches=(),
                 frames_left=None, calls_left=None, memory_visible=True,
                 local_memory_visible=None,
                 allow_read_window=True, legality_only=False,
                 show_read_set=True, synthetic_protocol=False,
                 memory_use=False, menu_mode=False, menu_explore=False,
                 global_stage=False, global_memory=False,
                 force_answer_stage=False, menu_layout="grouped",
                 memory_menu_order="use_first",
                 memory_decision_timing="pre_read",
                 scope_rotation=0,
                 card_list_visible=False, card_local_visible=True,
                 card_global_visible=True, cards_per_scope=4,
                 global_source="timeline", read_steer=None,
                 facts_block=False,
                 force_card_check=False, max_reads_per_question=None,
                 budget_answer=False, force_budget_answer=False,
                 time_left_seconds=None, min_check_to_final_seconds=None,
                 reader_min_runtime_seconds=None,
                 policy_turn_reserve_seconds=POLICY_TURN_RESERVE_SECONDS,
                 final_answer_reserve_seconds=FINAL_ANSWER_RESERVE_SECONDS):
    if menu_layout not in ("grouped", "flat"):
        raise ValueError("unknown menu layout: %s" % menu_layout)
    if memory_menu_order not in ("use_first", "defer_first"):
        raise ValueError("unknown memory menu order: %s" % memory_menu_order)
    if memory_decision_timing not in MEMORY_DECISION_TIMINGS:
        raise ValueError("unknown memory decision timing: %s"
                         % memory_decision_timing)
    if card_list_visible and not (card_local_visible and card_global_visible):
        # A single-scope card arm must not see the other scope's stage text.
        global_stage = False
    opened_branches = set(opened_branches)
    if local_memory_visible is None:
        local_memory_visible = memory_visible
    def prior_answer_visible(row):
        same_question = int(row.get("question_number") or 0) == int(
            question_number)
        return same_question or (local_memory_visible and not card_list_visible)

    answers = sorted(
        (rid for rid, row in tree.revisions.items()
         if answer_bearing(row) and not tree.review_stale(rid)
         and prior_answer_visible(row)),
        key=lambda value: int(value[1:]))
    stale_answers = sorted(
        (rid for rid, row in tree.revisions.items()
         if answer_bearing(row) and tree.review_stale(rid)
         and prior_answer_visible(row)),
        key=lambda value: int(value[1:]))

    def revision_text(handles):
        shown = handles[-TURN_HANDLES:]
        text = ", ".join(shown) or "none"
        if len(handles) > len(shown):
            text += " [%d older revisions omitted]" % (len(handles) - len(shown))
        return text

    answer_text = revision_text(answers)
    context = _turn_context(
        tree, branches, question_number, open_branch, frames_left=frames_left,
        calls_left=calls_left, memory_visible=memory_visible,
        local_memory_visible=local_memory_visible,
        allow_read_window=allow_read_window,
        memory_use=memory_use and not card_list_visible,
        allow_reread=menu_mode, global_memory=global_memory,
        force_answer_stage=force_answer_stage, question_time=question_time,
        read_steer=read_steer,
        memory_decision_timing=memory_decision_timing,
        max_reads_per_question=max_reads_per_question,
        budget_answer=budget_answer,
        force_budget_answer=force_budget_answer,
        time_left_seconds=time_left_seconds,
        reader_min_runtime_seconds=reader_min_runtime_seconds,
        policy_turn_reserve_seconds=policy_turn_reserve_seconds,
        final_answer_reserve_seconds=final_answer_reserve_seconds)
    context["answer_options"] = list(
        "ABCDEFGH"[:len(C.options_of(question))])
    effective = context["effective"]
    review_lines = ["%s %s %s %s%s" % (
        row["review_id"], row["revision_id"], row["option"], row["outcome"],
        ((' "%s"' % row["review_quote"]) if row["review_quote"] else "")
        + (" [stale]" if tree.review_stale(row["revision_id"]) else "")
        + (" [prior question]" if not _review_pair_applies(
            tree, row, question_number) else ""))
        for row in sorted(effective, key=lambda value: value["sequence"])[-TURN_HANDLES:]
        if (not card_list_visible or int(row.get("question_number") or 0)
            == int(question_number))]
    if len(effective) > TURN_HANDLES:
        review_lines.insert(
            0, "[%d older reviews omitted]" % (len(effective) - TURN_HANDLES))
    use_lines = [] if card_list_visible else [
        "%s %s %s %s%s" % (
            row["use_id"], row["revision_id"], row["action"], row["outcome"],
            ((' (%s) "%s"' % (row["option"], row["quote"]))
             if row.get("option") and row.get("quote") else ""))
        for row in sorted(tree.memory_uses.values(),
                          key=lambda value: int(value.get("sequence") or 0))
        if int(row.get("question_number") or 0) == int(question_number)
    ]
    card_block = ""
    card_handles = {}
    if card_list_visible:
        cards, rendered_cards, card_handles = _memory_card_view(
            tree, question, question_number, card_local_visible,
            card_global_visible, per_scope=cards_per_scope,
            scope_rotation=scope_rotation, global_source=global_source)
        card_block = ("\nPrior memory cards:\n" +
                      (rendered_cards or "(none)") + "\n")

    candidate = context["memory_candidate"]
    candidate_text = "(none)"
    if candidate:
        candidate_text = (
            "R%s source=Q%d branch=%s time=%s-%ss stored_frames=%d\n"
            "Preview: %s\nUSE frames needed: %d (fits: %s)" % (
                candidate["revision_id"][1:],
                candidate["source_question_number"], candidate["branch_id"],
                _seconds(candidate["window"][0]),
                _seconds(candidate["window"][1]), candidate["stored_frames"],
                candidate["preview"], context["memory_use_frames"],
                "yes" if context["memory_use_fits"] else "no"))
    if candidate:
        memory_forms = []
        if context["memory_use_fits"]:
            memory_forms.append(
                "ACTION: USE R%s" % candidate["revision_id"][1:])
        memory_forms.append(
            "ACTION: DEFER R%s" % candidate["revision_id"][1:])
        memory_rule = (
            "\nMemory decision must be exactly one of: %s."
            % " or ".join(memory_forms))
    else:
        memory_rule = ""
    candidate_block = ("" if card_list_visible else
                       "\nPrior memory candidate (one decision maximum):\n%s%s"
                       % (candidate_text, memory_rule) if memory_use else "")
    memory_use_block = ("\nMemory uses:\n%s" % ("\n".join(use_lines) or "(none)")
                        if memory_use else "")
    empty_open = context["empty_branch"] is not None
    empty_required = empty_open and context["can_read"] and not context["budget_abstain"]
    if open_branch is None:
        opened = "(none)"
    elif not memory_visible:
        opened = "(memory ablation: hidden)"
    else:
        opened = tree.render_branch(
            open_branch, question_time=question_time,
            question_number=(question_number if card_list_visible else
                             None if local_memory_visible else question_number))
    budget_evidence = (_current_read_evidence(tree, question_number)
                       if context.get("budget_answer_available") else "")
    budget_evidence_block = (
        "\nCurrent-question read evidence:\n%s\n" % budget_evidence
        if context.get("budget_answer_available") else "")
    if empty_required:
        opened += "\nNo memory exists in this branch."
        if synthetic_protocol or legality_only:
            opened += " This branch has no successful read."
        else:
            opened += " Next send ACTION: READ B%d." % open_branch
    elif empty_open and context["budget_abstain"] and not legality_only:
        opened += "\nThe call budget is nearly spent; ACTION: ABSTAIN is required."
    elif empty_open and context["budget_abstain"]:
        opened += "\nThe call budget is nearly spent."
    elif empty_open and not context["can_read"]:
        opened += "\nNo frame budget remains; this empty branch cannot be read."
    if legality_only or menu_mode:
        required_action = (
            "follow the synthetic protocol priority above"
            if synthetic_protocol else "follow the lifecycle rule above")
        if context["budget_abstain"]:
            actions = ["ACTION: ABSTAIN <reason>  required: the budget is nearly spent"]
        elif context["terminal_required"]:
            actions = []
            if context["review_required"]:
                actions.append("ACTION: REVIEW %s" % context["review_required"])
            if context["submit_available"]:
                for row in sorted(context["accepted"], key=lambda value: value["sequence"]):
                    actions.append(
                        "ACTION: SUBMIT %s @ %s | %s"
                        % (row["option"], row["revision_id"],
                           json.dumps(row["review_quote"])))
            if context["abstain_available"]:
                actions.append("ACTION: ABSTAIN <reason>  finish without task credit")
        else:
            actions = []
            if menu_mode and not context["review_required"]:
                actions = [
                    "ACTION: OPEN B%d" % branch
                    for branch in context["openable_branches"][:MENU_BRANCH_LIMIT]
                    if branch not in opened_branches]
            elif not menu_mode:
                actions = (["ACTION: OPEN B<chunk>  show one unopened branch"]
                           if any(branch not in opened_branches
                                  for branch in context["openable_branches"]) else [])
            if context["can_read"] and not context["review_required"]:
                for branch in context["readable_branches"][:MENU_BRANCH_LIMIT]:
                    actions.append(
                        "ACTION: READ B%d  call one reader on chunk %s-%ss%s"
                        % (branch, _seconds(branches[branch][0]),
                           _seconds(branches[branch][1]),
                           "" if allow_read_window
                           else "; do not add a time window"))
            if context["review_required"]:
                if menu_mode:
                    actions.append("ACTION: REVIEW %s"
                                   % context["review_required"])
                else:
                    actions.append(
                        "ACTION: REVIEW R<number>  review one answer-bearing revision")
            if context["submit_available"]:
                for row in sorted(context["accepted"], key=lambda value: value["sequence"]):
                    actions.append(
                        "ACTION: SUBMIT %s @ %s | %s"
                        % (row["option"], row["revision_id"],
                           json.dumps(row["review_quote"])))
            if not context["review_required"] and context["abstain_available"]:
                actions.append(
                    "ACTION: ABSTAIN <reason>  finish without task credit")
    elif empty_required:
        required_action = "READ B%d" % open_branch
        actions = ["ACTION: " + required_action + "  required: read the whole branch"]
    elif context["review_required"]:
        required_action = "REVIEW " + context["review_required"]
        actions = ["ACTION: " + required_action + "  required: review the new report"]
    elif context["scan_required"]:
        next_branch = context["next_read_branch"]
        required_action = "READ B%d" % next_branch
        actions = ["ACTION: " + required_action + "  required: read the next unread branch"]
    elif context["budget_abstain"]:
        required_action = "ABSTAIN"
        actions = ["ACTION: ABSTAIN <reason>  required: the call budget is nearly spent"]
    elif context["terminal_required"]:
        required_action = "none; choose one action below"
        actions = []
        if context["review_required"]:
            actions.append("ACTION: REVIEW %s" % context["review_required"])
        if context["submit_available"]:
            for row in sorted(context["accepted"], key=lambda value: value["sequence"]):
                actions.append(
                    "ACTION: SUBMIT %s @ %s | %s"
                    % (row["option"], row["revision_id"],
                       json.dumps(row["review_quote"])))
        actions.append("ACTION: ABSTAIN <reason>  finish without an answer")
    else:
        required_action = "none; choose one action below"
        actions = (["ACTION: OPEN B<chunk>  show one unopened branch"]
                   if context["accepted"] and any(branch not in opened_branches
                          for branch in context["openable_branches"])
                   else [])
        if context["can_read"]:
            actions.append("ACTION: READ B<chunk>  call one reader on the whole chunk"
                           + ("" if allow_read_window
                              else "; do not add a time window"))
        if context["review_required"]:
            actions.append("ACTION: REVIEW R<number>  review one answer-bearing revision")
        if context["submit_available"]:
            for row in sorted(context["accepted"], key=lambda value: value["sequence"]):
                actions.append(
                    "ACTION: SUBMIT %s @ %s | %s"
                    % (row["option"], row["revision_id"],
                       json.dumps(row["review_quote"])))
        if context["abstain_available"]:
            actions.append("ACTION: ABSTAIN <reason>  finish without an answer")
    if candidate and not legality_only:
        required_action = "none; choose one action below"
    candidate_actions = []
    if candidate:
        use_action = None
        defer_action = None
        if context["memory_use_fits"]:
            use_action = (
                "ACTION: USE R%s" % candidate["revision_id"][1:]
                if menu_mode else
                "ACTION: USE R%s  inspect this prior memory with stored frames"
                % candidate["revision_id"][1:])
        defer_action = (
            "ACTION: DEFER R%s" % candidate["revision_id"][1:]
            if menu_mode else
            "ACTION: DEFER R%s  continue without this prior memory"
            % candidate["revision_id"][1:])
        candidate_actions = (
            [use_action, defer_action] if use_action and
            memory_menu_order == "use_first" else
            [defer_action, use_action] if use_action else
            [defer_action])
    promotable = (tree.promotable_local_revisions(question_number)
                  if global_stage else [])
    claims = tree.active_global_claims() if global_stage else []
    global_actions = []
    for revision in promotable:
        global_actions.append("ACTION: EVOLVE_GLOBAL ADD %s"
                              % revision["revision_id"])
        for claim in claims:
            global_actions.append(
                "ACTION: EVOLVE_GLOBAL SUPERSEDE %s %s"
                % (claim["claim_id"], revision["revision_id"]))
    if global_stage:
        global_actions.append("ACTION: EVOLVE_GLOBAL KEEP")
    if global_stage:
        candidate_actions = []
        actions = global_actions
        required_action = "EVOLVE_GLOBAL"
        global_block = (
            "\nGlobal-memory decision stage. Choose exactly one operation.\n"
            "Promotable accepted local revisions: %s\n"
            "Active global claims: %s\n"
            "KEEP is always legal and writes no new global fact.\n"
            % (", ".join(row["revision_id"] for row in promotable) or "(none)",
               ", ".join(row["claim_id"] for row in claims) or "(none)"))
    else:
        global_block = ""
    card_actions = []
    if card_list_visible and not global_stage:
        card_actions = _card_action_lines(
            card_handles, question, force_card_check,
            frames_left=frames_left, calls_left=calls_left,
            filter_budget=bool(context.get("budget_answer_available")
                               or context.get("answer_stage")
                               or context.get("terminal_required")),
            time_left_seconds=time_left_seconds,
            min_action_seconds=min_check_to_final_seconds,
            min_submit_seconds=(
                0.0 if context.get("accepted_card_checks")
                else final_answer_reserve_seconds),
            min_calls_for_check=(
                3 if context["current_question_read"] else 6))
        if context["accepted"] or context.get("accepted_card_checks"):
            card_actions = [line for line in card_actions
                            if not line.startswith("ACTION: CHECK")]
    if context.get("budget_answer_available"):
        actions = ["ACTION: SUBMIT %s" % option
                   for option in context["answer_options"]]
        required_action = "SUBMIT <option>"
        menu_rule = "The evidence-search budget is exhausted. Choose one option."
    prefer_card_action = bool(
        not global_stage and card_list_visible and card_actions
        and not context["can_read"]
        and context.get("budget_answer_enabled"))
    if prefer_card_action:
        actions = [line for line in actions
                   if not line.startswith("ACTION: ABSTAIN")]
    actions = card_actions + candidate_actions + actions
    recommended_objects = []
    if menu_mode and (global_stage or context.get("answer_stage")
                      or not card_list_visible or not card_actions):
        if global_stage:
            menu_stage, _recommended_verbs = "global", {
                "evolve_global"}
            recommended_objects = []
            if global_actions:
                try:
                    recommended_objects = [parse_action(global_actions[0])]
                except ActionError:
                    recommended_objects = []
        else:
            menu_stage, _recommended_verbs = _recommended_action_menu(context)
            recommended_objects = _recommended_action_objects(
                context, menu_stage, memory_menu_order=memory_menu_order)
        recommended_signatures = {
            _action_signature(action) for action in recommended_objects}
        recommended = []
        other = [line for line in actions
                 if _action_line_signature(line) not in recommended_signatures]
        for line in actions:
            try:
                parsed_line = parse_action(line)
            except ActionError:
                continue
            if _action_signature(parsed_line) in recommended_signatures:
                recommended.append(line)
        if menu_layout == "grouped":
            actions = (recommended + (["Other legal actions (lower probability):"]
                                      if other else []) + other)
            required_action = "none; choose one recommended or other legal action"
            menu_rule = (
                "Menu mode: explore. If reasonable, choose one action from Other "
                "legal actions; otherwise choose the best recommended action. You decide."
                if menu_explore else
                "Menu mode: guided. Choose exactly one action from Recommended now. "
                "Do not invent a target and do not repeat a read target unless it is "
                "shown again. "
                "When SUBMIT is recommended, prefer it over ABSTAIN.")
        else:
            actions = recommended + other
            required_action = "none; choose one displayed legal action"
            recommended_count = len(recommended)
            if recommended_count == 1:
                preferred = "the first displayed action"
            else:
                preferred = "one of the first %d displayed actions" % (
                    recommended_count)
            if not menu_explore:
                menu_rule = (
                    "Menu mode: flat. Choose %s unless it is clearly wrong."
                    % preferred)
            elif other:
                menu_rule = (
                    "Menu mode: flat explore. Choose an action after the first "
                    "%d unless it is clearly wrong; otherwise choose %s. You decide."
                    % (recommended_count, preferred))
            else:
                menu_rule = (
                    "Menu mode: flat explore. Choose %s. You decide."
                    % preferred)
    else:
        menu_rule = ""
    if card_list_visible and not global_stage:
        if prefer_card_action:
            required_action = "one displayed memory action; prefer CHECK"
            menu_rule = (
                "Another READ does not fit. Prefer one displayed memory CHECK "
                "or cited SUBMIT. Choose exactly one displayed action.")
        else:
            required_action = "none; choose one displayed action"
            menu_rule = (
                "Memory cards remain available after OPEN or READ. "
                "Choose any displayed legal action.")
    elif global_stage:
        required_action = "none; choose one displayed action"
        menu_rule = (
            "Choose any displayed global-memory operation. "
            "KEEP writes no new global fact.")
    if context.get("budget_answer_available"):
        required_action = "SUBMIT <option>"
        menu_rule = (
            "The evidence-search budget is exhausted. Choose exactly one "
            "SUBMIT option using the current-question evidence above.")

    if context.get("budget_answer_available"):
        required_display = "ACTION: SUBMIT <option>"
    elif context["review_required"]:
        required_display = "ACTION: REVIEW %s" % context["review_required"]
    elif global_stage and global_actions and not menu_mode:
        required_display = global_actions[0]
    elif global_stage and menu_mode:
        required_display = "none; choose one displayed action"
    elif force_answer_stage and context["submit_available"] \
            and context["accepted"]:
        row = sorted(context["accepted"], key=lambda value: value["sequence"])[-1]
        required_display = "ACTION: SUBMIT %s @ %s | %s" % (
            row["option"], row["revision_id"], json.dumps(row["review_quote"]))
    else:
        required_display = (required_action
                            if legality_only or required_action.startswith("none")
                            else "ACTION: " + required_action)
    read_set = ("" if not show_read_set else
                "Read branches already inspected: %d\n"
                "Unread branches: %s\nBlocked branches: %s\n"
                "Exhausted READ branches (hidden, not legal): %d\n"
                "Calls remaining: %s\nFrames remaining: %s\n" % (
                    len(context["read_branches"]),
                    ", ".join(
                        ["B%d" % branch for branch in context["unread_branches"]
                         if branch not in context["exhausted_branches"]]
                        [:MENU_BRANCH_LIMIT])
                    + ((" [%d more omitted]" % max(
                        0, len(context["unread_branches"])
                        - MENU_BRANCH_LIMIT))
                       if len(context["unread_branches"]) > MENU_BRANCH_LIMIT
                       else "")
                    if context["unread_branches"] else "(none)",
                    ", ".join("B%d" % branch
                              for branch in context["blocked_branches"])
                    or "(none)",
                    len(context["exhausted_branches"]),
                    calls_left if calls_left is not None else "unknown",
                    frames_left if frames_left is not None else "unknown"))
    if facts_block:
        read_set += ("Facts read so far in this question:\n%s\n" % (
            "\n".join(_facts_lines(tree, question_number)) or "(none yet)"))
    lifecycle_rule = (
        (BUDGET_ANSWER_RULE
         if (context.get("budget_answer_available") or context.get("answer_stage")) else
         GLOBAL_LIFECYCLE_RULE if global_memory else LIFECYCLE_RULE)
                      + "\n\n"
                      if legality_only and not synthetic_protocol else "")
    if context.get("budget_answer_available"):
        turn_rule = (
            "You are O. The evidence-search budget is exhausted. Choose exactly "
            "one final answer action from the options below.")
    else:
        turn_rule = (
            "You are O. Choose exactly one atomic action.\n"
            "Do not answer the question in this turn.")
    return """{turn_rule}
Displayed ACTION lines are the guided menu. In guided mode choose one displayed
action. In exploration mode, another existing branch target may still be legal;
invalid or exhausted targets are refused.

Question and options:
{question}

{lifecycle_rule}{card_block}{global_tree}{global_block}

Open branch:
{opened}

{read_set}Answer-bearing revisions: {answers}
Reviewable current-question revisions: {reviewable}
Stale revisions (not reviewable): {stale_answers}
Effective reviews: {reviews}{candidate_block}{memory_use_block}{budget_evidence_block}

{evidence_rule}
Last refusal: {last_refusal}
Required next action: {required_action}{menu_rule_line}

{available_label}
{actions}

Reply with an optional WHY line and exactly one action line beginning with ACTION:.
Use exact action words: ABSTAIN (not ABSTAINS), SUBMIT, and EVOLVE_GLOBAL ADD R<number>.
Do not add unexplained text.
No batching and no extra action tokens. Answer option or answer text is not an action.
A submitted quote must equal the accepted review quote.""" .format(
        turn_rule=turn_rule,
        question=question,
        lifecycle_rule=lifecycle_rule,
        global_tree=(
            "Prior memory is represented only by the cards above."
            if card_list_visible else
            tree.render_global(
                question_time, include_local_evidence=local_memory_visible)
            if memory_visible else "(memory ablation: hidden)"),
        card_block=card_block,
        global_block=global_block,
        opened=opened,
        read_set=read_set,
        answers=answer_text,
        reviewable=revision_text(context["unreviewed_answer_revisions"]),
        stale_answers=revision_text(stale_answers),
        reviews="\n".join(review_lines) or "(none)",
        candidate_block=candidate_block,
        memory_use_block=memory_use_block,
        budget_evidence_block=budget_evidence_block,
        evidence_rule=(
            "Use only the current-question READ summaries and displayed memory "
            "to choose one option. No further READ or CHECK is available."
            if context.get("budget_answer_available") else
            "The environment checks object existence, evidence, and budgets. "
            "The video is available only through the READ tool; OPEN alone does "
            "not inspect it. Follow the synthetic protocol priority above."
            if synthetic_protocol else
            "The environment checks object existence, evidence, and budgets. "
            "The video is available only through the READ tool; OPEN alone does "
            "not inspect it. Follow the lifecycle rule above."
            if legality_only else
            "ACTION: READ B%d is required before any other action." % open_branch
            if empty_required else
            "ACTION: REVIEW %s before acting on its answer." % context["review_required"]
            if context["review_required"] else
            "No answer-bearing report exists yet. ACTION: READ B%d is required; "
            "unread branches remain." % context["next_read_branch"]
            if context["scan_required"] else
            "The call budget is nearly spent. ACTION: ABSTAIN is required."
            if context["budget_abstain"] else
            "No current-question READ or accepted review is available. SUBMIT and "
            "ABSTAIN will be refused."
            if not context["terminal_available"] else "A terminal action is available."),
        required_action=required_display,
        menu_rule_line=("\n" + menu_rule if menu_rule else ""),
        last_refusal=(("%s: %s" % (last_refusal.get("kind"), last_refusal.get("why")))
                      if last_refusal else "(none)"),
        available_label=("Recommended now:"
                         if menu_mode and menu_layout == "grouped"
                         and (context.get("answer_stage") or not card_list_visible
                              or not card_actions)
                         else "Available now:"),
        actions="\n".join(actions)).replace(
            "A submitted quote must equal the accepted review quote.",
            "A submitted quote must equal the accepted review or USE quote."
            if memory_use else
            "A submitted quote must equal the accepted review quote.")


def _repair_suggestion(prompt, context):
    """Avoid turning an order test into a forced first-menu-action arm."""
    if (context.get("memory_candidate") is not None
            or context.get("budget_answer_available")
            or context.get("answer_stage")):
        return ""
    return _first_menu_action(prompt)


def _first_menu_action(prompt):
    """Return the first parseable recommended or other legal action line."""
    text = str(prompt or "")
    if "strict single-action repair" in text:
        for line in text.splitlines():
            if not line.startswith("ACTION:"):
                continue
            try:
                parse_action(line)
            except ActionError:
                continue
            return line.strip()
    for marker in ("Recommended now:", "Other legal actions (lower probability):",
                   "Available now:"):
        if marker not in text:
            continue
        section = text.split(marker, 1)[1]
        for stop in ("\n\nReply with", "Calls remaining:"):
            section = section.split(stop, 1)[0]
        for line in section.splitlines():
            if not line.startswith("ACTION:"):
                continue
            try:
                parse_action(line)
            except ActionError:
                continue
            return line.strip()
    return ""


def _action_repair_prompt(prompt, reply, error, suggested_action=""):
    """One strict retry after a malformed or truncated policy turn."""
    excerpt = " ".join(str(reply or "").split())[:480]
    if suggested_action:
        return ("Previous reply was invalid.\n"
                "Previous reply: %s\nParser error: %s\n"
                "This is a strict single-action repair. Use only this exact "
                "currently legal action:\n%s\nReply with exactly that action "
                "line and nothing else." % (excerpt, error, suggested_action))
    return (str(prompt) + "\n\nPrevious reply was invalid.\n"
            "Previous reply: %s\nParser error: %s\n"
            "Reply now with exactly one line beginning ACTION:. Copy one complete "
            "action from the displayed menu exactly; do not add another action or "
            "explanation." % (excerpt, error))


def _save_frames(root, video, branch, span, images, revision_id):
    record = TREE.save_frames(root, video, branch, span, images,
                              name=revision_id.lower())
    return record


def _revision_key(value):
    text = str(value or "").upper()
    return text if text.startswith("R") else "R" + text


def _json_sha256(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _policy_refusal_count(refusals):
    return sum(
        str(row.get("kind") or "") not in NON_POLICY_REFUSAL_KINDS
        for row in refusals)


def _global_stage_open(global_memory, global_stage_done, context,
                       calls_left):
    """Promotion waits until the current answer-bearing read is reviewed."""
    return bool(
        global_memory and not global_stage_done and context["accepted"]
        and not context["review_required"]
        and context["current_question_read"]
        and calls_left is not None and int(calls_left) >= 2)


def run_question(ask, tree, question, chunk_windows, frames_of, max_turns=8,
                 frame_cap=2048, root="", question_number=0, call_cap=None,
                 memory_visible=True, policy_temperature=0.0, policy_seed=None,
                 max_reads_per_question=None,
                 local_memory_visible=None,
                 reader_temperature=0.0, review_temperature=0.0,
                 reader_seed=None, reader_policy=False,
                 reader_schema_version=1,
                 allow_read_window=True, legality_only=False,
                 on_read_precommit=None, on_question_start=None,
                 memory_use=False, menu_mode=False,
                 off_menu_probability=None, global_memory=False,
                 menu_layout="grouped", memory_menu_order="use_first",
                 memory_decision_timing="pre_read", reader_allow_keep=True,
                 card_list_visible=False, card_local_visible=True,
                 card_global_visible=True, cards_per_scope=4,
                 global_source="timeline", read_order="positional",
                 facts_block=False,
                 force_card_check=False, budget_answer=False,
                 reader_min_runtime_seconds=None,
                 policy_turn_reserve_seconds=POLICY_TURN_RESERVE_SECONDS,
                 final_answer_reserve_seconds=FINAL_ANSWER_RESERVE_SECONDS,
                 min_check_to_final_seconds=MIN_CHECK_TO_FINAL_SECONDS):
    """Run one question with one atomic action per O turn."""
    if menu_layout not in ("grouped", "flat"):
        raise ValueError("unknown menu layout: %s" % menu_layout)
    if memory_menu_order not in ("use_first", "defer_first"):
        raise ValueError("unknown memory menu order: %s" % memory_menu_order)
    if memory_decision_timing not in MEMORY_DECISION_TIMINGS:
        raise ValueError("unknown memory decision timing: %s"
                         % memory_decision_timing)
    wall_time_values = {
        "reader_min_runtime_seconds": reader_min_runtime_seconds,
        "policy_turn_reserve_seconds": policy_turn_reserve_seconds,
        "final_answer_reserve_seconds": final_answer_reserve_seconds,
        "min_check_to_final_seconds": min_check_to_final_seconds,
    }
    if any(value is not None and float(value) < 0.0
           for value in wall_time_values.values()):
        raise ValueError("wall-time policy values must be non-negative")
    frame_cap = 2048 if frame_cap is None else int(frame_cap)
    if local_memory_visible is None:
        local_memory_visible = memory_visible
    branches = {int(branch): (float(first), float(last))
                for branch, first, last in chunk_windows}
    options = C.options_of(question)
    question_time = C.clock_reference(question, tree.duration)
    if read_order not in ("positional", "steered"):
        raise ValueError("unknown read order: %s" % read_order)
    read_steer = (
        [list(cue) for cue in C.M.time_references(question, tree.duration)]
        if read_order == "steered" else None)
    turns, refusals, review_records, memory_use_records = [], [], [], []
    memory_decision_expired = False
    open_branch = None
    opened_branches = set()
    # A normal question starts at zero. A pre-Reader anchor resumes from an
    # already-spent question budget, so initialize from the callable's absolute
    # counter instead of losing that prefix.
    frames_used = int(getattr(ask, "state", {}).get("frames") or 0)
    read_counts = {}
    ended_by, answer = "", ""
    commit = None
    submitted_revision = None
    submitted_option = ""
    submitted_quote = ""
    state_seen = {}
    last_policy_request_id = ""
    repair = None
    pending_repair_call = None
    attempted_repair = False
    repair_count = 0
    protocol_fallback = False
    budget_submit = False
    harness_recovery = False
    policy_request_id = ""
    global_stage_done = False
    card_supported = False
    tree.questions[int(question_number)] = str(question)
    tree.save()
    if on_question_start is not None:
        on_question_start({
            "tree": tree.to_dict(),
            "question_number": int(question_number),
            "question": question,
        })

    def turn(action, reply, observation, calls, frames, refusal=None):
        item = {"n": len(turns) + 1, "actor": "o",
                "action": action, "reply": reply,
                "observation": observation, "calls": int(calls),
                "frames": int(frames), "state": copy.deepcopy(state_seen),
                "refusals": [refusal] if refusal else [],
                "request_id": last_policy_request_id,
                "memory_sha256": _json_sha256(tree.to_dict())}
        turns.append(item)
        return item

    def mark_repair_recovered():
        nonlocal pending_repair_call
        if not attempted_repair or pending_repair_call is None:
            return
        request_id = pending_repair_call.get("request_id")
        if hasattr(ask, "mark_recovered"):
            ask.mark_recovered(request_id, policy_request_id)
        else:
            pending_repair_call["recovered_action"] = True
            if pending_repair_call.get("finish_reason") == "length":
                pending_repair_call["recovered_turn_cut"] = True
            pending_repair_call["recovered_by_request_id"] = policy_request_id
        pending_repair_call = None

    def result(ended_by, answer="", commit=None, supported=False):
        return {
            "turns": turns, "refusals": refusals, "ended_by": ended_by,
            "answer": answer, "capped": False, "truncated": False,
            "fault": ("turn_limit" if ended_by == "turn_limit" else
                      "protocol_abstain" if ended_by == "protocol_abstain" else ""),
            "complete": ended_by in ("agent_submit", "agent_abstain",
                                     "protocol_abstain"),
            "answer_source": "budget_submit" if budget_submit else
                             "loop_commit" if supported else
                             "abstain" if ended_by == "agent_abstain" else
                             "protocol_abstain" if ended_by == "protocol_abstain" else "",
            "commit": commit, "commit_ok": supported,
            "citation_tier": "cited" if supported else "none",
            "grounding": "checked-cited" if supported else "none",
            "submitted_unit": commit["unit"] if commit else "",
            "submitted_quote": commit["quote"] if commit else "",
            "submitted_entry_id": commit["entry_id"] if commit else None,
            "verified": supported,
            "memory_terminal": "supported" if supported else
                               "uncited" if budget_submit else
                               "abstained" if ended_by in ("agent_abstain",
                                                           "protocol_abstain")
                               else "incomplete",
            "memory_positive": supported,
            "reviews": review_records,
            "memory_uses": memory_use_records,
            "atomic_memory": tree.to_dict(),
            "question_number": int(question_number),
            "legality_only": bool(legality_only),
            "protocol_fallback": bool(protocol_fallback),
            "harness_recovery": bool(harness_recovery),
        }

    for _ in range(int(max_turns)):
        if _policy_refusal_count(refusals) >= MAX_POLICY_REFUSALS \
                and not harness_recovery:
            ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
            protocol_fallback = bool(menu_mode)
            break
        if sum(item.get("kind") in (
                "memory_use_refused", "memory_use_invalid")
               for item in refusals) >= MAX_MEMORY_ACTION_REFUSALS:
            ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
            protocol_fallback = bool(menu_mode)
            break
        before_calls = int(getattr(ask, "state", {}).get("n") or 0)
        before_frames = int(getattr(ask, "state", {}).get("frames") or 0)
        fallback_refusal = None
        frames_left = max(0, int(frame_cap) - frames_used)
        calls_left = (max(0, int(call_cap) - before_calls)
                      if call_cap is not None else None)
        deadline = getattr(ask, "state", {}).get("deadline")
        time_left_seconds = (
            max(0.0, float(deadline) - time.time())
            if deadline is not None else None)
        answer_stage = bool(global_memory and global_stage_done)
        context = _turn_context(
            tree, branches, question_number, open_branch, frames_left=frames_left,
            calls_left=calls_left, memory_visible=memory_visible,
            local_memory_visible=local_memory_visible,
            allow_read_window=allow_read_window,
            memory_use=memory_use and not memory_decision_expired,
            allow_reread=menu_mode, global_memory=global_memory,
            force_answer_stage=answer_stage, question_time=question_time,
            read_steer=read_steer,
            memory_decision_timing=memory_decision_timing,
            max_reads_per_question=max_reads_per_question,
            budget_answer=budget_answer,
            time_left_seconds=time_left_seconds,
            reader_min_runtime_seconds=reader_min_runtime_seconds,
            policy_turn_reserve_seconds=policy_turn_reserve_seconds,
            final_answer_reserve_seconds=final_answer_reserve_seconds)
        global_stage = bool(_global_stage_open(
            global_memory, global_stage_done, context, calls_left)
            and not (card_list_visible and not (
                card_local_visible and card_global_visible)))
        card_rows, _card_text, card_handles = (
            _memory_card_view(
                tree, question, question_number, card_local_visible,
                card_global_visible, per_scope=cards_per_scope,
                scope_rotation=int(policy_seed or 0),
                global_source=global_source)
            if card_list_visible else ([], "", {}))
        card_actions = _card_action_lines(
            card_handles, question, force_card_check,
            frames_left=frames_left, calls_left=calls_left,
            filter_budget=True, time_left_seconds=time_left_seconds,
            min_action_seconds=min_check_to_final_seconds,
            min_submit_seconds=(
                0.0 if context.get("accepted_card_checks")
                else final_answer_reserve_seconds),
            min_calls_for_check=(
                3 if context["current_question_read"] else 6))
        if context["accepted"] or context.get("accepted_card_checks"):
            card_actions = [line for line in card_actions
                            if not line.startswith("ACTION: CHECK")]
        card_action_verbs = set()
        for line in card_actions:
            try:
                card_action_verbs.add(parse_action(line).verb)
            except ActionError:
                continue
        budget_answer_stage = _budget_answer_stage(
            context, card_actions, budget_answer, calls_left,
            time_left_seconds=time_left_seconds,
            final_answer_reserve_seconds=final_answer_reserve_seconds)
        time_terminal_stage = _time_terminal_stage(
            context, budget_answer, time_left_seconds, calls_left,
            final_answer_reserve_seconds=final_answer_reserve_seconds)
        if budget_answer_stage:
            context = _turn_context(
                tree, branches, question_number, open_branch,
                frames_left=frames_left, calls_left=calls_left,
                memory_visible=memory_visible,
                local_memory_visible=local_memory_visible,
                allow_read_window=allow_read_window,
                memory_use=memory_use and not memory_decision_expired,
                allow_reread=menu_mode, global_memory=global_memory,
                force_answer_stage=answer_stage, question_time=question_time,
                read_steer=read_steer,
                memory_decision_timing=memory_decision_timing,
                max_reads_per_question=max_reads_per_question,
                budget_answer=True, force_budget_answer=True)
            global_stage = False
        elif time_terminal_stage:
            context = _turn_context(
                tree, branches, question_number, open_branch,
                frames_left=frames_left, calls_left=calls_left,
                memory_visible=memory_visible,
                local_memory_visible=local_memory_visible,
                allow_read_window=allow_read_window,
                memory_use=memory_use and not memory_decision_expired,
                allow_reread=menu_mode, global_memory=global_memory,
                force_answer_stage=True, question_time=question_time,
                read_steer=read_steer,
                memory_decision_timing=memory_decision_timing,
                max_reads_per_question=max_reads_per_question,
                budget_answer=True)
            global_stage = False
        context["answer_options"] = list(
            "ABCDEFGH"[:len(C.options_of(question))])
        global_promotable = (tree.promotable_local_revisions(question_number)
                             if global_stage else [])
        global_claims = tree.active_global_claims() if global_stage else []
        menu_explore = bool(menu_mode and _menu_explore_turn(
            policy_seed, len(turns), off_menu_probability
            if off_menu_probability is not None else 1.0))
        menu_stage, recommended_verbs = _recommended_action_menu(context)
        recommended_specs = _recommended_action_objects(
            context, menu_stage, memory_menu_order=memory_menu_order)
        legal_verbs = _legal_action_verbs(context)
        if card_list_visible and not global_stage:
            legal_verbs |= card_action_verbs
        if global_stage:
            menu_stage = "global"
            recommended_verbs = {"evolve_global"}
            legal_verbs = {"evolve_global"}
            recommended_specs = []
            if global_promotable:
                recommended_specs = [Action(
                    "evolve_global", global_operation="add",
                    revision=int(global_promotable[-1]["revision_id"][1:]))]
            else:
                recommended_specs = [Action(
                    "evolve_global", global_operation="keep")]
        def visible(row):
            return local_memory_visible or int(row.get("question_number") or 0) == int(
                question_number)

        current_revision_ids = {
            rid for rid, revision in tree.revisions.items() if visible(revision)}
        state_seen = {
            "question": question,
            "legality_only": bool(legality_only),
            "memory_ablation": "on" if memory_visible else "off",
            "local_memory_visible": bool(local_memory_visible),
            "global_memory": bool(global_memory),
            "open_branch": open_branch,
            "opened_branches": sorted(opened_branches),
            "branches": sorted(branches),
            "answer_bearing_revisions": [
                rid for rid, revision in tree.revisions.items()
                if visible(revision) and answer_bearing(revision)
                and not tree.review_stale(rid)],
            "stale_revisions": sorted(
                (rid for rid, revision in tree.revisions.items()
                 if visible(revision) and answer_bearing(revision)
                 and tree.review_stale(rid)),
                key=lambda value: int(value[1:]))[-TURN_HANDLES:],
            "stale_revisions_omitted": max(0, sum(
                1 for rid, revision in tree.revisions.items()
                if visible(revision) and answer_bearing(revision)
                and tree.review_stale(rid))
                - TURN_HANDLES),
            "current_events": [event["event_id"]
                               for event in tree.current_events()
                               if event["revision_id"] in current_revision_ids],
            "reviews": [
                {"review_id": row["review_id"],
                 "revision_id": row["revision_id"], "option": row["option"],
                 "outcome": row["outcome"], "quote": row["review_quote"]}
                for row in tree.reviews.values()
                if tree.effective_review(row["revision_id"], row["option"]) is row
                and (local_memory_visible or int(row.get("question_number") or 0)
                     == int(question_number))],
            "current_revisions": [
                row["revision_id"] for row in tree.revisions.values()
                if int(row.get("question_number") or 0) == int(question_number)],
            "accepted_reviews": [
                row["review_id"] for row in tree.reviews.values()
                if row["outcome"] == "accept"
                and tree.effective_review(row["revision_id"], row["option"]) is row
                and not tree.review_stale(row["revision_id"])
                and (local_memory_visible or int(row.get("question_number") or 0)
                     == int(question_number))],
            "memory_candidate": copy.deepcopy(context["memory_candidate"]),
            "card_list_visible": bool(card_list_visible),
            "card_local_visible": bool(card_local_visible),
            "card_global_visible": bool(card_global_visible),
            "force_card_check": bool(force_card_check),
            "memory_cards": [
                {key: copy.deepcopy(card.get(key)) for key in (
                    "card_id", "scope", "source_revision_id",
                    "selected_anchor_revision_id", "global_claim_id",
                    "source_question", "claim", "quote", "option",
                    "check_status", "memory_kind", "time_span",
                    "stored_frames", "source_revision_ids",
                    "timeline_entries", "coverage", "timeline_version",
                    "selected_anchor_entry_id", "timeline_bucket",
                    "compression")}
                for card in card_rows],
            "global_stage": bool(global_stage),
            "global_stage_done": bool(global_stage_done),
            "global_promotable_revisions": [
                row["revision_id"] for row in global_promotable],
            "global_claims": [row["claim_id"] for row in global_claims],
            "memory_use_fits": bool(context["memory_use_fits"]),
            "memory_uses": [
                {"use_id": row["use_id"], "revision_id": row["revision_id"],
                 "action": row["action"], "outcome": row["outcome"],
                 "option": row["option"], "quote": row["quote"]}
                for row in tree.memory_uses.values()
                if int(row.get("question_number") or 0) == int(question_number)
            ],
            "terminal_available": context["terminal_available"],
            "submit_available": context["submit_available"],
            "accepted_card_checks": len(
                context.get("accepted_card_checks") or ()),
            "current_question_read": context["current_question_read"],
            "accepted": bool(context["accepted"]),
            "can_read": bool(context["can_read"]),
            "abstain_available": context["abstain_available"],
            "required_action": "" if (context["memory_candidate"] or
                                      card_list_visible) else (
                "EVOLVE_GLOBAL" if global_stage else
                "READ B%d" % context["empty_branch"]
                if context["empty_branch"] is not None and context["can_read"]
                and not context["budget_abstain"]
                else "REVIEW " + context["review_required"]
                if context["review_required"] else
                "READ B%d" % context["next_read_branch"]
                if context["scan_required"] else
                "ABSTAIN" if context["budget_abstain"] else ""),
            "unread_branches": context["unread_branches"],
            "blocked_branches": context["blocked_branches"],
            "terminal_required": context["terminal_required"],
            "budget_answer_available": context["budget_answer_available"],
            "budget_answer_options": list(context["answer_options"]),
            "time_terminal_stage": time_terminal_stage,
            "time_left_seconds": time_left_seconds,
            "wall_time_policy": {
                "reader_min_runtime_seconds": float(
                    reader_min_runtime_seconds)
                if reader_min_runtime_seconds is not None else None,
                "policy_turn_reserve_seconds": float(
                    policy_turn_reserve_seconds),
                "final_answer_reserve_seconds": float(
                    final_answer_reserve_seconds),
                "min_check_to_final_seconds": float(
                    min_check_to_final_seconds),
            },
            "unreviewed_answer_revisions": context["unreviewed_answer_revisions"],
            "empty_open_branch": context["empty_branch"],
            "frames_left": frames_left,
            "calls_left": calls_left,
            "stage": menu_stage,
            "recommended_action_verbs": sorted(recommended_verbs),
            "recommended_actions": [action.as_dict()
                                    for action in recommended_specs],
            "recommended_action_signatures": [
                list(_action_signature(action)) for action in recommended_specs],
            "legal_action_verbs": sorted(legal_verbs),
            "other_legal_action_verbs": sorted(
                legal_verbs - recommended_verbs),
            "menu_mode": bool(menu_mode),
            "menu_layout": str(menu_layout),
            "memory_menu_order": str(memory_menu_order),
            "memory_decision_timing": str(memory_decision_timing),
            "menu_decision_mode": ("explore" if menu_explore else "guided"),
            "off_menu_probability": (float(off_menu_probability)
                                     if off_menu_probability is not None else 1.0),
        }
        try:
            prompt = _turn_prompt(
                tree, question, branches, open_branch, question_time=question_time,
                question_number=question_number,
                last_refusal=refusals[-1] if refusals else None,
                opened_branches=opened_branches, frames_left=frames_left,
                calls_left=calls_left, memory_visible=memory_visible,
                local_memory_visible=local_memory_visible,
                allow_read_window=allow_read_window,
                legality_only=legality_only,
                memory_use=memory_use and not memory_decision_expired,
                menu_mode=menu_mode, menu_explore=menu_explore,
                global_stage=global_stage, global_memory=global_memory,
                force_answer_stage=(answer_stage or time_terminal_stage),
                menu_layout=menu_layout,
                memory_menu_order=memory_menu_order,
                memory_decision_timing=memory_decision_timing,
                scope_rotation=int(policy_seed or 0),
                card_list_visible=card_list_visible,
                card_local_visible=card_local_visible,
                card_global_visible=card_global_visible,
                cards_per_scope=cards_per_scope,
                global_source=global_source, read_steer=read_steer,
                facts_block=facts_block,
                force_card_check=force_card_check,
                max_reads_per_question=max_reads_per_question,
                budget_answer=budget_answer,
                force_budget_answer=budget_answer_stage,
                time_left_seconds=time_left_seconds,
                min_check_to_final_seconds=min_check_to_final_seconds,
                reader_min_runtime_seconds=reader_min_runtime_seconds,
                policy_turn_reserve_seconds=policy_turn_reserve_seconds,
                final_answer_reserve_seconds=final_answer_reserve_seconds)
            attempted_repair = repair is not None
            if attempted_repair:
                prompt = _action_repair_prompt(prompt, repair["reply"],
                                               repair["why"],
                                               repair.get("suggestion", ""))
                repair = None
                ask.extra = {"repair": True}
            ask.extra = dict(getattr(ask, "extra", None) or {})
            displayed_completions = []
            if menu_mode:
                displayed_completions = _allowed_action_completions(prompt)
                state_seen["legal_action_completions"] = list(
                    displayed_completions)
                state_seen["legal_action_signatures"] = [
                    list(_action_signature(parse_action(text)))
                    for text in displayed_completions]
                state_seen["legal_action_verbs"] = sorted({
                    parse_action(text).verb for text in displayed_completions})
                allowed_completions = displayed_completions
                if not allowed_completions:
                    raise ActionError("menu prompt displays no legal action")
                ask.extra["allowed_completions"] = allowed_completions
            reply = ask(prompt, (),
                role="turn", temperature=(0.0 if attempted_repair
                                          else float(policy_temperature)),
                seed=policy_seed)
        except BaseException as error:
            error.atomic_row = result("")
            raise
        policy_calls = int(getattr(ask, "state", {}).get("n") or 0) - before_calls
        last_policy_request_id = getattr(ask, "last_request_id", "")
        policy_request_id = last_policy_request_id
        last_call = getattr(ask, "last_call", None)
        normalized_reply = _complete_submit_quote(
            _normalize_policy_reply(reply), context)
        try:
            action = parse_action(normalized_reply)
        except ActionError as error:
            refusal = {"kind": "action_refused", "why": str(error)}
            refusals.append(refusal)
            turn("refused", reply, str(error), policy_calls, 0, refusal)
            if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            repair = {"reply": reply, "why": str(error),
                      "suggestion": _repair_suggestion(prompt, context)}
            pending_repair_call = last_call
            repair_count += 1
            if calls_left is not None and int(calls_left) <= 1:
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            continue
        if normalized_reply != reply or getattr(action, "normalized", False):
            if hasattr(ask, "mark_normalized"):
                ask.mark_normalized(
                    policy_request_id, normalized_reply, action.verb)
            elif last_call is not None:
                last_call["normalized_action"] = True
                last_call["normalized_reply"] = normalized_reply
                last_call["parsed_action"] = action.verb
                last_call["action_line_count"] = 1
        menu_class = _menu_class(prompt, action)
        state_seen["selected_action"] = action.as_dict()
        state_seen["selected_action_signature"] = list(_action_signature(action))
        state_seen["selected_menu_class"] = menu_class
        state_seen["selected_preferred_target"] = (
            _action_signature(action) in {
                _action_signature(value) for value in recommended_specs})
        semantic_error = ""
        semantic_kind = "action_refused"
        if action.verb in ("read", "open") and action.branch not in branches:
            semantic_error = "branch does not exist"
            semantic_kind = "open_refused" if action.verb == "open" else "action_refused"
        elif action.verb in ("use", "defer") \
                and context["memory_candidate"] is not None \
                and action.revision != int(
                    context["memory_candidate"]["revision_id"][1:]):
            semantic_error = "action must name the offered candidate %s" % (
                context["memory_candidate"]["revision_id"])
            semantic_kind = "memory_use_refused"
        elif action.verb == "review":
            revision_id = "R%d" % action.revision
            revision = tree.revisions.get(revision_id)
            if revision is None:
                semantic_error = "revision does not exist"
                semantic_kind = "review_refused"
            elif not memory_visible and int(
                    revision.get("question_number") or 0) != int(question_number):
                semantic_error = "prior-question review is hidden"
                semantic_kind = "memory_ablation"
            elif not _review_applies(tree, revision, question_number):
                semantic_error = "review belongs to another question"
                semantic_kind = "review_scope"
            elif tree.review_stale(revision_id):
                semantic_error = "revision has been superseded"
                semantic_kind = "review_refused"
            elif not answer_bearing(revision):
                semantic_error = "revision is not answer-bearing"
                semantic_kind = "review_refused"
            elif revision_id not in context["unreviewed_answer_revisions"]:
                semantic_error = "revision is not currently reviewable"
                semantic_kind = "review_refused"
        elif action.verb == "evolve_global" and not global_stage:
            semantic_error = "global-memory stage is not available"
            semantic_kind = "global_stage_refused"
        elif context["review_required"] and action.verb != "review":
            semantic_error = "review the answer-bearing revision before another action"
            semantic_kind = "review_required"
        if semantic_error:
            refusal = {"kind": semantic_kind, "why": semantic_error}
            refusals.append(refusal)
            turn("refused", reply, semantic_error, policy_calls, 0, refusal)
            if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            repair = {"reply": reply, "why": semantic_error,
                      "suggestion": _repair_suggestion(prompt, context)}
            pending_repair_call = last_call
            repair_count += 1
            if calls_left is not None and int(calls_left) <= 1:
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            continue
        if global_stage and action.verb != "evolve_global":
            refusal = {
                "kind": "global_stage_required",
                "why": "choose one EVOLVE_GLOBAL action before the answer stage",
            }
            refusals.append(refusal)
            turn(action.verb, reply, refusal["why"], policy_calls, 0, refusal)
            if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            repair = {"reply": reply, "why": refusal["why"],
                      "suggestion": _repair_suggestion(prompt, context)}
            pending_repair_call = last_call
            repair_count += 1
            continue
        if action.verb == "evolve_global":
            provenance = {
                "actor": "o",
                "adapter": "o",
                "call_id": str(last_policy_request_id or ""),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "reply_sha256": hashlib.sha256(str(reply).encode("utf-8")).hexdigest(),
                "policy_seed": policy_seed,
            }
            accepted, why, claim, operation = tree.evolve_global(
                action.global_operation, question_number,
                source_revision_id=("R%d" % action.revision
                                    if action.revision is not None else ""),
                target_claim_id=("G%d" % action.global_target
                                 if action.global_target is not None else ""),
                provenance=provenance)
            memory_op = {
                "op": "evolve_global",
                "operation": action.global_operation,
                "operation_id": operation["operation_id"],
                "claim_id": (claim or {}).get("claim_id", ""),
                "target_claim_id": operation.get("target_claim_id", ""),
                "source_revision_id": operation.get("source_revision_id", ""),
                "accepted": bool(accepted),
                "committed": bool(accepted),
                "global_memory_reward": GLOBAL_REWARD if accepted else 0.0,
            }
            global_turn = turn(
                "evolve_global", reply,
                "%s %s" % (operation["operation_id"],
                           "accepted" if accepted else why),
                policy_calls, 0,
                refusal=None if accepted else {
                    "kind": "global_operation_refused", "why": why})
            global_turn["memory_ops"] = [memory_op]
            if accepted:
                global_stage_done = True
                mark_repair_recovered()
                continue
            refusal = {"kind": "global_operation_refused", "why": why}
            refusals.append(refusal)
            if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            repair = {"reply": reply, "why": why,
                      "suggestion": _repair_suggestion(prompt, context)}
            pending_repair_call = last_call
            repair_count += 1
            continue
        final_stage = bool(state_seen.get("budget_answer_available")
                           or state_seen.get("time_terminal_stage"))
        final_error = ""
        if final_stage:
            final_completions = set(_allowed_action_completions(prompt))
            try:
                selected_final = _action_text(action)
            except ActionError:
                selected_final = ""
            if selected_final not in final_completions:
                final_error = "the final budget stage requires one displayed SUBMIT"
        if final_error:
            refusal = {"kind": "budget_answer_required", "why": final_error}
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1) \
                    or (calls_left is not None and int(calls_left) <= 1):
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            repair = {"reply": reply, "why": refusal["why"],
                      "suggestion": _repair_suggestion(prompt, context)}
            pending_repair_call = last_call
            repair_count += 1
            continue
        if action.verb == "check":
            card = card_handles.get(action.memory_handle)
            if not card_list_visible or card is None:
                refusal = {"kind": "check_refused",
                           "why": "memory card handle is not offered"}
                refusals.append(refusal)
                turn("check", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            prior_checks = [
                row for row in tree.memory_uses.values()
                if str(row.get("card_id") or "") == str(card.get("card_id") or "")
                and str(row.get("revision_id") or "") == str(
                    card.get("source_revision_id") or "")
                and int(row.get("question_number") or 0) == int(question_number)
                and row.get("action") == "check"]
            if prior_checks:
                refusal = {"kind": "check_refused",
                           "why": "memory card was already checked this question"}
                refusals.append(refusal)
                turn("check", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            stored = TREE.load_frames(
                card.get("frame_handle"), limit=MEMORY_USE_FRAMES)
            frames_left = max(0, int(frame_cap) - frames_used)
            if (not stored or len(stored) > frames_left
                    or (calls_left is not None and int(calls_left) < 2)):
                refusal = {
                    "kind": "check_refused",
                    "why": ("memory card has no stored frames" if not stored else
                            "CHECK does not fit the remaining call/frame budget"),
                }
                refusals.append(refusal)
                turn("check", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            prompt = _card_check_prompt(card, question)
            outcome = "invalid"
            option = ""
            quote = ""
            reason = ""
            invalid_reason = ""
            check_reply = ""
            check_prompt_used = prompt
            agreement = {}
            card_revision = str(card.get("source_revision_id") or "")
            first_invalid_reason = ""
            for attempt in range(2):
                attempt_prompt = (prompt if attempt == 0 else
                                  _memory_use_repair_prompt(
                                      prompt, invalid_reason,
                                      revision=(int(card_revision[1:])
                                                if re.fullmatch(
                                                    r"R\d+", card_revision)
                                                else None)))
                check_prompt_used = attempt_prompt
                role = "check-card-%s%s" % (
                    str(card["card_id"]).replace(":", ""),
                    "" if attempt == 0 else "-retry")
                check_reply = ask(
                    attempt_prompt, stored, role=role,
                    temperature=float(review_temperature), seed=policy_seed)
                try:
                    parsed = parse_review(check_reply)
                    source_revision = str(
                        card.get("source_revision_id") or "")
                    expected = (int(source_revision[1:])
                                if re.fullmatch(r"R\d+", source_revision) else None)
                    if parsed["revision"] != expected:
                        raise ActionError("CHECK named another revision")
                    if parsed["outcome"] == "accept":
                        option = str(parsed.get("option") or "").upper()
                        quote = str(parsed.get("quote") or "")
                        if not option_is_valid(option, options):
                            raise ActionError("CHECK named an unknown option")
                        agreement = MEMORY_CLAIM.card_agreement(
                            question, card, options, option, quote)
                        if not agreement["ok"]:
                            raise ActionError(
                                "blind CHECK evidence does not agree with the card")
                    else:
                        reason = parsed["why"]
                    outcome = parsed["outcome"]
                    invalid_reason = ""
                    reason = "" if outcome == "accept" else reason
                    break
                except ActionError as error:
                    invalid_reason = str(error)
                    reason = invalid_reason
                    if attempt == 0:
                        first_invalid_reason = invalid_reason
                    remaining_calls = (None if call_cap is None else
                                       int(call_cap) - int(ask.state["n"]))
                    remaining_frames = int(frame_cap) - int(ask.state["frames"])
                    check_deadline = getattr(ask, "state", {}).get("deadline")
                    check_time_left = (
                        max(0.0, float(check_deadline) - time.time())
                        if check_deadline is not None else None)
                    if attempt == 0 and (remaining_calls is None
                                         or remaining_calls > 1) \
                            and remaining_frames >= len(stored) \
                            and (check_time_left is None
                                 or check_time_left >= float(
                                     final_answer_reserve_seconds)):
                        continue
                    break
            total_calls = int(ask.state["n"]) - before_calls
            total_frames = int(ask.state["frames"]) - before_frames
            frames_used += total_frames
            candidate = {
                "revision_id": str(card.get("source_revision_id") or ""),
                "source_question_number": int(card.get("source_question") or 0),
                "stored_frames": int(card.get("stored_frames") or 0),
                "preview": str(card.get("claim") or ""),
                "preview_sha256": hashlib.sha256(
                    str(card.get("claim") or "").encode("utf-8")).hexdigest(),
            }
            record = tree.record_memory_use(
                candidate, question_number, "check", outcome,
                option=option, quote=quote, reason=reason,
                frames=len(stored), frames_from_store=True,
                prompt_sha256=hashlib.sha256(
                    check_prompt_used.encode()).hexdigest(),
                reply_sha256=hashlib.sha256(check_reply.encode()).hexdigest(),
                call_id=getattr(ask, "last_request_id", ""),
                card_id=str(card.get("card_id") or ""),
                scope=str(card.get("scope") or ""),
                global_claim_id=str(card.get("global_claim_id") or ""),
                agreement=agreement)
            memory_use_records.append(record)
            check_turn = turn(
                "check", check_reply, "%s %s" % (record["use_id"], outcome),
                total_calls, total_frames)
            check_turn["memory_ops"] = [{
                "op": "check", "entry_id": record["use_id"],
                "card_id": record["card_id"], "scope": record["scope"],
                "target_revision_id": record["revision_id"],
                "committed": True,
            }]
            if invalid_reason:
                invalid = {"kind": "check_invalid", "why": invalid_reason}
                if first_invalid_reason != invalid_reason:
                    invalid["first_why"] = first_invalid_reason
                refusals.append(invalid)
                continue
            mark_repair_recovered()
            continue

        if (action.verb == "submit"
                and state_seen.get("budget_answer_available")
                and not action.memory_handle and not action.revision
                and not action.quote):
            answer_error = ""
            if not option_is_valid(action.option, options):
                answer_error = "budget SUBMIT named an option outside this question"
            if answer_error:
                refusal = {"kind": "submit_refused", "why": answer_error}
                refusals.append(refusal)
                turn("refused", reply, answer_error, policy_calls, 0, refusal)
                if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                repair = {"reply": reply, "why": answer_error,
                          "suggestion": _repair_suggestion(prompt, context)}
                pending_repair_call = last_call
                repair_count += 1
                if calls_left is not None and int(calls_left) <= 1:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            answer = C.option_text(action.option, options)
            budget_submit = True
            ended_by = "agent_submit"
            turn("submit", reply, "budget submission %s" % action.option,
                 policy_calls, 0)
            mark_repair_recovered()
            break

        if (action.verb == "submit" and not action.memory_handle
                and not action.revision and not action.quote):
            refusal = {
                "kind": "submit_refused",
                "why": "bare SUBMIT is available only at the budget-final stage",
            }
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            continue

        if context["memory_candidate"] and action.verb not in ("use", "defer"):
            memory_decision_expired = True
        if context["budget_abstain"] and action.verb != "abstain":
            refusal = {
                "kind": "abstain_required",
                "why": "the remaining budget is reserved for the terminal action",
            }
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            ended_by = "protocol_abstain"
            answer = ""
            break
        if context["terminal_required"] and not (
                action.verb == "abstain"
                or action.verb == "review" and bool(context["review_required"])
                or action.verb == "submit" and (
                    state_seen["submit_available"] or action.memory_handle
                    or state_seen.get("budget_answer_available"))):
            refusal = {
                "kind": "terminal_required",
                "why": "the remaining budget is reserved for the terminal action",
            }
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            ended_by = "protocol_abstain"
            answer = ""
            break
        required_action = state_seen["required_action"]
        empty_branch = state_seen["empty_open_branch"]
        allowed_empty_open = bool(
            empty_branch is not None and action.verb == "open"
            and action.branch == empty_branch)
        if (not card_list_visible and not legality_only and required_action
                and not allowed_empty_open
                and not _required_action_matches(action, required_action, branches)):
            refusal = {
                "kind": ("review_required" if required_action.startswith("REVIEW")
                         else "abstain_required" if required_action == "ABSTAIN"
                         else "read_required"),
                "why": "next action must be %s" % required_action,
            }
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            continue
        if action.verb in ("use", "defer"):
            candidate = context["memory_candidate"]
            expected_revision = (candidate or {}).get("revision_id")
            if candidate is None or "R%d" % action.revision != expected_revision:
                refusal = {
                    "kind": "memory_use_refused",
                    "why": ("no prior memory candidate is offered"
                            if candidate is None else
                            "action must name the offered candidate %s"
                            % expected_revision),
                }
                refusals.append(refusal)
                turn(action.verb, reply, refusal["why"], policy_calls, 0, refusal)
                if sum(item.get("kind") == "memory_use_refused"
                       for item in refusals) >= MAX_MEMORY_ACTION_REFUSALS:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            if action.verb == "defer":
                record = tree.record_memory_use(
                    candidate, question_number, "defer", "defer")
                memory_use_records.append(record)
                use_turn = turn(
                    "defer", reply, "%s deferred" % record["use_id"],
                    policy_calls, 0)
                use_turn["memory_ops"] = [{
                    "op": "defer", "entry_id": record["use_id"],
                    "target_revision_id": record["revision_id"],
                    "committed": True,
                }]
                if context["calls_left"] is not None \
                        and int(context["calls_left"]) <= 1:
                    ended_by = "protocol_abstain"
                    answer = ""
                    break
                mark_repair_recovered()
                continue
            if not context["memory_use_fits"]:
                refusal = {
                    "kind": "memory_use_refused",
                    "why": "USE does not fit the remaining call and frame budget",
                }
                refusals.append(refusal)
                turn(action.verb, reply, refusal["why"], policy_calls, 0, refusal)
                if (context["calls_left"] is not None
                        and int(context["calls_left"]) <= 2) \
                        or sum(item.get("kind") == "memory_use_refused"
                               for item in refusals) \
                        >= MAX_MEMORY_ACTION_REFUSALS:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            revision = tree.revisions.get(candidate["revision_id"])
            stored = (TREE.load_frames(revision.get("frame_handle"),
                                       limit=MEMORY_USE_FRAMES)
                      if revision else [])
            frames_left = max(0, int(frame_cap) - frames_used)
            if not stored or len(stored) > frames_left:
                refusal = {
                    "kind": "memory_use_refused",
                    "why": ("prior memory has no stored frames"
                            if not stored else
                            "prior memory needs %d frames; %d remain"
                            % (len(stored), frames_left)),
                }
                refusals.append(refusal)
                turn(action.verb, reply, refusal["why"], policy_calls, 0, refusal)
                if sum(item.get("kind") == "memory_use_refused"
                       for item in refusals) >= MAX_MEMORY_ACTION_REFUSALS:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            memory_prompt = _memory_use_prompt(
                tree, candidate, question,
                local_memory_visible=local_memory_visible)
            use_reason = ""
            use_option = ""
            use_quote = ""
            use_outcome = "invalid"
            invalid_reason = ""
            memory_reply = ""
            for attempt in range(2):
                attempt_prompt = (memory_prompt if attempt == 0 else
                                  _memory_use_repair_prompt(
                                      memory_prompt, invalid_reason))
                role = "review-use-%s%s" % (
                    candidate["revision_id"][1:], "" if attempt == 0 else "-retry")
                try:
                    memory_reply = ask(
                        attempt_prompt, stored, role=role,
                        temperature=float(review_temperature), seed=policy_seed)
                except BaseException as error:
                    error.atomic_row = result("")
                    raise
                try:
                    parsed = parse_review(memory_reply)
                    if parsed["revision"] != action.revision:
                        raise ActionError("USE check named another revision")
                    if parsed["outcome"] == "accept":
                        use_option = str(parsed.get("option") or "").upper()
                        use_quote = str(parsed.get("quote") or "")
                        if not option_is_valid(use_option, options):
                            raise ActionError("USE check named an unknown option")
                        grounding = " ".join((
                            str(revision.get("branch_summary") or ""),
                            str(revision.get("quote") or ""),
                            str(tree.events[candidate["event_id"]].get("text") or "")))
                        if not use_quote or not C._quote_grounded(use_quote, grounding):
                            raise ActionError("USE quote is not in the prior memory")
                        if not MEMORY_CLAIM.covers_question_attribute(
                                question, use_quote, options, option=use_option):
                            raise ActionError(
                                "USE quote does not cover a question attribute")
                        use_outcome = "accept"
                    else:
                        use_outcome = parsed["outcome"]
                        use_reason = parsed["why"]
                    invalid_reason = ""
                    break
                except ActionError as error:
                    invalid_reason = str(error)
                    use_reason = invalid_reason
                    remaining_calls = (None if call_cap is None else
                                       int(call_cap) - int(ask.state["n"]))
                    remaining_frames = int(frame_cap) - int(ask.state["frames"])
                    if attempt == 0 and (remaining_calls is None
                                         or remaining_calls > 1) \
                            and remaining_frames >= len(stored):
                        continue
                    break
            total_calls = int(ask.state["n"]) - before_calls
            total_frames = int(ask.state["frames"]) - before_frames
            frames_used += total_frames
            record = tree.record_memory_use(
                candidate, question_number, "use", use_outcome,
                option=use_option, quote=use_quote, reason=use_reason,
                frames=len(stored), frames_from_store=True,
                prompt_sha256=hashlib.sha256(memory_prompt.encode()).hexdigest(),
                reply_sha256=hashlib.sha256(memory_reply.encode()).hexdigest(),
                call_id=getattr(ask, "last_request_id", ""))
            memory_use_records.append(record)
            use_turn = turn(
                "use", memory_reply,
                "%s %s" % (record["use_id"], record["outcome"]),
                total_calls, total_frames)
            use_turn["memory_ops"] = [{
                "op": "use", "entry_id": record["use_id"],
                "target_revision_id": record["revision_id"],
                "committed": True,
            }]
            if invalid_reason:
                refusals.append({
                    "kind": "memory_use_invalid", "why": invalid_reason})
                ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                protocol_fallback = bool(menu_mode)
                answer = ""
                break
            mark_repair_recovered()
            continue
        if action.verb == "read" and action.window is not None \
                and not allow_read_window and action.branch in branches \
                and tuple(action.window) != tuple(branches[action.branch]):
            refusal = {"kind": "read_refused",
                       "why": "direct-O first rung allows whole-branch READ only"}
            refusals.append(refusal)
            turn("read", reply, refusal["why"], policy_calls, 0, refusal)
            continue
        if (action.verb in ("submit", "abstain")
                and not action.memory_handle
                and state_seen["accepted"]
                and not state_seen["current_question_read"]
                and state_seen["can_read"]):
            refusal = {"kind": "current_read_required",
                       "why": "read a current-question branch before ending the question"}
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            continue
        if (not legality_only and action.verb == "submit"
                and not action.memory_handle
                and not state_seen["terminal_available"]
                and not state_seen.get("budget_answer_available")):
            refusal = {"kind": "no_evidence",
                       "why": "read at least one branch before ending the question"}
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            continue
        if (not legality_only and action.verb == "abstain"
                and not state_seen["abstain_available"]):
            refusal = {"kind": "no_evidence",
                       "why": "read at least one branch before ending the question"}
            refusals.append(refusal)
            turn("refused", reply, refusal["why"], policy_calls, 0, refusal)
            continue
        if action.verb == "open":
            if action.branch not in branches:
                refusal = {"kind": "open_refused", "why": "branch does not exist"}
                refusals.append(refusal)
                turn("open", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if action.branch in opened_branches:
                empty = not tree.latest_summary(action.branch)["text"] and not (
                    tree.current_events(action.branch))
                refusal = {
                    "kind": "empty_branch" if empty else "open_refused",
                    "why": ("branch B%d has no memory; next send READ B%d"
                            % (action.branch, action.branch)) if empty else
                           "branch B%d is already open; choose READ, REVIEW or a terminal"
                           % action.branch}
                refusals.append(refusal)
                turn("open", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            open_branch = action.branch
            opened_branches.add(action.branch)
            if not memory_visible:
                rendered = "(memory ablation: hidden)"
            else:
                rendered = tree.render_branch(
                    open_branch, question_time=question_time,
                    question_number=(None if local_memory_visible
                                     else question_number))
            if _empty_branch(tree, open_branch) is not None:
                rendered += "\nNo memory exists in this branch."
                if legality_only:
                    rendered += " READ B%d retrieves frames from it." % open_branch
                else:
                    rendered += " Next send ACTION: READ B%d." % open_branch
            turn("open", reply,
                 rendered, policy_calls, 0)
            mark_repair_recovered()
            continue
        if action.verb == "read":
            if int(read_counts.get(action.branch, 0)) >= MAX_READS_PER_BRANCH:
                refusal = {
                    "kind": "read_refused",
                    "why": ("branch B%d has already been read twice; choose "
                            "another unread branch or review its evidence"
                            % action.branch),
                }
                refusals.append(refusal)
                safe_branch = context["next_read_branch"] if menu_mode else None
                can_recover = bool(
                    repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1)
                    and safe_branch is not None
                    and int(safe_branch) != int(action.branch))
                if can_recover:
                    action = Action("read", branch=int(safe_branch))
                    reply = "ACTION: READ B%d" % int(safe_branch)
                    state_seen["selected_action"] = action.as_dict()
                    state_seen["selected_menu_class"] = "in_menu"
                    state_seen["selected_preferred_target"] = True
                    state_seen["harness_recovery"] = True
                    harness_recovery = True
                    fallback_refusal = refusal
                else:
                    turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                if not can_recover and repair_count >= (
                        MAX_POLICY_REPAIRS if menu_mode else 1):
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                if not can_recover:
                    repair = {"reply": reply, "why": refusal["why"],
                              "suggestion": _repair_suggestion(prompt, context)}
                    pending_repair_call = last_call
                    repair_count += 1
                if not can_recover and calls_left is not None \
                        and int(calls_left) <= 1:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                if not can_recover:
                    continue
            if action.branch not in branches:
                refusal = {"kind": "read_refused", "why": "branch does not exist"}
                refusals.append(refusal)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if action.branch in set(state_seen.get("blocked_branches") or ()):
                refusal = {
                    "kind": "branch_blocked",
                    "why": ("B%d has two non-ACCEPT reviews; read another "
                            "unread branch or abstain" % action.branch),
                }
                refusals.append(refusal)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                if _policy_refusal_count(refusals) >= MAX_POLICY_REFUSALS:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            live_deadline = getattr(ask, "state", {}).get("deadline")
            live_time_left = (
                max(0.0, float(live_deadline) - time.time())
                if live_deadline is not None else None)
            if (reader_min_runtime_seconds is not None
                    and live_time_left is not None
                    and live_time_left < (
                        float(reader_min_runtime_seconds)
                        + float(final_answer_reserve_seconds))):
                refusal = {
                    "kind": "read_time_refused",
                    "why": ("READ and final answer need at least %.0f seconds "
                            "remaining; "
                            "%.1f remain" % (
                                float(reader_min_runtime_seconds)
                                + float(final_answer_reserve_seconds),
                                float(live_time_left))),
                }
                refusals.append(refusal)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            first, last = (branches[action.branch] if not allow_read_window
                           else action.window or branches[action.branch])
            base_first, base_last = branches[action.branch]
            if first < base_first or last > base_last:
                refusal = {"kind": "read_refused", "why": "window is outside the branch"}
                refusals.append(refusal)
                open_branch = action.branch
                opened_branches.add(action.branch)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            requested = max(1, round((last - first) * READ_FPS))
            n = min(requested, max(0, int(frame_cap) - frames_used))
            if not allow_read_window and n < requested:
                refusal = {
                    "kind": "read_refused",
                    "why": ("whole branch needs %d frames; %d remain"
                            % (requested, max(0, int(frame_cap) - frames_used))),
                }
                refusals.append(refusal)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if n < 1:
                refusal = {"kind": "read_refused", "why": "no frame budget remains"}
                refusals.append(refusal)
                turn("read", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            revision_id = tree.allocate_revision_id()
            read_counts[int(action.branch)] = int(
                read_counts.get(int(action.branch), 0)) + 1
            images = []
            reader_calls_before = int(ask.state["n"])
            reader_frames_before = int(ask.state["frames"])
            try:
                images = frames_of("atomic-%d" % action.branch, first, last,
                                   n / float(last - first), n)
                ask.extra = {"reader_span": [float(first), float(last)]}
                reader_reply = ask(
                    _reader_prompt(
                        tree, action.branch, first, last,
                        n / float(last - first), n, question,
                        question_number=question_number,
                        memory_visible=memory_visible,
                        local_memory_visible=local_memory_visible,
                        reader_policy=reader_policy,
                        reader_schema_version=reader_schema_version,
                        reader_allow_keep=reader_allow_keep),
                    images, role="loopread-%d" % action.branch,
                    temperature=float(reader_temperature),
                    seed=(policy_seed if reader_seed is None else reader_seed))
                parsed = parse_reader_report(
                    reader_reply, first=first, last=last,
                    reader_policy=reader_policy,
                    reader_schema_version=reader_schema_version,
                    allow_keep=reader_allow_keep,
                    max_options=len(options))
                frame_handle = _save_frames(root, tree.video, action.branch,
                                            (first, last), images, revision_id)
                frame_status = "ok" if frame_handle else "failed"
                frame_error = "" if frame_handle else "no frames stored"
                if on_read_precommit is not None:
                    on_read_precommit({
                        "tree": tree.to_dict(),
                        "reader_reply": reader_reply,
                        "parsed_report": parsed,
                        "frame_handle": frame_handle,
                        "branch": int(action.branch),
                        "span": [float(first), float(last)],
                        "revision_id": revision_id,
                        "question_number": int(question_number),
                        "question": question,
                        "action": action.as_dict(),
                        "call_cap": call_cap,
                        "frame_cap": int(frame_cap),
                        "max_turns": int(max_turns),
                        "calls_before_read": reader_calls_before,
                        "frames_before_read": reader_frames_before,
                        "policy_seed": policy_seed,
                        "reader_seed": reader_seed,
                        "policy_temperature": policy_temperature,
                        "reader_temperature": reader_temperature,
                        "memory_visible": bool(memory_visible),
                        "local_memory_visible": bool(local_memory_visible),
                        "global_memory": bool(global_memory),
                        "menu_layout": str(menu_layout),
                        "memory_menu_order": str(memory_menu_order),
                        "memory_decision_timing": str(memory_decision_timing),
                    })
            except BaseException as error:
                refusal = {"kind": "read_failed", "why": "%s: %s" % (
                    type(error).__name__, error)}
                refusals.append(refusal)
                tree.append_read(
                    action.branch, first, last, n / float(last - first),
                    "", {"parse_status": "failed"}, None, revision_id=revision_id,
                    frame_status="failed", requested_frames=requested,
                    actual_frames=len(images), rendered_frames=len(images),
                    effective_fps=(len(images) / float(last - first)
                                   if images else 0.0), frame_error="%s: %s" % (
                        type(error).__name__, error), question_number=question_number)
                read_turn = turn(
                    "read", reply, refusal["why"], policy_calls,
                    len(images), refusal)
                read_turn["revision_id"] = revision_id
                error.atomic_row = result("")
                raise
            actual_frames = len(images)
            effective_fps = actual_frames / float(last - first)
            revision = tree.append_read(
                action.branch, first, last, n / float(last - first),
                reader_reply, parsed, frame_handle, revision_id=revision_id,
                frame_status=frame_status, requested_frames=requested,
                actual_frames=actual_frames, rendered_frames=actual_frames,
                effective_fps=effective_fps,
                frame_error=frame_error,
                question_number=question_number)
            revision["question_number"] = int(question_number)
            frames_used += len(images)
            calls = int(ask.state["n"]) - before_calls
            frames = int(ask.state["frames"]) - before_frames
            read_turn = turn("read", reply,
                             "%s %s" % (revision_id, revision["read_status"]),
                             calls, frames, refusal=fallback_refusal)
            read_turn["revision_id"] = revision_id
            read_turn["memory_ops"] = [{
                "op": "append", "entry_id": revision_id,
                "branch": int(action.branch),
                "time_span": [float(first), float(last)],
                "committed": frame_status == "ok",
            }]
            mark_repair_recovered()
            continue
        if action.verb == "review":
            revision = tree.revisions.get("R%d" % action.revision)
            if (not memory_visible and revision is not None
                    and int(revision.get("question_number") or 0)
                    != int(question_number)):
                refusal = {"kind": "memory_ablation",
                           "why": "prior-question review is hidden"}
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if revision is not None and not _review_applies(
                    tree, revision, question_number):
                refusal = {"kind": "review_scope",
                           "why": "revision belongs to another question; read current evidence"}
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if revision is None or not answer_bearing(revision):
                refusal = {"kind": "review_refused", "why": "revision is not answer-bearing"}
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if "R%d" % action.revision not in state_seen[
                    "unreviewed_answer_revisions"]:
                refusal = {
                    "kind": "review_refused",
                    "why": "revision is not the current unreviewed answer; read or abstain",
                }
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            if tree.review_stale(action.revision):
                refusal = {"kind": "review_refused", "why": "revision has been superseded"}
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            stored = TREE.load_frames(revision.get("frame_handle"), limit=REVIEW_FRAMES)
            if not stored:
                refusal = {"kind": "review_refused", "why": "revision has no stored frames"}
                refusals.append(refusal)
                turn("review", reply, refusal["why"], policy_calls, 0, refusal)
                continue
            review_prompt = _review_prompt(
                tree, revision, question, question_time=question_time,
                memory_visible=memory_visible,
                local_memory_visible=local_memory_visible,
                question_number=question_number)
            review_turn = turn("review", reply, "review pending", policy_calls, 0)
            review_error = None
            record = None
            for attempt in range(2):
                attempt_prompt = (review_prompt if attempt == 0 else
                                  _review_repair_prompt(
                                      tree, revision, question, review_error,
                                      question_time=question_time,
                                      memory_visible=memory_visible,
                                      local_memory_visible=local_memory_visible,
                                      question_number=question_number))
                role = "review-%s%s" % (
                    revision["branch_id"][1:], "" if attempt == 0 else "-retry")
                try:
                    review_reply = ask(
                        attempt_prompt, stored, role=role,
                        temperature=float(review_temperature), seed=policy_seed)
                except BaseException as error:
                    error.atomic_row = result("")
                    raise
                review_turn.update(
                    calls=int(ask.state["n"]) - before_calls,
                    frames=int(ask.state["frames"]) - before_frames)
                try:
                    parsed = parse_review(review_reply)
                    if parsed["revision"] != action.revision:
                        raise ActionError("review named another revision")
                    reviewed_option = (parsed["option"]
                                       if parsed["outcome"] == "accept" else "")
                    reviewed_quote = (parsed["quote"]
                                      if parsed["outcome"] == "accept" else "")
                    if (parsed["outcome"] == "accept"
                            and not option_is_valid(reviewed_option, options)):
                        raise ActionError(
                            "accepted review named an option outside this question")
                    record = tree.append_review(
                        action.revision, reviewed_option, parsed["outcome"],
                        reviewed_quote, question_number=question_number,
                        reason=parsed["why"], frames=len(stored),
                        frames_from_store=True,
                        call_id=getattr(ask, "last_request_id", ""),
                        stored_frames=int(
                            (revision.get("frame_handle") or {}).get("n") or 0),
                        prompt_sha256=hashlib.sha256(
                            attempt_prompt.encode()).hexdigest(),
                        reply_sha256=hashlib.sha256(
                            review_reply.encode()).hexdigest())
                    review_records.append(record)
                    break
                except ActionError as error:
                    review_error = error
                    remaining_calls = (None if call_cap is None else
                                       int(call_cap) - int(ask.state["n"]))
                    remaining_frames = (int(frame_cap)
                                        - int(ask.state["frames"]))
                    if attempt == 0 and (remaining_calls is None
                                         or remaining_calls > 1) \
                            and remaining_frames >= len(stored):
                        continue
                    refusal = {"kind": "review_refused", "why": str(error)}
                    refusals.append(refusal)
                    review_turn["observation"] = refusal["why"]
                    review_turn["refusals"] = [refusal]
                    break
            frames_used += int(ask.state["frames"]) - before_frames
            if record is None:
                if call_cap is not None and int(ask.state["n"]) >= int(call_cap):
                    ended_by = "protocol_abstain"
                    answer = ""
                    break
                continue
            review_turn.update(
                observation="%s %s" % (record["review_id"], record["outcome"]))
            mark_repair_recovered()
            continue
        if action.verb == "submit":
            if not option_is_valid(action.option, options):
                refusal = {
                    "kind": "option_outside_question",
                    "why": "submit option is outside the current question",
                }
                refusals.append(refusal)
                turn("submit", reply, refusal["why"], policy_calls, 0, refusal)
                if context["terminal_required"]:
                    ended_by = "protocol_abstain"
                    answer = ""
                    break
                continue
            if action.memory_handle:
                card = card_handles.get(action.memory_handle)
                if not card_list_visible or card is None:
                    refusal = {"kind": "submit_refused",
                               "why": "memory card handle is not offered"}
                    refusals.append(refusal)
                    turn("submit", reply, refusal["why"], policy_calls, 0,
                         refusal)
                    continue
                ok, why, prior = tree.memory_card_supports(
                    card, action.option, action.quote, question_number,
                    options=options, require_check=force_card_check)
                if not ok:
                    refusal = {"kind": "submit_refused", "why": why}
                    refusals.append(refusal)
                    turn("submit", reply, why, policy_calls, 0, refusal)
                    continue
                source_revision = str(card.get("source_revision_id") or "")
                if not re.fullmatch(r"R\d+", source_revision):
                    refusal = {"kind": "submit_refused",
                               "why": "memory card has no source revision"}
                    refusals.append(refusal)
                    turn("submit", reply, refusal["why"], policy_calls, 0,
                         refusal)
                    continue
                candidate = {
                    "revision_id": source_revision,
                    "source_question_number": int(
                        card.get("source_question") or 0),
                    "stored_frames": int(card.get("stored_frames") or 0),
                    "preview": str(card.get("claim") or ""),
                    "preview_sha256": hashlib.sha256(
                        str(card.get("claim") or "").encode()).hexdigest(),
                }
                record = tree.record_memory_use(
                    candidate, question_number, "cite", "accept",
                    option=action.option, quote=action.quote,
                    frames=0, frames_from_store=True,
                    card_id=str(card.get("card_id") or ""),
                    scope=str(card.get("scope") or ""),
                    global_claim_id=str(card.get("global_claim_id") or ""))
                memory_use_records.append(record)
                submitted_revision = int(source_revision[1:])
                submitted_option = action.option
                submitted_quote = action.quote
                answer = C.option_text(action.option, options)
                commit = {
                    "unit": source_revision, "option": answer,
                    "quote": action.quote, "entry_id": source_revision,
                    "card_id": record["card_id"], "card_scope": record["scope"],
                    "global_claim_id": record["global_claim_id"],
                }
                card_supported = True
                ended_by = "agent_submit"
                submit_turn = turn(
                    "submit", reply, "cited %s" % action.memory_handle,
                    policy_calls, 0)
                submit_turn["memory_ops"] = [{
                    "op": "cite", "entry_id": record["use_id"],
                    "card_id": record["card_id"], "scope": record["scope"],
                    "target_revision_id": record["revision_id"],
                    "committed": True,
                }]
                mark_repair_recovered()
                break
            revision = tree.revisions.get("R%d" % action.revision)
            if (not memory_visible and revision is not None
                    and int(revision.get("question_number") or 0)
                    != int(question_number)):
                refusal = {"kind": "memory_ablation",
                           "why": "prior-question submission is hidden"}
                refusals.append(refusal)
                turn("submit", reply, refusal["why"], policy_calls, 0, refusal)
                if context["terminal_required"]:
                    ended_by = "protocol_abstain"
                    answer = ""
                    break
                continue
            review = tree.effective_review(action.revision, action.option)
            revision_question = int((revision or {}).get("question_number") or 0)
            if review is not None and revision_question == int(question_number) \
                    and not _review_pair_applies(tree, review, question_number):
                refusal = {"kind": "review_scope",
                           "why": "review belongs to another question; read and review current evidence"}
                refusals.append(refusal)
                turn("submit", reply, refusal["why"], policy_calls, 0, refusal)
                if context["terminal_required"]:
                    ended_by = "protocol_abstain"
                    answer = ""
                    break
                continue
            ok, why = tree.can_submit(
                action.revision, action.option, action.quote,
                question_number=question_number)
            if not ok:
                refusal = {"kind": "submit_refused", "why": why}
                refusals.append(refusal)
                turn("submit", reply, why, policy_calls, 0, refusal)
                if calls_left is None or int(calls_left) > 1:
                    if repair_count >= (MAX_POLICY_REPAIRS if menu_mode else 1):
                        ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                        protocol_fallback = bool(menu_mode)
                        answer = ""
                        break
                    repair = {"reply": reply, "why": why,
                              "suggestion": _repair_suggestion(prompt, context)}
                    pending_repair_call = last_call
                    repair_count += 1
                    continue
                if context["terminal_required"]:
                    ended_by = "agent_abstain" if menu_mode else "protocol_abstain"
                    protocol_fallback = bool(menu_mode)
                    answer = ""
                    break
                continue
            support_quote = review["review_quote"] if review else ""
            if revision_question != int(question_number):
                _ok, _why, use_record = tree.memory_use_supports(
                    action.revision, action.option, action.quote,
                    question_number)
                support_quote = (use_record or {}).get("quote") or support_quote
            answer = C.option_text(action.option, options)
            commit = {"unit": "R%d" % action.revision, "option": answer,
                      "quote": support_quote, "entry_id": action.revision}
            submitted_revision = action.revision
            submitted_option = action.option
            submitted_quote = action.quote
            ended_by = "agent_submit"
            turn("submit", reply, "cited R%d" % action.revision, policy_calls, 0)
            mark_repair_recovered()
            break
        ended_by = "agent_abstain"
        answer = ""
        turn("abstain", reply, action.reason, policy_calls, 0)
        mark_repair_recovered()
        break

    if not ended_by:
        ended_by = "turn_limit"
    supported = bool(
        commit and (card_supported or tree.can_submit(
            submitted_revision, submitted_option, submitted_quote,
            question_number=question_number)[0]))
    return result(ended_by, answer=answer, commit=commit, supported=supported)

