"""Pure state rules for BiFocal's fast local and slow global memory.

This module deliberately has no model, prompt or runner dependency. It defines
the claim normalization, actor permissions, global transitions and immutable
question snapshots that the live harness must obey. Older replay records remain
valid; adopting this module starts a new memory schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation


SCHEMA_VERSION = "bifocal-memory-v3"
MAX_GLOBAL_RECORDS = 32

PREDICATES = {
    "count", "identity", "attribute", "location", "time", "action", "order",
    "existence", "other",
}

UNIT_ALIASES = {
    "": "",
    "count": "count", "counts": "count",
    "rank": "rank", "ranks": "rank",
    "percent": "percent", "percentage": "percent", "%": "percent",
    "second": "second", "seconds": "second", "sec": "second", "secs": "second",
    "s": "second",
    "minute": "minute_input", "minutes": "minute_input",
    "min": "minute_input", "mins": "minute_input",
    "hour": "hour_input", "hours": "hour_input", "h": "hour_input",
    "meter": "meter_input", "meters": "meter_input", "metre": "meter_input",
    "metres": "meter_input", "m": "meter_input",
    "centimeter": "centimeter_input", "centimeters": "centimeter_input",
    "centimetre": "centimeter_input", "centimetres": "centimeter_input",
    "cm": "centimeter_input",
    "kilometer": "kilometer_input", "kilometers": "kilometer_input",
    "kilometre": "kilometer_input", "kilometres": "kilometer_input",
    "km": "kilometer_input",
}

NUMERIC_UNITS = {
    "count": ("count", Decimal(1)),
    "rank": ("rank", Decimal(1)),
    "percent": ("percent", Decimal(1)),
    "second": ("second", Decimal(1)),
    "minute_input": ("second", Decimal(60)),
    "hour_input": ("second", Decimal(3600)),
    "meter_input": ("meter", Decimal(1)),
    "centimeter_input": ("meter", Decimal("0.01")),
    "kilometer_input": ("meter", Decimal(1000)),
}

PREDICATE_UNITS = {
    "count": {"count"},
    "order": {"rank"},
    "time": {"second"},
    "location": {"", "meter"},
    "attribute": {"", "percent"},
    "identity": {""},
    "action": {""},
    "existence": {""},
    "other": {""},
}

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}

REVIEW_STATUSES = {"unreviewed", "accepted", "rejected", "unsure", "no_frames",
                   "retracted"}
GLOBAL_STATUSES = {"active", "disputed", "superseded", "retracted"}
GLOBAL_RECORD_FIELDS = {
    "global_id", "video", "claim_key", "claim_value", "status",
    "first_visible_question", "support_local_ids", "source_frame_handles",
    "source_provenance", "contradiction_provenance", "contradiction_with",
    "citations", "versions", "display", "display_sha256", "display_local_id",
}

_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"))


def canonical_sha256(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_text(value, name, required=True):
    if not isinstance(value, str):
        raise ValueError("%s must be text" % name)
    value = " ".join(unicodedata.normalize("NFKC", value)
                     .casefold().split())
    if required and not value:
        raise ValueError("%s must be non-empty text" % name)
    return value


def _normalize_identifier(value, name):
    if not isinstance(value, str):
        raise ValueError("%s must be text" % name)
    value = " ".join(unicodedata.normalize("NFKC", value)
                     .casefold().split())
    if not value:
        raise ValueError("%s must be non-empty text" % name)
    return value


def _require_role(actor, actor_roles, expected):
    if not isinstance(actor_roles, dict):
        raise ValueError("actor_roles must be a mapping")
    normalized_roles = {}
    for name, role in actor_roles.items():
        normalized = _normalize_identifier(name, "actor_roles key")
        if normalized in normalized_roles and normalized_roles[normalized] != role:
            raise ValueError("actor identity has conflicting roles")
        normalized_roles[normalized] = role
    actor = _normalize_identifier(actor, "actor")
    role = normalized_roles.get(actor)
    if role is None:
        raise ValueError("unknown actor %s" % actor)
    if role != expected:
        raise ValueError("%s is %s, not %s" % (actor, role, expected))
    return role


def _seal_call_log(entries):
    if not isinstance(entries, list) or not entries:
        raise ValueError("sealed review call log must be a non-empty list")
    if any(not isinstance(entry, dict) or not entry for entry in entries):
        raise ValueError("sealed review call log entries must be non-empty objects")
    return canonical_sha256(entries)


def _parse_number(value):
    text = _normalize_text(value, "claim.value")
    if _NUMBER.fullmatch(text):
        try:
            return Decimal(text)
        except InvalidOperation as error:
            raise ValueError("cannot parse claim.value as a number") from error
    if text in ORDINALS:
        return Decimal(ORDINALS[text])
    words = text.replace("-", " ").split()
    total, current = 0, 0
    consumed = False
    for word in words:
        if word == "and":
            continue
        if word == "hundred":
            current = (current or 1) * 100
            consumed = True
        elif word in NUMBER_WORDS:
            current += NUMBER_WORDS[word]
            consumed = True
        else:
            raise ValueError("cannot parse claim.value as a number")
    total += current
    if not consumed:
        raise ValueError("cannot parse claim.value as a number")
    return Decimal(total)


def _canonical_decimal(value):
    value = value.normalize()
    if value == value.to_integral_value():
        return str(int(value))
    return format(value, "f").rstrip("0").rstrip(".")


def normalize_claim(video, subject, predicate, value, unit):
    """Normalize one structured claim into a comparable key and value."""
    video = _normalize_text(video, "video")
    subject = _normalize_text(subject, "claim.subject")
    predicate = _normalize_text(predicate, "claim.predicate")
    if predicate not in PREDICATES:
        raise ValueError("unknown claim.predicate %s" % predicate)
    unit = _normalize_text(unit, "claim.unit", required=False)
    if unit not in UNIT_ALIASES:
        raise ValueError("unknown claim.unit %s" % unit)
    unit = UNIT_ALIASES[unit]
    if not unit and predicate == "count":
        unit = "count"
    elif not unit and predicate == "order":
        unit = "rank"

    if unit in NUMERIC_UNITS:
        canonical_unit, factor = NUMERIC_UNITS[unit]
        number = _parse_number(value) * factor
        if canonical_unit not in PREDICATE_UNITS[predicate]:
            raise ValueError("predicate %s does not accept canonical unit %s"
                             % (predicate, canonical_unit))
        if canonical_unit in ("count", "rank") and number != number.to_integral_value():
            raise ValueError("cannot parse claim.value as an integer")
        if canonical_unit == "rank" and number < 1:
            raise ValueError("claim.value rank must be at least one")
        if canonical_unit in ("count", "second", "meter") and number < 0:
            raise ValueError("claim.value must be non-negative")
        canonical_value = _canonical_decimal(number)
    else:
        canonical_unit = ""
        if canonical_unit not in PREDICATE_UNITS[predicate]:
            raise ValueError("predicate %s does not accept text values" % predicate)
        canonical_value = _normalize_text(value, "claim.value")

    body = {
        "key": [video, subject, predicate, canonical_unit],
        "value": canonical_value,
        "unit": canonical_unit,
        "type": "number" if canonical_unit else "text",
    }
    body["key_sha256"] = canonical_sha256(body["key"])
    return body


def _valid_span(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                    for item in value)
            and float(value[1]) >= float(value[0]))


def _contains(outer, inner):
    return (outer and inner and float(outer[0]) <= float(inner[0])
            and float(inner[1]) <= float(outer[1]))


def make_local_leaf(local_id, video, question_number, reader, chunk, turn,
                    read_attempt, assigned_span, sighting_span, chunk_span,
                    description, frame_handle, stored_frames, raw_claim=None,
                    provisional=False, version=1, parent_local_id=None,
                    session_entry_id=None, raw_reply=""):
    """Create an immutable raw local observation, including invalid evidence."""
    video = _normalize_text(video, "video")
    local_id = _normalize_identifier(local_id, "local_id")
    reader = _normalize_identifier(reader, "reader")
    session_entry_id = (session_entry_id if session_entry_id is not None
                        else local_id)
    for name, value in (("local_id", local_id), ("video", video), ("reader", reader)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("%s must be non-empty text" % name)
    for name, value, minimum in (
            ("question_number", question_number, 1), ("chunk", chunk, 0),
            ("turn", turn, 1), ("read_attempt", read_attempt, 1),
            ("stored_frames", stored_frames, 0), ("version", version, 1)):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError("%s must be an integer >= %d" % (name, minimum))
    errors = []
    if not isinstance(description, str) or not description.strip():
        errors.append("description is empty")
    if not frame_handle:
        errors.append("frame_handle is missing")
    if int(stored_frames) <= 0:
        errors.append("stored_frames is zero")
    if not _valid_span(assigned_span):
        errors.append("assigned_span is invalid")
    if not _valid_span(sighting_span):
        errors.append("sighting_span is invalid")
    if not _valid_span(chunk_span):
        errors.append("chunk_span is invalid")
    if not errors:
        if not _contains(assigned_span, sighting_span):
            errors.append("sighting_span is outside assigned_span")
        if not _contains(chunk_span, assigned_span):
            errors.append("assigned_span is outside chunk_span")

    claim, claim_error = None, ""
    if raw_claim is None:
        claim_error = "missing structured claim"
    else:
        try:
            claim = normalize_claim(
                video, raw_claim.get("subject"), raw_claim.get("predicate"),
                raw_claim.get("value"), raw_claim.get("unit"))
        except (AttributeError, ValueError) as error:
            claim_error = str(error)

    return {
        "schema_version": SCHEMA_VERSION,
        "local_id": local_id,
        "video": video,
        "question_number": int(question_number),
        "reader": reader,
        "chunk": int(chunk),
        "turn": int(turn),
        "read_attempt": int(read_attempt),
        "assigned_span": [float(value) for value in assigned_span],
        "sighting_span": [float(value) for value in sighting_span],
        "chunk_span": [float(value) for value in chunk_span],
        "description": str(description or "").strip(),
        "raw_reply": str(raw_reply or ""),
        "frame_handle": copy.deepcopy(frame_handle) if frame_handle else None,
        "stored_frames": int(stored_frames),
        "claim": claim,
        "claim_valid": claim is not None,
        "claim_error": claim_error,
        "format_valid": not errors,
        "validation_errors": errors,
        "review_status": "unreviewed",
        "reviewer": None,
        "review_frames_from_store": False,
        "review_reason": "",
        "review_evidence": None,
        # Raw observations are saved first. Only a support review can make one answer-capable.
        "answer_support": False,
        "citation_allowed": False,
        "provisional": bool(provisional),
        "version": int(version),
        "parent_local_id": parent_local_id,
        "session_entry_id": copy.deepcopy(session_entry_id),
    }


def _handle_equal(left, right):
    return canonical_sha256(left) == canonical_sha256(right)


def _handles_cover(reviewed_handles, required_handles):
    reviewed = list(reviewed_handles or ())
    return all(any(_handle_equal(candidate, required) for candidate in reviewed)
               for required in required_handles or ())


def review_local(leaf, reviewer, actor_roles, status, frames_from_store, reason,
                 review_call_id, reviewed_frame_handles, prompt_tokens=0,
                 stored_frames_read=None, review_call_log_entries=None,
                 answer_support=False, reader_answer="", reader_quote=""):
    """Review a local leaf without using terminal correctness or gold."""
    reviewer = _normalize_identifier(reviewer, "reviewer")
    _require_role(reviewer, actor_roles, "orchestrator")
    if status not in REVIEW_STATUSES - {"unreviewed"}:
        raise ValueError("unknown local review status %s" % status)
    if reviewer == _normalize_identifier(leaf.get("reader"), "local.reader"):
        raise ValueError("a review needs a different actor from the leaf author")
    if status == "accepted" and not frames_from_store:
        raise ValueError("acceptance needs stored frames read directly")
    if status == "accepted" and not leaf.get("format_valid"):
        raise ValueError("an invalid local leaf cannot be accepted")
    if status == "accepted" and not str(reason or "").strip():
        raise ValueError("acceptance needs an explicit review reason")
    review_call_id = _normalize_text(review_call_id, "review call id")
    reviewed_frame_handles = copy.deepcopy(list(reviewed_frame_handles or ()))
    if frames_from_store:
        if not reviewed_frame_handles:
            raise ValueError("a frame-backed review needs frame handles")
        if not _handles_cover(reviewed_frame_handles, [leaf.get("frame_handle")]):
            raise ValueError("review frames do not include the leaf frame handle")
    if not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool) \
            or prompt_tokens < 0:
        raise ValueError("prompt_tokens must be a non-negative integer")
    if status == "accepted" and prompt_tokens <= 0:
        raise ValueError("acceptance needs positive review prompt tokens")
    answer_support = bool(status == "accepted" and answer_support)
    reader_answer = str(reader_answer or "").strip().upper()[:1]
    reader_quote = " ".join(str(reader_quote or "").split())
    if answer_support and not reader_answer:
        raise ValueError("answer support needs the reader's answer")
    if answer_support and not reader_quote:
        raise ValueError("answer support needs the reader's quote")
    if stored_frames_read is None:
        stored_frames_read = leaf.get("stored_frames") if frames_from_store else 0
    if not isinstance(stored_frames_read, int) or isinstance(stored_frames_read, bool) \
            or stored_frames_read < 0:
        raise ValueError("stored_frames_read must be a non-negative integer")
    if frames_from_store and stored_frames_read <= 0:
        raise ValueError("a frame-backed review must read stored frames")
    if frames_from_store and stored_frames_read > int(leaf.get("stored_frames") or 0):
        raise ValueError("review cannot read more stored frames than the leaf contains")
    review_call_log_sha256 = _seal_call_log(review_call_log_entries)
    evidence = {
        "actor": reviewer,
        "actor_role": "orchestrator",
        "reviewer": reviewer,
        "frames_from_store": bool(frames_from_store),
        "review_call_id": review_call_id,
        "review_call_log_sha256": review_call_log_sha256,
        "reviewed_frame_handles": reviewed_frame_handles,
        "prompt_tokens": prompt_tokens,
        "stored_frames_read": stored_frames_read,
        "reason": str(reason or "").strip(),
        "reader_answer": reader_answer,
        "reader_quote": reader_quote,
    }
    out = copy.deepcopy(leaf)
    out.update({
        "review_status": status,
        "reviewer": reviewer,
        "review_frames_from_store": bool(frames_from_store),
        "review_reason": evidence["reason"],
        "review_evidence": evidence,
        "answer_support": answer_support,
        "citation_allowed": answer_support,
    })
    return out


def promotion_eligible(leaf):
    reasons = []
    if not leaf.get("format_valid"):
        reasons.append("local leaf is not format/frame valid")
    if not leaf.get("claim_valid") or not leaf.get("claim"):
        reasons.append("local leaf has no valid structured claim")
    if leaf.get("review_status") != "accepted":
        reasons.append("local leaf has no accepted review")
    if not leaf.get("review_frames_from_store"):
        reasons.append("review did not read stored frames")
    if not leaf.get("reviewer") or leaf.get("reviewer") == leaf.get("reader"):
        reasons.append("reviewer is not independent")
    evidence = leaf.get("review_evidence") or {}
    if evidence.get("actor_role") != "orchestrator":
        reasons.append("review actor is not the orchestrator")
    if not evidence.get("review_call_id"):
        reasons.append("review call is missing")
    if not re.fullmatch(r"[0-9a-f]{64}",
                        str(evidence.get("review_call_log_sha256") or "")):
        reasons.append("review call log hash is missing")
    if not _handles_cover(evidence.get("reviewed_frame_handles") or (),
                          [leaf.get("frame_handle")]):
        reasons.append("review provenance does not cover the leaf frame")
    if leaf.get("provisional"):
        reasons.append("local leaf is still provisional")
    return not reasons, reasons


def _global_id(claim, suffix=""):
    digest = canonical_sha256({"key": claim["key"], "value": claim["value"]})
    return "G-" + digest[:16] + (str(suffix) if suffix else "")


def _leaf_display(leaf):
    claim = leaf["claim"]
    return {
        "text": leaf["description"],
        "claim_text": " ".join([
            claim["key"][1], claim["key"][2], claim["value"],
            claim["key"][3],
        ]).strip(),
        "question_number": int(leaf["question_number"]),
        "chunk": int(leaf["chunk"]),
        "span": copy.deepcopy(leaf["sighting_span"]),
        "assigned_span": copy.deepcopy(leaf["assigned_span"]),
        "sighting_span": copy.deepcopy(leaf["sighting_span"]),
        "stored_frames": int(leaf["stored_frames"]),
        "source": "reader",
        "source_reader": leaf["reader"],
        "answer_support": bool(leaf.get("answer_support", False)),
        "citation_allowed": bool(leaf.get("citation_allowed", False)),
    }


def _new_global_record(leaf, next_question, status="active"):
    claim = leaf["claim"]
    evidence = copy.deepcopy(leaf["review_evidence"])
    display = _leaf_display(leaf)
    provenance = {
        "local_id": leaf["local_id"],
        "local_version": int(leaf["version"]),
        "frame_handle": copy.deepcopy(leaf["frame_handle"]),
        "review_call_id": evidence["review_call_id"],
        "review_call_log_sha256": evidence["review_call_log_sha256"],
        "reviewer": evidence["reviewer"],
        "actor": evidence["actor"],
        "actor_role": evidence["actor_role"],
        "reader": leaf["reader"],
        "session_entry_id": copy.deepcopy(leaf["session_entry_id"]),
        "display_sha256": canonical_sha256(display),
        "prompt_tokens": evidence["prompt_tokens"],
        "stored_frames_read": evidence["stored_frames_read"],
    }
    record_id = _global_id(claim)
    return {
        "global_id": record_id,
        "video_record_id": record_id,
        "scope": "video",
        "video": claim["key"][0],
        "claim_key": list(claim["key"]),
        "claim_value": claim["value"],
        "status": status,
        "first_visible_question": int(next_question),
        "support_local_ids": [leaf["local_id"]],
        "source_frame_handles": [copy.deepcopy(leaf["frame_handle"])],
        "source_provenance": [provenance],
        "contradiction_provenance": [],
        "contradiction_with": [],
        "citations": [],
        "display": display,
        "display_sha256": canonical_sha256(display),
        "display_local_id": leaf["local_id"],
        "answer_support": bool(leaf.get("answer_support", False)),
        "citation_allowed": bool(leaf.get("citation_allowed", False)),
        "versions": [{
            "version": 1,
            "status": status,
            "claim_value": claim["value"],
            "local_ids": [leaf["local_id"]],
            "question_number": int(leaf["question_number"]),
            "reviewer": leaf["reviewer"],
            "reason": leaf["review_reason"],
            "source_provenance": [copy.deepcopy(provenance)],
            "review_evidence": copy.deepcopy(evidence),
        }],
    }


def _append_version(record, leaf, status=None, supports_record=True):
    record = copy.deepcopy(record)
    version = len(record["versions"]) + 1
    record["versions"].append({
        "version": version,
        "status": status or record["status"],
        "claim_value": leaf["claim"]["value"],
        "local_ids": [leaf["local_id"]] if supports_record else [],
        "question_number": int(leaf["question_number"]),
        "reviewer": leaf["reviewer"],
        "reason": leaf["review_reason"],
        "source_provenance": ([{
            "local_id": leaf["local_id"],
            "local_version": int(leaf["version"]),
            "frame_handle": copy.deepcopy(leaf["frame_handle"]),
            "review_call_id": leaf["review_evidence"]["review_call_id"],
            "review_call_log_sha256": leaf["review_evidence"]["review_call_log_sha256"],
            "reviewer": leaf["review_evidence"]["reviewer"],
            "actor": leaf["review_evidence"]["actor"],
            "actor_role": leaf["review_evidence"]["actor_role"],
            "reader": leaf["reader"],
            "session_entry_id": copy.deepcopy(leaf["session_entry_id"]),
            "prompt_tokens": leaf["review_evidence"]["prompt_tokens"],
            "stored_frames_read": leaf["review_evidence"]["stored_frames_read"],
            "display_sha256": canonical_sha256(_leaf_display(leaf)),
        }] if supports_record else []),
        "review_evidence": copy.deepcopy(leaf["review_evidence"]),
    })
    if not supports_record:
        record["versions"][-1]["contradicted_by_local_id"] = leaf["local_id"]
    elif supports_record:
        record["answer_support"] = bool(record.get("answer_support")
                                        or leaf.get("answer_support", False))
        record["citation_allowed"] = bool(record.get("citation_allowed")
                                          or leaf.get("citation_allowed", False))
    return record


def promote_local(records, leaf, next_question, actor, actor_roles):
    """Apply the fixed boundary promotion rule to one reviewed local leaf."""
    _require_role(actor, actor_roles, "orchestrator")
    eligible, reasons = promotion_eligible(leaf)
    if not eligible:
        raise ValueError("local leaf is not promotable: %s" % "; ".join(reasons))
    if int(next_question) <= int(leaf["question_number"]):
        raise ValueError("promotion must become visible after its question")
    out = copy.deepcopy(records)
    if any(row.get("video") != leaf["video"] for row in out):
        raise ValueError("global memory contains a record from another video")

    all_local_ids = {local_id for row in out
                     for local_id in row.get("support_local_ids") or ()}
    if leaf["local_id"] in all_local_ids:
        existing = next(row for row in out if leaf["local_id"] in row["support_local_ids"])
        if existing.get("claim_key") == list(leaf["claim"]["key"]) \
                and existing.get("claim_value") == leaf["claim"]["value"]:
            return sorted(out, key=lambda row: row["global_id"])
        raise ValueError("local_id is already linked to a different global claim")
    key = list(leaf["claim"]["key"])
    value = leaf["claim"]["value"]
    same_key = [row for row in out
                if row.get("claim_key") == key
                and row.get("status") in ("active", "disputed")]
    same_value = next((row for row in same_key
                       if row.get("claim_value") == value), None)
    conflicts = [row for row in same_key if row.get("claim_value") != value]

    if same_value is not None:
        updated = _append_version(same_value, leaf)
        updated["support_local_ids"].append(leaf["local_id"])
        updated["source_frame_handles"].append(copy.deepcopy(leaf["frame_handle"]))
        updated["source_provenance"].append(copy.deepcopy(
            updated["versions"][-1]["source_provenance"][0]))
        out = [updated if row["global_id"] == same_value["global_id"] else row
               for row in out]
        return sorted(out, key=lambda row: row["global_id"])

    active_count = sum(1 for row in out
                       if row.get("status") in ("active", "disputed"))
    if active_count >= MAX_GLOBAL_RECORDS:
        raise ValueError("global memory is at its declared record cap")

    if conflicts:
        new_record = _new_global_record(leaf, next_question, status="disputed")
        existing_ids = {row["global_id"] for row in conflicts}
        new_record["contradiction_with"] = sorted(existing_ids)
        updated = []
        for row in out:
            row = copy.deepcopy(row)
            if row["global_id"] in existing_ids:
                row["status"] = "disputed"
                row.setdefault("contradiction_with", [])
                row["contradiction_with"] = sorted(
                    set(row["contradiction_with"]) | {new_record["global_id"]})
                row = _append_version(row, leaf, status="disputed",
                                      supports_record=False)
                row["contradiction_provenance"].append(copy.deepcopy(
                    row["versions"][-1]["review_evidence"]))
            updated.append(row)
        updated.append(new_record)
        return sorted(updated, key=lambda row: row["global_id"])

    existing_id = _global_id(leaf["claim"])
    if any(row.get("global_id") == existing_id for row in out):
        suffix = 2
        while any(row.get("global_id") == _global_id(leaf["claim"], suffix)
                  for row in out):
            suffix += 1
        new_record = _new_global_record(leaf, next_question)
        new_record["global_id"] = _global_id(leaf["claim"], suffix)
    else:
        new_record = _new_global_record(leaf, next_question)
    out.append(new_record)
    return sorted(out, key=lambda row: row["global_id"])


def _record(records, global_id):
    found = next((row for row in records if row.get("global_id") == global_id), None)
    if found is None:
        raise ValueError("unknown global_id %s" % global_id)
    return found


def is_citable(record):
    return bool(record.get("status") == "active" and record.get("versions")
                and record.get("answer_support") is True
                and record.get("citation_allowed") is True)

