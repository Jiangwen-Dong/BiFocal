"""Promote reviewed structured local claims at a question boundary."""

from __future__ import annotations

import copy

from . import local as B


ACTOR_ROLES = {"O": "orchestrator"}


def _entry(session, entry_id):
    for rows in (session.get("chunks") or {}).values():
        for entry in rows or ():
            if str(entry.get("id")) == str(entry_id):
                return entry
    return None


def _raw_claim(entry):
    claim = entry.get("claim")
    if not isinstance(claim, dict):
        raise ValueError("local entry has no structured claim")
    key = claim.get("key")
    if not isinstance(key, list) or len(key) != 4:
        raise ValueError("local entry claim key is malformed")
    return {
        "subject": key[1], "predicate": key[2],
        "value": claim.get("value"), "unit": claim.get("unit"),
    }


def _chunk_span(chunk_spans, chunk):
    if isinstance(chunk_spans, dict):
        value = chunk_spans.get(chunk)
    else:
        value = next((row for row in chunk_spans or () if int(row[0]) == int(chunk)), None)
        value = value[1:] if value else None
    if not value or len(value) != 2:
        raise ValueError("chunk %s has no declared span" % chunk)
    return [float(value[0]), float(value[1])]


def _accepted(review):
    return (review.get("outcome") == "accept"
            and review.get("support", review.get("accept")) is True
            and review.get("accept") is True
            and review.get("frames_from_store") is True
            and int(review.get("frames") or 0) > 0)


def _local_leaf(session, entry, review, chunk_spans):
    evidence = review.get("sealed_call") or {}
    if not evidence or B.canonical_sha256(evidence) != str(review.get("review_call_id") or ""):
        raise ValueError("review has no valid sealed call provenance")
    frames = entry.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("local entry has no stored-frame handle")
    return B.make_local_leaf(
        local_id="local-%s" % entry["id"],
        video=session.get("video"),
        question_number=int(entry["question_number"]),
        reader=str(entry.get("author") or "reader"),
        chunk=int(review["chunk"]),
        turn=int(review.get("round") or 1),
        read_attempt=int(entry["id"]),
        assigned_span=list(entry.get("assigned") or ()),
        sighting_span=list(entry.get("seconds") or ()),
        chunk_span=_chunk_span(chunk_spans, review["chunk"]),
        description=str(entry.get("description") or ""),
        frame_handle=copy.deepcopy(frames),
        stored_frames=int(frames.get("n") or 0),
        raw_claim=_raw_claim(entry),
        provisional=bool(entry.get("provisional")),
        version=1,
        parent_local_id=None,
        session_entry_id=entry["id"],
        raw_reply=str(entry.get("raw_reply") or ""),
    )


def _review_leaf(leaf, review):
    evidence = review["sealed_call"]
    return B.review_local(
        leaf, reviewer="O", actor_roles=ACTOR_ROLES, status="accepted",
        frames_from_store=True,
        reason=str(review.get("why") or "review accepted stored frames"),
        review_call_id=str(review["review_call_id"]),
        review_call_log_entries=[copy.deepcopy(evidence)],
        reviewed_frame_handles=[copy.deepcopy(leaf["frame_handle"])],
        prompt_tokens=int(evidence.get("prompt_tokens") or 0),
        stored_frames_read=int(review.get("frames") or 0),
        answer_support=bool(review.get("answer_support")),
        reader_answer=str(review.get("reader_answer") or ""),
        reader_quote=str(review.get("reader_quote") or ""),
    )


def promote_question_claims(session, row, question_number, next_question,
                            chunk_spans):
    """Apply all reviewed promotions once and store the resulting records."""
    if int(next_question) <= int(question_number):
        raise ValueError("promotion must commit after its question")
    records = copy.deepcopy(
        session.get("bifocal_video_records")
        or session.get("bifocal_global_records") or [])
    events = []
    for review in row.get("reviews") or ():
        entry_id = review.get("entry_id")
        event = {"entry_id": entry_id, "chunk": review.get("chunk"),
                 "unit": review.get("unit")}
        if not _accepted(review):
            event.update(status="not_accepted", reason="report support check did not pass")
            events.append(event)
            continue
        entry = _entry(session, entry_id)
        if entry is None:
            event.update(status="deferred", reason="local entry is missing")
            events.append(event)
            continue
        if int(entry.get("question_number") or 0) != int(question_number):
            event.update(status="deferred", reason="entry belongs to another question")
            events.append(event)
            continue
        if entry.get("retracted"):
            event.update(status="deferred", reason="local entry is retracted")
            events.append(event)
            continue
        if entry.get("provisional"):
            event.update(status="deferred", reason="local entry is still provisional")
            events.append(event)
            continue
        if not isinstance(entry.get("claim"), dict):
            event.update(status="deferred", reason="local entry has no structured claim")
            events.append(event)
            continue
        local_id = "local-%s" % entry_id
        already = any(local_id in record.get("support_local_ids") or ()
                      for record in records)
        if already:
            event.update(status="already_present", reason="claim already promoted")
            events.append(event)
            continue
        try:
            leaf = _local_leaf(session, entry, review, chunk_spans)
            leaf = _review_leaf(leaf, review)
            updated = B.promote_local(
                records, leaf, next_question=int(next_question),
                actor="O", actor_roles=ACTOR_ROLES)
        except ValueError as error:
            reason = str(error)
            if "sealed call" in reason or "call log" in reason or "prompt tokens" in reason:
                reason = "review has no valid sealed call provenance: " + reason
            event.update(status="deferred", reason=reason)
            events.append(event)
            continue
        records = updated
        entry["global"] = True
        entry["status"] = "verified"
        entry["reviewed"] = True
        entry["global_record_ids"] = sorted(
            record["global_id"] for record in records
            if local_id in record.get("support_local_ids") or ())
        entry["video_record_ids"] = sorted(
            record.get("video_record_id", record["global_id"]) for record in records
            if local_id in record.get("support_local_ids") or ())
        event.update(status="promoted",
                     global_ids=sorted(record["global_id"] for record in records
                                       if local_id in record.get("support_local_ids") or ()),
                     video_record_ids=sorted(
                         record.get("video_record_id", record["global_id"])
                         for record in records
                         if local_id in record.get("support_local_ids") or ()))
        events.append(event)
    session["bifocal_global_records"] = records
    session["bifocal_video_records"] = copy.deepcopy(records)
    session.setdefault("bifocal_promotion_events", []).extend(events)
    return {"records": copy.deepcopy(records), "events": events}
