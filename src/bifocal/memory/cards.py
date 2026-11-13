"""Build short local observation cards and team-promoted global fact cards."""

from __future__ import annotations

import collections
import copy
import re

from ..harness import actions as A
from ..harness import protocol as C
from . import claim as CLAIM

SCHEMA_VERSION = "memory-card-v2"


def _norm(value):
    return CLAIM.norm(value)


def is_generic_claim(value):
    return CLAIM.is_generic_claim(value)


def _sentences(value):
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", _norm(value))
        if part.strip()
    ]


def _claim_from(event_text, quote, summary, option):
    candidates = [
        (str(event_text or ""), "event"),
        *((sentence, "summary") for sentence in _sentences(summary)),
        (str(quote or ""), "quote"),
    ]
    for candidate, source in candidates:
        if not is_generic_claim(candidate) and len(_norm(candidate).split()) >= 2:
            return _norm(candidate), source
    for candidate, _source in candidates:
        if _norm(candidate):
            return _norm(candidate), "generic"
    return "", "missing"


def _time_span(event, revision):
    start = event.get("start")
    end = event.get("end")
    if start is None or end is None:
        window = revision.get("window") or ()
        if len(window) == 2:
            start, end = window
    try:
        start, end = float(start), float(end)
    except (TypeError, ValueError):
        return None
    if not start < end:
        return None
    return [start, end]


def _accepted_review(tree, revision_id, revision):
    if not A.answer_bearing(revision):
        return None
    review = tree.effective_review(revision_id, revision.get("answer"))
    source_question = int(revision.get("question_number") or 0)
    if (review is None or review.get("outcome") != "accept"
            or int(review.get("question_number") or 0) != source_question
            or review.get("frames_from_store") is not True
            or int(review.get("stored_frames") or 0) < 1):
        return None
    return review


def _legacy_answer_eligible(tree, revision_id, revision):
    if not A.answer_bearing(revision) or tree.review_stale(revision_id):
        return False
    if int((revision.get("frame_handle") or {}).get("n") or 0) < 1:
        return False
    event_ids = list(revision.get("events") or ())
    if len(event_ids) != 1 or event_ids[0] not in tree.events:
        return False
    return _accepted_review(tree, revision_id, revision) is not None


def _exclude(report, revision_id, reason):
    report["excluded_revisions"] += 1
    report["excluded_by_reason"][reason] += 1
    report["excluded_revision_ids"].append(str(revision_id))


def _observation_card(tree, revision_id, revision):
    source_question = int(revision.get("question_number") or 0)
    event_ids = [str(value) for value in revision.get("events") or ()
                 if str(value) in tree.events]
    events = [tree.events[event_id] for event_id in event_ids]
    event_text = " ".join(
        str(event.get("text") or "").strip() for event in events
        if str(event.get("text") or "").strip())
    claim, claim_source = _claim_from(
        event_text, revision.get("quote"), revision.get("branch_summary"),
        revision.get("answer"))
    if claim_source == "missing":
        return None, "missing_claim"
    if claim_source == "generic":
        return None, "generic_claim"
    if len(events) == 1:
        span = _time_span(events[0], revision)
    else:
        span = _time_span({}, revision)
    review = _accepted_review(tree, revision_id, revision)
    accepted = review is not None
    answer_bearing = bool(A.answer_bearing(revision))
    return {
        "schema_version": SCHEMA_VERSION,
        "card_id": "L:%s" % revision_id,
        "scope": "local",
        "source_revision_id": str(revision_id),
        "global_claim_id": "",
        "source_question": source_question,
        "reader": str(revision.get("reader") or ""),
        "branch_id": str(revision.get("branch_id") or ""),
        "time_span": span,
        "claim": claim,
        "claim_source": claim_source,
        "generic_claim": False,
        "quote": str(revision.get("quote") or ""),
        "detail": _norm(revision.get("branch_summary"))[:600],
        "option": (str(revision.get("answer") or "").upper()
                   if accepted else ""),
        "state": "active",
        "frame_handle": copy.deepcopy(revision.get("frame_handle") or {}),
        "stored_frames": int(
            (revision.get("frame_handle") or {}).get("n") or 0),
        "review_status": "accept" if accepted else "not_reviewed",
        "check_status": "accepted_review" if accepted else "not_checked",
        "source_local_stale": bool(tree.review_stale(revision_id)),
        "memory_kind": "answer" if accepted else "observation",
        "answer_bearing": answer_bearing,
        "legacy_answer_card": _legacy_answer_eligible(
            tree, revision_id, revision),
        "event_ids": event_ids,
        "duplicate_revision_ids": [],
    }, ""


def local_card_supply(tree, question_number, include_superseded=False):
    """Return deduplicated local cards and an explicit supply report."""
    question_number = int(question_number)
    report = {
        "revisions_seen": len(tree.revisions),
        "prior_revisions": 0,
        "successful_reads": 0,
        "eligible_revisions": 0,
        "legacy_answer_eligible_revisions": 0,
        "recovered_non_legacy_revisions": 0,
        "answer_cards": 0,
        "observation_cards": 0,
        "duplicate_revisions_removed": 0,
        "superseded_revisions_included": 0,
        "excluded_revisions": 0,
        "excluded_by_reason": collections.Counter(),
        "excluded_revision_ids": [],
    }
    candidates = []
    checked_scopes = {"local", "global"} if include_superseded else {"local"}
    checked_revisions = {
        str(row.get("revision_id") or "")
        for row in tree.memory_uses.values()
        if row.get("action") == "check"
        and row.get("outcome") == "accept"
        and int(row.get("question_number") or 0) == int(question_number)
        and row.get("scope") in checked_scopes}
    for revision_id, revision in tree.revisions.items():
        source_question = int(revision.get("question_number") or 0)
        if source_question < 1 or source_question >= question_number:
            _exclude(report, revision_id, "source_question_out_of_range")
            continue
        report["prior_revisions"] += 1
        if (revision.get("read_status") != "success"
                or revision.get("parse_status") != "ok"):
            _exclude(report, revision_id, "read_not_successful")
            continue
        report["successful_reads"] += 1
        if revision.get("frame_status") != "ok":
            _exclude(report, revision_id, "frame_not_ok")
            continue
        if int((revision.get("frame_handle") or {}).get("n") or 0) < 1:
            _exclude(report, revision_id, "no_frame_handle")
            continue
        if revision.get("keep_noop"):
            _exclude(report, revision_id, "keep_noop")
            continue
        source_local_stale = bool(tree.review_stale(revision_id))
        if source_local_stale and not include_superseded:
            _exclude(report, revision_id, "superseded")
            continue
        report["eligible_revisions"] += 1
        if _legacy_answer_eligible(tree, revision_id, revision):
            report["legacy_answer_eligible_revisions"] += 1
        card, reason = _observation_card(tree, revision_id, revision)
        if reason:
            _exclude(report, revision_id, reason)
            continue
        if source_local_stale:
            report["superseded_revisions_included"] += 1
        if not card["legacy_answer_card"]:
            report["recovered_non_legacy_revisions"] += 1
        candidates.append(card)

    cards_by_key = {}
    checked_duplicates = []
    for card in candidates:
        span = tuple(card.get("time_span") or ())
        key = (
            str(card.get("branch_id") or ""),
            _norm(card.get("claim")).lower(),
            span,
            str(card.get("memory_kind") or ""),
            str(card.get("option") or ""),
        )
        existing = cards_by_key.get(key)
        if existing is None:
            cards_by_key[key] = card
            continue
        report["duplicate_revisions_removed"] += 1
        def recency(value):
            return (
                int(value.get("source_question") or 0),
                int(str(value.get("source_revision_id") or "R0")[1:] or 0),
            )

        def prefer(value):
            return (
                str(value.get("source_revision_id") or "") in checked_revisions,
                value.get("memory_kind") == "answer",
                recency(value))

        keep, drop = (
            (existing, card) if prefer(existing) >= prefer(card)
            else (card, existing))
        keep["duplicate_revision_ids"] = sorted(set(
            list(keep.get("duplicate_revision_ids") or [])
            + list(drop.get("duplicate_revision_ids") or [])
            + [str(drop.get("source_revision_id") or "")]))
        cards_by_key[key] = keep

    for card in candidates:
        if str(card.get("source_revision_id") or "") not in checked_revisions:
            continue
        span = tuple(card.get("time_span") or ())
        key = (
            str(card.get("branch_id") or ""),
            _norm(card.get("claim")).lower(),
            span,
            str(card.get("memory_kind") or ""),
            str(card.get("option") or ""),
        )
        kept = cards_by_key.get(key)
        if kept is not None and str(kept.get("source_revision_id") or "") != str(
                card.get("source_revision_id") or ""):
            checked_duplicates.append(card)

    cards = sorted(
        list(cards_by_key.values()) + checked_duplicates,
        key=lambda card: (
            int(card.get("source_question") or 0),
            int(str(card.get("source_revision_id") or "R0")[1:] or 0),
            str(card.get("card_id") or "")))
    report["answer_cards"] = sum(
        card.get("memory_kind") == "answer" for card in cards)
    report["observation_cards"] = len(cards) - report["answer_cards"]
    report["retrieval_cards"] = len(cards)
    report["excluded_by_reason"] = dict(sorted(
        report["excluded_by_reason"].items()))
    return cards, report


def local_cards(tree, question_number):
    return local_card_supply(tree, question_number)[0]


GLOBAL_BUCKET_SECONDS = 128
GLOBAL_MAX_ENTRIES = 12


def _compressed_source_claim(value, words=12):
    parts = _norm(value).split()
    text = " ".join(parts[:words]).strip(" .")
    return text + ("..." if len(parts) > words else "")


def _timeline_entry(card, bucket_start):
    span = [float(value) for value in card["time_span"]]
    summary = _compressed_source_claim(
        card.get("detail") or card.get("claim"), words=16)
    return {
        "entry_id": "T:%d:%s" % (
            int(bucket_start), str(card.get("source_revision_id") or "")),
        "chapter": [float(bucket_start),
                    max(float(bucket_start + GLOBAL_BUCKET_SECONDS), span[1])],
        "start": span[0],
        "end": span[1],
        "summary": summary,
        "claim": summary,
        "source_revision_id": str(
            card.get("source_revision_id") or ""),
        "source_question": int(card.get("source_question") or 0),
        "reader": str(card.get("reader") or ""),
        "branch_id": str(card.get("branch_id") or ""),
        "event_ids": list(card.get("event_ids") or ()),
        "frame_handle": copy.deepcopy(card.get("frame_handle") or {}),
        "stored_frames": int(card.get("stored_frames") or 0),
        "source_local_stale": bool(card.get("source_local_stale")),
    }


def _entry_rank(card):
    return (
        len(concrete_tokens(card.get("detail") or card.get("claim"))),
        int(card.get("memory_kind") == "answer"),
        int(bool(card.get("quote"))),
        int(card.get("source_question") or 0),
        int(str(card.get("source_revision_id") or "R0")[1:] or 0),
    )


def _merge_windows(spans):
    windows = []
    for first, last in sorted(
            (float(span[0]), float(span[1])) for span in spans
            if span and len(span) == 2 and float(span[0]) < float(span[1])):
        if not windows or first > windows[-1][1]:
            windows.append([first, last])
        else:
            windows[-1][1] = max(windows[-1][1], last)
    return windows


def _covered_seconds(spans, window=None):
    windows = _merge_windows(spans)
    if window and len(window) == 2:
        asked_first, asked_last = float(window[0]), float(window[1])
        windows = [
            [max(first, asked_first), min(last, asked_last)]
            for first, last in windows
            if min(last, asked_last) > max(first, asked_first)]
    return sum(last - first for first, last in windows)


def _sample_timeline_entries(entries, limit=GLOBAL_MAX_ENTRIES,
                             pinned_revision_ids=()):
    if len(entries) <= int(limit):
        return entries
    limit = max(1, int(limit))
    pinned = {
        str(value) for value in pinned_revision_ids or () if str(value)}
    pinned_indices = {
        index for index, entry in enumerate(entries)
        if str(entry.get("source_revision_id") or "") in pinned}
    remaining_slots = max(0, limit - len(pinned_indices))
    available = [
        index for index in range(len(entries)) if index not in pinned_indices]
    if remaining_slots <= 0:
        indices = set(pinned_indices)
    elif remaining_slots == 1:
        indices = pinned_indices | {available[-1]}
    else:
        indices = pinned_indices | {
            available[int(round(index * (len(available) - 1)
                                / float(remaining_slots - 1)))]
            for index in range(remaining_slots)}
    return [entry for index, entry in enumerate(entries) if index in indices]


def _timeline_card(entries, eligible, report, claim_backed=False):
    first = min(float(entry["start"]) for entry in entries)
    last = max(float(entry["end"]) for entry in entries)
    coverage_windows = _merge_windows([
        [entry["start"], entry["end"]] for entry in entries])
    coverage_seconds = _covered_seconds(coverage_windows)
    source_tokens = sum(
        len(concrete_tokens(card.get("claim"))) for card in eligible)
    entry_tokens = sum(
        len(concrete_tokens(entry.get("summary"))) for entry in entries)
    source_times = [
        [float(entry["start"]), float(entry["end"])] for entry in entries]
    source_revisions = sorted(
        {str(entry.get("source_revision_id") or "") for entry in entries})
    source_questions = sorted({
        int(entry.get("source_question") or 0) for entry in entries})
    claim = ("Shared notes: %d claims from %gs to %gs"
             if claim_backed else
             "Observed timeline: %d events from %gs to %gs") % (
        len(entries), first, last)
    default = entries[-1]
    report.update({
        "global_cards": 1,
        "sources_used": len(entries),
        "sources_collapsed": max(0, len(eligible) - len(entries)),
        "source_claim_tokens": source_tokens,
        "global_claim_tokens": entry_tokens,
        "timeline_entries": len(entries),
    })
    card = {
        "schema_version": SCHEMA_VERSION,
        "card_id": "G:TL:VIDEO",
        "scope": "global",
        "source_revision_id": str(
            default.get("source_revision_id") or ""),
        "selected_anchor_revision_id": str(
            default.get("source_revision_id") or ""),
        "selected_anchor_entry_id": str(default.get("entry_id") or ""),
        "global_claim_id": "",
        "source_question": int(default.get("source_question") or 0),
        "reader": "harness",
        "branch_id": "TIMELINE",
        "time_span": [first, last],
        "claim": claim,
        "anchor_source_claim": str(default.get("summary") or ""),
        "shared_evidence_terms": [],
        "claim_source": ("claims" if claim_backed else "timeline_overview"),
        "generic_claim": False,
        "quote": "",
        "detail": "",
        "option": "",
        "state": "active",
        "frame_handle": copy.deepcopy(
            default.get("frame_handle") or {}),
        "stored_frames": int(default.get("stored_frames") or 0),
        "review_status": "timeline",
        "check_status": "not_checked",
        "source_local_stale": False,
        "memory_kind": "global_timeline",
        "timeline_version": 2,
        "answer_bearing": False,
        "legacy_answer_card": False,
        "event_ids": sorted({
            str(event_id) for entry in entries
            for event_id in entry.get("event_ids") or ()}),
        "duplicate_revision_ids": [],
        "timeline_bucket": [first, last],
        "source_revision_ids": source_revisions,
        "source_questions": source_questions,
        "source_readers": sorted({
            str(entry.get("reader") or "") for entry in entries}),
        "source_claims": [str(entry.get("summary") or "")
                          for entry in entries],
        "source_spans": source_times,
        "timeline_entries": entries,
        "anchor_candidates": entries,
        "coverage_windows": coverage_windows,
        "coverage": {
            "observed_seconds": coverage_seconds,
            "span_seconds": last - first,
            "asked_seconds": None,
            "asked_coverage": None,
        },
        "compression": {
            "source_claims": len(eligible),
            "source_claim_tokens": source_tokens,
            "global_claim_tokens": entry_tokens,
            "ratio": (entry_tokens / float(source_tokens)
                      if source_tokens else 1.0),
        },
    }
    if claim_backed:
        card["claim_backed"] = True
    return [card], report


def global_timeline_supply(tree, question_number):
    """Build one chronological overview from local observations.

    The global layer is not a second local card. It keeps at most one concrete
    observation per 128 second chapter, in time order, and exposes the whole
    observed chronology as one broad memory. Local cards retain all details and
    exact spans.
    """
    # Global memory preserves the observed chronology. A local detail can be
    # superseded, but that later correction must not erase the earlier global
    # observation and its stored frames.
    local, local_report = local_card_supply(
        tree, question_number, include_superseded=True)
    eligible = [
        card for card in local
        if card.get("time_span") and len(card["time_span"]) == 2]
    buckets = collections.defaultdict(list)
    for card in eligible:
        start = float(card["time_span"][0])
        bucket_start = int(start // GLOBAL_BUCKET_SECONDS) * GLOBAL_BUCKET_SECONDS
        buckets[bucket_start].append(card)
    report = {
        "eligible_local_cards": len(eligible),
        "local_cards_excluded": int(local_report.get("excluded_revisions") or 0),
        "superseded_sources_included": int(
            local_report.get("superseded_revisions_included") or 0),
        "buckets": len(buckets),
        "singleton_buckets": 0,
        "global_cards": 0,
        "sources_used": 0,
        "sources_collapsed": 0,
        "source_claim_tokens": 0,
        "global_claim_tokens": 0,
        "timeline_entries": 0,
    }
    if len(buckets) < 2:
        return [], report
    entries = []
    checked_revisions = {
        str(row.get("revision_id") or "")
        for row in tree.memory_uses.values()
        if row.get("action") == "check"
        and row.get("outcome") == "accept"
        and int(row.get("question_number") or 0) == int(question_number)
        and row.get("scope") == "global"}
    for bucket_start in sorted(buckets):
        bucket_cards = sorted(
            buckets[bucket_start],
            key=lambda card: (
                str(card.get("source_revision_id") or "") in checked_revisions,
                _entry_rank(card)),
            reverse=True)
        checked_cards = [
            card for card in bucket_cards
            if str(card.get("source_revision_id") or "") in checked_revisions]
        representatives = checked_cards or [bucket_cards[0]]
        entries.extend(
            _timeline_entry(card, bucket_start) for card in representatives)
        if len(bucket_cards) == 1:
            report["singleton_buckets"] += 1
    entries = _sample_timeline_entries(
        [entry for entry in entries if entry.get("stored_frames")],
        pinned_revision_ids=checked_revisions)
    if len(entries) < 2:
        return [], report
    revisions = {
        str(entry.get("source_revision_id") or "") for entry in entries}
    times = {
        (float(entry["start"]), float(entry["end"])) for entry in entries}
    if len(revisions) < 2 or len(times) < 2:
        return [], report
    return _timeline_card(entries, eligible, report)


def _claim_entry(tree, claim):
    """One global entry from a claim O promoted. No option, no quote."""
    revision = tree.revisions.get(str(claim.get("source_revision_id") or ""))
    span = claim.get("time_span") or ()
    if not revision or len(span) != 2:
        return None
    handle = revision.get("frame_handle") or {}
    if not int(handle.get("n") or 0) or float(span[0]) >= float(span[1]):
        return None
    text, _source = _claim_from(
        claim.get("event_text"), "", revision.get("branch_summary"), "")
    summary = _compressed_source_claim(
        CLAIM.strip_option_markers(text), words=16)
    bucket = int(float(span[0]) // GLOBAL_BUCKET_SECONDS) * GLOBAL_BUCKET_SECONDS
    return {
        "entry_id": "C:%s" % claim["claim_id"],
        "claim_id": str(claim["claim_id"]),
        "chapter": [float(bucket), max(float(bucket + GLOBAL_BUCKET_SECONDS),
                                       float(span[1]))],
        "start": float(span[0]), "end": float(span[1]),
        "summary": summary, "claim": summary,
        "source_revision_id": revision["revision_id"],
        "source_question": int(claim.get("source_question") or 0),
        "reader": str(revision.get("reader") or ""),
        "branch_id": str(claim.get("source_branch") or ""),
        "event_ids": list(claim.get("source_event_ids") or ()),
        "frame_handle": copy.deepcopy(handle),
        "stored_frames": int(handle.get("n") or 0),
        "source_local_stale": bool(
            claim.get("source_local_stale")
            or tree.review_stale(revision["revision_id"])),
    }


def global_claims_supply(tree, question_number):
    """Global memory is only what O promoted through EVOLVE_GLOBAL.

    A claim exists only after O chose it from an accepted, frame-backed local
    revision, so nothing reaches this card without orchestrator review. Active
    claims from earlier questions persist; a superseded claim is gone.
    """
    entries = [
        entry for entry in (
            _claim_entry(tree, claim) for claim in tree.active_global_claims()
            if 1 <= int(claim.get("source_question") or 0)
            < int(question_number))
        if entry is not None and entry["summary"]]
    report = {"active_claims": len(tree.active_global_claims()),
              "claim_entries": len(entries), "global_cards": 0,
              "timeline_entries": 0}
    if not entries:
        return [], report
    entries.sort(key=lambda row: (row["start"], row["end"]))
    pinned = {str(row.get("revision_id") or "")
              for row in tree.memory_uses.values()
              if row.get("action") == "check" and row.get("outcome") == "accept"
              and row.get("scope") == "global"
              and int(row.get("question_number") or 0) == int(question_number)}
    entries = _sample_timeline_entries(entries, pinned_revision_ids=pinned)
    return _timeline_card(entries, entries, report, claim_backed=True)


def global_cards(tree, question_number, source="timeline"):
    if source == "claims":
        return global_claims_supply(tree, question_number)[0]
    return global_timeline_supply(tree, question_number)[0]


def select_global_anchor(card, question, duration=None,
                         exclude_revision_id="", entry_id=""):
    merged = copy.deepcopy(card)
    if isinstance(exclude_revision_id, str):
        excluded = {exclude_revision_id} if exclude_revision_id else set()
    else:
        excluded = {str(value) for value in exclude_revision_id or ()}
    candidates = [
        candidate for candidate in merged.get("anchor_candidates") or ()
        if int(candidate.get("stored_frames") or 0) > 0
        and str(candidate.get("revision_id") or
               candidate.get("source_revision_id") or "") not in excluded
        and (not entry_id or str(candidate.get("entry_id") or "") == entry_id)]
    if not candidates:
        candidates = [
            candidate for candidate in merged.get("anchor_candidates") or ()
            if int(candidate.get("stored_frames") or 0) > 0]
    if not candidates:
        return merged

    def score(candidate):
        source_card = {
            "scope": "local",
            "claim": candidate.get("claim") or candidate.get("summary"),
            "detail": candidate.get("detail"),
            "quote": candidate.get("quote"),
            "time_span": candidate.get("time_span") or [
                candidate.get("start"), candidate.get("end")],
            "source_question": candidate.get("source_question"),
            "check_status": "not_checked",
        }
        relevance = card_relevance(question, source_card, duration=duration)
        return (
            int(relevance["accepted"]), int(relevance["direct_match"]),
            len(relevance["strong_terms"]), int(relevance["time_match"]),
            len(concrete_tokens(source_card["claim"])),
            int(candidate.get("source_question") or 0),
            str(candidate.get("revision_id") or
                candidate.get("source_revision_id") or ""))

    selected = max(candidates, key=score)
    revision_id = str(selected.get("revision_id") or
                      selected.get("source_revision_id") or "")
    merged["source_revision_id"] = revision_id
    merged["selected_anchor_revision_id"] = revision_id
    merged["selected_anchor_entry_id"] = str(
        selected.get("entry_id") or "")
    merged["frame_handle"] = copy.deepcopy(
        selected.get("frame_handle") or {})
    merged["stored_frames"] = int(selected.get("stored_frames") or 0)
    merged["anchor_source_claim"] = str(
        selected.get("claim") or selected.get("summary") or "")
    if selected.get("claim_id"):
        merged["global_claim_id"] = str(selected["claim_id"])
    return merged

def build_cards(tree, question_number, local_visible=True, global_visible=True,
                global_source="timeline"):
    if global_source not in ("timeline", "claims"):
        raise ValueError("unknown global source: %s" % global_source)
    cards = []
    if local_visible:
        cards.extend(local_cards(tree, question_number))
    if global_visible:
        cards.extend(global_cards(tree, question_number, global_source))
    return cards


def _timeline_card_invalid(card):
    entries = list(card.get("timeline_entries") or ())
    revisions = {
        str(entry.get("source_revision_id") or "") for entry in entries}
    times = [
        (float(entry.get("start") or 0.0), float(entry.get("end") or 0.0))
        for entry in entries]
    ordered = times == sorted(times)
    selected_entry = next((
        entry for entry in entries
        if str(entry.get("entry_id") or "")
        == str(card.get("selected_anchor_entry_id") or "")), None)
    minimum = 1 if card.get("claim_backed") else 2
    return bool(
        len(entries) < minimum
        or len(revisions) < minimum
        or len(set(times)) < minimum
        or not ordered
        or any(
            not entry.get("frame_handle")
            or int(entry.get("stored_frames") or 0) < 1
            or not entry.get("summary")
            or not (
                float(entry.get("chapter", [0, 0])[0])
                <= float(entry.get("start") or 0.0)
                < float(entry.get("end") or 0.0)
                <= float(entry.get("chapter", [0, 0])[1]))
            for entry in entries)
        or selected_entry is None
        or str(selected_entry.get("source_revision_id") or "")
        != str(card.get("selected_anchor_revision_id") or "")
        or str(card.get("source_revision_id") or "")
        != str(card.get("selected_anchor_revision_id") or "")
        or str(selected_entry.get("summary") or "")
        != str(card.get("anchor_source_claim") or ""))


def audit(cards):
    return {
        "schema_version": "memory-card-audit-v3",
        "cards": len(cards),
        "local_cards": sum(card["scope"] == "local" for card in cards),
        "global_cards": sum(card["scope"] == "global" for card in cards),
        "generic_claims": sum(bool(card.get("generic_claim")) for card in cards),
        "missing_claims": sum(not card.get("claim") for card in cards),
        "missing_time_spans": sum(not card.get("time_span") for card in cards),
        "missing_quotes": sum(
            not card.get("quote")
            and card.get("memory_kind") != "global_timeline"
            for card in cards),
        "missing_readers": sum(not card.get("reader") for card in cards),
        "missing_branches": sum(not card.get("branch_id") for card in cards),
        "missing_source_revisions": sum(
            not card.get("source_revision_id") for card in cards),
        "missing_frame_handles": sum(
            not card.get("frame_handle") for card in cards),
        "missing_check_status": sum(
            not card.get("check_status") for card in cards),
        "stale_source_cards": sum(
            bool(card.get("source_local_stale")) for card in cards),
        "global_timeline_cards": sum(
            card.get("memory_kind") == "global_timeline" for card in cards),
        "global_answer_letters": sum(
            bool(card.get("option")) for card in cards
            if card.get("scope") == "global"),
        "global_timeline_multi_source": sum(
            len(set(card.get("source_revision_ids") or ())) >= 2
            for card in cards if card.get("memory_kind") == "global_timeline"),
        "global_timeline_sources": sum(
            len(set(card.get("source_revision_ids") or ()))
            for card in cards if card.get("memory_kind") == "global_timeline"),
        "global_timeline_entries": sum(
            len(card.get("timeline_entries") or ())
            for card in cards if card.get("memory_kind") == "global_timeline"),
        "global_timeline_invalid_entries": sum(
            _timeline_card_invalid(card) for card in cards
            if card.get("memory_kind") == "global_timeline"),
        "global_timeline_anchor_mismatch": sum(
            _timeline_card_invalid(card) for card in cards
            if card.get("memory_kind") == "global_timeline"),
        "global_timeline_source_tokens": sum(
            int((card.get("compression") or {}).get(
                "source_claim_tokens") or 0)
            for card in cards if card.get("memory_kind") == "global_timeline"),
        "global_timeline_claim_tokens": sum(
            int((card.get("compression") or {}).get(
                "global_claim_tokens") or 0)
            for card in cards if card.get("memory_kind") == "global_timeline"),
    }


STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been this that "
    "these those what which who whom whose why how when where from as at by it "
    "its into about over under after before during while do does did can could "
    "should would may might will".split())
GENERIC_TERMS = CLAIM.GENERIC_TERMS
WEAK_COLOR_TERMS = CLAIM.WEAK_COLOR_TERMS


def _tokens(value):
    words = set()
    for word in re.findall(r"[a-z0-9]+", str(value or "").lower()):
        if word.isdigit() or (len(word) > 2 and word not in STOPWORDS):
            words.add(word)
    return words


def concrete_tokens(value):
    return _tokens(value) - GENERIC_TERMS


def question_stem(question):
    return re.split(r"\s*\(A\)", str(question or ""), maxsplit=1)[0].strip()


def _named_terms(value):
    names = set()
    for match in re.finditer(r"\b([A-Z][A-Za-z0-9'-]{2,})\b", str(value or "")):
        word = match.group(1)
        if match.start() == 0:
            continue
        token = word.lower()
        if token not in STOPWORDS and token not in GENERIC_TERMS:
            names.add(token)
    return names


def _timeline_source_windows(card):
    if card.get("memory_kind") == "global_timeline":
        return [
            [float(entry["start"]), float(entry["end"])]
            for entry in card.get("timeline_entries") or ()
            if entry.get("start") is not None and entry.get("end") is not None]
    span = card.get("time_span") or ()
    return [[float(span[0]), float(span[1])]] if len(span) == 2 else []


def _time_match_detail(question, duration, card):
    windows = _timeline_source_windows(card)
    try:
        asked = C.clock_reference(question, float(duration or 0.0))
    except (TypeError, ValueError, AttributeError):
        asked = None
    if not windows or not asked:
        return {"score": 0, "reference": asked, "coverage": 0.0,
                "overlap_seconds": 0.0}
    asked_seconds = max(0.0, float(asked[1]) - float(asked[0]))
    covered = _covered_seconds(windows, asked)
    coverage = covered / asked_seconds if asked_seconds else 0.0
    if asked_seconds <= 60.0:
        score = 2 if covered > 0.0 else 0
    else:
        score = 2 if coverage >= 0.75 else 1 if coverage > 0.0 else 0
    return {
        "score": int(score),
        "reference": [float(asked[0]), float(asked[1])],
        "coverage": min(1.0, max(0.0, coverage)),
        "overlap_seconds": covered,
    }


def searchable_text(card):
    entries = " ".join(
        str(entry.get("summary") or entry.get("claim") or "")
        for entry in card.get("timeline_entries") or ())
    return " ".join((
        str(card.get("claim") or ""),
        str(card.get("quote") or ""),
        entries))


def card_relevance(question, card, duration=None):
    """Return a conservative, gold-blind relevance decision for one card."""
    stem = question_stem(question)
    options = " ".join(C.options_of(question))
    stem_tokens = _tokens(stem)
    option_tokens = _tokens(options)
    evidence_tokens = _tokens(searchable_text(card))
    question_shared = stem_tokens.intersection(evidence_tokens)
    question_strong = question_shared - GENERIC_TERMS - WEAK_COLOR_TERMS
    option_shared = option_tokens.intersection(evidence_tokens)
    option_strong = option_shared - GENERIC_TERMS - WEAK_COLOR_TERMS
    weak_colors = question_shared.intersection(WEAK_COLOR_TERMS)
    novel_evidence = evidence_tokens - stem_tokens - GENERIC_TERMS
    proper_shared = question_strong.intersection(_named_terms(stem))
    option_colors = option_strong.intersection(WEAK_COLOR_TERMS)
    direct = bool(
        question_strong
        and novel_evidence
        and (
            len(question_shared & question_strong) >= 2
            or bool(proper_shared)
            or (len(question_strong) == 1 and bool(question_shared))
            or (bool(question_shared) and bool(option_colors))
        ))
    temporal = _time_match_detail(stem, duration, card)
    time_score = int(temporal["score"])
    return {
        "accepted": bool(direct or time_score == 2),
        "direct_match": direct,
        "strong_terms": sorted(question_strong),
        "option_terms": sorted(option_strong),
        "weak_color_terms": sorted(weak_colors),
        "novel_evidence_terms": sorted(novel_evidence),
        "matched_terms": sorted(question_shared.union(option_shared)),
        "proper_name_terms": sorted(proper_shared),
        "time_match": time_score,
        "time_coverage": float(temporal["coverage"]),
        "time_overlap_seconds": float(temporal["overlap_seconds"]),
        "time_reference": temporal["reference"],
        "reason": ("direct" if direct else
                   "time" if time_score == 2 else
                   "time_partial" if time_score else "none"),
    }


def _card_score(question, duration, card):
    relevance = card_relevance(question, card, duration=duration)
    return (
        int(relevance["direct_match"]),
        len(relevance["strong_terms"]),
        len(relevance["weak_color_terms"]),
        float(relevance["time_coverage"]),
        int(relevance["time_match"]),
        int(card.get("source_question") or 0),
        int(card.get("check_status") in (
            "accepted_review", "check_accept", "cite_accept")),
    )

def rank_cards(cards, question, duration=None, per_scope=4,
               avoid_global_sources=(), interleave_scopes=False,
               scope_order=("local", "global")):
    per_scope = int(per_scope)
    if per_scope < 1:
        raise ValueError("per_scope must be positive")
    ranked = []
    def checked(card):
        # A review from an earlier question is not a check of the current
        # question. It must still pass the semantic relevance filter.
        return str(card.get("check_status") or "") in (
            "check_accept", "cite_accept")

    local_rows = [
        card for card in cards if card.get("scope") == "local"
        and (checked(card)
             or card_relevance(question, card, duration=duration)["accepted"])]
    local_rows.sort(key=lambda card: (
        int(checked(card)),
        _card_score(question, duration, card), str(card.get("card_id"))),
        reverse=True)
    local_limit = max(
        per_scope, sum(checked(card) for card in local_rows))
    selected_local = local_rows[:local_limit]
    excluded = {str(value) for value in avoid_global_sources or ()}
    global_rows = [
        select_global_anchor(
            card, question, duration=duration,
            exclude_revision_id=excluded)
        for card in cards if card.get("scope") == "global"]
    global_rows = [
        card for card in global_rows
        if checked(card) or card.get("claim_backed")
        or card_relevance(question, card, duration=duration)["accepted"]]
    global_rows.sort(key=lambda card: (
        int(checked(card)),
        _card_score(question, duration, card), str(card.get("card_id"))),
        reverse=True)
    global_limit = max(
        per_scope, sum(checked(card) for card in global_rows))
    selected_global = global_rows[:global_limit]
    if interleave_scopes:
        if set(scope_order) != {"local", "global"}:
            raise ValueError("scope_order must name local and global")
        selected = {"local": selected_local, "global": selected_global}
        for index in range(max(len(selected_local), len(selected_global))):
            for scope in scope_order:
                if index < len(selected[scope]):
                    ranked.append(selected[scope][index])
    else:
        ranked.extend(selected_local)
        ranked.extend(selected_global)
    return ranked


def _entry_check_order(question, duration, entries):
    def score(entry):
        card = {
            "scope": "local",
            "claim": entry.get("summary") or entry.get("claim"),
            "quote": "",
            "time_span": [entry.get("start"), entry.get("end")],
            "source_question": entry.get("source_question"),
            "check_status": entry.get("check_status") or "not_checked",
        }
        return _card_score(question or "", duration, card)
    return sorted(
        enumerate(entries, 1),
        key=lambda pair: (score(pair[1]), -pair[0]),
        reverse=True)


def render_cards(cards, question="", duration=None, entry_handles=4):
    lines = []
    handles = {}
    for index, card in enumerate(cards, 1):
        handle = "M%d" % index
        span = card.get("time_span") or ()
        when = ("%g-%gs" % (float(span[0]), float(span[1]))
                if len(span) == 2 else "time unknown")
        status = str(card.get("check_status") or "unchecked")
        lines.append(
            "%s [%s] Q%s %s %s | %s" % (
                handle, card.get("scope"), card.get("source_question"),
                when, status, card.get("claim") or "(missing claim)"))
        if card.get("memory_kind") == "global_timeline":
            entries = list(card.get("timeline_entries") or ())
            checkable = {
                entry_index for entry_index, _entry in _entry_check_order(
                    question, duration, entries)[:max(1, int(entry_handles))]}
            checkable.update(
                entry_index for entry_index, entry in enumerate(entries, 1)
                if str(entry.get("check_status") or "") in (
                    "accepted_review", "check_accept", "cite_accept"))
            for entry_index, entry in enumerate(entries, 1):
                entry_handle = "%s.%d" % (handle, entry_index)
                if entry_index in checkable:
                    anchor = select_global_anchor(
                        card, question, duration=duration,
                        entry_id=str(entry.get("entry_id") or ""))
                    anchor["check_status"] = str(
                        entry.get("check_status") or "not_checked")
                    if entry.get("checked_option"):
                        anchor["checked_option"] = str(
                            entry.get("checked_option") or "")
                    else:
                        anchor.pop("checked_option", None)
                    if entry.get("checked_quote"):
                        anchor["checked_quote"] = str(
                            entry.get("checked_quote") or "")
                    else:
                        anchor.pop("checked_quote", None)
                    handles[entry_handle] = anchor
                    marker = entry_handle
                else:
                    marker = "-"
                entry_status = str(
                    entry.get("check_status") or card.get("check_status")
                    or "unchecked")
                lines.append(
                    "  %s %g-%gs %s | %s" % (
                        marker, float(entry.get("start") or 0.0),
                        float(entry.get("end") or 0.0), entry_status,
                        entry.get("summary") or "(missing event)"))
        else:
            handles[handle] = card
            detail = str(card.get("detail") or "").strip()
            if detail and _norm(detail).lower() != _norm(
                    card.get("claim")).lower():
                lines.append("  Detail: %s" % detail[:500])
            if card.get("quote"):
                lines.append("  Quote: %s" % card["quote"])
    return "\n".join(lines), handles
