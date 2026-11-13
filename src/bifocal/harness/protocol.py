#!/usr/bin/env python3
"""**BiFocal's two-pass chunk framework.**

Memory: written whenever an agent finishes reading; always presented to whoever is
reading or answering; **session-only**, keyed by video. The orchestrator may read any
chunk's memory; a reader sees only its own chunk's memory and never the global one.

Everything here is pure or takes an injected `ask(text, images, role)`, so the whole
flow is testable offline with a scripted model."""

import collections
import copy
import hashlib
import os
import re

from . import media as M
from ..memory import local as B
from ..memory import session as BS
from ..memory.replay import canonical_sha256

CHUNK_SECONDS = 64
COARSE_CHUNK = 0
SELECTIVE_FPS = 0.5
ALL_FPS = 1.0                # denser, fixed, for the all-chunks pass
MAX_PICK = 6
EVIDENCE_LIMIT = 3          # how many earlier leaves the commit block may list as E<k>
SCHEMA_VERSION = 1

MAX_TURNS = 8
LOOP_READS = 3
ZOOM_MIN_FRAMES = 8
SCAN_CHUNKS = 12
SCAN_FRAMES_PER_CHUNK = 3
SCAN_NOTE_CHARS = 240
STATE_CHARS = 2400           # the turn history's declared ceiling, for prompt and trainer
# **And the whole turn's**: the history is one part of what the policy sees (the question, the
# menu, the map and the budget are the rest), so the trainer needs a ceiling on the call itself.
# A row above it is named in the run's record rather than silently trained on.
STATE_CHARS_HARD = 12000
MAX_MEMORY_ENTRIES = 3       # per reader prompt
MAX_GLOBAL_MEMORIES = 8      # chunk memories shown to the orchestrator
BELIEF_HANDLE = "T1"
#                              belief for this question is a place in its own right - and the tier is
#                              **claimed** (its words are the readers' claims), never provenance.
BELIEF_CHARS = 300
#                              carries the readers' own words, bounded - a distillation, not an index.
#                              Set **inside `INDEX_CHARS`** (320) so a belief with a claim is never
#                              dropped by the section bound: at 380 it was, and 5 of 11 stopped rendering.
REASON_CHARS = 32            # one reason's own words, clipped
MAX_REASONS = 2              # at most this many supporting reasons, and the rival's words
MAX_SIGHTINGS = 3
FRAMES_MAX_WORDS = 30
SIGHTING_LINE = re.compile(
    r"(?:PLACE:)?\s*(\d+(?:\.\d+)?)\s*s?\s*-\s*(\d+(?:\.\d+)?)\s*s?\s*\|\s*"
    r"ANSWER:\s*\(?([A-H])\)?\s*\|\s*QUOTE:\s*(.+)", re.I)
INDEX_CHARS = 320            # the always-loaded index in one O call
LONG_VIDEO_CONTEXT_CHARS = 900
QUESTION_HISTORY_TEXT = "(question asked; no evidence saved)"
MAX_FREE_REFUSALS = 3       # refused actions re-asked inside one turn
INDEX_MIN_HANDLES = 2        # map lines the index always keeps room for
CUE_WORDS = 4
INDEX_SETTLED_LINES = 3      # settled questions rendered, the most recent ones
NOTE_LINES = 3               # notes rendered, best-matching first
NOTE_CHARS = 60              # one note's own clip
LEAF_SIGHTING_CHARS = 120
COVERAGE_CHARS = 48          # the coverage line's own clip, tight: it grows with the reads
PROPOSAL_CHARS = 90          # the proposal line's own clip
TREE_HEADER_CHARS = 64
TREE_OFFER_CHARS = INDEX_CHARS - (PROPOSAL_CHARS + COVERAGE_CHARS + 2) - TREE_HEADER_CHARS - 4
PROPOSAL_PREFIX = "proposed next read: "
PROPOSAL_SIGHTING_CHARS = 40
INDEX_MIN_NAV = 2            # navigational head lines the index always keeps room for
INDEX_TOPIC_CHARS = 44
INDEX_QUESTION_WORDS = 6
READER_MEMORY_CHARS = 400    # a reader's own chunk memory in one call
READER_NOTE_CHARS = 240      # the part of it that is verbatim
GLOBAL_NOTE_LIMIT = 3             # earlier notes one reader's block may carry
READER_GLOBAL_MEMORY_CHARS = 500
READER_TOTAL_MEMORY_CHARS = READER_MEMORY_CHARS + READER_GLOBAL_MEMORY_CHARS + 1
GLOBAL_NOTE_CHARS = 160           # one note's own words, clipped, in the reader's block
GLOBAL_NEIGHBOUR_TERMS = 2
GLOBAL_HANDLE_PREFIX = "G"        # the reader block's own handle namespace: never O's M<k>/E<k>
PLACE_MIN_WINDOW = 8.0
MAX_PLACE_FRAMES = 64        # what one check read may attach
MAX_CHECKS = 3               # claimed reports one check round may look at
MAX_SELECTION_ROUNDS = 1
MAX_ESCALATION_READS = 2

# **How a reader writes a place.** Four forms: `t=1302s-1307s`, `42:04`, `12-40` (the
# declared `start-end inside your chunk`), and `NONE`. The seconds forms are absolute; clock forms are absolute; bare pairs are
# offsets inside the chunk. **`media.time_references` cannot do this**: its clock
# pattern requires the words "at/from/around" before the time, so a bare `42:04` -
# exactly what sample 2's reader wrote - would not parse.
PLACE_SECONDS_RANGE = re.compile(
    r"(?<![\d.])(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s?\s*(?:-|to|until)\s*"
    r"(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s\b", re.I)
PLACE_SECONDS_ONE = re.compile(r"t\s*=\s*(\d+(?:\.\d+)?)\s*s", re.I)
PLACE_CLOCK_RANGE = re.compile(
    r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:-|to|until)\s*"
    r"(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
PLACE_CLOCK_ONE = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d:])")
PLACE_DECIMAL_RANGE = re.compile(
    r"(?<![\d.])(?:t\s*=\s*)?(\d+\.\d+)\s*s?\s*(?:-|to|until)\s*"
    r"(?:t\s*=\s*)?(\d+\.\d+)\s*s?", re.I)
PLACE_OFFSET_RANGE = re.compile(
    r"(?<![\d.:])(\d{1,4})\s*-\s*(\d{1,4})(?![\d.:])")
FIELD_NAMES = ("THINK", "FRAMES", "PLACE", "CAN_ANSWER", "ANSWER", "KIND", "QUOTE",
               "CLAIM_SUBJECT", "CLAIM_PREDICATE", "CLAIM_VALUE", "CLAIM_UNIT",
               "MEMORY_USED", "MEMORY_UPDATE", "ABOUT", "PICK", "DRAFT", "CHECK")
QUOTE_MAX_WORDS = 60
DEGENERATE_BELOW = 2 * CHUNK_SECONDS   # coarse read no longer sparser than readers


# --------------------------------------------------------------------------
# the unit

def chunk_list(duration, size=CHUNK_SECONDS):
    """`[(id, first, last), ...]` - numbered once per video, the last one shorter."""
    duration = float(duration)
    out, start, index = [], 0.0, 1
    while start < duration - 1e-9:
        last = min(duration, start + size)
        out.append((index, round(start, 3), round(last, 3)))
        start, index = last, index + 1
    return out or [(1, 0.0, round(duration, 3))]


def chunk_menu(chunks):
    """The numbered list the orchestrator chooses from."""
    return " | ".join("%d: %g-%gs" % (cid, first, last) for cid, first, last in chunks)


def frames_for(first, last, fps, cap=64):
    """How many frames a window read at `fps` would attach, capped like the client."""
    # **`+ 0.5`, not `round`**: Python rounds halves to even, so a 65-second window
    # at 0.5 fps would silently lose a frame. The count a reader is given must be
    # the count the prompt states.
    return max(1, min(cap, int((float(last) - float(first)) * float(fps) + 0.5)))


def _clock(hours, minutes, seconds=None):
    """`mm:ss` or `h:mm:ss` -> seconds."""
    hours, minutes = int(hours), int(minutes)
    if seconds is None:
        return hours * 60 + minutes
    return hours * 3600 + minutes * 60 + int(seconds)


def place_spans(text, first, last):
    """The spans a reader's `PLACE` names, **clamped to its own chunk**.

    Absolute forms (`t=1302s-1307s`, `03:15`) are kept only when they fall inside
    the chunk; plain offsets (`12-40`) are read inside it. A span that lies entirely
    outside is dropped rather than mapped - sample 2's reader named `42:04` while
    reading chunk 4, and moving that into the chunk would invent evidence.
    """
    first, last = float(first), float(last)
    found = []
    for match in PLACE_SECONDS_RANGE.finditer(text or ""):
        found.append((float(match.group(1)), float(match.group(2))))
    for match in PLACE_DECIMAL_RANGE.finditer(text or ""):
        found.append((float(match.group(1)), float(match.group(2))))
    for match in PLACE_CLOCK_RANGE.finditer(text or ""):
        found.append((_clock(*match.group(1, 2, 3)), _clock(*match.group(4, 5, 6))))
    for match in PLACE_SECONDS_ONE.finditer(text or ""):
        found.append((float(match.group(1)), float(match.group(1))))
    for match in PLACE_CLOCK_ONE.finditer(text or ""):
        at = _clock(*match.group(1, 2, 3))
        found.append((at, at))
    for match in PLACE_OFFSET_RANGE.finditer(text or ""):
        a, b = float(match.group(1)), float(match.group(2))
        if first <= a and b <= last:
            found.append((a, b))
        elif a < last - first:
            # an offset whose end runs past the chunk is clamped, as it always was: the last
            # chunk of a video is often shorter than the grid (a 150 s video ends at 150 s)
            found.append((first + a, min(last, first + b)))
    out = []
    for a, b in found:
        if b < a:
            a, b = b, a
        if b < first or a > last:            # entirely outside the chunk
            continue
        out.append((max(first, a), min(last, b)))
    # **A range's own ends are not separate places.** `t=1302s-1307s` is read by the
    # range pattern *and* by the single-second pattern, so the point spans inside a
    # real span are dropped here rather than inflating the record.
    wide = [(a, b) for a, b in out if b > a]
    kept = [(a, b) for a, b in out
            if b > a or not any(w[0] <= a and b <= w[1] for w in wide)]
    seen, unique = set(), []
    for span in sorted(kept):
        if span not in seen:
            seen.add(span)
            unique.append(span)
    return [(round(a, 3), round(b, 3)) for a, b in unique]


def window_for_place(text, first, last, min_width=PLACE_MIN_WINDOW):
    """One window for the check read: the bound of the spans, or the whole chunk.

    A place narrower than `min_width` is widened symmetrically and shifted inside
    the chunk, the rule `lvb_k1.widened` already uses for a gold window - one second
    of a 1 fps read is a single frame, which checks nothing.
    """
    first, last = float(first), float(last)
    spans = place_spans(text, first, last)
    if not spans:
        return (first, last)
    a, b = min(span[0] for span in spans), max(span[1] for span in spans)
    return widen_inside(a, b, first, last, min_width)


def widen_inside(a, b, first, last, min_width=PLACE_MIN_WINDOW):
    """**A narrow place widened and shifted inside its chunk** - one second of a 1 fps read is a
    single frame, which checks nothing."""
    a, b, first, last = float(a), float(b), float(first), float(last)
    if b - a < min_width:
        centre = (a + b) / 2.0
        a, b = centre - min_width / 2.0, centre + min_width / 2.0
        if a < first:
            a, b = first, min(last, first + min_width)
        if b > last:
            a, b = max(first, last - min_width), last
    return (round(a, 3), round(b, 3))


def bare_range_that_fits_neither_form(text, first, last):
    """`(value, chunk seconds)` when a bare range is neither the seconds nor an offset.

    `place_spans` accepts a bare `a-b` that lies inside the chunk as seconds (the form the
    frames are labelled with, written without the `s`), and otherwise one whose start is an
    offset inside the chunk. A range that fits neither - `150-157` read in a 22-second chunk -
    is the reader using the wrong form, and the refusal names it rather than saying the place
    is outside a window the reader may never have been thinking about.
    """
    if PLACE_SECONDS_RANGE.search(text or "") or PLACE_SECONDS_ONE.search(text or "") \
            or PLACE_CLOCK_RANGE.search(text or "") or PLACE_CLOCK_ONE.search(text or ""):
        return None
    first, last = float(first), float(last)
    for match in PLACE_OFFSET_RANGE.finditer(text or ""):
        a, b = float(match.group(1)), float(match.group(2))
        if not (first <= a and b <= last) and not a < last - first:
            return match.group(0), last - first
    return None


# --------------------------------------------------------------------------
# the session memory

def _read_context(session, question_number):
    """Group actual prior reads; this is a view, not new state."""
    source = list(session.get("reads") or ())
    groups = {}
    for chunk, entries in sorted(session.get("chunks", {}).items()):
        if int(chunk) < 1:
            continue
        for entry in entries:
            source.append({
                "question_number": entry.get("question_number"), "chunk": int(chunk),
                "first": (entry.get("assigned") or entry.get("seconds") or [None, None])[0],
                "last": (entry.get("assigned") or entry.get("seconds") or [None, None])[-1],
                "frames": int((entry.get("frames") or {}).get("n") or 0),
                "answer": entry.get("answer") or "", "entry_id": entry.get("id"),
            })
    for record in source:
        chunk = record.get("chunk")
        if chunk is None or int(chunk) < 1:
            continue
        if int(record.get("question_number") or 0) >= int(question_number or 1):
            continue
        span = ([record.get("first"), record.get("last")]
                if record.get("first") is not None and record.get("last") is not None
                else record.get("seconds"))
        entry = next((entry for entry in session.get("chunks", {}).get(int(chunk), ()) or ()
                      if str(entry.get("id")) == str(record.get("entry_id"))), None)
        author = str(record.get("author") or (entry or {}).get("author") or "")
        if not re.fullmatch(r"W\d+", author):
            continue
        key = (record.get("question_number"), int(chunk), author,
               tuple(span or ()))
        group = groups.setdefault(key, {
            "question_number": record.get("question_number"), "chunk": int(chunk),
            "seconds": list(span) if span else None, "frames": int(record.get("frames") or 0),
            "answer": "", "citable": False})
        group["answer"] = group["answer"] or str(record.get("answer") or "")[:1]
        group["citable"] = group["citable"] or bool(
            entry and entry.get("review_status") == "accepted" and entry.get("citable", True))
    return sorted(groups.values(), key=lambda row: (
        int(row.get("question_number") or 0), int(row.get("chunk") or 0),
        tuple(row.get("seconds") or ())))


def _remember_question_marker(session, question, question_number, duration):
    """Keep the question count resumable when the question saved no leaf."""
    if session.get("memory_disabled") or (session.get("store") or {}).get("mode") != "files":
        return
    if any(int(entry.get("question_number") or 0) == int(question_number)
           for entries in (session.get("chunks") or {}).values()
           for entry in entries or ()):
        return
    remember_chunk(
        session, COARSE_CHUNK, "question history", QUESTION_HISTORY_TEXT,
        touch_map=False, citable=False, author="O", seconds=[0.0, float(duration)],
        assigned=[0.0, float(duration)], question_number=question_number, kind="method")


def tell_the_tree(session, entry, span=None, status=None, by="", why="", fields=None):
    """**The harness tells the store what happened**."""
    store = session.get("store") or {"mode": "session"}
    # **The id may travel as `entry_id`** (the index view hands out projections with `handle` and
    # `entry_id` and no `id`), so both names resolve here - a fetch that silently skipped the write
    # would leave the tree telling a different story from the session.
    entry_id = entry.get("id") if entry else None
    if entry_id is None and entry:
        entry_id = entry.get("entry_id")
    if store.get("mode") != "files" or entry_id is None:
        return None
    from ..memory import global_store as _tree_store
    try:
        marked = _tree_store.mark_leaf(store["root"], store.get("video") or session.get("video"),
                                       entry_id, status=status, span=span, by=by, why=why,
                                       fields=fields)
    except KeyError:
        session.setdefault("events", []).append(
            {"kind": "tree_miss", "entry_id": entry_id, "status": status,
             "span": list(span) if span else None, "by": by, "why": why})
        return None
    # **A dead claim leaves the map** (coherence audit): if nothing live is left in that chunk, its
    # line goes - the session view and the tree must say the same thing, and the tree's own
    # `tree_view` invariant is what compares them.
    if status in ("retracted", "superseded", "merged", "rejected", "unsure",
                  "unverified", "verified"):
        chunk = marked.get("parent")
        # **The door that tells the tree tells the session too** (coherence audit): whichever
        # caller marked the leaf, the session entry carries the same death, or the two copies
        # disagree and the map keeps a claim nothing stands behind.
        for item in session.get("chunks", {}).get(chunk) or []:
            if item.get("id") == entry_id:
                for key, value in (fields or {}).items():
                    item[key] = copy.deepcopy(value)
                if status is not None:
                    item["status"] = status
                if status in ("retracted", "superseded", "merged", "rejected", "unsure"):
                    if status == "merged":
                        item["citable"] = False
                        item["merged_into_tree"] = True
                    else:
                        item["citable"] = False
                        if "review_status" not in (fields or {}):
                            item["review_status"] = status
        refresh_line(session, chunk)
    return marked


def refresh_line(session, chunk):
    """**Recompute a chunk's map line from its live leaves** (the 50-item run's `tree_view`
    violation): the session's line was the last write, while the tree's rebuild takes the highest-id
    live leaf - so any death or fold that is not followed by a fresh write left the two copies
    saying different things. One function, called by the door that marks a leaf and by the
    evolution pass, so a line can only ever be a live claim (or the marker)."""
    from ..memory import global_store as _tree_store
    entries = session.get("chunks", {}).get(chunk) or []
    live = [item for item in entries if not item.get("retracted") and item.get("citable", True)]
    if live:
        survivor = max(live, key=lambda item: int(item["id"]))
        # **The same truncation the write gate uses** (the 50-item run's `tree_view` violation): the
        # session entry keeps the reader's full description while the tree stores the map's one-line
        # form, so a line refreshed from `description` was longer than the leaf's own sighting and
        # the two copies disagreed by construction.
        line = map_line(survivor.get("description") or survivor.get("update") or "")
        session.setdefault("chunk_map", {})[chunk] = line or _tree_store.DEAD_LINE
        session.setdefault("notes", {})[chunk] = {
            "chunk": chunk,
            "sighting": session["chunk_map"][chunk],
            "span": ([float(value) for value in survivor["seconds"]]
                     if survivor.get("seconds") else None),
            "family": survivor.get("family") or "",
            "question_number": survivor.get("question_number")}
    else:
        session.setdefault("chunk_map", {})[chunk] = _tree_store.DEAD_LINE
        session.get("notes", {}).pop(chunk, None)
    return session["chunk_map"][chunk]


def map_line(text):
    """The map's one-line form of a sighting - **the same rule `remember_chunk` writes with**
    (first sentence, capped), so a line can be recomputed from an entry without drifting."""
    return str(text or "").strip().split(".")[0][:120]


def _entry_text(entry):
    return ((entry.get("description") or "") + " "
            + (entry.get("update") or "")).strip()


def remember_chunk(session, chunk_id, question, description, update=None,
                   touch_map=True, citable=True, author="", seconds=None,
                   question_number=None, provisional=False, verified=None,
                   answer_side="", kind=None, family=None, assigned=None, answer="",
                   frames=None, claim=None, raw_reply="", quote=""):
    """A reader finished reading: its observation is kept **with the question**.

    `touch_map=False` writes the entry but leaves the chunk's one-line map alone -
    a reader that reported nothing should still be in the failing history, but it
    must not overwrite what that chunk is known to hold.

    The provenance fields are what make the memory auditable and what let a later
    read retract or re-confirm an entry (BiFocal v2): `author`, `seconds`,
    `question_number`, `provisional` (written during *this* question, so it may not
    be cited in it), `verified` and `answer_side` (the retrieval key)."""
    if session.get("memory_disabled"):
        return None                     # the no-memory ablation writes nothing
    entry = {"id": session.setdefault("next_id", 1), "question": question,
             "description": (description or "").strip(),
             "update": (update or "").strip(), "citable": bool(citable),
             "frames": dict(frames) if frames else None,
             "claim": copy.deepcopy(claim) if claim else None,
             "author": author, "seconds": list(seconds) if seconds else None,
             "assigned": list(assigned) if assigned else None,
             "question_number": question_number,
             "provisional": bool(provisional), "verified": verified,
             "status": "fresh" if provisional else None,
             "answer_side": answer_side,
             "answer": (answer or "").strip().upper()[:1],
             # **A raw finding is saved before O reviews it, but it cannot support an
             # answer yet.** These flags are set only by a support review.
             "answer_support": False, "citation_allowed": False,
             "review_status": "unreviewed",
             "raw_reply": str(raw_reply or ""), "quote": str(quote or ""),
             "kind": kind or ("observation" if citable else "note"),
             "retracted": False, "retracted_by": "", "retracted_on": ""}
    session["next_id"] = entry["id"] + 1
    session["chunks"].setdefault(chunk_id, []).append(entry)
    line = map_line(update or description or "")
    store = session.get("store") or {"mode": "session"}
    if store.get("mode") == "files" and (line or raw_reply or frames):
        from ..memory import global_store as _tree_store           # local: the store is only needed when it is on
        stored_frames = None
        # (the record is the **caller's**: one read's frames are stored once and every leaf of that
        # read points at them, so a leaf writes no pixels of its own. See `apply_reader_memory`.)
        _tree_store.save_leaf(store["root"], store.get("video") or session.get("video"),
                              chunk_id,
                              {"id": entry["id"], "question": question,
                               "sighting": map_line(description or update or ""),
                               "kind": entry["kind"], "family": family or "",
                               "author": author, "question_number": question_number,
                               "citable": entry["citable"], "raw_reply": entry["raw_reply"],
                               "quote": entry["quote"], "answer": entry["answer"],
                               "claim": copy.deepcopy(entry.get("claim"))},
                              seconds or _tree_store.chunk_span(chunk_id, store.get("duration")),
                              assigned=assigned,
                              duration=store.get("duration"),
                              frames=frames)
    if line and touch_map:
        session["chunk_map"][chunk_id] = line
        session.setdefault("notes", {})[chunk_id] = {
            "chunk": chunk_id, "sighting": line,
            "span": [float(value) for value in seconds] if seconds else None,
            "family": family if family is not None else (session.get("family") or ""),
            "question_number": question_number}
        if seconds:
            return move_for_placebo(session, chunk_id, entry, seconds)
        if callable(session.get("placebo_move")):
            entry["placebo_pending"] = True
    return entry


def note_of(session, chunk_id):
    """**A chunk's note, as its four fields**: chunk, seconds span, sighting, question family.

    The span is what the writer resolved when it had one, and otherwise what the store's own
    read record says - one source of truth for "where in the video is this", so a note and the
    read that produced it cannot disagree.
    """
    stored = (session.get("notes") or {}).get(chunk_id) or {}
    span = stored.get("span") or latest_read_span(session, chunk_id)
    return {"chunk": chunk_id,
            "span": span,
            "sighting": stored.get("sighting")
            or (session.get("chunk_map") or {}).get(chunk_id) or "",
            "family": stored.get("family") or ""}


def latest_read_span(session, chunk_id):
    """The seconds the most recent read of this chunk resolved, or `None`."""
    for report in reversed(list(session.get("reads") or ())):
        if report.get("chunk") == chunk_id and report.get("first") is not None:
            return [float(report["first"]), float(report["last"])]
    return None


def coverage_spans(session):
    """**The coverage memory**: every read this session records - chunk, resolved seconds,
    question, depth - plus the whole-video surveys."""
    out = []
    for report in session.get("reads") or ():
        if report.get("first") is None or report.get("last") is None:
            continue
        out.append({"chunk": report.get("chunk"),
                    "seconds": [float(report["first"]), float(report["last"])],
                    "question_number": report.get("question_number"),
                    "depth": str(report.get("pass") or "read"),
                    "density": report.get("fps")})
    for glance in session.get("surveys") or ():
        out.append({"chunk": None, "seconds": list(glance.get("seconds") or [0.0, 0.0]),
                    "question_number": glance.get("question_number"),
                    "depth": "survey", "density": glance.get("density")})
    return out


def chunk_bounds(chunk_id, chunks):
    """The `(first, last)` of one chunk in a chunk list, or `None`."""
    for cid, first, last in chunks or ():
        if cid == chunk_id:
            return [float(first), float(last)]
    return None


def render_coverage_line(session, chunks, limit=COVERAGE_CHARS):
    """**What the team has read, and what it has not** - one line, inside its own bound."""
    ids = [_chunk_text(chunk_id) for chunk_id, _first, _last in chunks or ()]
    id_set = set(ids)
    seen, spans, surveyed = [], [], False
    for span in coverage_spans(session):
        if span["chunk"] is None:
            surveyed = True
            continue
        label = _chunk_text(span["chunk"])
        if label not in id_set or label in seen:
            continue
        seen.append(label)
        spans.append("%g-%gs" % (span["seconds"][0], span["seconds"][1]))
    line = "read: %d/%d chunks" % (len(seen), len(ids))
    if spans:
        line += " (last %s)" % spans[-1]
    if surveyed:
        line += " | surveyed coarse"
    return _clip(line, limit)


def render_proposal_line(session, question, chunks, read_now=(), limit=PROPOSAL_CHARS):
    """**Where the next read should go**: the best-matching note for a chunk this question has
    not read yet, with its own seconds."""
    wanted = _terms(question)
    read_now = set(read_now or ())
    named = sorted(c for c in clock_chunks(question, chunks) if c not in read_now)
    clock_pick = named[0] if named else None
    if clock_pick is None:
        return ""
    chunk_id = clock_pick
    text = " ".join(str((session.get("chunk_map") or {}).get(chunk_id) or "").split())
    note = note_of(session, chunk_id)
    span = note["span"] or chunk_bounds(chunk_id, chunks) or [0.0, 0.0]
    # **The sighting is re-rendered through `note_form`, at the proposal's own clip** (the I15
    # class, and the reviewers' round on it): the audit excuses a note by removing *every* string
    # the map was seen to render, so this shorter form is excused as it stands - and the fix is on
    # the audit's side (`run_chunked` subtracts each visible form), because a renderer that had
    # to match the audit's single clip to the character would make the bound the author of the
    # prompt. What must not happen is a render form the audit does not know about at all.
    return _clip("%schunk %s, %g-%gs, unread: %s"
                 % (PROPOSAL_PREFIX, _chunk_text(chunk_id), span[0], span[1],
                    note_form(text, PROPOSAL_SIGHTING_CHARS)), limit)


def proposal_chunk(proposal):
    """The chunk id a rendered proposal named, or `None` - its own form, parsed once.

    **`search`, not `match`** (its own test caught this): the caller hands it a whole rendered
    block, so an anchored match answered `None` for every proposal the map carried - the turn's
    `nav` flag would have said "no proposal" while the prompt showed one.
    """
    found = PROPOSAL_LINE.search(proposal or "")
    return found.group(1) if found else None


def retract(session, chunk_id, target="", by="", question="", second=None,
            basis=""):
    """Mark entries retracted - **never deleted**, per the reviewers' rule.

    `target` is `""` for the chunk's current entries, or words naming the ones to
    retire. The entries stay in the session as history, rendered `superseded`, and
    they are not citable while retracted; a later supporting read can re-confirm one.
    """
    touched = []
    wanted = _normalise(target)
    for entry in session["chunks"].get(chunk_id, []):
        if entry.get("retracted") or entry.get("provisional"):
            continue
        if wanted and wanted not in _normalise(_entry_text(entry)):
            continue
        entry["retracted"] = True
        entry["retracted_by"] = by
        entry["retracted_on"] = question
        entry["retracted_second"] = second
        tell_the_tree(session, entry, status="retracted", by=by or "retract",
                      why=basis or question or "retracted")
        touched.append(entry["id"])
    session.setdefault("events", []).append(
        {"kind": "retraction", "chunk": chunk_id, "by": by, "question": question,
         "second": second, "basis": basis, "ids": touched})
    return touched


READER_MEMORY_FLAG = re.compile(r"^\s*(NONE|HOLDS|WRONG)\b\s*:?\s*(.*)$", re.I)


def apply_reader_memory(session, chunk_id, question, used_line, update_line,
                        reports=(), options=(), read_window=(0.0, 0.0),
                        author="", question_number=None, description="",
                        provisional=True, place="", answer="", images=None,
                        claim=None, raw_reply="", quote=""):
    """One reader's memory bookkeeping for one read, from its own two lines."""
    event = {"kind": "reader_memory", "chunk": chunk_id, "author": author,
             "question": question, "question_number": question_number}
    flag = READER_MEMORY_FLAG.match(used_line or "")
    if flag and flag.group(1).upper() == "WRONG":
        second = seconds_of((flag.group(2) or "").split()[0]
                            if (flag.group(2) or "").strip() else "")
        inside = (second is not None and float(read_window[0]) <= second
                  <= float(read_window[1]))
        if inside:
            event["retracted"] = retract(session, chunk_id, "", by=author,
                                         question=question, second=second,
                                         basis=(used_line or "").strip())
        else:
            event["refused_retraction"] = (
                "the second %s is not inside the window this read covered (%g-%g)"
                % (second, read_window[0], read_window[1]))
            session.setdefault("events", []).append(
                {"kind": "retraction_refused", "chunk": chunk_id, "by": author,
                 "question": question, "why": event["refused_retraction"]})
    note = (update_line or "").strip()
    kind, why = "", ""
    if note and note.upper() != "NONE":
        existing = [_entry_text(entry) for entry in session["chunks"].get(chunk_id, [])
                    if not entry.get("retracted")]
        kind, why = gate_write(note, reports, options, existing)
        session.setdefault("events", []).append(
            {"kind": "write_gate", "chunk": chunk_id, "author": author,
             "question": question, "question_number": question_number,
             "verdict": kind, "why": why, "text": note})
        event["write"] = kind
        event["write_why"] = why
    keep = note if kind in ("observation", "method") else ""
    stored_frames = None
    if images:
        store = session.get("store") or {}
        if store.get("mode") == "files":
            from ..memory import global_store as _tree_store
            stored_frames = _tree_store.save_frames(
                store["root"], store.get("video") or session.get("video"), chunk_id,
                read_window or (0.0, 0.0), images)
    if description or keep or raw_reply:
        spans = place_spans(place, read_window[0], read_window[1]) if read_window else []
        seconds = ([min(span[0] for span in spans), max(span[1] for span in spans)]
                   if spans else (list(read_window) if read_window else None))
        # **The address is the window the read was taken at**, and the sighting is what the reader
        # named inside it. When they are equal the read was a whole-chunk look and the leaf is a hull;
        # when the sighting is narrower, this is an event - which is the share `locate_licence.py`
        # exists to count.
        entry = remember_chunk(session, chunk_id, question, description, keep,
                               citable=bool(description), author=author,
                               seconds=seconds,
                               assigned=(list(read_window) if read_window else None),
                               question_number=question_number,
                               provisional=provisional,
                               answer=answer,
                               frames=stored_frames,
                               claim=claim,
                               kind="observation" if description else "method",
                               raw_reply=raw_reply, quote=quote)
        event["entry_id"] = entry["id"] if entry else None
    # **The frames travel with the event** (one read, one store): the caller's sighting loop writes
    # further leaves from the same frames, and they must point at the same directory rather than
    # storing it again.
    event["frames"] = stored_frames
    return event


def remember_qa(session, question, option, unit, quote, chunk, seconds,
                verified=False, support="", rejected=(), answer_side=None,
                support_second=None):
    """The team's QA ledger entry - written **once**, at the end of a question.

    The QA pair is what a later question can actually read: the question, the answer
    with the unit it came from, the chunk and seconds, whether O checked it, and the
    attempts that were looked at and dropped.
    """
    if session.get("memory_disabled"):
        return None
    entry = {"n": len(session.setdefault("qa", [])) + 1, "question": question,
             "option": option, "unit": unit, "quote": quote, "chunk": chunk,
             "seconds": list(seconds or []), "verified": bool(verified),
             "support": support, "rejected": list(rejected or ()),
             "support_second": support_second,
             "answer_side": (answer_side if answer_side is not None
                             else ((option or "") + " " + (quote or "")).strip())}
    session["qa"].append(entry)
    return entry


def settle_question(session, question_number=None, verified=False):
    """The question is over: its provisional writes join the team's memory."""
    settled = []
    for entries in (session.get("chunks") or {}).values():
        for entry in entries:
            if (entry.get("question_number") == question_number
                    and entry.get("provisional")):
                entry["provisional"] = False
                # A rejected, unsure, no-frame, malformed or answerless review is saved, but
                # the row becoming verified must not relabel it as answer evidence.
                supported = (entry.get("review_status") in (None, "", "unreviewed")
                             or (entry.get("review_status") == "accepted"
                                 and entry.get("answer_support") is True))
                if not supported:
                    settled.append(entry["id"])
                    continue
                entry["verified"] = bool(verified)
                entry["status"] = "verified" if verified else "unverified"
                tell_the_tree(
                    session, entry, status=entry["status"], by="settle",
                    why="question settled", fields={"verified": bool(verified)})
                settled.append(entry["id"])
    session.setdefault("events", []).append(
        {"kind": "settle", "question_number": question_number,
         "verified": bool(verified), "ids": settled})
    return settled


def mark_promoted_entries_verified(session, promotion):
    """Keep the live tree status coherent with the two-level boundary commit."""
    for event in (promotion or {}).get("events") or ():
        if event.get("status") != "promoted":
            continue
        entry = next((item for entries in (session.get("chunks") or {}).values()
                      for item in entries or ()
                      if str(item.get("id")) == str(event.get("entry_id"))), None)
        if entry is None:
            continue
        entry["status"] = "verified"
        entry["reviewed"] = True
        tell_the_tree(session, entry, status="verified", by="O",
                      why="promoted to video memory after reviewed claim",
                      fields={"global": True})


def remember_global(session, about):
    """The orchestrator finished its coarse read: the picture of the video."""
    if session.get("memory_disabled"):
        return session.get("topic") or ""
    if (about or "").strip():
        session["topic"] = about.strip()
    return session["topic"]


def _terms(text):
    stop = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "from",
            "and", "or", "is", "are", "was", "were", "what", "when", "where",
            "which", "who", "why", "how", "does", "did", "do", "this", "that"}
    # **Three letters, not four**: `hat`, `car`, `dog` are exactly the words a
    # question uses to point at a chunk, and dropping them made the memory silent.
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) >= 3 and w not in stop}


OPTION_MARK = re.compile(r"\(([A-H])\)\s*(.*?)(?=\s*\([A-H]\)|\Z)", re.S)


def options_of(question):
    """The option texts a question carries, in order.

    LVBench writes its options inside the question (`(A) ...` on their own lines), so
    they have to be pulled back out: an answer that merely **restates an option** is
    not evidence, and the check gate and the write gate both need the list to say so.
    """
    return [match.group(2).strip() for match in OPTION_MARK.finditer(question or "")]


def _normalise(text):
    """Lowercase, punctuation to spaces, runs collapsed - the echo comparison."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _quote_grounded(quote, sighting):
    """Whether a quote is exact or a paraphrase grounded in one sighting.

    An exact normalized substring is enough, even for a number or a short phrase.
    Otherwise the quote needs at least one content word, and every content word must
    occur in the sighting. This one rule keeps the typed record and the W/G/M commit
    gates from disagreeing about the same evidence.
    """
    quoted = _normalise(quote)
    said = _normalise(sighting)
    if not quoted:
        return False
    if quoted in said:
        return True
    terms = _terms(quote)
    return bool(terms) and terms <= _terms(sighting)


def is_option_echo(text, options):
    """Whether `text` is (or is contained in) one of the options.

    A reader's `QUOTE` that is an option said back verbatim is not a description of what
    the video showed, so it cannot ground a citation.
    """
    said = _normalise(text)
    if not said:
        return False
    return any(said in _normalise(option) for option in options or ())


def _stored_frame_count(frames):
    if isinstance(frames, dict):
        return int(frames.get("n") or 0)
    if isinstance(frames, (list, tuple)):
        return len(frames)
    try:
        return max(0, int(frames or 0))
    except (TypeError, ValueError):
        return 0


def current_question_number(session, explicit=None):
    """One current-question source for every memory view."""
    return int(explicit or session.get("_active_question_number")
               or (len(session.get("question_history") or ()) + 1))


def memory_layer(entry, question_number):
    """The accepted three-way layer: current local, earlier prior, promoted video."""
    written = entry.get("question_number")
    try:
        written = int(written) if written is not None else None
    except (TypeError, ValueError):
        written = None
    if written is not None and 1 <= written < int(question_number)             and entry.get("global") is True:
        return "video"
    if written is None or written < int(question_number):
        return "prior"
    return "local"


SCENE_PREDICATES = {"action", "time", "order", "existence", "location"}


def memory_kind(entry):
    """Classify one structured claim as scene or fact without a second store."""
    claim = entry.get("claim") if isinstance(entry, dict) else None
    key = claim.get("key") if isinstance(claim, dict) else None
    if not isinstance(key, list) or len(key) != 4:
        key = entry.get("claim_key") if isinstance(entry, dict) else None
    if not isinstance(key, list) or len(key) != 4:
        return ""
    return "scene" if str(key[2]) in SCENE_PREDICATES else "fact"


def supported_memory(entry):
    """Whether a saved note may support an answer, independent of question verdict."""
    return bool(
        entry.get("citable", True)
        and not entry.get("retracted")
        and entry.get("review_status") == "accepted"
        and entry.get("answer_support") is True
        and entry.get("citation_allowed") is True
        and _stored_frame_count(entry.get("frames")) > 0)


def _entry_for_id(session, entry_id):
    if entry_id is None:
        return None
    return next((entry for entries in (session.get("chunks") or {}).values()
                 for entry in entries or ()
                 if str(entry.get("id")) == str(entry_id)), None)


def _entry_review_backed(row, entry_id, option):
    """Latest terminal REVIEW for this stable entry and submitted answer."""
    wanted = _letter_of(option or "")
    if not wanted or entry_id is None:
        return False
    terminal = ("accept", "reject", "unsure", "no_frames")
    latest = None
    for review in row.get("reviews") or ():
        if str(review.get("entry_id")) != str(entry_id):
            continue
        if str(review.get("outcome") or "") not in terminal:
            continue
        latest = review
    return bool(latest and latest.get("accept") and latest.get("frames_from_store")
                and not latest.get("harness")
                and _letter_of(latest.get("option") or "") == wanted)


def memory_terminal(row, session):
    """Return `(memory_terminal, memory_positive)` for a finished row."""
    ended = str(row.get("ended_by") or "")
    answer = str(row.get("answer") or "").strip()
    if not answer:
        source = str(row.get("answer_source") or "")
        return ("abstained" if ended == "agent_abstain"
                or source.startswith("abstain") or source.endswith("abstain")
                else "incomplete"), False
    if ended != "agent_submit":
        return "uncited", False
    commit = row.get("commit") or {}
    entry_id = row.get("submitted_entry_id") or commit.get("entry_id")
    entry = _entry_for_id(session, entry_id)
    option = commit.get("option") or answer
    supported = bool(
        row.get("citation_tier") == "cited"
        and commit.get("unit") and commit.get("quote")
        and entry is not None and supported_memory(entry)
        and _entry_review_backed(row, entry_id, option))
    return ("supported", True) if supported else ("uncited", False)


def memory_kind_counts(session, row):
    """Count scene/fact memory at offered, reviewed, promoted and cited stages."""
    counts = {stage: {"scene": 0, "fact": 0}
              for stage in ("offered", "reviewed", "promoted", "cited")}

    def add(stage, entry_id):
        entry = _entry_for_id(session, entry_id)
        kind = memory_kind(entry) if entry is not None else ""
        if kind in ("scene", "fact"):
            counts[stage][kind] += 1
            return True
        return False

    offered = set()
    for turn in row.get("turns") or ():
        for item in turn.get("offered") or ():
            entry_id = item.get("entry_id")
            if entry_id is None or str(entry_id) in offered:
                continue
            if add("offered", entry_id):
                offered.add(str(entry_id))
    reviewed = set()
    for review in row.get("reviews") or ():
        entry_id = review.get("entry_id")
        if entry_id is None or str(entry_id) in reviewed:
            continue
        if add("reviewed", entry_id):
            reviewed.add(str(entry_id))
    promoted = set()
    for event in row.get("bifocal_promotions") or ():
        entry_id = event.get("entry_id")
        if event.get("status") != "promoted" or entry_id is None                 or str(entry_id) in promoted:
            continue
        if add("promoted", entry_id):
            promoted.add(str(entry_id))
    if row.get("memory_positive"):
        add("cited", row.get("submitted_entry_id")
            or (row.get("commit") or {}).get("entry_id"))
    return counts


def memory_group(entry, question_number):
    """Order shown memory by support first, then current local, then older leads."""
    layer = memory_layer(entry, question_number)
    if supported_memory(entry) and layer == "video":
        return 0
    if supported_memory(entry) and layer == "prior":
        return 1
    if layer == "local":
        return 2
    return 3


def matching_memories(session, question, limit=MAX_GLOBAL_MEMORIES,
                      question_number=None):
    """The chunk memories O is shown, with one local/prior/video order."""
    wanted = _terms(question)
    current = current_question_number(session, question_number)
    found = []
    if session.get("memory_disabled"):
        return found
    for chunk_id in sorted(session["chunks"]):
        for entry in session["chunks"][chunk_id]:
            if not entry.get("citable", True) or entry.get("retracted"):
                continue
            layer = memory_layer(entry, current)
            if session.get("memory_protocol") == "two_level":
                # Two-level O may open only reviewed, frame-backed video memory.
                if layer != "video" or not supported_memory(entry):
                    continue
            hits = len(wanted & _terms(_entry_text(entry)))
            key = "content"
            if wanted & _terms(entry.get("answer_side") or ""):
                hits += 2
                key = "answer-side"
            if hits:
                found.append((memory_group(entry, current), -hits, chunk_id,
                              int(entry.get("id") or 0), entry, key, layer))
    found.sort(key=lambda item: item[:4])
    out = []
    for index, (group, _neg_hits, chunk_id, _entry_id, entry, key, layer) in enumerate(
            found[:limit], start=1):
        out.append({"handle": "M%d" % index, "chunk": chunk_id,
                    "entry_id": entry.get("id"),
                    "question": entry["question"],
                    "description": entry["description"],
                    "update": entry["update"], "match_key": key,
                    "author": entry.get("author", ""),
                    "status": entry.get("status"),
                    "answer_support": bool(entry.get("answer_support", False)),
                    "citation_allowed": bool(entry.get("citation_allowed", False)),
                    "review_status": str(entry.get("review_status") or "unreviewed"),
                    "seconds": entry.get("seconds"),
                    "address": entry.get("address"),
                    "question_number": entry.get("question_number"),
                    "layer": layer, "memory_group": group,
                    "supported": supported_memory(entry),
                    "memory_kind": memory_kind(entry),
                    "global": bool(entry.get("global"))})
    return out


# --------------------------------------------------------------------------
# the four prompts

PLAN_PROMPT = """You are O, the orchestrator of a reading team. The video is {dur} \
seconds long and is cut into {k} chunks: {menu}
{coarse}
The question and its options: {question}
{memory}
THINK: <one line, at most 40 words>
ABOUT: <one or two sentences about the video; never NONE>
PICK: <up to {pick} chunk ids from the list, comma separated, or ALL>
DRAFT: <the option you would answer from this look, or NONE>
Reply with exactly these four lines and nothing after DRAFT. DRAFT is a draft and is \
never the answer. PICK must use ids from the list; ALL means every chunk will be read."""

COARSE_WITH_FRAMES = ("You are shown {n} frames spread over the whole video, each "
                      "labelled with the second it shows.")
COARSE_WITHOUT_FRAMES = ("You are shown no frames for this question - the team has "
                         "read parts of this video before, and what it knows is below.")

DECIDE_PROMPT = """You are O. The question and its options: {question}
What the team already knows about this video:
{memory}
Chunks: {menu}
THINK: <one line>
Reply with exactly one of these:
{actions}
{action_note}"""

DECIDE_ACTIONS = {
    "push": ("  ANSWER: <option> | M<k> | \"<words from memory entry M<k>>\"\n"
             "  COARSE_PLAN\n"
             "  PLAN"),
    "index": ("  COARSE_PLAN\n"
              "  PLAN\n"
              "  NO_USABLE: name what the index showed and why none of it is enough"),
}
DECIDE_ACTION_NOTE = {
    "push": ("An ANSWER must name the memory entry it uses, M<k> from the list above, and "
             "quote\nthat entry. It is checked against the entry it names, and it costs "
             "one read: the\nchunk that entry is about is read again before the answer "
             "counts."),
    "index": ("No record's text is shown here, so a quote cannot be given from this call: "
              "choose the\nread you want, or say nothing usable was shown."),
}


def rank_read_candidates(session, question, chunks, question_number=None,
                         read_now=(), protected=()):
    """Rank current-question READ targets without proposing a watched chunk first."""
    current = current_question_number(session, question_number)
    current_reads = {int(chunk) for chunk in read_now or ()}
    prior_reads = {int(row["chunk"]) for row in _read_context(session, current)}
    protected = {int(chunk) for chunk in protected or ()}
    legal = [(int(cid), float(first), float(last)) for cid, first, last in chunks
             if int(cid) not in current_reads and int(cid) not in protected]
    by_id = {cid: (cid, first, last) for cid, first, last in legal}
    clocked = [cid for cid in sorted(clock_chunks(question, legal)) if cid in by_id]
    ranked, used = [], set()

    def add(cid, reason):
        if cid in by_id and cid not in used:
            ranked.append({"chunk": cid, "span": [by_id[cid][1], by_id[cid][2]],
                           "reason": reason})
            used.add(cid)

    for cid in clocked:
        add(cid, "question time")
    anchor = current_reads
    fresh = [cid for cid in by_id if cid not in prior_reads and cid not in used]
    if anchor:
        for cid in sorted(fresh, key=lambda value: (
                min(abs(value - seen) for seen in anchor), value)):
            add(cid, "near current read")
    for cid in sorted(fresh):
        add(cid, "uncovered")
    for cid in sorted(by_id):
        add(cid, "rewatch")
    return ranked


def open_actions(session, question, chunks, question_reports=(),
                 read_now=(), reviewed=(), surveyed=False, survey_available=None,
                 turns_left=1, rendered=None, question_number=None):
    """**What this state actually offers**.

    Keys are the verbs; a verb whose value is a non-empty list is offered **with those units** (the
    chunks `READ` may take, the units `REVIEW` may open). `survey`, `submit` and `abstain` are offered
    without units, so their value is `[]`.

    `read` remains the full legal set for the gate. `read_ranked` is the shorter display order
    after rendered supported-memory chunks are removed. `submit` and `abstain` stay offered:
    a turn-one uncited answer is legal."""
    current = current_question_number(session, question_number)
    shown = [item for item in shown_memories(
                 session, question, question_number=current)
             if item.get("handle") and str(item["handle"]).upper().startswith("M")
             and (rendered is None or item["handle"] in rendered)]
    supported = [item["handle"] for item in shown
                 if item.get("supported")
                 and item.get("layer") in ("prior", "video")]
    # Current reader findings remain reviewable. Supported earlier memory leads them.
    reviews = supported + ["W%d" % report["chunk"] for report in (question_reports or ())]
    reviews += [item["handle"] for item in reader_global_named_notes(question_reports)
                if item.get("handle")]
    reviews = [unit for unit in dict.fromkeys(reviews)
               if unit not in set(reviewed or ())]
    doors = [item["handle"] for item in shown
             if item["handle"] not in reviews
             and item["handle"] not in set(reviewed or ())]
    protected = {item["chunk"] for item in shown if item.get("supported")}
    ranked = rank_read_candidates(
        session, question, chunks, question_number=current,
        read_now=read_now, protected=protected)
    unread = [cid for cid, _first, _last in chunks if cid not in set(read_now or ())]
    clock = sorted(cid for cid in clock_chunks(question, chunks) if cid in set(unread))
    if clock:
        unread = list(dict.fromkeys(clock + unread))
    state = {}
    show_survey = (not surveyed) if survey_available is None else bool(survey_available)
    if show_survey:
        state["survey"] = []
    if unread and int(turns_left) > 0:
        state["read"] = unread
        state["read_ranked"] = [row["chunk"] for row in ranked]
        state["read_reasons"] = {str(row["chunk"]): row["reason"] for row in ranked}
    if (reviews or doors) and int(turns_left) > 0:
        state["review"] = reviews + doors
    if (doors or supported) and int(turns_left) > 0:
        state["search"] = []
    state["submit"] = []
    state["abstain"] = []
    return state


def decide_menu(view):
    """The consult's action list and its note, for the view in use."""
    view = "index" if view == "index" else "push"
    return DECIDE_ACTIONS[view], DECIDE_ACTION_NOTE[view]

READER_PROMPT = """You are reader-{cid}. Your chunk is seconds {first} to {last}; you \
are shown {n} frames of it, one frame every {every} second(s). Report only what is \
inside your chunk.
The question and its options: {question}
Your memory of this chunk from earlier questions:
{memory}
{global_memory}Reply with the fields below, **in this order, and nothing else** - the fields are what
is read, so the core report comes first:
FRAMES: <what you see in those seconds, **in words**, **at most {frames_words} words** -
    on-screen text, people, what they do - not only the timestamps. Keep it short: the
    fields below it are what the record is built from, and a long description pushes them
    out of your reply>
PLACE: <the sighting's own seconds - the start and end of the moment you are describing.
    A sighting that fills your whole chunk is written as the whole chunk. Write them the way
    your frames are labelled (for example 150s-155s), or as an offset from the start of your
    chunk when the range is shorter than your chunk (for example 10-20 in a 64-second chunk)
    - or NONE>
CAN_ANSWER: YES | NO | POINTS
ANSWER: <the option letter your own sighting supports - A, B, C or D - or NONE>
KIND: <one word from the declared vocabulary: who, what, where, attribute, action,
    participant, time - the kind of evidence you saw, or NONE>
QUOTE: <three to eight words copied from your FRAMES line - not a sentence - or NONE>
CLAIM_SUBJECT: <the person, object or scene fact your sighting is about, or NONE>
CLAIM_PREDICATE: <one of count, identity, attribute, location, time, action, order, existence, other - or NONE>
CLAIM_VALUE: <the short value you saw, without explanation, or NONE>
CLAIM_UNIT: <count, second, minute, meter, percent or rank when the value is numeric - otherwise NONE>
MEMORY_USED: <what your memory told you and whether it still holds, or NONE>
MEMORY_UPDATE: <one line worth keeping about this chunk, or NONE>
SIGHTINGS: <**after** the core report above, up to 3 **distinct** moments that bear on the question,
one per line as: PLACE: <t0>-<t1> | ANSWER: <letter> | QUOTE: "<3-8 words>" - then **STOP**. A
distinct moment is a different place, not the same line again; each names the option it supports.
If you saw one such moment, one line; if none, NONE>
THINK: <one short line only, and only **after** the fields above>"""

READER_REREASK_CLAIMED = """
Your reply said you can answer, and it named no evidence that carries words you copied. \
So say where the evidence is, in seconds inside your chunk:
  PLACE: <start>-<end> the sighting's own seconds, written the way your frames are labelled \
(for example 150s-155s) - the whole chunk only if the sighting fills it
  QUOTE: three to eight words, copied from the words you put on your FRAMES line
  CLAIM_SUBJECT: the person, object or scene fact your sighting is about
  CLAIM_PREDICATE: one of count, identity, attribute, location, time, action, order, existence, other
  CLAIM_VALUE: the short value you saw, without explanation
  CLAIM_UNIT: count, second, minute, meter, percent or rank when numeric, otherwise NONE
If the evidence is not in this window after all, withdraw the claim: CAN_ANSWER: NO.
Write only those correction lines. Do not write FRAMES again, do not repeat a sentence, \
and stop after the last correction line."""

READER_REREASK_EMPTY = """
Your reply carried no PLACE and no CAN_ANSWER line, so it cannot be read as a report. \
Reply with the same lines again and fill both in this time.
  PLACE: <start>-<end> the sighting's own seconds, written the way your frames are labelled \
(for example 150s-155s) - the whole chunk only if the sighting fills it - or NONE if there is \
no evidence here
  CAN_ANSWER: YES | NO | POINTS
  QUOTE: three to eight words copied from your own FRAMES line, if you have evidence
  CLAIM_SUBJECT: the person, object or scene fact your sighting is about, or NONE
  CLAIM_PREDICATE: count, identity, attribute, location, time, action, order, existence or other, or NONE
  CLAIM_VALUE: the short value you saw, without explanation, or NONE
  CLAIM_UNIT: count, second, minute, meter, percent or rank when numeric, otherwise NONE
NONE is a fine answer here. Saying nothing is not.
Write only those correction lines. Do not repeat FRAMES, do not repeat a sentence, and \
stop after the last correction line."""

READER_REREASK_HULL = """
Your PLACE is the whole window you were given, so your report says the evidence is **somewhere in
those seconds** and not where. Name the moment inside it - the start and end seconds of the \
*specific* thing that bears on the question, written the way your frames are labelled (for example \
150s-155s) - and the words you copied from it:
  PLACE: <start>-<end> the moment **inside your window** - not the whole window
  QUOTE: three to eight words, copied from the words you put on your FRAMES line
If what bears on the question really does fill the whole window, keep your place and say so in \
your THINK line; a sighting that fills the window is an answer, not a failure. But if you named \
the window only because you did not look for the question, look again now and name the moment.
Write only those correction lines. Do not repeat FRAMES, do not repeat a sentence, and stop \
after the last correction line."""


def names_only_the_window(record, window):
    """Whether the place the reader named **is the window it was given**, not a moment inside it."""
    if not window:
        return False
    spans = place_spans(record.get("place") or "", window[0], window[1])
    if not spans:
        return False
    return (abs(min(span[0] for span in spans) - float(window[0])) < 1.0
            and abs(max(span[1] for span in spans) - float(window[1])) < 1.0)


def reader_rereask(record, window=None):
    """Which sharper ask a read gets, from what the first reply actually said."""
    if (record.get("claimed") or record.get("points")) and names_only_the_window(record, window):
        return READER_REREASK_HULL
    if record.get("claimed") or record.get("points"):
        return READER_REREASK_CLAIMED
    return READER_REREASK_EMPTY


def needs_a_place(record):
    """Whether the reply is unusable **for want of a field**, not for want of evidence.

    A reply that carried neither a `PLACE` nor a `CAN_ANSWER` line, or that claimed (`YES`)
    while naming nowhere, is re-asked. A reply that declared its own negative
    (`PLACE: NONE, CAN_ANSWER: NO`) is never re-asked - it answered.
    """
    if record.get("claimed") or record.get("points"):
        return not record.get("place")               # a claim must say where
    return not record.get("place_field") and not record.get("can_answer_field")


def needs_a_second_ask(record, why, family="", window=None):
    """Whether **the record's refusal is one the reader can repair when told**."""
    kind = refusal_kind(why)
    if (record.get("claimed") or record.get("points")) and names_only_the_window(record, window):
        return True
    if kind == "family":
        return False              # the question is out of the map's scope: nothing helps
    if kind == "no_reply_fields":
        return True
    if kind == "kind":
        # **Only a claim needs a kind**: a reader that says it can answer and names no kind
        # from the vocabulary has something to correct, while a reply that claims nothing
        # and writes `KIND: NONE` is an honest negative with a place.
        return bool(record.get("claimed") or record.get("points"))
    if kind == "place_outside":
        return True
    return needs_a_place(record)


def carries_a_field(record):
    """Whether a re-ask earned its call: it named a place, or at least declared a verdict.

    A second chance that answers nothing does not overwrite the first reply.
    """
    return bool(record.get("place") or record.get("place_none")
                or record.get("can_answer_field"))


def place_read_as(report):
    """Which form a reader's `PLACE` was read as: `none`, `absolute`, `clock` or `offset`."""
    text = report.get("place") or ""
    if not text or report.get("place_none"):
        return "none"
    if PLACE_SECONDS_RANGE.search(text) or PLACE_SECONDS_ONE.search(text):
        return "absolute"
    if PLACE_CLOCK_RANGE.search(text) or PLACE_CLOCK_ONE.search(text):
        return "clock"
    if PLACE_OFFSET_RANGE.search(text):
        return "offset"
    return "none"


def merge_retry(first, second):
    """**The report a re-ask leaves: the retry's fields, and the first reply's sighting.**

        `PLACE: 150-155 | QUOTE: RAYMOND & JAYSON (COUSINS) | CAN_ANSWER: YES | ...`

    and **no `FRAMES` line at all** - and taking that reply alone replaced a sighting with an
    empty one, so every check afterwards refused (*"the phrase is not in chunk W3's
    description of what it saw"*) and three of the five questions abstained. Both replies
    describe the same frames: the re-ask is the same prompt with the same images, so the
    sighting stands and only the fields the retry actually carries are replaced. The
    derived fields are recomputed from the merged text rather than copied, so a retry that
    declares `CAN_ANSWER: NO` still withdraws the claim (and is counted as a flip), and a
    merge cannot invent a phrase that the sighting does not contain."""
    out = dict(first)
    for key, value in (second or {}).items():
        if value:
            out[key] = value
    if "claim" in (second or {}):
        out["claim"] = copy.deepcopy(second.get("claim"))
        out["claim_error"] = str(second.get("claim_error") or "")
    verdict = str(out.get("can_answer_field") or "").upper()
    out["claimed"] = verdict.startswith("Y") and bool(out.get("answer"))
    out["points"] = verdict.startswith("P") and bool(out.get("answer"))
    # **A withdrawn claim cannot keep its old answer quote.** The retry may
    # replace FRAMES while omitting QUOTE or writing NONE; retaining the first
    # quote then couples words to a different sighting and creates a false
    # phrase refusal. Once the final verdict is non-claiming, the answer quote
    # belongs to the withheld claim and is cleared.
    if not out["claimed"] and not out["points"]:
        out["answer"] = ""
        out["quote"] = ""
    place_field = out.get("place_field")
    out["place_none"] = bool(place_field) and str(place_field).upper().startswith(
        ("NONE", "N/A"))
    out["place"] = "" if out["place_none"] else (place_field or "")
    return out

COMMIT_PROMPT = """You are O. What the team already knows about this video:
{memory}
The readers' reports:
{ledger}
The question and its options: {question}
THINK: <one line>
Name the unit you cite as W<k> for a reader (for example W3 - the reader's number,
not "chunk 3"), S for your own stored look, or M<k> for a memory entry.
The quote must be **one short phrase of three to eight words, copied exactly as
that unit wrote it** - do not join separate phrases into one quote, because the
check is that the words are found in the unit's own report.
FINAL W<k> | <option> | "<quote from that unit>\""""

CHECK_PROMPT = """You are O. **Before any answer is given**, decide whether the \
evidence a reader reported can support an answer to the question.
The question and its options: {question}
What the team already knows about this video:
{memory}
Reader {unit}'s report about its own chunk, in its own words:
{ledger}
You are also shown the frames of the place that reader named. Look at them.
Reply with exactly one of these three lines, **with a bar between the fields**. Write
your own words in each field - **never copy this description back**:
  CHECK: SUPPORT | {unit} | the option's letter - one capital letter | the second the
  evidence is at, as a number | the reader's own three to eight words, in quotation
  marks
  CHECK: REFUTE | one line on what the evidence shows instead
  CHECK: UNSURE | one line on what you cannot tell
The quoted words must be the reader's own description of what it saw, three to eight
words, copied exactly - **not the option said back**, and not words from the reader's
QUOTE line unless the reader also described them as seen."""

REVIEW_PROMPT = """You are O, reviewing a reader's report before an answer is given.

A reader read seconds {first} to {last} of the video. The reader saw the raw video; you see the
stored sample below only to confirm that a readable frame record exists. Do not say the reader
was wrong merely because your sample looks different. Review the report's internal support:
  CAN_ANSWER: {can}
  ANSWER: {answer}
  KIND: {kind}
  QUOTE: "{quote}"
  IT SAID: "{words}"
And this is what the team already knows about this video:
{memory}

The question and its options: {question}

Check whether the reader's answer is supported by its own quote and whether the description and
claim are internally consistent. A missing frame record is a validity failure, not evidence that
the observation is false. A report that does not answer can still be saved as context, but it
must not support an answer.

SUPPORT: YES | <the reader's answer letter, if it gave one> | "<the reader's own evidence quote>"
SUPPORT: NO  | <the failed check: fields, answer/quote, claim/description, span, or readable frames>
SUPPORT: UNSURE | <why the report cannot be judged internally>
WHAT: <one line about the report's support, not a new visual claim>
WHY: <one line explaining the support verdict>

Stop after the WHY line. Keep WHAT and WHY together under 60 words. Do not list
the other answer options, do not repeat a sentence, and do not write a second
review. A valid decision needs only the SUPPORT line and one short WHAT line and
one short WHY line. A failed support check does not delete the reader's memory."""


def parse_review(text):
    """`{accept, option, quote, what, why}` from the review call, or `None` without a verdict.

    **Only an explicit verdict counts** (the same rule `parse_verify` follows): a reply that names
    neither YES nor NO has decided nothing, and a missing decision must never be read as an
    acceptance - which is the one thing this call exists to prevent.

    **A support verdict must carry evidence from the reader's report.** For the current
    `SUPPORT:` protocol, `_review_round` binds the option to the reader's answer and requires the
    quoted words to appear in what the reader wrote. The legacy `ACCEPT:` parser remains readable
    for old records and tests, but new code does not use O's visual option as truth.
    """
    found = re.search(
        r"(?mi)^[\s\-\*\#]*\**\s*(?:SUPPORT|ACCEPT)\s*\**\s*:\s*[ \t]*([^\n]*)",
        text or "")
    if not found:
        return None
    rest = (found.group(1) or "").strip()
    word = rest.split()[0].upper().strip(".,;:*|") if rest.split() else ""
    if not word.startswith(("Y", "N", "U")):
        return None
    if word.startswith("U"):
        return {"accept": False, "support": False, "unsure": True, "option": None, "quote": "",
                "what": _field(text, "WHAT"), "why": _field(text, "WHY")}
    accept = word.startswith("Y")
    option = ""
    if accept:
        # the letter may be written `| B |`, `B`, `(B)` or `B) a red hat`
        letter = _letter_of(rest[len(word):].strip().lstrip("|").strip())
        option = (letter or "").upper()
    quotes = re.findall(r'"([^"]*)"', rest)
    legacy = not re.search(r"(?mi)^[\s\-\*\#]*\**\s*SUPPORT\s*", text or "")
    return {"accept": accept, "support": accept, "unsure": False,
            "legacy": legacy, "option": option or None,
            "quote": (quotes[0].strip() if quotes else ""),
            "what": _field(text, "WHAT"),
            "why": _field(text, "WHY")}


SELECT_PROMPT = """You are O. Nothing you have seen grounds an answer yet: {why}
The video is {dur} seconds long and is cut into {k} chunks: {menu}
The question and its options: {question}
What the team already knows about this video:
{memory}
What you may cite **once you have checked it** (name one of these in your submission):
{citable}
{looked}
What has already been tried for this question:
{history}
You have {rounds_left} selection round(s) left, {calls_left} calls and {frames_left}
frames in the budget.
Choose exactly one line - **you decide what to look at next**:
  READ <chunk ids>       (up to {pick} chunks you have not read, at 0.5 fps)
  LOOK M<k>              (that memory entry's own words; it costs one more round)
  RECALL M<k>            (that chunk's memory and a fresh read of it)
  ALL                    (every chunk at 1 fps - the expensive one)"""


SUBMIT_PROMPT = """You are O, and the team has finished reading. **You decide what to \
submit.**

The question and its options: {question}
What the team already knows about this video:
{memory}
{looked}
Your own draft, from your first look at the whole video:
DRAFT: {draft}
What the readers reported for this question:
{reports}
What the checks said about those reports:
{verdicts}
**What the checks grounded: {grounded}**

Reply with exactly one line and nothing else, **with a bar between the fields**. Write
your own words in each field - **never copy this description back**:
  DECIDE: COMMIT | the reader you name as W and its number | the option's letter -
  one capital letter | that reader's own three to eight words, in quotation marks
  DECIDE: DRAFT | the option's letter - one capital letter | one line on why your own
  read stands
  DECIDE: ABSTAIN | one line on what is missing
**COMMIT must name the reader named above as grounded** - another reader's report \
was refused, so citing it is not a commit - and its quoted words must be that \
reader's own description of what it saw, three to eight words, not the option said \
back. DRAFT must repeat your own draft option. If nothing is grounded and you have \
no draft, ABSTAIN. Decide on the evidence, not on which line looks safer."""

TURN_PROMPT = """You are O, and **you choose what happens next**.

The question and its options: {question}
The video is {dur} seconds long and is cut into {k} chunks: {menu}
What the team already knows about this video:
{memory}
{refine}
What you may cite **once you have checked it** (name one of these in your submission):
{citable}
{must_check}
What you have already looked up this question:
{looked}
Your draft from your own survey of the video: {draft}
**That draft is a hypothesis, not your answer** - it comes from 64 frames spread over the whole video, so it cannot show what was said, what a sign read, or a detail inside a minute. A reader's report or a memory entry that disagrees with it is the better evidence: submit the answer the evidence supports, even when that means changing the draft.
**When you may answer.** The team's belief above is built from what your readers have read. If it \
has **no answer yet** for this question, **READ a window before you submit** - an answer with no \
belief behind it is a guess, and the record marks it as one. If it **has** an answer, you may SUBMIT \
from it; reading on is still allowed, and a REVIEW of the frames behind it is what earns the \
higher tier. Read where a reader can **state an option** - aim at the window the question is \
about, not the whole chunk.
What you have done so far this question:
{history}
You have {turns_left} turn(s), {calls_left} calls and {frames_left} frames left.

**What each action can and cannot see.** Your survey is {n_survey} frames over the whole \
{dur} seconds - about one frame every {survey_every} seconds, so it shows the shape of the \
video and almost no detail. A reader's chunk read is 32 frames over 64 seconds - one frame \
every 2 seconds - and a reader reports in words what is inside them. So a question about \
what was said, what a sign read, who was there or what changed cannot be answered from the \
survey, however often you look: READ is how you see it. A claim is only usable once **you have reviewed it** \
(`REVIEW W<k>` opens the frames the reader saw), and the map can be searched with SEARCH. \
**Choosing to survey again when nothing has changed shows you the same frames** - do it only \
after you have learned something.

Choose **exactly one** action. Write one line first saying why - `WHY: <at most 20 words>` - \
then the action line. Nothing is done for you: if you never survey, you never see the video; \
if you never read, no reader reports; if you never review, nothing you answer rests on evidence.

{open_actions}

**What each verb means.** The listing above is what is open *now*. The READ line shows at most \
{reads} ranked unread chunks; other unread chunks remain legal, and they must be named in groups of \
at most {reads}. The REVIEW line names every unit it offers. A unit you have already used this \
question is spent; the menu will not offer it and a line that names one anyway is refused.
  SURVEY                your own look at the whole video, 64 frames - the only way you see it
  READ <chunk>, <chunk> readers read up to {reads} chunks at 0.5 fps and report back
  ZOOM <chunk> <t0>-<t1> read the SECONDS YOU NAME in that chunk at a dense rate - this is how
                        a vague entry becomes a precise one, and the window you name is what a
                        reader reads. Name no window and the question's own seconds are read; if
                        the question names none, the read is refused (`ZOOM M<k> <t0>-<t1>` is
                        still accepted and resolves the handle to its chunk)
  SEARCH <your words>   the map of memory, rendered around words you choose
  REVIEW W<k>           **your own look at a reader's memory**: the frames it saw, out of the
                        store, and the words it wrote. Then accept or reject it - a claim you
                        accept is what your answer rests on
  SUBMIT <letter> @ S | "quote"      your answer, at your coarse whole-video look (your survey)
  SUBMIT <letter> @ W<k> | "quote"   your answer, at a chunk you read, with words from it
                                     **after a REVIEW accepted it** - that is what makes it grounded
  SUBMIT <letter> @ E<k> | "quote"   your answer, at an earlier question's note (listed below)
  SUBMIT <letter> @ M<k> | "quote"   your answer, at a stored entry the map showed, its own words
  SUBMIT <letter> @ T1 | "quote"    your answer, at the team's belief for this question (quote one of
                                     its reasons, not the whole line)
  SUBMIT <letter> @ W<k>             names a place only; without REVIEW it is recorded uncited
  SUBMIT <letter>       your answer with no support named - legal, and recorded as unsupported
  ABSTAIN <one line>    no answer

**An address must be a place this run offered**: `W<k>` a chunk read this question, `E<k>` a leaf
the evidence block below lists, `M<k>` a live entry the map above showed, `S` your own survey.
`M<k>` is cited exactly as `W<k>` is - a memory entry you were shown, with its own words.
Anything else is refused once, and the answer then stands uncited.
**A bare `W<k>` address is legal but does not support memory.** To make it support, first send \
`REVIEW W<k>`; if that review accepts the option, submit it at `W<k>` with the review's words.
Reply with the WHY line first, then exactly one action line, and nothing else. SHORT lines only."""

SURVEY_PROMPT = """You are O. This is your own look at the whole video - {n} frames spread \
across its {dur} seconds - and nobody else has seen it.

The question and its options: {question}
What the team already knows about this video:
{memory}
Look at the frames, then answer in exactly three lines:
THINK: <one line, at most 40 words>
ABOUT: <one or two sentences about what this video is; never NONE>
DRAFT: <the option you would answer from this look, or NONE>"""

COMMIT_MENU = """You have {turns_left} turn(s) left. A read is not open any more: the harness no \
longer returns a reader's report for a chunk this question has not already read. **What is open:**
  SUBMIT <letter> @ S | "quote"       your answer, at your coarse whole-video look (your survey)
  SUBMIT <letter> @ W<k> | "quote"    your answer, at a chunk read this question, with its words
  SUBMIT <letter> @ E<k> | "quote"    your answer, at a leaf of the memory the block below lists
  SUBMIT <letter> @ M<k> | "quote"    your answer, at a stored entry the map showed, its own words
  SUBMIT <letter> @ T1 | "quote"     your answer, at the team's belief for this question (quote one of
                                      its reasons, not the whole line)
  SUBMIT <letter> @ W<k>              names a place only; without REVIEW it is recorded uncited
  SUBMIT <letter>                     your answer with no support named - legal, and recorded
  ABSTAIN <one line>                  no answer
  REVIEW W<k>           open the memory the reader left - **its own frames, out of the store, and
                        its words** - and accept or reject it. A claim you accept is what your
                        answer may rest on.
{extra}
**An address must be a place this run offered** - `W<k>` a chunk read this question, `E<k>` a leaf \
the block below lists, `M<k>` a live entry the map above showed, `S` your own survey; anything else \
is refused once, and the answer then stands uncited. **`M<k>` is cited exactly as `W<k>` is** - the \
words in it, or a `REVIEW` of the footage behind it.

**A `W<k>` can only back an answer if its reader claimed one, and the listing marks each one.** A \
`REVIEW` of a reader that wrote `CAN_ANSWER: NO` grounds nothing, however good its description is. \
**A review that accepts names the option its frames support: submit that option, with the review's own \
words.** A bare `W<k>` address is legal but is recorded uncited; first `REVIEW W<k>` to make it \
answer support."""

COMMIT_EXTRA = """  READ <chunk>, <chunk> readers read up to {reads} chunks at 0.5 fps and report back
  ZOOM <chunk> <t0>-<t1> read the SECONDS YOU NAME in that chunk at a dense rate - this is how a vague
                        entry becomes a precise one: the window you name is what a reader
                        reads; the leaf it writes is keyed by **the reader's own PLACE inside
                        that window**, so the leaf can be narrower than the window you sent. Name none and the
                        entry's own window is read (usually the whole 64s chunk, which is
                        rarely what the question asks about)
**Your last submission was refused: it had no check behind it. The reading actions above are open \
now, and a unit with no check behind it is refused again.**"""

TURN_PROMPT_COMMIT = """You are O, and **this row is in its commit stage**.

The question and its options: {question}
What the team already knows about this video:
{memory}
{refine}
What you may cite **once you have checked it** (name one of these in your submission):
{citable}
{must_check}
Your draft from your own survey of the video: {draft}
**That draft is a hypothesis, not your answer** - it comes from 64 frames spread over the whole video, so it cannot show what was said, what a sign read, or a detail inside a minute. A reader's report or a memory entry that disagrees with it is the better evidence: submit the answer the evidence supports, even when that means changing the draft.
What you have done so far this question:
{history}
{commit_menu}

**An uncited answer is a legal answer** - it keeps the tier it earns, and the run reports what
share of answers were cited - so abstain only when you would rather say nothing than answer.
Write one line first saying why - `WHY: <at most 20 words>` - then the action line, and nothing
else. SHORT lines only."""

# One label covers both deployment prompts and changes automatically when either prompt changes.
PROMPT_SCHEMA = ("bifocal-turn-reason-then-action-v2:"
                 + hashlib.sha256((TURN_PROMPT + "\n---\n" + TURN_PROMPT_COMMIT).encode()).hexdigest()[:16])

SCAN_PROMPT = """You are O. A **rough look** at a stretch of the video: {n} frames from {k} \
chunks, {per} frames per chunk, each labelled with the second it shows. This is navigation, \
not evidence - it can tell you roughly where something happens, never what was said. A READ \
is still what shows you that.

The question and its options: {question}
What the team already knows about this video:
{memory}
The chunks you asked to scan: {menu}
Reply with exactly these lines and nothing else:
THINK: <one line, at most 40 words>
SEES: <one line per scanned chunk - the chunk id and, in at most eight words, what it roughly \
shows>
READ NEXT: <up to {reads} chunk ids this look makes you want to read, or NONE>"""


VERIFY_PROMPT = """You are O. A note is stored in the team's memory. It is a claim until \
you look at the footage behind it, and that is what this call is for.

The question and its options: {question}
The note, in the words written when it was stored ({unit}, seconds {first} to {last}):
"{claim}"
You are shown the frames of those seconds. Look at them, and answer **both** halves:
  1. does the note still hold in these frames?
  2. which option of the question above do these frames support, if any?
**Reply with exactly one line**:
  VERIFY: <VERDICT> | <LETTER> | <your own words>
where `<VERDICT>` is `HOLDS`, `FAILS` or `UNSURE`; `<LETTER>` is the option letter those \
frames support (`-` when they support none of them); and `<your own words>` is **three to eight \
words you can see in the frames** for HOLDS, one line on what is there instead for FAILS, or one \
line on what you cannot tell for UNSURE.
For example: `VERIFY: HOLDS | B | the man in green crouches at the net`
**Write one line and stop. Do not write the other verdicts** - a reply with more than one VERIFY \
line names no decision, and this call is spent on nothing.
**FAILS means the frames say something else, and only that** - a window that simply shows \
too little to tell is UNSURE, never FAILS. Never copy this note back."""


def parse_verify(text):
    """`{verdict, option, words, why}` from a `VERIFY` reply, or `None`."""
    lines = re.findall(r"(?im)^\s*(?:[-*#]+[ \t]*)?(?:\*\*|__)?VERIFY(?:\*\*|__)?[ \t]*:[ \t]*(.*)$",
                       text or "")
    if not lines:
        return None
    parsed = []
    for line in lines:
        part = _verify_line(line)
        if part:
            parsed.append(part)
    if not parsed:
        return None
    if len(set(item["verdict"] for item in parsed)) > 1:
        return {"verdict": "ambiguous", "option": "", "words": "",
                "why": "the reply named more than one verdict (%s)"
                       % ", ".join(item["verdict"] for item in parsed)}
    return parsed[0]


def _verify_line(line):
    """One `VERIFY:` line's `{verdict, option, words, why}`, or `None`."""
    fields = line.split("|")
    head = fields[0].strip().upper()
    rest = fields[1:]
    option = ""
    if len(rest) >= 2:
        option = _letter_of(rest[0].strip())
        tail = "|".join(rest[1:])
    else:
        tail = rest[0] if rest else ""
    if head.startswith("HOLD"):
        first, last = tail.find('"'), tail.rfind('"')
        words = tail[first + 1:last] if 0 <= first < last else tail.strip()
        return {"verdict": "holds", "option": option, "words": words.strip(), "why": ""}
    if head.startswith("FAIL"):
        return {"verdict": "fails", "option": option, "words": "", "why": tail.strip()}
    if head.startswith("UNSURE"):
        return {"verdict": "unsure", "option": option, "words": "", "why": tail.strip()}
    return None


TURN_LINE = re.compile(
    r"^\s*(SURVEY|SCAN|READ|ZOOM|SEARCH|REVIEW|SUBMIT|ABSTAIN)\b\s*:?(.*)$",
    re.I | re.M)
TURN_ACTIONS = ("survey", "scan", "read", "zoom", "search", "review", "submit", "abstain")


def action_line_count(text):
    """How many lines in a policy reply look like an action.

    This is a measurement only. `parse_turn` still acts on the first action line, exactly as before.
    """
    return len(TURN_LINE.findall(text or ""))


def parse_turn(text):
    """`{action, ids, handle, unit, option, quote, terms, why}` from one turn's reply."""
    found = TURN_LINE.search(text or "")
    if not found:
        return {"action": "unparsed", "ids": [], "handle": "", "unit": "", "option": "",
                "quote": "", "terms": "", "why": "the reply named no action"}
    action = found.group(1).upper()
    rest = found.group(2).strip().lstrip("|").strip()
    _override = re.search(r"(?im)\bOVERRIDE\s*:\s*(.+)$", text or "")
    base = {"ids": [], "handle": "", "unit": "", "option": "", "quote": "", "terms": "",
            "why": _field(text, "WHY") or "",
            "override": (_override.group(1).strip() if _override else "")}
    if action == "SURVEY":
        return dict(base, action="survey")
    if action == "READ":
        return dict(base, action="read",
                    ids=list(dict.fromkeys(int(n) for n in re.findall(r"\d+", rest))))
    if action == "SCAN":
        # **A scan is batched and bounded by `SCAN_CHUNKS`**, and it names ids: a scan of
        # nothing is not an action, so it is refused as unparsed rather than guessed at.
        ids = [int(n) for n in re.findall(r"\d+", rest)][:SCAN_CHUNKS]
        if not ids:
            return dict(base, action="unparsed", why="scan named no chunk")
        return dict(base, action="scan", ids=ids)
    if action == "ZOOM":
        span = re.search(r"(\d+(?:\.\d+)?)\s*s?\s*[-\u2013:]\s*(\d+(?:\.\d+)?)\s*s?", rest)
        window, head = None, rest
        if span and float(span.group(2)) > float(span.group(1)):
            window = [float(span.group(1)), float(span.group(2))]
            head = rest.replace(span.group(0), " ", 1)
        handle = re.search(r"\bM\s*(\d+)\b", head, re.I)
        if handle:
            out = dict(base, action="zoom", handle="M%s" % handle.group(1))
        else:
            chunk = re.search(r"\b(\d+)\b", head)
            if not chunk:
                return dict(base, action="unparsed", why="zoom named no chunk or memory handle")
            out = dict(base, action="zoom", chunk=int(chunk.group(1)))
        # **The window key is absent when none was named** (the pre-Phase-4 contract): an offer
        # that advertised a template window must be distinguishable from a `ZOOM` that named none.
        if window:
            out["window"] = window
        return out
    if action == "SEARCH":
        return dict(base, action="search", terms=rest)
    if action == "REVIEW":
        unit = re.search(r"\b(W|M|G)\s*(\d+)\b", rest, re.I)
        if not unit:
            return dict(base, action="unparsed", why="review named no reader unit")
        return dict(base, action="review",
                    unit="%s%s" % (unit.group(1).upper(), unit.group(2)))
    if action == "ABSTAIN":
        return dict(base, action="abstain", why=rest or "no reason given")
    if re.match(r"^\s*ABSTAIN\b", rest, re.I):
        return dict(base, action="abstain",
                    why=(rest[7:].strip(" |:") or "no reason given"))
    # SUBMIT: the letter first, then an optional unit and an optional quoted span.
    option = _letter_of(rest)
    if not option:
        return dict(base, action="unparsed", why="submit named no option letter")
    head = rest.split('"', 1)[0]
    unit = re.search(r"\b([WMETG])\s*(\d+)\b", head, re.I)
    if unit:
        unit_name = "%s%s" % (unit.group(1).upper(), unit.group(2))
    elif re.search(r"(?:@|\|)\s*S\b", head, re.I):
        unit_name = "S"
    else:
        unit_name = ""
    first, last = rest.find('"'), rest.rfind('"')
    quote = rest[first + 1:last] if 0 <= first < last else ""
    return dict(base, action="submit", option=option, unit=unit_name,
                quote=quote.strip())


CONFLICT_PROMPT = """You are O. Your own look at the whole video drafted {draft}; the \
check grounded {unit}'s report for {option}. The two disagree, and only one can be \
submitted.

The question and its options: {question}
What the team already knows about this video:
{memory}
Reader {unit}'s report about its own chunk, in its own words:
{ledger}
You are shown the frames of the place that reader named. Look at them, and say \
whether they really support {option}.

Reply with exactly one of these two lines, **with a bar between the fields**, writing
your own words in each field - **never copy this description back**:
  CONFLICT: KEEP | {option} | the second the evidence is at, as a number | the
  reader's own three to eight words, in quotation marks
  CONFLICT: DROP | one line on what the frames show instead."""


def parse_decision(text):
    """`{action, unit, option, quote, reason}` from the decision call, or `None`.

    Strict on purpose: the decision is the only place an answer is chosen, so an
    unusable reply must be visible as one rather than half-parsed.
    """
    found = re.search(r"^\s*DECIDE:\s*\|?\s*(COMMIT|DRAFT|ABSTAIN)\b(.*)$",
                      text or "", re.I | re.M)
    if not found:
        return None
    action, rest = found.group(1).upper(), (found.group(2) or "").strip()
    if rest.startswith("|"):
        rest = rest[1:]
    rest = rest.strip()
    parts = [part.strip() for part in rest.split("|")]
    if action == "COMMIT":
        if len(parts) < 2:
            return None
        unit, unit_at, quotes_seen = "", None, 0
        for at, part in enumerate(parts):
            outside_quotes = (quotes_seen % 2 == 0)
            quotes_seen += part.count('"')
            if outside_quotes and re.fullmatch(r"W\d+", part.strip().upper()):
                unit, unit_at = part.strip().upper(), at
                break
        if unit_at is None:
            unit, unit_at = parts[0], 0
        remainder = "|".join(part for at, part in enumerate(parts)
                             if at != unit_at).strip()
        # **Any quoted run parses, however short**: a one-character quote is the gate's
        # refusal to make ("a citation must quote at least two words"), not the parser's -
        # otherwise the record loses the real reason, which is how the 729/126 fix read.
        quotes = re.findall(r'"([^"]*)"', remainder)
        quote = quotes[0].strip() if quotes else ""
        option = remainder.split('"')[0].strip().strip("|-").strip()
        if (not unit or unit.startswith("<") or not option
                or option.startswith("<") or not quote):
            return None
        return {"action": "commit", "unit": unit.upper(), "option": option,
                "quote": quote, "reason": ""}
    if action == "DRAFT":
        if len(parts) < 2 or not parts[0] or parts[0].startswith("<"):
            return None
        return {"action": "draft", "unit": "", "option": parts[0],
                "quote": "", "reason": "|".join(parts[1:]).strip()}
    return {"action": "abstain", "unit": "", "option": "", "quote": "",
            "reason": rest}


def parse_conflict(text):
    """`{keep, quote, why}` from a conflict re-look, or `None`."""
    found = re.search(r"^\s*CONFLICT:\s*(KEEP|DROP)\b(.*)$", text or "", re.I | re.M)
    if not found:
        return None
    kind, rest = found.group(1).upper(), (found.group(2) or "").strip()
    if rest.startswith("|"):
        rest = rest[1:]
    if kind == "DROP":
        return {"keep": False, "quote": "", "why": rest}
    parts = [part.strip() for part in rest.split("|")]
    if len(parts) < 3:
        return None
    return {"keep": True, "why": "", "quote": "|".join(parts[2:]).strip().strip('"')}


def _letter_of(text):
    """The option letter in a string, upper-cased, or `""`."""
    found = re.search(r"\b([A-Ha-h])\b", str(text or ""))
    return found.group(1).upper() if found else ""


def unit_of(text):
    """`W<k>` from `W32`, `w32`, `32` or `chunk 32`, or `""`.

    **The unit is an identifier, not a claim.** A reader is `W32` whatever the model
    writes around the number, so a missing prefix is a formatting difference and not
    a wrong citation - refusing a commit over it would be a trap, and the gate still
    decides whether the quoted words are that reader's.
    """
    found = re.search(r"[Ww]?\s*(\d+)", str(text or ""))
    return "W%s" % found.group(1) if found else ""


def default_submission(committed, draft, fault):
    """The deterministic rule, and only the default."""
    if committed:
        return {"answer": committed["option"], "answer_source": "default_on_unparse",
                "fault": fault, "why": "the default kept the grounded commit %s"
                % committed["unit"]}
    if draft:
        return {"answer": draft, "answer_source": "default_on_unparse", "fault": fault,
                "why": "the default fell back to O's own draft"}
    return {"answer": "", "answer_source": "abstain", "fault": fault,
            "why": "no grounded commit and no draft"}


def resolve_submission(text, draft, committed, reports=(), options=(),
                       memory=None, about="", commit_why=""):
    """`{answer, answer_source, fault, why, decision}` - what the run submits.

    `answer_source` is one of `commit` (the only verified row), `draft`, `abstain`,
    or `default_on_unparse` (the deterministic rule applied because the decision
    reply was unusable). **A `COMMIT` is executor-validated**: it may name only a
    unit whose check actually grounded, its option must be that commit's option, and
    its quote must pass the same description gate a check's support passes. A `DRAFT`
    selects the recorded draft and may not rewrite it - the decision is a selection
    over evidence, not a new place to answer from the option text.
    """
    decision = parse_decision(text)
    if decision is None:
        return dict(default_submission(committed, draft, "unparsable decision reply"),
                    decision=None, commit_refused=False)
    if decision["action"] == "abstain":
        if _letter_of(draft) and not committed:
            return {"answer": draft, "answer_source": "draft", "fault": "",
                    "why": "the decision abstained over its own recorded draft",
                    "decision": decision, "commit_refused": bool(committed),
                    "abstain_over_draft": True}
        return {"answer": "", "answer_source": "abstain", "fault": "",
                "why": decision["reason"] or "O abstained", "decision": decision,
                "commit_refused": bool(committed)}
    if decision["action"] == "draft":
        if not draft:
            return dict(default_submission(committed, draft,
                                           "the decision chose a draft and there is "
                                           "none"), decision=decision)
        if _letter_of(decision["option"]) != _letter_of(draft):
            return dict(default_submission(committed, draft,
                                           "the decision rewrote the draft (%s, not %s)"
                                           % (decision["option"], draft)),
                        decision=decision)
        return {"answer": draft, "answer_source": "draft", "fault": "",
                "why": decision["reason"] or "O chose its own draft",
                "decision": decision, "commit_refused": bool(committed)}
    if not committed:
        reason = commit_why or "no grounded commit stood"
        return dict(default_submission(
            committed, draft, "the decision committed %s but %s"
            % (decision["unit"] or "nothing", reason)), decision=decision,
            commit_refused=False)
    if _letter_of(decision["option"]) != _letter_of(committed["option"]):
        return dict(default_submission(
            committed, draft,
            "the decision committed %s under the %s citation"
            % (decision["option"], committed["option"])), decision=decision,
            commit_refused=False)
    return {"answer": committed["option"], "answer_source": "commit", "fault": "",
            "why": "O committed %s" % committed["unit"], "decision": decision,
            "commit_refused": False, "named_unit": unit_of(decision["unit"]),
            "named_quote": decision["quote"]}


MEMORY_HEADER = "What the team already knows about this video:"
READER_MEMORY_HEADER = "Your memory of this chunk from earlier questions:"
NO_MEMORY_BLOCK = "(the team knows nothing about this video yet)"
LOOKED_HEADER = "What you have already looked up this question:"
REFINE_HEADER = "The refine this question's own seconds offer:"

MEMORY_TERMINATORS = ("Chunks: ", "Reader ", "What has already been tried",
                      "Your own draft", "THINK:", "You are also shown", "Reply with",
                      "The question and its options:", LOOKED_HEADER,
                      "Look at the frames",
                      "Your draft", "What you have done so far", "You have ",
                      "Choose exactly one",
                      "The chunks you asked to scan",
                      "What you may cite",
                      REFINE_HEADER)


def _clip(text, limit):
    """`text` cut to `limit` characters at a word boundary, with no ellipsis games."""
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    out = ""
    for word in text.split(" "):
        if len(out) + len(word) + 1 > limit:
            break
        out = (out + " " + word).strip()
    return out or text[:limit]


def _fit_lines(lines, limit):
    """Shrink `lines` until they fit `limit`, by clipping the longest line first."""
    lines = list(lines)
    while len("\n".join(lines)) > limit and lines:
        index = max(range(len(lines)), key=lambda at: len(lines[at]))
        words = lines[index].split(" ")
        if len(words) > 3:
            lines[index] = " ".join(words[:-1])
        else:
            lines.pop(index)
    return "\n".join(lines)[:limit]


def _fit_head(lines, limit):
    """A head kept in **priority order**: whole lines, dropped from the end.

    `_fit_lines` clips the longest line first, which is right for a bag of lines and wrong
    for a list that has an order: the chunk list is the one head line the handle lines
    already carry (a handle says `chunk 7`), so it is the first to go, and the topic line
    and the counts are the two that stay.
    """
    lines = [str(line) for line in lines]
    kept = []
    for line in lines:
        if len("\n".join(kept + [line])) > limit:
            break
        kept.append(line)
    if not kept and lines:
        kept = [_clip(lines[0], max(limit, 1))]
    return kept


def _few_words(text, count):
    words = re.findall(r"[A-Za-z0-9']+", str(text or ""))
    return " ".join(words[:count])


def _span_of(address):
    """A stored address as a reader sees it: `1856-1920s`, or `no address`."""
    try:
        if isinstance(address, str):
            # A string is not a span: its `[0]` and `[-1]` are characters, and `float`
            # reads `1856-1920` as the pair `1` and `0`. Nothing the writer produces
            # reaches here, and a guess would be a wrong pointer.
            return "no address"
        first, last = ((address, address) if isinstance(address, (int, float))
                       else (address[0], address[-1]))
        return "%d-%ds" % (round(float(first)), round(float(last)))
    except (TypeError, ValueError, IndexError, KeyError):
        return "no address"


def memory_by_handle(shown, handle):
    """The entry a named handle refers to, or `None`."""
    wanted = str(handle or "").strip().upper()
    for item in shown or ():
        if str(item.get("handle") or "").upper() == wanted:
            return item
    return None


def _chunk_text(chunk_id):
    """A chunk id as text, whether it arrived as an int or from a JSON store."""
    try:
        return str(int(chunk_id))
    except (TypeError, ValueError):
        return str(chunk_id)


def rendered_handles(text):
    """**The handles an index actually showed.**"""
    forms = (
        r"^([ME]\d+)\s+chunk\s+\d+",     # the index view: `M1 chunk 3: "words"`
        r"^(M\d+)\s+\d+\s+[\d.]+-",      # the tree view:  `M1 3 128-136 unverified "cue"`
    )
    out, seen = [], set()
    for form in forms:
        for handle in re.findall(form, text or "", re.M):
            if handle not in seen:
                seen.add(handle)
                out.append(handle)
    return out


def question_leaves(session, question):
    """The citable, settled reads **for this question**.

    The team layer must not mix questions. A video's leaves accumulate across its questions, and a
    leaf's `answer` is the option its reader gave for **its** question - so a line rendered during
    question k from a leaf written during question j < k would report a **stale answer to a different
    question**, so the question is passed through to the team lines. A seeded note
    (`question_number == 0`) belongs to no question and always carries."""
    out = []
    for chunk_id in sorted(session.get("chunks") or {}):
        for entry in session["chunks"][chunk_id]:
            if entry.get("retracted") or not entry.get("citable", True):
                continue
            if (entry.get("question_number") != 0 and question is not None
                    and entry.get("question") != question):
                continue
            out.append((chunk_id, entry))
    return out


def _entry_words(entry, limit=LEAF_SIGHTING_CHARS):
    """**The read's own words, from the session entry** (found on `belief3-eval`: every belief's `why`
    rendered empty, because `leaf_sighting` reads a **reader record** - `quote`/`claimed`/`points` -
    while the session entry carries the sighting as `description`/`update`). The belief's reason must be
    the reader's words, so it reads them off the entry it actually has.
    """
    text = " ".join(str(entry.get("description") or entry.get("update") or "").split())
    return text[:limit]


def _span_text(place):
    """A place, as a person would say it: `128-192s`, or `?` when the read named none."""
    if place and len(place) >= 2:
        return "%g-%gs" % (float(place[0]), float(place[1]))
    return "?"


def _scope_of(question):
    """The **scope** the claim is true over, when the question asks for an order - so the belief is not a
    true-sounding claim with the wrong temporal meaning."""
    found = re.search(r"\b(first|last|before|after|earliest|latest|begin|end|next)\b",
                      question or "", re.I)
    return found.group(1).lower() if found else ""


def bearing_leaves(session, question):
    """**Which reads bear on the question**: this question's own reads, **plus** the video's other
    questions' reads whose **content shares a content term with the question** - a **declared** rule
    (`_terms`), never a learner (rule 2).

    Another question's **answer** must not be shown as this question's. A leaf whose
    **content** bears on this question is **evidence**, not an answer: it is *marked* with the question it
    was written for, so the belief and its citation stay honest. This is the memory's **distinctive**
    content - what the agent did not itself read - and it is the `E` citations' own material."""
    out = list(question_leaves(session, question))
    seen = {id(entry) for _cid, entry in out}
    wanted = _terms(question)
    if not wanted:
        return out
    for chunk_id in sorted(session.get("chunks") or {}):
        for entry in session["chunks"][chunk_id]:
            if id(entry) in seen or entry.get("question_number") == 0:
                continue
            if entry.get("retracted") or not entry.get("citable", True):
                continue
            words = "%s %s" % (entry.get("description") or "", entry.get("update") or "")
            if wanted & _terms(words):
                out.append((chunk_id, entry))
    return out


def question_belief(session, question, limit=MAX_GLOBAL_MEMORIES):
    """**The belief a human would keep**: not a bag of notes, but one object - **the claim**, the
    places that support it **each naming its read**, the **rival** (remembered, not deleted), the places
    **looked at with nothing there** (the negative that stops a person re-reading), and the **newest
    read** so a revision is visible."""
    leaves = bearing_leaves(session, question)
    claimed = [(cid, entry) for cid, entry in leaves if entry.get("answer")]
    looked = [(cid, entry) for cid, entry in leaves if not entry.get("answer")]
    if not claimed and not looked:
        return None
    if not claimed:
        # **Only the negative** (a person's "I looked there and it wasn't there"): every read this
        # question reported nothing. The belief carries the places that were checked and held nothing,
        # which is what stops the same region being re-read.
        return {"answer": "", "agree": 0, "reads": 0, "regions": [], "support": [], "rival": None,
                "negative": [list(e.get("seconds") or e.get("assigned") or ()) for _cid, e in looked],
                "negative_regions": sorted({int(cid) for cid, _e in looked}),
                "folders_read": sorted({int(cid) for cid, _e in looked}), "independent": 0,
                "latest": None, "scope": _scope_of(question), "chunks": []}
    by_option = {}
    for cid, entry in claimed:
        place = tuple(entry.get("seconds") or entry.get("assigned") or (entry["id"],))
        by_option.setdefault(str(entry["answer"]), {}).setdefault(place, (cid, entry))
    ranked = sorted(by_option.items(), key=lambda item: (-len(item[1]), item[0]))
    claim, support = ranked[0]
    cells = list(support.values())
    regions = sorted({int(cid) for cid, _entry in cells})
    folders_read = sorted({int(cid) for cid, _entry in leaves})
    rival = None
    if len(ranked) > 1:
        option, other = ranked[1]
        entries = list(other.values())
        rival = {"answer": option,
                 "places": [list(e.get("seconds") or e.get("assigned") or ()) for _cid, e in entries],
                 "regions": sorted({int(cid) for cid, _e in entries}),
                 "quote": _entry_words(entries[-1][1])}
    readers = sorted({str(e.get("author") or "") for _cid, e in cells if e.get("author")})
    latest_cid, latest = max(claimed, key=lambda pair: pair[1]["id"])
    negative_regions = sorted({int(cid) for cid, _entry in looked if int(cid) not in regions})
    return {
        "answer": claim,
        "agree": len(cells),
        "reads": sum(len(option_map) for option_map in by_option.values()),
        "regions": regions,
        "folders_read": folders_read,
        "support": [{"place": list(e.get("seconds") or e.get("assigned") or ()),
                     "quote": _entry_words(e), "read": e.get("id"), "region": int(cid),
                     "from_question": e.get("question_number"),
                     "author": str(e.get("author") or ""), "verified": bool(e.get("verified"))}
                    for cid, e in cells],
        "rival": rival,
        "negative": [list(e.get("seconds") or e.get("assigned") or ()) for _cid, e in looked],
        "negative_regions": negative_regions,
        "independent": len(readers),
        "latest": {"answer": str(latest.get("answer") or ""), "read": latest.get("id"),
                   "region": int(latest_cid),
                   "place": list(latest.get("seconds") or latest.get("assigned") or ()),
                   "agrees": str(latest.get("answer")) == claim},
        "scope": _scope_of(question),
        "chunks": sorted({cid for cid, _entry in claimed}),
    }


def team_lines(session, question, limit=MAX_GLOBAL_MEMORIES):
    """**The team's settled findings for this question** - one line per chunk: the option the team
    established, the span it rests on, and how many reads agree. The **global layer** the memory was
    missing: a leaf holds one reader's seconds, while the settled line holds what the **team** has
    established. Deterministic: the majority option among the live claimed leaves of **this question**,
    the latest claimed leaf's span, and the agreement count over **distinct places**. A chunk no reader
    claimed leaves no line.
    """
    by_chunk = {}
    for chunk_id, entry in question_leaves(session, question):
        by_chunk.setdefault(chunk_id, []).append(entry)
    out = []
    for chunk_id in sorted(by_chunk):
        claimed = [entry for entry in by_chunk[chunk_id] if entry.get("answer")]
        if not claimed:
            continue
        by_place = {}
        for entry in claimed:
            place = tuple(entry.get("seconds") or entry.get("assigned") or (entry["id"],))
            by_place.setdefault(place, entry)
        distinct = list(by_place.values())
        answers = [entry["answer"] for entry in distinct]
        settled, agree = collections.Counter(answers).most_common(1)[0]
        latest = max(distinct, key=lambda entry: entry["id"])
        out.append({"chunk": chunk_id, "answer": settled, "agree": agree, "reads": len(distinct),
                    "span": latest.get("seconds") or latest.get("assigned"),
                    "sighting": latest.get("description") or latest.get("update") or ""})
    return out[:limit]


def render_team_memory(session, question, limit=MAX_GLOBAL_MEMORIES):
    """**The belief as the memory's content**: the distilled layer the
    section shows, bounded at `BELIEF_CHARS`, so using the memory is reasoning over content, not a fetch
    chore. The per-read leaf handles are **not** shown beside it - that was the redundancy: the belief is
    the reads' own majority, so showing both made a citation of the belief no different from citing the
    read. The reads stay in the store and are reached by a fetch (rule 3)."""
    belief = question_belief(session, question, limit=limit)
    if not belief:
        return ""
    if session.get("belief_hidden"):
        return ""
    if not belief["answer"]:
        text = ("the team's belief for this question: no answer yet | looked, nothing: %s"
                % ", ".join("region %d" % r for r in belief["negative_regions"][:3]))
        return text[:BELIEF_CHARS]
    bits = ["the team's belief for this question: the answer is (%s)" % belief["answer"]]
    def _src(item):
        text = "read %s" % (item.get("read") or "?")
        if item.get("from_question"):
            text += " from q%s" % item["from_question"]
        if not item["verified"]:
            text += ", unverified"
        return text

    why = ['"%s" at region %d %s (%s)'
           % (_clip(item.get("quote") or "", REASON_CHARS), item["region"],
              _span_text(item["place"]), _src(item))
           for item in belief["support"][:MAX_REASONS]]
    if why:
        bits.append("because: " + "; ".join(why))
    if belief["agree"] and belief["reads"]:
        bits.append("%d of %d reads, over %d region(s) of %d read"
                    % (belief["agree"], belief["reads"], len(belief["regions"]),
                       len(belief["folders_read"])))
    if belief["rival"]:
        rival_regions = belief["rival"]["regions"] or []
        where = ("region %d" % rival_regions[0]) if rival_regions else "another region"
        rival_quote = _clip(belief["rival"].get("quote") or "", REASON_CHARS)
        # The quote comes first, as in a reason, so the audit's one declared form reads both. The
        # declaration also carries the form from the belief object when the prompt clip cuts the
        # location text; see `belief_note_forms`.
        extra = ('"%s" at %s' % (rival_quote, where)) if rival_quote else where
        bits.append("rival (%s): %s" % (belief["rival"]["answer"], extra))
    if belief["negative_regions"]:
        bits.append("looked, nothing: %s"
                    % ", ".join("region %d" % r for r in belief["negative_regions"][:2]))
    if belief["latest"] and not belief["latest"]["agrees"]:
        bits.append("newest read (region %d) says (%s)"
                    % (belief["latest"]["region"], belief["latest"]["answer"]))
    if belief["scope"]:
        bits.append("scope: %s" % belief["scope"])
    text = " | ".join(bits)
    if len(text) > BELIEF_CHARS:
        text = text[:BELIEF_CHARS - 1].rstrip() + "\u2026"
    return text


def shown_belief(session, question):
    """**The belief the section actually showed**: the ablation arm hides
    the belief, and a `T1` citation is eligible only when it was **rendered** - the off arm cited a belief it
    never saw and one of the three was accepted at the pointer tier, because eligibility asked whether a
    belief **exists** rather than whether it was **shown**. One place decides, and `shown_memories` and the
    gate both read it."""
    if session.get("belief_hidden"):
        return None
    return question_belief(session, question)


def shown_memories(session, question, shown=None, question_number=None):
    """The matched entries **the prompt actually showed**, for a call that names one.

    Under the index view the 200-character bound can drop handle lines from the end, so the
    list the matcher returned and the list the model saw are not the same thing. The
    consult's `NO_USABLE` refusal has been validated against what was rendered since
    `0210749`; this is the same rule for the three doors that **take** a handle - the
    consult's `ANSWER` branch, `LOOK` and `RECALL`. Serving a handle the cap removed reaches
    an entry outside the index's own space, which is the one thing the map exists to bound.
    The push view renders every match, so there the matcher's list is what was shown.
    """
    explicit = shown is not None
    frozen = session.get("_frozen_shown_memories")
    if frozen is not None and not explicit:
        return copy.deepcopy(frozen)
    shown = (list(shown) if explicit else
             matching_memories(session, question, question_number=question_number))
    if session.get("memory_view") == "tree" and not explicit:
        out = tree_offers(session, question, question_number=question_number) or shown
    elif session.get("memory_view") != "index":
        out = shown
    else:
        rendered = set(rendered_handles(render_memory_index(
            session, question, question_number=question_number)))
        out = [item for item in shown if item["handle"] in rendered]
    belief = shown_belief(session, question)
    if belief and belief.get("answer") and not session.get("belief_hidden"):
        text = render_team_memory(session, question)
        top = belief["support"][0] if belief["support"] else {}
        out = [{"handle": BELIEF_HANDLE, "entry_id": top.get("read"), "chunk": top.get("region"),
                "address": list(top.get("place") or []) or None, "assigned": None,
                "question": question, "status": None,
                "cue": text, "description": text, "update": "", "words": text,
                "belief": True}] + list(out)
    return out


def render_looks(row):
    """**What a served look returned**, as the observation the next call sees.

    A fetch is a read of the memory: the entry's own words enter the agent's context and
    nothing else happens - no answer is produced, no citation is earned. The block is
    labelled so the artifact shows the dependency, and it is never passed to the check.
    """
    lines = []
    for look in (row or {}).get("looks") or []:
        # An `unreviewable` look is recorded with no chunk (uid 730). Rendering the record is not a
        # second review and must not raise: the honest line names that the chunk is absent.
        chunk = look.get("chunk")
        where = ("chunk %s" % _chunk_text(chunk)) if chunk is not None else "no chunk"
        lines.append('[look %s] %s %s: "%s"'
                     % (look.get("handle") or "?", where, _span_of(look.get("address")),
                        look.get("text") or ""))
    return "\n".join(lines)


def store_counts(session):
    """`(chunks with citable records, records, records carrying an address)`."""
    chunks, records, addressed = [], 0, 0
    for chunk_id, entries in (session.get("chunks") or {}).items():
        keep = [entry for entry in entries
                if entry.get("citable", True) and not entry.get("retracted")]
        if not keep:
            continue
        try:
            chunks.append(int(chunk_id))
        except (TypeError, ValueError):
            continue
        records += len(keep)
        addressed += sum(1 for entry in keep if entry.get("address"))
    return sorted(chunks), records, addressed


def supply_counts(reports, family=""):
    """The supply bar's own arithmetic, **on the right denominator**."""
    reports = list(reports or ())
    claiming = [report for report in reports
                if report.get("claimed") or report.get("points")]
    written = [report for report in reports if report.get("typed")]
    written_claiming = [report for report in claiming if report.get("typed")]
    rate = (round(len(written_claiming) / float(len(claiming)), 4) if claiming else 0.0)
    first = [report for report in reports if not report.get("reread_field")]
    first_claiming = [report for report in first
                      if report.get("claimed") or report.get("points")]
    first_written = [report for report in first_claiming if report.get("typed")]
    first_rate = (round(len(first_written) / float(len(first_claiming)), 4)
                  if first_claiming else 0.0)
    return {"reads": len(reports), "claiming": len(claiming),
            "written": len(written), "written_claiming": len(written_claiming),
            "unmapped_reads": len(reports) if not family else 0,
            "addressed_claim_rate": rate,
            "claiming_first_reply": len(first_claiming),
            "written_claiming_first_reply": len(first_written),
            "addressed_claim_rate_first_reply": first_rate}


def index_rows(session, question, limit=MAX_GLOBAL_MEMORIES, question_number=None):
    """**The only thing the index renderer is allowed to see.**"""
    return [{"handle": item["handle"], "chunk": item["chunk"],
             "question": _few_words(item["question"], INDEX_QUESTION_WORDS),
             "cue": _few_words(item.get("description") or item.get("update") or "", CUE_WORDS),
             "address": item.get("address"), "layer": item.get("layer"),
             "memory_group": item.get("memory_group"),
             "supported": item.get("supported", False),
             "memory_kind": item.get("memory_kind", ""),
             "question_number": item.get("question_number"),
             "global": item.get("global", False)}
            for item in matching_memories(
                session, question, limit, question_number=question_number)]


# **The proposal line's own form**, parsed by the same audit path as a note: the prefix, the
# chunk it proposes, its span, and the sighting it re-renders. One definition, used by the
# renderer and by `rendered_notes`, so the excuse is written against the string that rendered.
PROPOSAL_LINE = re.compile(r"^" + re.escape(PROPOSAL_PREFIX)
                           + r"chunk (\d+), [\d.]+-[\d.]+s(?:, unread)?: (.*)$", re.M)
TREE_LINE = re.compile(r"^(?:M\d+ )?(\d+) \d+-\d+ \w+(?: \"(.*)\")?$")
NOTE_LINE = re.compile(r"^chunk (\d+): (.*)$")
BELIEF_REASON = re.compile(r'"([^"]{4,})"\s+(?:at|in)\s+region\s+(\d+)')
COVERAGE_LINE = re.compile(r"^read: \d+/\d+ chunks", re.M)


def leaf_sighting(record, limit=LEAF_SIGHTING_CHARS):
    """**The leaf's sighting is the reader's evidence, not its frame description**. The map cue is not the opening of the reader's `FRAMES` text, which is a generic
    description and gives a fetched leaf nothing to verify. When the reader **claimed**
    (`CAN_ANSWER: YES|POINTS` with an `ANSWER`), the leaf's one line is the **option and the resting
    words** (the `QUOTE`) - the evidence the leaf exists to carry; otherwise it is the frame text, as
    before, because a read that claimed nothing still leaves a lead."""
    answer = str(record.get("answer") or "").strip()
    quote = str(record.get("quote") or "").strip()
    kind = str(record.get("kind") or "").strip()
    if answer and (record.get("claimed") or record.get("points")):
        parts = []
        if quote:
            parts.append('"%s"' % quote)
        parts.append("answer %s" % answer)
        if kind:
            parts.append("(%s)" % kind)
        return _clip(" ".join(parts), limit)
    return _clip(" ".join(str(record.get("frames_text") or "").split()), limit)


def note_form(text, limit=NOTE_CHARS):
    """**The note exactly as the index renders it**: one line, clipped at a word boundary."""
    return _clip(" ".join(str(text or "").split()), limit)


def note_lines(session, question, limit=NOTE_LINES, skip=None):
    """**The video's shape, as one line per chunk** - the notes the question's words match.

    `chunk_map` is written at every read (`remember_chunk`), so the video already carries a
    one-line-per-chunk map; the index view simply did not show it. Showing it is what makes the
    map *navigable*: a handle says "there is a record", a note says "this is what that part of
    the video holds", which is what a planner needs in order to decide where to read.

    The notes are ranked by the same term overlap the record matcher uses, and a chunk whose
    note shares nothing with the question is left out rather than padding the bound."""
    wanted = _terms(question)
    scored = []
    for chunk_id, text in (session.get("chunk_map") or {}).items():
        if skip is not None and _chunk_text(chunk_id) == _chunk_text(skip):
            continue
        text = " ".join(str(text or "").split())
        if not text:
            continue
        hits = len(wanted & _terms(text))
        if not hits:
            continue
        scored.append((-hits, chunk_id, text))
    scored.sort(key=lambda item: (item[0], item[1]))
    out = []
    for _, chunk_id, text in scored[:limit]:
        out.append("chunk %s: %s" % (_chunk_text(chunk_id), note_form(text)))
    return out


def rendered_notes(text):
    """**The note lines a rendered index actually showed**: their chunk ids and their text.

    The audit needs both: the text is what the model was told (so I15 must not fingerprint it
    twice), and the chunk ids are what a reading counts."""
    out = []
    for line in (text or "").splitlines():
        found = (NOTE_LINE.match(line.strip()) or PROPOSAL_LINE.match(line.strip())
                 or TREE_LINE.match(line.strip()))
        if found:
            out.append((found.group(1), found.group(2)))
        for reason in BELIEF_REASON.finditer(line):
            out.append((reason.group(2), reason.group(1)))
    return out


def render_memory_index(session, question, rows=None, chunks=None, read_now=None,
                        question_number=None):
    """What O is shown instead of the block: a bounded, derived index of the same store.

    The head is a **priority list** - topic, counts, the settled questions, and last which
    chunks hold records - and the map follows it. The head is fitted to what is left after
    a reservation for `INDEX_MIN_HANDLES` map lines, and the map is then dropped from the
    end until the whole thing fits `INDEX_CHARS`: the best-matching handle is the one that
    survives, and a handle now always survives when the matcher found one."""
    rows = (index_rows(session, question, question_number=question_number)
            if rows is None else list(rows))
    holds_records, records, addressed = store_counts(session)
    head = []
    head.append("records: %d (%d addressed) | settled: %d"
                % (records, addressed, len(session.get("qa") or [])))
    proposal = ""
    if chunks is not None and not session.get("memory_disabled"):
        proposal = render_proposal_line(session, question, chunks, read_now=read_now or ())
    nav = [line for line in (proposal, ) if line]
    if chunks is not None and not session.get("memory_disabled"):
        nav.append(render_coverage_line(session, chunks))
    head += nav[:INDEX_MIN_NAV]
    # **The topic comes after the navigation** (the same rule): it is a sentence about the
    # video, and a bound that has to choose keeps the line that says where to look next.
    if session.get("topic"):
        head.append("the video: %s" % _clip(session["topic"], INDEX_TOPIC_CHARS))
    ordered = sorted(rows, key=lambda row: (
        not row.get("address"),
        int(row["memory_group"]) if row.get("memory_group") is not None else 9))
    handles = ["%s chunk %s %s \"%s\"" % (row["handle"], _chunk_text(row["chunk"]),
                                          _span_of(row.get("address")),
                                          row.get("cue") or row["question"])
               for row in ordered]
    reserve = min(sum(len(line) + 1 for line in handles[:INDEX_MIN_HANDLES]),
                  INDEX_CHARS // 2)
    tail = note_lines(session, question, skip=proposal_chunk(proposal))
    settled = render_qa_ledger(session)
    if settled and not settled.startswith("("):
        tail += settled.splitlines()
    if holds_records:
        tail.append("chunks with records: %s"
                    % ",".join(str(cid) for cid in holds_records))
    room = max(0, INDEX_CHARS - reserve - len("\n".join(head)))
    kept = []
    for line in tail:
        if len("\n".join(kept + [line])) > room:
            break
        kept.append(line)
    head += kept
    while handles and len("\n".join(head + handles)) > INDEX_CHARS:
        handles.pop()
    while len("\n".join(head + handles)) > INDEX_CHARS and head:
        head = _fit_head(head[:-1], INDEX_CHARS)
    return _fit_lines(head + handles, INDEX_CHARS)


def render_memory_block(session, question, memories=None, consult=False, chunks=None,
                        read_now=None, question_number=None):
    """What O is shown - and, given `memories`, exactly the entries the gate will
    check. Passing the list in is what keeps `M<k>` one thing per question.

    **`consult=True` is the only door left open when steering is off**: the memory-first
    call still sees the block it is asked about, and every plan, check, select, conflict
    and submit call is shown the empty placeholder instead.
    """
    if not consult and session.get("memory_steering") is False:
        return NO_MEMORY_BLOCK
    if session.get("memory_view") == "index":
        # **Even the caller's list is projected**, so the index path never holds an object
        # carrying entry text, whichever door it came through.
        rows = None if memories is None else [
            {"handle": item["handle"], "chunk": item["chunk"],
             "question": _few_words(item.get("question"), INDEX_QUESTION_WORDS),
             "address": item.get("address"), "layer": item.get("layer"),
             "memory_group": item.get("memory_group"),
             "supported": item.get("supported", False),
             "memory_kind": item.get("memory_kind", ""),
             "question_number": item.get("question_number"),
             "global": item.get("global", False)} for item in memories]
        return render_memory_index(session, question, rows, chunks=chunks,
                                   read_now=read_now, question_number=question_number)
    def with_frozen_handles(block):
        if memories is not None and not rendered_handles(block):
            handles = [tree_offer_line(item) for item in memories
                       if str(item.get("handle") or "").upper().startswith("M")
                       and item.get("chunk") is not None
                       and int(item.get("chunk")) >= 1]
            if handles:
                return block + "\n" + "\n".join(handles)
            index = render_memory_index(
                session, question, rows=memories, chunks=chunks,
                read_now=read_now, question_number=question_number)
            if rendered_handles(index):
                return index
        return block

    if session.get("memory_view") == "tree":
        rows = None if memories is None else [
            {"handle": item["handle"], "chunk": item["chunk"],
             "entry_id": item.get("entry_id"),
             "question": _few_words(item.get("question"), INDEX_QUESTION_WORDS),
             "address": item.get("address"), "layer": item.get("layer"),
             "memory_group": item.get("memory_group"),
             "supported": item.get("supported", False),
             "memory_kind": item.get("memory_kind", ""),
             "question_number": item.get("question_number"),
             "global": item.get("global", False)} for item in memories]
        block = render_tree_index(
            session, question, rows, chunks=chunks, read_now=read_now,
            question_number=question_number)
        team = render_team_memory(session, question)
        belief = question_belief(session, question)
        if team and belief and belief.get("answer") and len(team) <= INDEX_CHARS:
            nav = render_tree_index(session, question, rows, chunks=chunks, read_now=read_now,
                                    show_leaves=False)
            room = INDEX_CHARS - len(team) - 1
            lines = nav.split("\n")
            while lines and len("\n".join(lines)) > room:
                lines.pop()
            fitted = "\n".join(lines)
            return with_frozen_handles((team + "\n" + fitted) if fitted else team)
        if team and len(team) + 1 + len(block) <= INDEX_CHARS:
            return with_frozen_handles(team + "\n" + block)
        return with_frozen_handles(block)
    lines = []
    if session.get("topic"):
        lines.append("the video: %s" % session["topic"])
    for chunk_id in sorted(session.get("chunk_map") or {}):
        lines.append("chunk %d holds: %s" % (chunk_id, session["chunk_map"][chunk_id]))
    if session.get("qa"):
        lines.append("Questions already settled in this session:")
        lines.append(render_qa_ledger(session))
    for entry in (matching_memories(session, question) if memories is None
                  else memories):
        lines.append("%s (chunk %d, from the question: %s%s): %s %s"
                     % (entry["handle"], entry["chunk"], entry["question"],
                        ", matched by " + entry["match_key"]
                        if entry.get("match_key") else "",
                        entry["description"], entry["update"]))
    return "\n".join(lines) or NO_MEMORY_BLOCK


def tree_offer_line(offer):
    """**One rendered leaf line**, in the one form the renderer and the audit both know."""
    span = list(offer.get("address") or [])
    chunk = offer.get("chunk") or 0
    return "%s%s %d-%d %s%s" % (
        ("%s " % offer["handle"]) if offer.get("handle") else "", chunk,
        int(span[0]) if span else CHUNK_SECONDS * (chunk - 1),
        int(span[1]) if span else CHUNK_SECONDS * chunk,
        offer.get("status") or "unverified",
        (" \"%s\"" % offer["cue"]) if offer.get("cue") else "")


def tree_offers(session, question, limit=MAX_GLOBAL_MEMORIES, chunks=None, read_now=None,
                budget=TREE_OFFER_CHARS, question_number=None):
    """**The tree's addressable leaves**.

    The tree view rendered the store's leaves with a handle **only for the entries `index_rows`
    word-matched** - so on a question whose words matched nothing, every leaf was shown **without an
    address**, and the 50-item run records what that buys: **0 `M` citations** and one guess at a
    handle the prompt never showed. The word match is not a ranker (the relevance licence measured no
    signal), so addressability must not depend on it.

    This is the offered set: live eligible leaves, bounded, with a handle and a short
    cue. Supported video/prior memory leads, then the proposal, clock and question terms break
    ties. `shown_memories` returns exactly this under the tree view."""
    store = session.get("store") or {"mode": "session"}
    leaves = []
    if store.get("mode") == "files":
        from ..memory import global_store as _tree_store
        video = store.get("video") or session.get("video")
        for _path, leaf in _tree_store._leaf_files(
                os.path.join(str(store["root"]).rstrip("/"), str(video))):
            if leaf.get("retracted") or not leaf.get("citable"):
                continue
            if str(leaf.get("status") or "") in ("superseded", "retracted"):
                continue
            provenance = leaf.get("provenance") or {}
            leaves.append({
                "id": leaf.get("id"), "parent": leaf.get("parent"),
                "span": list(leaf.get("span") or []),
                "assigned": leaf.get("assigned"),
                "sighting": leaf.get("sighting") or "",
                "status": leaf.get("status") or "unverified",
                "question": provenance.get("question") or "",
                "question_number": provenance.get("question_number"),
                "global": bool(leaf.get("global")),
                "answer_support": bool(leaf.get("answer_support", False)),
                "citation_allowed": bool(leaf.get("citation_allowed", False)),
                "review_status": str(leaf.get("review_status") or "unreviewed"),
                "claim": copy.deepcopy(leaf.get("claim")),
                "frames": copy.deepcopy(leaf.get("frames")),
                "citable": bool(leaf.get("citable", True))})
    else:
        for chunk_id in sorted(session.get("chunks") or {}):
            for entry in session["chunks"][chunk_id]:
                if entry.get("retracted") or not entry.get("citable"):
                    continue
                leaves.append({"id": entry.get("id"), "parent": chunk_id,
                               "span": entry.get("seconds") or
                                       [CHUNK_SECONDS * (chunk_id - 1), CHUNK_SECONDS * chunk_id],
                               "assigned": entry.get("assigned"),
                               "sighting": entry.get("description") or entry.get("update") or "",
                               "status": ("verified" if entry.get("verified") else "unverified"),
                               "question": entry.get("question") or "",
                               "question_number": entry.get("question_number"),
                               "global": bool(entry.get("global")),
                               "answer_support": bool(entry.get("answer_support", False)),
                               "citation_allowed": bool(entry.get("citation_allowed", False)),
                               "review_status": str(entry.get("review_status") or "unreviewed"),
                               "claim": copy.deepcopy(entry.get("claim")),
                               "frames": copy.deepcopy(entry.get("frames")),
                               "citable": True})
    current = current_question_number(session, question_number)
    if session.get("memory_protocol") == "two_level":
        leaves = [leaf for leaf in leaves
                  if memory_layer(leaf, current) == "video" and supported_memory(leaf)]
    proposed = None
    if chunks is not None:
        line = render_proposal_line(session, question, chunks, read_now=read_now or ())
        proposed = proposal_chunk(line)
    clocked = clock_chunks(question, chunks) if chunks is not None else set()
    terms = _terms(question)
    leaves.sort(key=lambda leaf: (
        memory_group(leaf, current),
        (leaf.get("parent") not in clocked) if clocked else False,
        leaf.get("parent") != proposed,
        -len(terms & _terms(leaf.get("question") or "")),
        leaf.get("parent") or 0, (leaf.get("span") or [0.0])[0], leaf.get("id") or 0))
    offers = []
    spent = 0
    for index, leaf in enumerate(leaves[:limit], start=1):
        span = list(leaf.get("span") or [])
        offer = {"handle": "M%d" % index, "entry_id": leaf.get("id"),
                 "chunk": leaf.get("parent"),
                 "assigned": (list(leaf["assigned"]) if leaf.get("assigned") else None),
                 "question": str(leaf.get("question") or ""),
                 "description": str(leaf.get("sighting") or ""),
                 "update": "", "address": span or None,
                 "status": leaf.get("status") or "unverified",
                 "question_number": leaf.get("question_number"),
                 "layer": memory_layer(leaf, current),
                 "memory_group": memory_group(leaf, current),
                 "supported": supported_memory(leaf),
                 "memory_kind": memory_kind(leaf),
                 "global": bool(leaf.get("global")),
                 "answer_support": bool(leaf.get("answer_support", False)),
                 "citation_allowed": bool(leaf.get("citation_allowed", False)),
                 "review_status": str(leaf.get("review_status") or "unreviewed"),
                 "cue": _few_words(leaf.get("sighting") or "", CUE_WORDS)}
        if budget and spent + len(_offer_line(offer)) + 1 > int(budget):
            break
        spent += len(_offer_line(offer)) + 1
        offers.append(offer)
    return offers


def _offer_line(offer):
    """The rendered line of one offer - the same string `tree_offer_line` produces."""
    return tree_offer_line(offer)


def clock_reference(question, duration):
    """**The one place the question names**, chosen once and used by both the assignment and the
    score: the media rule is that *whatever the question names is read
    first*, so the **first** reference is the assigned place. Scoring a span against every reference
    while assigning from the first was a real inconsistency - a two-clock question was scored
    against a clock the read would never use."""
    refs = M.time_references(question or "", duration) if duration else []
    return [float(refs[0][0]), float(refs[0][1])] if refs else None


def assigned_span(question, duration, span, bounds=None, named=None):
    """**The span a `ZOOM` on this address reads**: the one function the offer and the handler both
    call, so the menu's line and the leaf the read writes cannot drift."""
    if not span:
        return None
    first, last = float(span[0]), float(span[-1])
    if named and len(named) >= 2:
        at = [float(named[0]), float(named[1])]
    else:
        ref = clock_reference(question, duration)
        at = list(ref) if ref else [first, last]
    if bounds:
        at = [max(at[0], float(bounds[0])), min(at[1], float(bounds[1]))]
    if at[1] <= at[0]:
        at = [first, last]
    if bounds:
        at = list(widen_inside(at[0], at[1], float(bounds[0]), float(bounds[1])))
    return [round(at[0], 3), round(at[1], 3)]


def clock_density(question, duration, span):
    """**How densely an address meets the one place the question names** - a **feature**, never a
    reward.

    **It is undefined on the unclocked class** (0.0 for every door), which is the class the
    interaction claim rests on - there the ranking is the map's own, so `candidate_doors` records
    the proposal flag and the term overlap beside it."""
    if not span:
        return 0.0
    first, last = float(span[0]), float(span[-1])
    if last <= first:
        return 0.0
    ref = clock_reference(question, duration)
    if not ref:
        return 0.0
    best = min(last, ref[1]) - max(first, ref[0])
    return round(max(0.0, best) / (last - first), 4)


MEMORY_DOORS = ("zoom",)
REFINE_CHARS = 200


def candidate_doors(session, question, duration, chunks=None, offers=None):
    """**Every door this turn could take**, with the exact line, the stable identity and the span
    the action would read.

    The line is **concrete** - `ZOOM M1 128-136`, not `ZOOM M<k> 150-155` - because the two 50-item
    runs measured the template: the refine verb fired **0 times** in both, with the menu documenting
    it. The span is the handler's own `assigned_span`, so the promise and the leaf agree."""
    bounds = {int(cid): (float(first), float(last)) for cid, first, last in (chunks or ())}
    if offers is None:
        offers = tree_offers(session, question, chunks=chunks)
    proposed = None
    if chunks is not None:
        proposed = proposal_chunk(render_proposal_line(session, question, chunks))
    terms = _terms(question)
    doors = []
    for offer in offers or ():
        handle = offer.get("handle")
        if not handle:
            continue
        address = offer.get("address")
        chunk = int(offer.get("chunk") or 0)
        assigned = assigned_span(question, duration, address, bounds.get(chunk))
        at = assigned
        if address and assigned and len(address) >= 2:
            span = [float(address[0]), float(address[-1])]
            if span[1] - span[0] < float(assigned[1]) - float(assigned[0]) \
                    and float(assigned[0]) <= span[0] and span[1] <= float(assigned[1]):
                # widened to the minimum place and shifted inside the chunk, exactly as the handler
                # will read it - the offered line must be the window the read takes
                lo, hi = bounds.get(chunk) or (float(assigned[0]), float(assigned[1]))
                at = list(widen_inside(span[0], span[1], lo, hi))
        for action in MEMORY_DOORS:
            if action == "zoom" and at:
                line = "ZOOM %d %g-%g" % (chunk, at[0], at[1])
            else:
                line = "%s %s" % (action.upper(), handle)
            doors.append({"action": action, "line": line, "handle": handle,
                          "entry_id": offer.get("entry_id"), "chunk": chunk,
                          # **What a `ZOOM` would read** (the assigned span, the handler's rule) -
                          # which is also the span the offline label scores for a door untaken.
                          "span": (list(at) if at else None),
                          "assigned": (list(assigned) if assigned else None),
                          "clock_density": clock_density(question, duration, at),
                          # **The leaf's stored address and its density**, recorded as a separate
                          # feature and never used to choose the line shown.
                          "address": (list(address) if address else None),
                          "address_density": clock_density(question, duration, address),
                          "proposal": bool(proposed) and chunk == proposed,
                          "term_overlap": len(terms & _terms(offer.get("question") or ""))})
    doors.sort(key=lambda door: (-door["clock_density"], not door["proposal"],
                                 -door["term_overlap"], door["chunk"] or 0,
                                 str(door["entry_id"] or ""), door["action"]))
    return doors


def render_tree_index(session, question, rows=None, chunks=None, read_now=None,
                      show_leaves=True, question_number=None):
    """**The Time Tree Memory, as the index**: the tree's own lines - folder
    paths as time ranges, each leaf's span and status, the handle only for the entries the caller
    rendered. No entry text: the same discipline the flat index keeps, so the tree cannot become a
    pushed summary.

    It renders from the **store**, not from a copy of it, and it falls back to the session when the
    run's store is not `files` (the control arm, and every stored-prompt test)."""
    store = session.get("store") or {"mode": "session"}
    # **The navigation lines travel with the view** (found by auditing a box run's prompts): the
    # coverage line and the proposal are what the policy is known to read, and switching the view
    # to the tree must not silently drop them - the tree lines come after, as the map does in the
    # flat index.
    nav = []
    if chunks is not None and not session.get("memory_disabled"):
        proposal = render_proposal_line(session, question, chunks, read_now=read_now or ())
        if proposal:
            nav.append(proposal)
        nav.append(render_coverage_line(session, chunks))
    offers = (tree_offers(session, question, chunks=chunks, read_now=read_now,
                          question_number=question_number) if show_leaves else [])
    if not show_leaves:
        return "\n".join(nav) if nav else NO_MEMORY_BLOCK
    handles = {item["entry_id"]: item["handle"] for item in offers
               if item.get("entry_id") is not None}
    cues = {item["entry_id"]: item.get("cue") for item in offers}
    rows = (index_rows(session, question, question_number=question_number)
            if rows is None else list(rows))
    if store.get("mode") == "files":
        from ..memory import global_store as _tree_store
        lines = _tree_store.tree_lines(store["root"], store.get("video") or session.get("video"),
                                       handles=handles, cues=cues, offers=offers)
        if lines:
            return "\n".join(nav + lines)
    lines = []
    for offer in offers:
        lines.append(tree_offer_line(offer))
    if not lines:
        return "\n".join(nav) if nav else NO_MEMORY_BLOCK
    return "\n".join(nav + ["tree: %d leaf/leaves" % len(lines)] + lines)


def render_qa_ledger(session):
    """The team's QA pairs, as a later question reads them."""
    if session.get("memory_view") == "index":
        lines = []
        for entry in session.get("qa") or []:
            where = ("%g-%gs" % tuple(entry["seconds"][:2]) if entry.get("seconds")
                     else "no seconds")
            letter = re.match(r"\s*\(?([A-H])\)?\s*(?=\s|$|[.)])",
                              entry.get("option") or "")
            lines.append("Q%d | %s | %s at %s | %s"
                     % (entry["n"], letter.group(1) if letter else "-",
                        entry.get("unit") or "no unit", where,
                        "verified" if entry.get("verified")
                        else "not double-checked"))
        return "\n".join(lines[-INDEX_SETTLED_LINES:]) or \
            "(no question has been settled yet)"
    lines = []
    for entry in session.get("qa") or []:
        question = (entry["question"] or "").strip().splitlines()[0]
        lines.append("Q%d: %s" % (entry["n"], question))
        where = ("%g-%gs" % tuple(entry["seconds"][:2]) if entry.get("seconds")
                 else "no seconds")
        lines.append("    -> %s from %s at %s [%s]"
                     % (entry["option"] or "no answer",
                        entry["unit"] or "no unit", where,
                        "verified" if entry["verified"] else "not double-checked"))
        for rejected in entry.get("rejected") or ():
            lines.append("    looked, not it: %s" % rejected)
    return "\n".join(lines) or "(no question has been settled yet)"


def render_chunk_memory(session, chunk_id, reader=None):
    """One reader's memory of its own chunk, as question/answer pairs."""
    entries = session["chunks"].get(chunk_id) or []
    if reader is not None:
        entries = [entry for entry in entries
                   if str(entry.get("author") or "") == str(reader)]
    entries = entries[-MAX_MEMORY_ENTRIES:]
    if not entries:
        return "(you have none)"
    if session.get("memory_view") == "index":
        return _compact_chunk_memory(chunk_id, entries)
    lines = []
    for entry in entries:
        what = entry.get("description") or entry.get("update") or "(nothing kept)"
        block = ["Q: %s" % (entry.get("question") or "")]
        if entry.get("retracted"):
            block.append("   W%d saw: %s  [superseded by %s - not current]"
                         % (chunk_id, what, entry.get("retracted_by") or "a later read"))
        else:
            where = (" [%g-%gs]" % tuple(entry["seconds"][:2])
                     if entry.get("seconds") else "")
            block.append("   W%d saw: %s%s" % (chunk_id, what, where))
            if entry.get("update") and entry["update"] != what:
                block.append("   kept: %s" % entry["update"])
        lines.append(block)
    if session.get("memory_view") == "push":
        return "\n".join(line for group in lines for line in group)
    clipped = [[_clip(line, max(60, (READER_MEMORY_CHARS - 60) // 3)) for line in group]
               for group in lines]
    kept = []
    for block in reversed(clipped):
        if kept and len("\n".join([line for group in [block] + kept for line in group])) \
                > READER_MEMORY_CHARS:
            break
        kept.insert(0, block)
    return "\n".join(line for group in kept for line in group)


def append_method_memory(prompt, session, roles):
    """Append reviewed cross-video method blocks, without changing off prompts."""
    blocks = (session.get("method_memory") or {}).get("prompt_blocks") or {}
    seen, selected = set(), []
    for role in roles or ():
        if role in seen:
            continue
        seen.add(role)
        block = str(blocks.get(role) or "")
        if block:
            selected.append(block)
    return prompt if not selected else prompt + "\n\n" + "\n\n".join(selected)


def _compact_chunk_memory(chunk_id, entries):
    """The reader's own chunk memory, bounded: the freshest note verbatim, the rest as one
    line each, dropped from the oldest end until the block fits."""
    last = entries[-1]
    note = last.get("description") or last.get("update") or "(nothing kept)"
    lines = ["your last note: %s" % _clip(note, READER_NOTE_CHARS)]
    earlier = []
    for entry in entries[:-1]:
        what = _clip(entry.get("description") or entry.get("update")
                     or "(nothing kept)", 40)
        where = ("%g-%gs" % tuple(entry["seconds"][:2]) if entry.get("seconds")
                 else "no address")
        mark = " [superseded]" if entry.get("retracted") else ""
        earlier.append("- %s | %s | %s -> %s%s"
                       % (where, entry.get("kind") or "observation",
                          _few_words(entry.get("question"), 5), what, mark))
    for line in reversed(earlier):           # keep the fresher one-liners first
        if len("\n".join(lines + [line])) > READER_MEMORY_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)



# **The global block's own label.** Distinct from the local header so a reader can tell the two layers
# apart, and so the audit can read the block off the prompt.
GLOBAL_MEMORY_HEADER = ("What earlier questions' reads found (video memory - evidence from "
                        "earlier questions on this same video, not this read):")
GLOBAL_NAMING_NOTE = ("If one of these notes informed your report, name it in your MEMORY_USED "
                      "line (for example `HOLDS {handle}`).")


def _note_span(entry):
    """The note's own seconds - the sighting first, the read's assigned span as its fallback."""
    span = entry.get("seconds") or entry.get("assigned")
    if not span or len(span) < 2:
        return None
    try:
        first, last = float(span[0]), float(span[-1])
    except (TypeError, ValueError):
        return None
    if last < first:
        first, last = last, first
    return [first, last]


def _span_overlap(a, b):
    if not a or not b:
        return 0.0
    return max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))


def _note_own_words(entry, limit=GLOBAL_NOTE_CHARS):
    """The note's own words - the sighting, then the kept update - clipped. **Not the answer letter**: rendering the option would bias the reader toward it."""
    text = " ".join(str(entry.get("description") or entry.get("update") or "").split())
    return text[:limit]


def _note_status(entry):
    """The note's status in words: a live status if it has one, else verified/unverified, never a
    filter on `verified`."""
    status = str(entry.get("status") or "").strip()
    if status:
        return status
    return "verified" if entry.get("verified") else "unverified"


def _global_handle(entry_id):
    """**The note's own id makes the alias unique per question**: a counter
    restarting at 1 in every reader's block would let two readers name different notes `G1`, and a
    review of `G1` could open the wrong one. The entry id is stable across the session, so the same
    note keeps one alias however many readers saw it, and different notes never collide."""
    return "%s%d" % (GLOBAL_HANDLE_PREFIX, int(entry_id))


def related_global_notes(session, question_number, chunk, question):
    """**The earlier questions' notes that bear on this reader's chunk and question**.

    `chunk` is the reader's `(chunk_id, first, last)`. A candidate is an entry of an **earlier
    question** (`0 <= question_number < the current one`, so a seeded note carries) that is citable,
    live (not retracted, not superseded, not merged), not this question's provisional write, from a
    **different chunk** than the reader's own (its local block already holds those, and deduplicating
    by the note's own id is that exclusion), and carrying a span.

    Returns the notes the block carries, plus the offered count, characters, truncation and whether a
    non-empty eligible set was starved. A zero therefore distinguishes *no related note* from *block
    too small* from *irrelevant retrieval*."""
    empty = {"notes": [], "offered": 0, "carried": 0, "characters": 0,
             "truncated": 0, "eligible": 0, "starved": False}
    if not session or session.get("memory_disabled"):
        return empty
    try:
        chunk_id, first, last = chunk[0], float(chunk[1]), float(chunk[2])
    except (TypeError, IndexError, ValueError):
        return empty
    window = [first, last]
    current = int(question_number or 0)
    wanted = _terms(question)
    candidates = []
    for cid in sorted(session.get("chunks") or {}):
        if _chunk_text(cid) == _chunk_text(chunk_id):
            continue
        for entry in session["chunks"][cid]:
            if entry.get("retracted") or entry.get("merged_into_tree"):
                continue
            if not entry.get("citable", True):
                continue
            if str(entry.get("status") or "") in ("superseded", "retracted"):
                continue
            if entry.get("provisional"):
                continue
            number = entry.get("question_number")
            if number is None or int(number) < 0 or int(number) >= current:
                continue
            span = _note_span(entry)
            if not span:
                continue
            words = _note_own_words(entry)
            if not words:
                continue
            overlap = _span_overlap(span, window)
            terms = sorted(wanted & _terms(words))
            if overlap <= 0 and len(terms) < GLOBAL_NEIGHBOUR_TERMS:
                continue
            candidates.append({
                "entry_id": entry.get("id"), "chunk": cid,
                "question_number": int(number),
                "question": entry.get("question") or "",
                "span": span, "status": _note_status(entry), "words": words,
                "relevance": {"overlap": round(overlap, 3), "terms": terms,
                              "score": round(overlap, 3) + len(terms)}})
    candidates.sort(key=lambda note: (note["relevance"]["overlap"] > 0,
                                      len(note["relevance"]["terms"]),
                                      note["question_number"],
                                      note["entry_id"] or 0), reverse=True)
    eligible = len(candidates)
    kept, characters, truncated = [], 0, 0
    note_budget = _global_note_budget()
    for note in candidates:
        handle = _global_handle(note["entry_id"])
        line = _global_note_line(handle, note)
        if len(kept) >= GLOBAL_NOTE_LIMIT:
            truncated += 1
            continue
        if kept and characters + len(line) + 1 > note_budget:
            truncated += 1
            continue
        if not kept and len(line) > note_budget:
            note = dict(note)
            note["words"] = note["words"][:max(0, note_budget - 40)]
            line = _global_note_line(handle, note)
        note = dict(note)
        note["handle"] = handle
        kept.append(note)
        characters += len(line) + 1
    starved = bool(candidates) and not kept
    return {"notes": kept, "offered": eligible, "carried": len(kept),
            "characters": characters, "truncated": truncated, "eligible": eligible,
            "starved": starved}


def _global_note_line(handle, note):
    """One note's line: its handle, its own span, its status and its own words."""
    span = note.get("span") or [0.0, 0.0]
    return "  %s %g-%gs [%s]: %s" % (handle, float(span[0]), float(span[1]),
                                      note.get("status") or "unverified",
                                      note.get("words") or "")


def render_global_notes(result):
    """The global block as the reader sees it, or `""` when there is nothing to show."""
    notes = (result or {}).get("notes") or []
    if not notes:
        return ""
    lines = [GLOBAL_MEMORY_HEADER,
             GLOBAL_NAMING_NOTE.format(handle=notes[0].get("handle") or "G1")]
    lines.extend(_global_note_line(note["handle"], note) for note in notes)
    text = "\n".join(lines) + "\n"
    if len(text) > READER_GLOBAL_MEMORY_CHARS:
        text = text[:READER_GLOBAL_MEMORY_CHARS]
        if not text.endswith("\n"):
            text = text.rstrip() + "\n"
    return text


def _global_note_budget():
    """The characters the note lines may take inside `READER_GLOBAL_MEMORY_CHARS`."""
    overhead = len(GLOBAL_MEMORY_HEADER) + len(GLOBAL_NAMING_NOTE) + 2   # the two newlines
    return max(0, READER_GLOBAL_MEMORY_CHARS - overhead)


def global_mentions(used_line, notes):
    """The global notes a reader named in its own `MEMORY_USED` line."""
    text = str(used_line or "")
    found = []
    for note in notes or ():
        handle = str(note.get("handle") or "")
        if handle and re.search(r"\b%s\b" % re.escape(handle), text, re.I):
            found.append({"handle": handle, "entry_id": note.get("entry_id"),
                          "chunk": note.get("chunk"),
                          "question_number": note.get("question_number"),
                          "words": note.get("words") or ""})
    return found


def named_global_note(unit, reports):
    """The global note a `G<k>` address names, by label, or `None`.

    The alias is minted per reader call, so the reports are its only registry. The label is the
    address (the same rule the `M`/`E` handles follow); position is never the fallback.
    """
    text = str(unit or "").strip().upper()
    if not text.startswith("G"):
        return None
    for report in reports or ():
        for item in report.get("global_notes_named") or []:
            if str(item.get("handle") or "").upper() == text:
                return item
    return None


def reader_global_named_notes(reports):
    """The global notes the readers named, as items a review can resolve by handle.

    The reader's `G<k>` namespace is its own, so these join the reviewable set under the same alias,
    and `_review_round` opens the frames the note stored. Deduplicated by handle and entry id.
    """
    out, seen = [], set()
    for report in reports or ():
        for named in report.get("global_notes_named") or []:
            key = (str(named.get("handle") or ""), named.get("entry_id"))
            if not key[0] or key in seen:
                continue
            seen.add(key)
            out.append(dict(named))
    return out


def _length_filler(n):
    """Neutral padding of exactly `n` characters, content-word free.

    The block-off arm withholds the reader's global block but must not be shorter by it, or the two
    arms differ in length as well as content. The filler is a run of em
    dashes, so it carries **no** content words and cannot inform the reader."""
    if n <= 0:
        return ""
    if n == 1:
        return "\n"
    return ("\u2014" * n)[:n - 1] + "\n"


def _match_length(text, target):
    """A block padded or clipped to exactly `target` characters, header included."""
    text = text or ""
    if target <= 0:
        return ""
    if len(text) > target:
        head = text[:target]
        if not head.endswith("\n"):
            # **Never cut a line into the text that follows** (the prompt's next paragraph is not
            # this block): trim back to the last word and close the line.
            cut = head.rfind(" ")
            head = (head[:cut] if cut > 0 else head).rstrip() + "\n"
        return head + _length_filler(max(0, target - len(head)))
    return text + _length_filler(target - len(text))


def _unrelated_global_notes(session, question_number, chunk, question):
    """Earlier notes that are **not** related: no span overlap and no shared content term."""
    empty = {"notes": [], "offered": 0, "carried": 0, "characters": 0,
             "truncated": 0, "eligible": 0, "starved": False}
    try:
        chunk_id, first, last = chunk[0], float(chunk[1]), float(chunk[2])
    except (TypeError, IndexError, ValueError):
        return empty
    window = [first, last]
    current = int(question_number or 0)
    wanted = _terms(question)
    candidates = []
    for cid in sorted(session.get("chunks") or {}):
        if _chunk_text(cid) == _chunk_text(chunk_id):
            continue
        for entry in session["chunks"][cid]:
            if entry.get("retracted") or entry.get("merged_into_tree"):
                continue
            if not entry.get("citable", True):
                continue
            if str(entry.get("status") or "") in ("superseded", "retracted"):
                continue
            if entry.get("provisional"):
                continue
            number = entry.get("question_number")
            if number is None or int(number) < 0 or int(number) >= current:
                continue
            span = _note_span(entry)
            words = _note_own_words(entry)
            if not span or not words:
                continue
            if _span_overlap(span, window) > 0 or (wanted & _terms(words)):
                continue
            candidates.append({"entry_id": entry.get("id"), "chunk": cid,
                               "question_number": int(number),
                               "question": entry.get("question") or "",
                               "span": span, "status": _note_status(entry), "words": words,
                               "relevance": {"overlap": 0.0, "terms": [], "score": 0.0}})
    candidates.sort(key=lambda note: (note["question_number"], note["entry_id"] or 0), reverse=True)
    kept = candidates[:GLOBAL_NOTE_LIMIT]
    for note in kept:
        note["handle"] = _global_handle(note["entry_id"])
    return {"notes": kept, "offered": len(candidates), "carried": len(kept),
            "characters": sum(len(_global_note_line(n["handle"], n)) + 1 for n in kept),
            "truncated": max(0, len(candidates) - len(kept)), "eligible": len(candidates),
            "starved": False}



EVIDENCE_KINDS = {"entity": ("who", "what", "where", "attribute"),
                  "event": ("action", "participant", "time"),
                  "mixed": ("who", "what", "where", "attribute",
                            "action", "participant", "time")}
FAMILY_TYPES = {"entity": ("entity recognition", "key information retrieval"),
                "mixed": ("1_plotqa", "2_needle", "3_ego", "4_count",
                          "5_order", "6_anomaly_reco", "7_topic_reasoning"),
                "event": ("event understanding", "temporal grounding", "summarization")}


def refusal_kind(why):
    """Which declared reason a record was refused for.

    An enum mismatch is a vocabulary effect and a phrase failure is a sighting effect, so the two
    are reported separately. The address reason is split further into three failures: a reply
    that named no field, a reader that declared `PLACE: NONE`, and a place named outside its own
    window."""
    text = str(why or "").lower()
    if "no place and no verdict" in text:
        return "no_reply_fields"
    if "declared no place" in text:
        return "place_none"
    if "named no place" in text:
        return "no_place_field"
    if "outside the read window" in text or "fits neither form" in text:
        return "place_outside"
    if "address" in text:
        return "no_address"          # kept: stored runs carry the old wording
    if "sighting" in text or "words" in text:
        return "phrase"
    if "kind" in text:
        return "kind"
    if "family" in text:
        return "family"
    return "other"


def typed_record(report, question, family, kind=None):
    """`(record, why)` - the typed record a read must leave, or the reason it cannot."""
    if family not in EVIDENCE_KINDS:
        return None, ("the question's family %r is not declared for this rung"
                      % (family or ""))
    first = float(report.get("first") or 0.0)
    last = float(report.get("last") or 0.0)
    spans = place_spans(report.get("place") or "", first, last)
    if not spans:
        place_field = report.get("place_field", report.get("place"))
        if not place_field and not report.get("can_answer_field"):
            return None, "the reply named no place and no verdict"
        if report.get("place_none"):
            return None, "the reader declared no place (PLACE: NONE)"
        if not place_field:
            return None, "the reply named no place"
        neither = bare_range_that_fits_neither_form(report.get("place") or "", first, last)
        if neither:
            return None, ("the place %r fits neither form: it is not inside this %g-second "
                          "chunk as seconds, and its start is not an offset inside it"
                          % (neither[0], neither[1]))
        return None, ("the place %r lies outside the read window %g-%g"
                      % (report.get("place"), first, last))
    address = [min(span[0] for span in spans), max(span[1] for span in spans)]
    quoted = str(report.get("quote") or "").strip().strip('"').strip()
    words = quoted.split()
    # **A withdrawn answer still leaves an observation.** A reader that says
    # `CAN_ANSWER: NO` and `QUOTE: NONE` can still describe what it saw in
    # `FRAMES`. Keep that description by using its own opening words as the
    # typed observation phrase. A claimed answer with `QUOTE: NONE` stays a
    # refusal because there is no reader-authored answer quote to preserve.
    unclaimed = not (report.get("claimed") or report.get("points")
                     or _letter_of(report.get("answer") or ""))
    if unclaimed and _normalise(quoted) in ("", "none"):
        words = str(report.get("frames_text") or "").split()[:8]
    if len(words) < 3:
        return None, "the phrase must be three to eight words, not %d" % len(words)
    if len(words) > QUOTE_MAX_WORDS:
        return None, ("the phrase is the whole description, not a phrase (%d words)"
                      % len(words))
    phrase = " ".join(words[:8])
    if not _quote_grounded(phrase, report.get("frames_text") or ""):
        return None, "the phrase is not in the reader's sighting"
    if is_option_echo(phrase, options_of(question)):
        return None, ("the phrase is the option said back, not the reader's sighting")
    kind = str(kind or "").strip().lower()
    if kind not in EVIDENCE_KINDS[family]:
        return None, ("the kind %r is not in the %s enum (%s)"
                      % (kind, family, ", ".join(EVIDENCE_KINDS[family])))
    return {"address": address, "phrase": phrase, "evidence_kind": kind,
            "question_family": family, "terms": sorted(_terms(question)),
            "source": "W%s" % report.get("chunk")}, ""


def attach_typed_record(session, chunk_id, typed):
    """Put a validated record's fields on the chunk's newest entry.

    The entry is the one the read just wrote; the typed fields ride on it so the stored
    memory carries the fine **address** the offline batch found missing in all 736
    entries of the run it measured. Returns the entry, or `None` when there is none.
    """
    entries = session.get("chunks", {}).get(chunk_id) or []
    if not entries:
        return None
    entries[-1].update(typed)
    entry = entries[-1]
    if entry.get("placebo_span"):
        entry["address"] = list(entry["placebo_span"])
        return entry
    if entry.get("address"):
        tell_the_tree(session, entry, span=entry["address"], by=entry.get("author") or "read",
                      why="the reader's own address")
        return move_for_placebo(session, chunk_id, entry, entry["address"])
    if entry.get("placebo_pending"):
        entry["placebo_pending"] = False
        stats = session.setdefault("placebo", {"moved": 0, "refused": 0, "occupied": 0,
                                              "unmoved_citable": 0, "moves": []})
        if entry.get("citable", True):
            stats["unmoved_citable"] = int(stats.get("unmoved_citable") or 0) + 1
    return entry



SEED_AUTHOR = "seed"
SEED_QUESTION = "an earlier question about this video"


def move_for_placebo(session, chunk_id, entry, seconds=None):
    """**The within-video placebo**: the record's address - and its note - move
    to a gold-disjoint window in a different chunk of the same video.

    The transform is the runner's, because only the runner may look at gold; this function does
    the mechanical half, and it moves **both** halves together: the note's
    rendering key (`chunk_map`), the note's own span, and the record's `address`, so the pointer
    the agent reads and the pointer the harness would check are the same placebo address. A
    refused move (no candidate) leaves the record where it was and is counted, and the refusal is
    visible in the run's record rather than averaged silently into the arm.

    **A move never lands on a chunk that already carries a note**. The
    destination's occupancy is handed to the runner's picker, which owns the choice of chunk; an
    occupied candidate is counted and skipped, never overwritten."""
    move = session.get("placebo_move")
    if not callable(move):
        return entry
    stats = session.setdefault("placebo", {"moved": 0, "refused": 0, "occupied": 0,
                                           "unmoved_citable": 0, "moves": []})
    address = seconds or entry.get("address") or None
    if address is None:
        # **Nothing to move yet** (a write whose address arrives with its typed record): deferring
        # costs nothing and counting a refusal here would count one write twice.
        return entry
    if entry.get("placebo_tried"):
        return entry
    entry["placebo_tried"] = True
    occupied = set(session.get("chunk_map") or {}) - {chunk_id}
    placed = move(address, occupied)
    if not placed:
        stats["refused"] += 1
        if entry.get("citable", True):
            # **The count that voids the arm**: a citable write whose pointer stayed real is a
            # real pointer inside the placebo's store, and B minus C then compares two stores
            # rather than two addresses.
            stats["unmoved_citable"] += 1
        return entry
    new_chunk = int(placed["chunk"])
    span = [float(placed["seconds"][0]), float(placed["seconds"][1])]
    entry["placebo_from"] = {"chunk": chunk_id, "address": entry.get("address")}
    entry["placebo_span"] = span
    entry["address"] = span
    if new_chunk != chunk_id:
        line = (session.get("chunk_map") or {}).pop(chunk_id, "")
        note = (session.get("notes") or {}).pop(chunk_id, None)
        if line:
            session["chunk_map"][new_chunk] = line
            note = dict(note or {"chunk": new_chunk, "sighting": line})
            note["chunk"] = new_chunk
            note["span"] = span
            session.setdefault("notes", {})[new_chunk] = note
    stats["moved"] += 1
    stats["moves"].append({"from_chunk": chunk_id, "to_chunk": new_chunk,
                           "from": entry.get("placebo_from", {}).get("address"),
                           "to": span, "entry_id": entry.get("id")})
    return entry


def validate_no_usable(reason, shown):
    """`(ok, why)` - a `NO_USABLE` must name every handle the index showed."""
    if not shown:
        return True, ""
    missing = [entry["handle"] for entry in shown
               if entry["handle"].lower() not in str(reason or "").lower()]
    if missing:
        return False, ("the refusal does not name %s, which the index showed"
                       % ", ".join(missing))
    return True, ""


def render_ledger(reports):
    if not reports:
        return "(no chunk has been read yet)"
    lines = []
    for report in reports:
        lines.append("chunk %d (seconds %g-%gs), reader W%d: FRAMES %s | PLACE %s | "
                     "CAN_ANSWER %s | ANSWER %s | QUOTE %s"
                     % (report["chunk"], report["first"], report["last"], report["chunk"],
                        report["frames_text"] or "(described nothing)",
                        report["place"] or "NONE",
                        "YES" if report["claimed"] else "NO",
                        report["answer"] or "NONE", report["quote"] or "NONE"))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# parsing, and the rules that make a reply safe to use

def _field_line(text, name):
    """The value of the first line that starts with `name:`, or **`None` when no such
    line is there at all**.

    **The colon is followed by spaces and tabs only.** A `\\s*` there let an empty
    field swallow the next line - a reader that reported nothing had `FRAMES:`
    parse as `PLACE: 10-20` - which corrupts the report and the memory built on it."""
    pattern = re.compile(
        r"(?:^|\|)[ \t]*(?:[-*#]+[ \t]*)?(?:\*\*|__)?%s(?:\*\*|__)?[ \t]*:"
        r"[ \t]*(.*?)(?=[ \t]*\|[ \t]*(?:%s)[ \t]*:|[ \t]*$)"
        % (name, "|".join(FIELD_NAMES)),
        re.I | re.M)
    found = pattern.search(text or "")
    return (found.group(1).strip() if found else None)


def _field(text, name):
    """The first line that starts with `name:` - prose after the block is ignored."""
    return _field_line(text, name) or ""


def spread(ids, limit):
    """`limit` of `ids`, **spread across the list** rather than taken from its front."""
    ids = list(ids)
    if len(ids) <= limit:
        return ids
    step = len(ids) / float(limit)
    return [ids[int(index * step)] for index in range(limit)]


def clamp_reads(ids, limit, why=""):
    """`(kept, clamped)` - **one rule for every read list**.

    The taxonomy found three unbounded doors into the read path, and only one of them
    was closed: the plan-less fallback read *every chunk* (items 5170/4312/738, 58/58/53
    reads, the cap spent, no decision), and the escalation's `READ <ids>` path was not
    bounded at all (item 4498 was told `READ 47, 48` and read six chunks) while `ALL`
    carried `MAX_ESCALATION_READS`. One helper, used at every door, with the clamp
    recorded where it fires - so a row can be read for what the bound cost it."""
    ids = list(dict.fromkeys(ids))
    if len(ids) <= limit:
        return ids, None
    return ids[:limit], {"asked": len(ids), "read": limit,
                         "why": why or "the read list is bounded at %d" % limit}


def parse_plan(text, chunks):
    """`{think, about, picks, draft, plan_source}` or `None` when there is no plan.

    `DRAFT` is returned for the log and **never** as the answer. An unusable `PICK`
    is not an error: it becomes the all-chunks fallback, recorded as `partition`.
    """
    about = _field(text, "ABOUT")
    if not about:
        return None
    raw = _field(text, "PICK")
    valid = {cid for cid, _, _ in chunks}
    if re.search(r"\bALL\b", raw or "", re.I):
        picks = spread([cid for cid, _, _ in chunks], MAX_PICK)
        source = "plan-all-clamped"
    else:
        picks = [int(n) for n in re.findall(r"\d+", raw or "") if int(n) in valid]
        picks = list(dict.fromkeys(picks))[:MAX_PICK]
        source = "plan" if picks else "partition-clamped"
    if not picks:
        picks = spread([cid for cid, _, _ in chunks], MAX_PICK)
    return {"think": _field(text, "THINK"), "about": about, "picks": picks,
            "draft": _field(text, "DRAFT"), "plan_source": source}


def seconds_of(text):
    """A second written as a number, `1302s`, or a clock `42:04` - or `None`."""
    text = (text or "").strip()
    clock = PLACE_CLOCK_ONE.fullmatch(text.replace(" ", ""))
    if clock:
        return float(_clock(*clock.group(1, 2, 3)))
    number = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s?", text, re.I)
    return float(number.group(1)) if number else None


CHECK_LINE = re.compile(r"^\s*CHECK\s*:\s*(SUPPORT|REFUTE|UNSURE)\b(.*)$", re.I | re.M)


def parse_check(text):
    """`{action, unit, second, quote, why}` from O's grounding reply, or `None`.

    `SUPPORT` carries the unit, the second and the phrase; `REFUTE` and `UNSURE` carry
    a reason and resolve to no unit, which the flow reads as **not verified**. A reply
    with no `CHECK:` line is not a verdict at all - the flow counts that as not
    verified too, never as agreement.
    """
    found = None
    first = None
    for candidate in CHECK_LINE.finditer(text or ""):
        action = candidate.group(1).lower()
        rest = candidate.group(2).strip().lstrip("|").strip()
        if action != "support":
            found = found or (action, rest, None)
            continue
        parsed = _parse_support(rest)
        if first is None:
            first = (action, rest, parsed)
        if parsed and not _placeholder(parsed):
            found = (action, rest, parsed)
            break
        found = found or (action, rest, parsed)
    if found is None:
        return None
    action, rest, check = found
    if action != "support":
        return {"action": action, "unit": "", "option": "", "second": None,
                "quote": "", "why": rest}
    if check is None:
        return None
    return check


CHECK_REREASK = ("\n\n**Your last reply copied this description back.** Write the option's "
                 "letter as one capital letter, the second the evidence is at as a "
                 "number, and the reader's own three to eight words in quotation marks "
                 "- your evidence, not this form, and with a bar between the fields.")


def _placeholder(check):
    """Whether a parsed support copied the prompt's grammar instead of the evidence."""
    option = str((check or {}).get("option") or "")
    quote = str((check or {}).get("quote") or "")
    marks = ("<option>", "<second>", "<words", "<one line")
    said = (option + " " + quote).lower()
    if any(mark in said for mark in marks):
        return True
    prose = ("option", "the second", "capital letter", "quotation marks",
             "the reader's own")
    return any(phrase in option.lower() for phrase in prose)


def _parse_support(rest):
    """One `SUPPORT` line's fields, or `None` when it has no usable shape."""
    fields = rest.split("|")
    unit = fields[0].strip().upper()
    rest_fields = fields[1:]
    option, second = "", None
    if rest_fields:
        if seconds_of(rest_fields[0]) is not None:
            second = seconds_of(rest_fields[0])
            rest_fields = rest_fields[1:]
        else:
            option = rest_fields[0].strip()
            rest_fields = rest_fields[1:]
            if rest_fields and seconds_of(rest_fields[0]) is not None:
                second = seconds_of(rest_fields[0])
                rest_fields = rest_fields[1:]
    tail = "|".join(rest_fields).strip()
    first_quote, last_quote = tail.find('"'), tail.rfind('"')
    quote = (tail[first_quote + 1:last_quote] if 0 <= first_quote < last_quote
             else tail).strip()
    return {"action": "support", "unit": unit, "option": option,
            "second": second, "quote": quote, "why": ""}


RECOVER_LINE = re.compile(r"^\s*(LOOK|MEMORY|READ|RECALL|REPLAN|ALL)\b\s*:?(.*)$",
                          re.I | re.M)


def parse_select(text):
    """`{action, handle, option, quote, why}` - what O does after a failed check."""
    found = RECOVER_LINE.search(text or "")
    if not found:
        return {"action": "all", "ids": [], "handle": "", "option": "", "quote": "",
                "why": "the reply named no recovery, so the all-chunks read runs"}
    action = found.group(1).upper()
    rest = found.group(2).strip().lstrip("|").strip()
    if action == "ALL":
        return {"action": "all", "ids": [], "handle": "", "option": "", "quote": "",
                "why": ""}
    if action in ("READ", "REPLAN"):
        ids = [int(n) for n in re.findall(r"\d+", rest)][:MAX_ESCALATION_READS]
        return {"action": "read", "ids": ids, "handle": "", "option": "",
                "quote": "", "why": ""}
    handle = re.search(r"\bM\s*(\d+)\b", rest, re.I)
    if action in ("LOOK", "MEMORY"):
        return {"action": "look" if handle else "all", "ids": [],
                "handle": ("M%s" % handle.group(1)) if handle else "",
                "option": "", "quote": "", "why": ""}
    if action == "RECALL":
        return {"action": "recall" if handle else "all", "ids": [],
                "handle": ("M%s" % handle.group(1)) if handle else "",
                "option": "", "quote": "", "why": ""}
    return {"action": "all", "ids": [], "handle": "", "option": "", "quote": "",
            "why": "%s named no handle" % action}


def parse_decide(text):
    """`{think, action, option, handle, quote}` from P2.

    The answer form **names the memory entry it uses** - `ANSWER: <option> | M<k>
    | "<quote>"` - which is the spec's §2 rule ("it must name the memory entry it
    cites"). The name is read anywhere in the answer line, and the quote is read
    **first quote mark to last**, so a bar or a `|` inside the quote cannot break
    the split. A reply with no `M<k>` still parses, with an empty handle, and the
    flow refuses it as an unverifiable answer.
    """
    think = _field(text, "THINK")
    answer = _field(text, "ANSWER")
    if answer:
        head, _, tail = answer.partition("|")
        option = head.strip()
        handle = ""
        for part in tail.split("|"):
            named = re.fullmatch(r"M\s*(\d+)", part.strip(), re.I)
            if named:
                handle = "M%s" % named.group(1)
                break
        first_quote, last_quote = tail.find('"'), tail.rfind('"')
        quote = (tail[first_quote + 1:last_quote]
                 if 0 <= first_quote < last_quote else "")
        return {"think": think, "action": "answer", "option": option,
                "handle": handle, "quote": quote.strip()}
    if re.search(r"(?im)^\s*COARSE_PLAN\s*(?:[|:].*)?$", text or ""):
        return {"think": think, "action": "coarse_plan", "option": "",
                "handle": "", "quote": ""}
    if re.search(r"(?im)^\s*PLAN\s*(?:[|:].*)?$", text or ""):
        return {"think": think, "action": "plan", "option": "", "handle": "",
                "quote": ""}
    refusal = re.search(r"(?im)^\s*NO_USABLE\s*(?:[|:]\s*(.*))?\s*$", text or "")
    if refusal:
        return {"think": think, "action": "no_usable", "option": "", "handle": "",
                "quote": "", "reason": (refusal.group(1) or "").strip()}
    return None


def parse_sightings(text, limit=MAX_SIGHTINGS):
    """**The reader's distinct findings**: the reply may carry 2-3
    question-bearing sightings, each `PLACE t0-t1 | ANSWER <letter> | QUOTE "words"`. Each is a distinct
    moment the reader found in the window it read - the several local findings the team layer needs to
    aggregate, where one-read-per-chunk gave it only one. A reversed span is dropped, never mapped in."""
    out = []
    for line in (text or "").splitlines():
        found = SIGHTING_LINE.search(line)
        if not found:
            continue
        first, last = float(found.group(1)), float(found.group(2))
        if last < first:
            continue
        out.append({"span": [first, last], "answer": found.group(3).upper(),
                    "quote": found.group(4).strip().strip('"').strip()})
        if len(out) >= limit:
            break
    return out


def parse_reader(text):
    """The reader's report. `PLACE` is kept, and so are the two memory lines."""
    frames_text = _field(text, "FRAMES")
    verdict = _field(text, "CAN_ANSWER").upper()
    answer = _field(text, "ANSWER")
    if answer.upper() in ("NONE", "N/A", ""):
        answer = ""
    quote = _field(text, "QUOTE")
    if quote.upper() in ("NONE", "N/A"):
        quote = ""
    place_field = _field_line(text, "PLACE")
    place_none = bool(place_field) and place_field.upper().startswith(("NONE", "N/A"))
    place = "" if place_none else (place_field or "")
    claim = {}
    missing_claim = []
    for field_name, key in (("CLAIM_SUBJECT", "subject"),
                            ("CLAIM_PREDICATE", "predicate"),
                            ("CLAIM_VALUE", "value"),
                            ("CLAIM_UNIT", "unit")):
        value = _field(text, field_name).strip()
        if value.upper() in ("NONE", "N/A"):
            value = ""
        claim[key] = value
        if key != "unit" and not value:
            missing_claim.append(field_name)
    claim_error = ("missing " + ", ".join(missing_claim)
                   if missing_claim else "")
    return {"think": _field(text, "THINK"), "frames_text": frames_text,
            "place": place, "place_field": place_field, "place_none": place_none,
            "claimed": verdict.startswith("Y") and bool(answer),
            "points": verdict.startswith("P") and bool(answer),
            "can_answer_field": verdict,
            "answer": answer, "quote": quote,
            "kind": _field(text, "KIND"),
            "memory_used": _field(text, "MEMORY_USED"),
            "memory_update": _field(text, "MEMORY_UPDATE"),
            "claim": claim if not missing_claim else None,
            "claim_error": claim_error}


def normalize_reader_claim(record, video):
    """Normalize the reader's structured claim without inferring missing fields."""
    claim = (record or {}).get("claim")
    if not isinstance(claim, dict):
        return {"claim": None,
                "claim_error": (record or {}).get("claim_error")
                or "missing structured claim"}
    try:
        normalized = B.normalize_claim(
            video, claim.get("subject"), claim.get("predicate"),
            claim.get("value"), claim.get("unit"))
    except ValueError as error:
        return {"claim": None, "claim_error": str(error)}
    return {"claim": normalized, "claim_error": ""}


def parsable_reader(record):
    """Did this reader reply actually report?"""
    return bool((record.get("frames_text") or "").strip()
                or (record.get("can_answer_field") or "").strip())


def clock_chunks(question, chunks):
    """**The chunks the question's own clock names**."""
    if not chunks:
        return set()
    duration = max(float(chunk[2]) for chunk in chunks)
    named = set()
    for first, last in M.time_references(question or "", duration):
        for chunk_id, cfirst, clast in chunks:
            if min(float(clast), last) - max(float(cfirst), first) >= 1.0:
                named.add(chunk_id)
    return named


def read_marked_menu(chunks, reports, question=""):
    """The chunk menu with each chunk's read state on it - where O chooses.

    The read state existed in the selection's state and in the failure history, and the
    model ignored both (question 1601 asked again for the chunks it had just read), so
    it now sits on the menu line itself, with the two lists spelled out."""
    named = clock_chunks(question, chunks)
    state = {}
    for report in reports or ():
        entry = state.setdefault(report["chunk"], {"density": 0.0, "parsable": False})
        entry["density"] = max(entry["density"], float(report.get("fps") or 0.0))
        entry["parsable"] = entry["parsable"] or bool(report.get("parsable"))
    parts, read, unread = [], [], []
    for chunk_id, first, last in chunks:
        info = state.get(chunk_id)
        if info and info["density"]:
            # **One definition of "read"**: a read whose reader never reported is not
            # closed - it still has its one second chance - and the mark says so.
            if re_read_eligible(reports, chunk_id, info["density"]):
                mark = ("[read %.1f, unreadable - one re-read left]" % info["density"])
            else:
                mark = "[read %.1f]" % info["density"]
            read.append(chunk_id)
        else:
            mark = "[unread]"
            unread.append(chunk_id)
        if chunk_id in named:
            mark += " [your question names this]"
        parts.append("%d: %g-%gs %s" % (chunk_id, first, last, mark))
    return "\n".join([
        " | ".join(parts),
        "already read: %s" % (", ".join(str(cid) for cid in read) or "none"),
        "not read: %s" % (", ".join(str(cid) for cid in unread) or "none")])


def gate_check(check, reports, options=None):
    """`(ok, why)` - may this support count as a verification?

    A support counts only when it is **grounded in what the reader said it saw**:

    * the unit is one of *this question's* reports;
    * the second is a number inside that unit's window;
    * the phrase is at least two words and appears in that unit's `FRAMES`/`PLACE`
      description - **not** in its `QUOTE` field, which is where the measured option
      echo lived;
    * the phrase is **not** a substring of any option: a reader that copies an option
      into its own description is still not evidence.

    `REFUTE`, `UNSURE`, a reply with no `CHECK:` line and a model error all come back
    as not verified - the flow must never read silence as agreement.
    """
    if not check:
        return False, "the check gave no CHECK line, so there is nothing verified"
    if check["action"] != "support":
        return False, "the check was %s: %s" % (check["action"],
                                                check.get("why") or "no reason given")
    unit = str(check.get("unit") or "")
    if not (unit.upper().startswith("W") and unit[1:].isdigit()):
        return False, "a support must name a reader as W<k>, not %r" % unit
    chunk = int(unit[1:])
    mine = [r for r in reports if r["chunk"] == chunk]
    if not mine:
        return False, "chunk %s was not read, so it cannot be verified" % unit
    # **Every report of that chunk counts**: a chunk the selective pass read and the
    # all pass read again has two, and the evidence may be in either.
    report = mine[0]
    option = str(check.get("option") or "").strip()
    if not option:
        # v2 fell back to the reader's own `ANSWER` here; v3 requires O's option, or the
        # reader's option echo walks back in through the side door.
        return False, "a support must name the option it grounds"
    if "<" in option or ">" in option:
        return False, "the option is a placeholder, not the option O grounds"
    if "option" in option.lower():
        return False, "the option field is the prompt's description, not a letter"
    letters = [chr(ord("A") + index) for index in range(len(options or ()))]
    if not letters and not re.match(r"^\s*\(?[A-H]\)?", option):
        return False, ("no options were passed to the gate, so the option must start "
                       "with a capital letter - %r does not" % option)
    if letters and _letter_of(option) not in letters:
        return False, ("the option %r is not one of this question's options (%s)"
                       % (option, ", ".join(letters)))
    second = check.get("second")
    if second is not None and not any(float(r["first"]) <= float(second)
                                      <= float(r["last"]) for r in mine):
        return False, ("the stated second %s is not inside chunk %s's window "
                       "%g-%g" % (second, unit, report["first"], report["last"]))
    quote = check.get("quote") or ""
    if len(quote.split()) < 2:
        return False, "a support must quote at least two words"
    description = _normalise(" ".join(r.get("frames_text") or "" for r in mine))
    if not _quote_grounded(quote, " ".join(r.get("frames_text") or "" for r in mine)):
        return False, ("the phrase is not in chunk %s's description of what it saw"
                       % unit)
    if is_option_echo(quote, options):
        return False, "the phrase is the option said back, not evidence"
    return True, "verified against chunk %s" % unit


def verified_bucket(check):
    """Which verified bucket a support falls in."""
    if not check or check.get("action") != "support":
        return ""
    return "verified" if check.get("second") is not None else "verified_no_second"


def gate_write(text, reports=(), options=(), existing=(), min_words=2):
    """`(kind, why)` for a note an agent wants to keep - kind is one of
    `observation`, `method`, `repeat`, `refused`.

    * `refused` - under `min_words`, or an option said back: never stored.
    * `repeat` - a near-duplicate of the chunk's existing text: counted, not stored.
    * `observation` - >= 2 of its words appear in a `FRAMES`/`PLACE` report of this
      session: stored and citable.
    * `method` - a habit that no report supports: stored, **never citable**, rendered
      in its own "how to read" section. Habits survive without becoming evidence."""
    said = (text or "").strip()
    if len(said.split()) < min_words:
        return "refused", "a note under %d words is not a note" % min_words
    if is_option_echo(said, options):
        return "refused", "the note is the option said back, not an observation"
    normalised = _normalise(said)
    for other in existing or ():
        other_normalised = _normalise(other)
        if other_normalised and (normalised in other_normalised
                                 or other_normalised in normalised):
            return "repeat", "the chunk's memory already holds this"
    seen = set()
    for report in reports or ():
        seen |= _terms((report.get("frames_text") or "") + " "
                       + (report.get("place") or ""))
    if len(_terms(said) & seen) >= 2:
        return "observation", "traceable to what a reader said it saw"
    return "method", "no report supports this, so it is kept as a method note"


def _offered_item(unit, items):
    """The offered leaf an `E<k>` address names, **by label, never by position** - the same rule
    `memory_by_handle` already carries: a renderer that shows
    addressed records first and drops the tail can leave a handle whose position is not its
    number, and resolving `E3` as `items[2]` would ground against another leaf's id. The label is
    the address; position is only the fallback for a record written before handles were carried."""
    unit = str(unit or "")
    for item in items or ():
        if str((item or {}).get("handle") or "") == unit:
            return item
    index = int(unit[1:]) - 1 if unit[1:].isdigit() else -1
    items = items or []
    return items[index] if 0 <= index < len(items) else None


def _check_backed(unit, row, standing, option=""):
    """**Does a check stand behind this unit, for THIS option?**.

    Two ways: a `VERIFY` that held that `M` (recorded in `row["held"]`), or a `CHECK` that grounded
    this unit this question (`standing`). Both are frame-bearing - a words-only `LOOK` is not a check
    and does not count - which is what keeps the evidence **useful rather than decorative**: the
    check can refute, and its verdict was reached from the footage.

    **And the option must match**: evidence that supports
    option B does not make a submitted C grounded. Without this rule the citation is a formality -
    any check behind any option would license any answer."""
    name = str(unit or "")
    if not name:
        return False
    wanted = _letter_of(option or "")
    if name in (row.get("held") or {}):
        held = row["held"][name] or {}
        held_option = _letter_of((held.get("check") or {}).get("option") or held.get("option") or "")
        return (not wanted) or held_option == wanted
    if standing and str((standing or {}).get("unit") or "") == name:
        standing_option = _letter_of((standing or {}).get("option") or "")
        return (not wanted) or standing_option == wanted
    return False


_OVERRIDE_STOPWORDS = frozenset((
    "the a an is are was were be been being of to in on at it its this that these those and or "
    "but for with as by from not no yes i we they he she them their our your his her us me you "
    "because so if then than over under after before there here has have had do does did can "
    "could will would should may might must my more most other some such only own same too very "
    "just about into while during").split())


def gate_commit(commit, reports, memory=None, about="", options=None, evidence=None):
    """`(ok, why)` - the quote must appear in the unit it cites, or in its memory."""
    if not commit:
        return False, "a commit must be FINAL <unit> | <option> | \"<quote>\""
    unit, quote = commit["unit"], (commit["quote"] or "").lower()
    if not quote:
        return False, "a commit without quoted words is a claim, not a citation"
    if len(quote.split()) < 2 or len(quote) < 6:
        return False, ("a citation must quote at least two words that carry the "
                       "answer")
    if unit.startswith("W") and unit[1:].isdigit():
        # **Any** report of that chunk from this question counts: a chunk the
        # selective pass read and the all pass read again has two, and a quote
        # from either is inside that chunk's report.
        #
        # **The description, not the reader's `QUOTE` field.** Round 3 widened this to
        # include `QUOTE` because the measured commits cited it; the v2 review named
        # that widening as the laundering channel - the `QUOTE` is exactly where the
        # sample's option echo lived. The evidence is what the reader says it *saw*.
        # **Evidence is what the reader says it saw**, and the reader's own QUOTE line
        # stays out of the gate: that field is where the measured option echo lived, and
        # this rule was reviewed and tested before (the v2 round). The memory-index run
        # shows the cost of it - 9 of 25 defaulted submissions quoted text that lived only
        # there - so reopening it is a decision for the reviewers, not a unilateral edit.
        said = [(r["frames_text"] or "") + " " + (r.get("place") or "")
                for r in reports if r["chunk"] == int(unit[1:])]
        if not said:
            return False, "chunk %s was not read, so it cannot be cited" % unit
        text = " ".join(said)
        if not _quote_grounded(quote, text):
            return False, "the quote is not in chunk %s's report" % unit
        if is_option_echo(quote, options):
            return False, "the quote is the option said back, not evidence"
        return True, "cited chunk %s" % unit
    if unit == "S":
        if quote not in (about or "").lower():
            return False, "the quote is not in your own stored look"
        if is_option_echo(quote, options):
            return False, "the quote is the option said back, not evidence"
        return True, "cited its own look"
    if unit.startswith("G") and unit[1:].isdigit():
        note = named_global_note(unit, reports)
        if note is None:
            return False, "no reader named global note %s" % unit
        text = str(note.get("words") or "").lower()
        if not _quote_grounded(quote, text):
            return False, "the quote is not in global note %s" % unit
        if is_option_echo(quote, options):
            return False, "the quote is the option said back, not evidence"
        return True, "cited global note %s" % unit
    if unit[:1] in ("M", "T") and unit[1:].isdigit():
        entries = memory or []
        item = memory_by_handle(entries, unit)
        if item is None:
            index = int(unit[1:]) - 1
            item = entries[index] if 0 <= index < len(entries) else None
        if item is None:
            return False, "no memory entry %s was shown" % unit
        if unit.startswith("M") and item.get("citation_allowed") is not True:
            return False, "memory entry %s has no support review for its answer" % unit
        text = ((item.get("description") or "") + " " + (item.get("update") or "")).lower()
        if not _quote_grounded(quote, text):
            return False, "the quote is not in memory entry %s" % unit
        if is_option_echo(quote, options):
            return False, "the quote is the option said back, not evidence"
        return True, "cited memory entry %s" % unit
    if unit.startswith("E") and unit[1:].isdigit():
        item = _offered_item(unit, evidence or [])
        if item is None:
            return False, "no earlier evidence %s was offered at the commit" % unit
        if item.get("answer_support") is not True \
                or item.get("citation_allowed") is not True \
                or item.get("review_status") != "accepted":
            return False, "earlier evidence %s has no support review for its answer" % unit
        words = str(item.get("words") or "")
        if quote not in words.lower():
            return False, "the quote is not in the shown words of %s" % unit
        if is_option_echo(quote, options):
            return False, "the quote is the option said back, not evidence"
        return True, "cited earlier evidence %s" % unit
    return False, "cite a reader (W<k>), your own look (S), or a memory entry (M<k>)"


# --------------------------------------------------------------------------
# the flow

def _read_chunks(ask, session, chunks, picks, question, fps, role_prefix, log,
                 frames_of=None, options=(), question_number=None, notes=None, windows=None):
    """Read the named chunks at `fps`; every read updates that chunk's memory.

    `frames_of(label, first, last, rate, n)` is the runner's seam: it returns the `n`
    frames to attach, spread over `[first, last)` at `rate` frames a second, where
    `label` is the chunk id a reader owns or the string `"coarse"` for the overview
    O takes itself. **The reviewer's warning is why it exists**: with a placeholder
    list, a runner that attaches the wrong window - or no frames at all - still scores
    and passes the tests. The real runner must pass this in, and the tests do.

    `rate` is the rate the frames really have, `n / (last - first)` - the nominal
    `fps` for a whole chunk, smaller when the cap shortens the window.
    """
    by_id = {cid: (first, last) for cid, first, last in chunks}
    if windows:
        by_id.update({cid: (float(first), float(last))
                      for cid, (first, last) in windows.items()})
    reports = []
    for chunk_id in picks:
        first, last = by_id[chunk_id]
        n = frames_for(first, last, fps)
        rate = n / (last - first) if last > first else fps
        images = (frames_of(chunk_id, first, last, rate, n) if frames_of
                  else [None] * n)
        reader_memory_block = render_chunk_memory(
            session, chunk_id, reader="W%d" % int(chunk_id))
        carried_local_memory = bool(reader_memory_block and reader_memory_block.strip()
                                    and "you have none" not in reader_memory_block.lower())
        global_mode = str(session.get("global_block") or "none")
        if global_mode == "none":
            global_result = {"notes": [], "offered": 0, "carried": 0, "characters": 0,
                             "truncated": 0, "eligible": 0, "starved": False}
        else:
            global_result = related_global_notes(
                session, question_number, (chunk_id, first, last), question)
        treatment_block = render_global_notes(global_result)
        rendered_notes = []
        if global_mode == "on":
            global_text, rendered_notes = treatment_block, list(global_result["notes"])
        elif global_mode == "off":
            global_text = _length_filler(len(treatment_block))
        elif global_mode == "placebo":
            placebo = _unrelated_global_notes(
                session, question_number, (chunk_id, first, last), question)
            global_text = _match_length(render_global_notes(placebo), len(treatment_block))
            rendered_notes = list(placebo["notes"])
        else:
            global_text = ""
        prompt = READER_PROMPT.format(
            cid=chunk_id, first=first, last=last, n=n, every=1.0 / fps,
            frames_words=FRAMES_MAX_WORDS, global_memory=global_text,
            question=question, memory=reader_memory_block)
        prompt = append_method_memory(prompt, session, ["read"])
        note = (notes or {}).get(chunk_id)
        if note:
            prompt = prompt + "\n" + note
        role = "%s-%d" % (role_prefix, chunk_id)
        reply = ask(prompt, images, role)
        record = parse_reader(reply)
        log.append({"role": role, "prompt": prompt, "reply": reply, "frames": n,
                    "finish_reason": getattr(ask, "last_finish_reason", None)})
        family = session.get("family") or ""
        _, first_why = typed_record({"first": first, "last": last, **record},
                                    question, family, kind=record.get("kind"))
        reread_field = False
        retry_fields = []
        retry_cut = False
        if needs_a_second_ask(record, first_why, family, window=(first, last)):
            reread_field = True
            claimed_first = bool(record.get("claimed") or record.get("points"))
            retry_prompt = prompt + reader_rereask(record, (first, last))
            retry = ask(retry_prompt, images, role + "-retry")
            log.append({"role": role + "-retry", "prompt": retry_prompt, "reply": retry,
                        "frames": n,
                        "finish_reason": getattr(ask, "last_finish_reason", None)})
            second = parse_reader(retry)
            retry_cut = getattr(ask, "last_finish_reason", None) == "length"
            if carries_a_field(second) and not retry_cut:
                retry_fields = sorted(key for key, value in second.items()
                                      if value and record.get(key) != value)
                record, reply = merge_retry(record, second), retry
            reread_flip = claimed_first and not (record.get("claimed")
                                                 or record.get("points"))
        else:
            reread_flip = False
        record.update(normalize_reader_claim(
            record, session.get("video") or ""))
        global_named = global_mentions(
            record.get("memory_used") if isinstance(record, dict) else "", rendered_notes)
        report = {"chunk": chunk_id, "question": question, "first": first,
                  "last": last, "fps": fps, "rate": rate, "frames": n,
                  "pass": role_prefix, "parsable": parsable_reader(record),
                  "reread_field": reread_field, "reread_flip": reread_flip,
                  "retry_fields": retry_fields, "retry_cut": bool(retry_cut),
                  "question_number": question_number,
                  "earlier_reads": len([r for r in session.get("reads") or ()
                                        if r["chunk"] == chunk_id
                                        and r.get("question_number") is not None
                                        and r.get("question_number")
                                        != question_number]),
                  "carried_local_memory": carried_local_memory,
                  "global_block": global_mode,
                  # **Real memory** only when the `on` arm showed related notes; a placebo block is not
                  # memory, and a control's opportunity is recorded separately below.
                  "carried_global_memory": bool(global_mode == "on" and rendered_notes),
                  # The **opportunity**: what the related retrieval would have offered, in every arm, so
                  # the denominator is arm-comparable. `global_notes_rendered`/`global_rendered_ids` are
                  # what the reader actually saw.
                  "global_notes_offered": global_result["offered"],
                  "global_opportunity_carried": global_result["carried"],
                  "global_notes_truncated": global_result["truncated"],
                  "global_opportunity_ids": [note.get("entry_id")
                                              for note in global_result["notes"]],
                  "global_starved": global_result["starved"],
                  "global_opportunity_relevance": [note.get("relevance")
                                                   for note in global_result["notes"]],
                  "global_notes_rendered": len(rendered_notes),
                  "global_rendered_ids": [note.get("entry_id") for note in rendered_notes],
                  "global_note_chars": len(global_text),
                  "global_notes_named": global_named,
                  "reader_memory_chars": (len(reader_memory_block) + len(global_text)
                                          + (1 if reader_memory_block and global_text else 0)),
                  **record}
        session["reads"].append(report)
        # **The memory's active update happens here**, at the read, through the write
        # gate - a reader's `WRONG: <second>` retracts, and its `MEMORY_UPDATE` is
        # stored only if it passes the gate (v2).
        record["memory"] = apply_reader_memory(
            session, chunk_id, question, record["memory_used"],
            record["memory_update"], reports=[report], options=options,
            read_window=(first, last), author="W%d" % chunk_id,
            question_number=question_number, description=leaf_sighting(record),
            place=record.get("place") or "", answer=record.get("answer") or "",
            images=images, claim=record.get("claim"), raw_reply=reply,
            quote=record.get("quote") or "")
        report["memory"] = copy.deepcopy(record["memory"])
        report["entry_id"] = (record["memory"] or {}).get("entry_id")
        for sighting in parse_sightings(reply):
            spans = place_spans("%g-%g" % (sighting["span"][0], sighting["span"][1]), first, last)
            if not spans:
                continue
            seconds = [min(span[0] for span in spans), max(span[1] for span in spans)]
            remember_chunk(session, chunk_id, question,
                           '"%s" answer %s' % (sighting["quote"], sighting["answer"]), "",
                           author="W%d" % chunk_id, seconds=seconds, assigned=[first, last],
                           question_number=question_number, provisional=True,
                           frames=(record["memory"] or {}).get("frames"),
                           answer=sighting["answer"], kind="observation")
        typed, typed_why = typed_record(report, question, session.get("family") or "",
                                        kind=record.get("kind"))
        report["typed"] = typed
        report["typed_why"] = typed_why
        if typed:
            attach_typed_record(session, chunk_id, typed)
        if not (record["frames_text"] or record["memory_update"]):
            history_entry = remember_chunk(
                session, chunk_id, question,
                "(read seconds %g-%g at %g fps and reported nothing)"
                % (first, last, fps), "", touch_map=False, citable=False,
                author="W%d" % chunk_id, question_number=question_number,
                seconds=[first, last], assigned=[first, last],
                provisional=True, frames=(record["memory"] or {}).get("frames"),
                raw_reply=reply)
            if history_entry is not None:
                report["entry_id"] = history_entry["id"]
                report["memory"] = dict(report.get("memory") or {},
                                        entry_id=history_entry["id"])
        reports.append(report)
    return reports


def _review_quote_problem(quote, report, options):
    """Why a support quote is not evidence from the reader's report, or `''` when it is."""
    quote = str(quote or "").strip()
    if len(quote.split()) < 2 or len(quote) < 6:
        return "the support quote is missing or too short"
    evidence = " ".join((str(report.get("quote") or ""),
                         str(report.get("frames_text") or ""))).strip()
    if not evidence:
        return "the reader report has no evidence words to quote"
    if _normalise(quote) not in _normalise(evidence):
        return "the support quote is not in the reader's report"
    if is_option_echo(quote, options):
        return "the support quote repeats an answer option"
    return ""


def _review_round(call, session, question, unit, reports, memories, log, round_no, row,
                  frame_limit=None):
    """**O's own look at a reader's memory**: "O will need to double check by
    itself, read the relevant reader memory, the coarse frame and descriptions. If O accepts, then
    submit the answer; if O rejects, then for the next turn."

    The frames are **read back out of the store** - `tree_store.load_frames` on the record the leaf
    carries - so this is the memory being used, and the video is never decoded. The decision is
    **O's**, in O's own call: there is no checker and no verifier in this loop.

    Returns `(verdict, unit, ok, why)`. `ok` is True only when O accepted the reader's report as
    internally supported. A rejection makes the observation uncitable; it never retracts or deletes
    the reader's words and frames."""
    from ..memory import global_store as _tree_store
    reviewed_id = None
    if str(unit).upper().startswith("M"):
        entry = memory_by_handle(memories, unit) or {}
        chunk = entry.get("chunk")
        reviewed_id = entry.get("entry_id")
        if chunk is None:
            return None, unit, "unreviewable", "that handle is not a memory entry this run showed"
        mine = [e for e in (session.get("chunks", {}).get(chunk) or [])
                if e.get("id") == entry.get("entry_id")] or \
               [e for e in (session.get("chunks", {}).get(chunk) or []) if not e.get("retracted")]
        report = {"chunk": chunk,
                  "first": float((entry.get("address") or [0.0])[0]),
                  "last": float((entry.get("address") or [0.0, 0.0])[-1]),
                  "can_answer_field": "NONE", "answer": (entry.get("answer") or ""),
                  "kind": entry.get("kind") or "NONE",
                  "quote": entry.get("cue") or "",
                  "frames_text": (mine[0].get("description") if mine else entry.get("cue")) or ""}
    elif str(unit).upper().startswith("G"):
        named = next((item for report in (reports or ())
                      for item in (report.get("global_notes_named") or [])
                      if str(item.get("handle") or "").upper() == str(unit).upper()), None)
        if named is None:
            return None, unit, "unreviewable", "no reader named that global note this question"
        chunk = named.get("chunk")
        reviewed_id = named.get("entry_id")
        if chunk is None:
            return None, unit, "unreviewable", "that note carries no chunk to open"
        mine = [e for e in (session.get("chunks", {}).get(chunk) or [])
                if e.get("id") == reviewed_id]
        if not mine:
            return None, unit, "unreviewable", "that note is no longer in the memory"
        span = _note_span(mine[0]) or [0.0, 0.0]
        report = {"chunk": chunk, "first": span[0], "last": span[1],
                  "can_answer_field": "NONE", "answer": (mine[0].get("answer") or ""),
                  "kind": mine[0].get("kind") or "NONE",
                  "quote": mine[0].get("update") or "",
                  "frames_text": mine[0].get("description") or ""}
    else:
        report = next((r for r in reports if "W%d" % r["chunk"] == unit), None)
        event_id = report.get("entry_id") if report else None
        candidates = list(session.get("chunks", {}).get(report["chunk"] if report else None) or [])
        mine = [e for e in candidates
                if event_id is not None and e.get("id") == event_id
                and not e.get("retracted")]
        if not mine:
            mine = [e for e in candidates if not e.get("retracted")]
        if mine:
            reviewed_id = mine[0].get("id")
    if report is None:
        return None, unit, "unreviewable", "that unit is not a memory this run made"
    if mine and all(entry.get("retracted") for entry in mine):
        return None, unit, "unreviewable", "that note was already retracted by an earlier review"
    if True:
        images, frames_why, shown, total = [], "this memory stored no frames", [], 0
        for entry in reversed(mine):
            if entry.get("frames"):
                spans = place_spans(report.get("place") or "", report["first"], report["last"])
                centre = min(span[0] for span in spans) if spans else None
                limit = _tree_store.MAX_REVIEW_FRAMES if frame_limit is None else int(frame_limit)
                images = _tree_store.load_frames(
                    entry["frames"], limit=max(0, min(limit, _tree_store.MAX_REVIEW_FRAMES)),
                    around=centre, span=[report["first"], report["last"]])
                shown = list(getattr(images, "indices", []) or [])
                total = int(getattr(images, "total", 0) or 0)
                frames_why = ""
            break
        prompt = REVIEW_PROMPT.format(
            first=report["first"], last=report["last"], question=question,
            can=(report.get("can_answer_field") or "NONE"),
            answer=(report.get("answer") or "NONE"), kind=(report.get("kind") or "NONE"),
            quote=_clip(" ".join(str(report.get("quote") or "").split()), 60),
            words=_clip(" ".join(str(report.get("frames_text") or "").split()), 300),
            memory=render_memory_block(session, question, memories))
        role = "review-%s" % report["chunk"]
        reply = call(prompt, images, role)
        log.append({"role": role, "prompt": prompt, "reply": reply, "frames": len(images)})
        verdict = parse_review(reply)
        reader_option = (report.get("answer") or "").upper()[:1] or None
        mine_option = (verdict or {}).get("option")
        usage = dict(getattr(call, "last_usage", {}) or {})
        prompt_tokens = usage.get("prompt_tokens")
        estimated_tokens = not isinstance(prompt_tokens, int) or prompt_tokens <= 0
        if estimated_tokens:
            prompt_tokens = max(1, len(prompt.split()))
        request_id = str(getattr(call, "last_request_id", "") or
                         hashlib.sha256((role + "\n" + prompt + "\n" + reply)
                                        .encode("utf-8")).hexdigest())
        sealed_call = {
            "request_id": request_id,
            "role": role,
            "round": int(round_no),
            "entry_id": reviewed_id,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "reply_sha256": hashlib.sha256(reply.encode("utf-8")).hexdigest(),
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": getattr(call, "last_finish_reason", None),
            "prompt_tokens_estimated": bool(estimated_tokens),
        }
        record = {"unit": unit, "chunk": report["chunk"], "round": round_no,
                  "entry_id": reviewed_id,
                  "frames": len(images), "frames_from_store": bool(images),
                  "frame_indices": shown, "frames_in_read": total,
                  "frames_why": frames_why,
                  "accept": None if verdict is None else verdict["accept"],
                  "support": None if verdict is None else verdict.get("support", verdict["accept"]),
                  "option": mine_option,
                  "reader_option": reader_option,
                  "reader_answer": (report.get("answer") or "").upper()[:1] or None,
                  "reader_quote": " ".join(str(report.get("quote") or "").split()),
                  "reader_quote_line": " ".join(str(report.get("quote") or "").split()),
                  "corrected": bool(mine_option and reader_option and mine_option != reader_option),
                  "supplied_option": bool(mine_option and not reader_option),
                  "quote": (verdict or {}).get("quote") or "",
                  "quote_missing": bool(verdict and verdict.get("accept") and not verdict.get("quote")),
                  "what": (verdict or {}).get("what") or "",
                  "why": (verdict or {}).get("why") or "",
                  "reply": _clip(" ".join(str(reply or "").split()), 200),
                  "review_call_id": canonical_sha256(sealed_call),
                  "sealed_call": sealed_call}
        row.setdefault("reviews", []).append(record)
        if verdict is None:
            record["outcome"] = "no_verdict"
            for entry in mine:
                entry["review_status"] = "no_verdict"
                entry["support_rejected"] = True
                entry["answer_support"] = False
                entry["citation_allowed"] = False
                entry["citable"] = False
                entry["status"] = "rejected"
                tell_the_tree(session, entry, status="rejected", by="O",
                              why="review named no support verdict",
                              fields={"answer_support": False, "citation_allowed": False,
                                      "citable": False, "review_status": "no_verdict",
                                      "support_rejected": True})
                break
            return record, unit, "no_verdict", "the review named no verdict (SUPPORT: YES, NO or UNSURE)"
        if verdict.get("unsure"):
            record["outcome"] = "unsure"
            for entry in mine:
                entry["review_status"] = "unsure"
                entry["support_rejected"] = True
                entry["answer_support"] = False
                entry["citation_allowed"] = False
                entry["citable"] = False
                entry["status"] = "unsure"
                tell_the_tree(session, entry, status="unsure", by="O",
                              why="support review was unsure",
                              fields={"answer_support": False, "citation_allowed": False,
                                      "citable": False, "review_status": "unsure",
                                      "support_rejected": True})
                break
            return (record, unit, "unsure",
                    "you could not tell from these frames (%s) - nothing is retracted and nothing "
                    "is grounded" % (verdict.get("why") or "no reason given"))
        if verdict["accept"]:
            if not images:
                record["outcome"] = "no_frames"
                for entry in mine:
                    entry["review_status"] = "no_frames"
                    entry["support_rejected"] = True
                    entry["answer_support"] = False
                    entry["citation_allowed"] = False
                    entry["citable"] = False
                    entry["status"] = "rejected"
                    tell_the_tree(session, entry, status="rejected", by="O",
                                  why="support review had no stored frames",
                                  fields={"answer_support": False, "citation_allowed": False,
                                          "citable": False, "review_status": "no_frames",
                                          "support_rejected": True})
                    break
                return (record, unit, "no_frames",
                        "this memory stored no frames (%s), so you cannot have looked at it - the "
                        "claim stands unverified" % frames_why)
            if not verdict.get("legacy"):
                # The support check binds to the reader's answer, not an option
                # O claims to have re-read from the stored sample.
                verdict["option"] = (report.get("answer") or "").upper()[:1] or None
                record["option"] = verdict.get("option")
                problem = _review_quote_problem(verdict.get("quote"), report,
                                                options_of(question))
                if problem:
                    record["outcome"] = "no_quote" if not verdict.get("quote") else "quote_mismatch"
                    for entry in mine:
                        entry["review_status"] = record["outcome"]
                        entry["support_rejected"] = True
                        entry["answer_support"] = False
                        entry["citation_allowed"] = False
                        entry["citable"] = False
                        entry["status"] = "rejected"
                        tell_the_tree(session, entry, status="rejected", by="O", why=problem,
                                      fields={"answer_support": False,
                                              "citation_allowed": False,
                                              "citable": False,
                                              "review_status": record["outcome"],
                                              "support_rejected": True})
                        break
                    return (record, unit, record["outcome"],
                            "%s - quote the reader's own words from its report" % problem)
                # The reader may put its evidence in the description even when its QUOTE line
                # is empty. Store the exact reader-written phrase this review used.
                record["reader_quote"] = " ".join(str(verdict.get("quote") or "").split())
            if not verdict.get("option") and verdict.get("legacy"):
                record["outcome"] = "no_option"
                for entry in mine:
                    entry["review_status"] = "no_option"
                    entry["support_rejected"] = True
                    entry["answer_support"] = False
                    entry["citation_allowed"] = False
                    entry["citable"] = False
                    entry["status"] = "rejected"
                    tell_the_tree(session, entry, status="rejected", by="O",
                                  why="review named no supported option",
                                  fields={"answer_support": False, "citation_allowed": False,
                                          "citable": False, "review_status": "no_option",
                                          "support_rejected": True})
                    break
                return (record, unit, "no_option",
                        "you accepted it but named no option, so it supports nothing yet - say which "
                        "letter the frames support "
                        "(`ACCEPT: YES | <letter> | \"words from the frames\"`)")
            for entry in mine:
                entry["answer_support"] = bool(verdict.get("option"))
                entry["citation_allowed"] = bool(verdict.get("option"))
                entry["support_rejected"] = False
                entry["review_status"] = "accepted"
                entry["status"] = "unverified"
                if not verdict.get("option"):
                    entry["citable"] = False
                    tell_the_tree(session, entry, status="unverified", by="O",
                                  why="accepted as context; no answer support",
                                  fields={"answer_support": False, "citation_allowed": False,
                                          "citable": False, "review_status": "accepted",
                                          "support_rejected": False})
                else:
                    entry["citable"] = True
                    tell_the_tree(session, entry, status="unverified", by="O",
                                  why="reader report supports its answer",
                                  fields={"answer_support": True, "citation_allowed": True,
                                          "citable": True, "review_status": "accepted",
                                          "support_rejected": False})
                break
            record["answer_support"] = bool(verdict.get("option"))
            record["citation_allowed"] = bool(verdict.get("option"))
            record["outcome"] = "accept"
            return record, unit, "accept", (verdict.get("why") or "O accepted it")
        record["outcome"] = "reject"
        # **A failed support check does not delete the observation.** The raw finding and
        # its pixels stay in the audit; only its eligibility for promotion/citation is removed.
        for entry in mine:
            entry["review_status"] = "rejected"
            entry["support_rejected"] = True
            entry["answer_support"] = False
            entry["citation_allowed"] = False
            entry["citable"] = False
            entry["status"] = "rejected"
            tell_the_tree(session, entry, status="rejected", by="O",
                          why="support review failed",
                          fields={"answer_support": False, "citation_allowed": False,
                                  "citable": False, "review_status": "rejected",
                                  "support_rejected": True})
            break
        return record, unit, "reject", "O rejected it (%s)" % (verdict.get("why") or "")


def _check_round(call, session, question, reports, options, frames_of, log, round_no,
                 memories=None):
    """One check round: for each claimed report, O is shown that reader's own report
    **and the frames of the place it named**, and must ground it or refuse it.

    Returns `(check, ok, why, reads)`. The first grounded candidate wins; a round in
    which nothing grounds returns `ok=False` and the flow takes its one recovery.
    Capped at `MAX_CHECKS` candidates, so a chatty selective pass cannot turn the
    check into a sweep.
    """
    reads, why = [], "no reader claimed anything to ground"
    for report in [r for r in reports
                   if r.get("claimed") or r.get("points")
                   or _letter_of(r.get("answer") or "").upper() in tuple("ABCDEFGH")][:MAX_CHECKS]:
        window = window_for_place(report["place"], report["first"], report["last"])
        n = frames_for(window[0], window[1], 1.0, cap=MAX_PLACE_FRAMES)
        label = "place-%d" % report["chunk"]
        frames = ([None] * n if frames_of is None
                  else frames_of(label, window[0], window[1], 1.0, n))
        unit = "W%d" % report["chunk"]
        prompt = CHECK_PROMPT.format(question=question,
                                     memory=render_memory_block(session, question,
                                                                memories),
                                     unit=unit, ledger=render_ledger([report]))
        role = "check-%d-r%d" % (report["chunk"], round_no)
        reply = call(prompt, frames, role)
        log.append({"role": role, "prompt": prompt, "reply": reply,
                    "frames": len(list(frames))})
        check = parse_check(reply)
        retried = False
        if check is None or _placeholder(check):
            retried = True
            sharp = prompt + CHECK_REREASK
            retry = call(sharp, frames, role + "-retry")
            log.append({"role": role + "-retry", "prompt": sharp, "reply": retry,
                        "frames": len(list(frames))})
            second = parse_check(retry)
            if second is not None and not _placeholder(second):
                check = second
        ok, why = gate_check(check, reports, options)
        reads.append({"chunk": report["chunk"], "window": list(window), "frames": n,
                      "label": label, "ok": bool(ok), "why": why,
                      "reread": retried})
        if ok:
            return check, True, why, reads
    return None, False, why, reads


def render_history(attempts):
    """What has already been tried for this question, for the selection's state."""
    if not attempts:
        return "(nothing yet)"
    lines = []
    for attempt in attempts:
        read = ", ".join("chunk %d at %g fps" % (chunk, fps)
                         for chunk, fps in attempt.get("read") or ())
        lines.append("round %d: read %s" % (attempt["round"], read or "nothing"))
        for chunk, answer in attempt.get("claimed") or ():
            lines.append("    chunk %d claimed %s" % (chunk, answer))
        lines.append("    the check said: %s" % attempt.get("check", "nothing"))
    return "\n".join(lines)


def _conflict_relook(call, log, frames_of, session, question, options, committed, row,
                     window, memory, reports):
    """`(kept, outcome, detail)` - **the one place the draft may override a gate.**

    It fires only when a grounded commit disagrees with O's own draft, and it decides
    on the **same place frames** the check saw, because the question is whether those
    pixels really support the committed option. A `KEEP` still has to pass the
    description gate, so the re-look cannot bless a citation the gate would refuse -
    and a `KEEP` cannot afterwards be vetoed by the decision call, or a verified row
    would depend on O's mood.
    """
    span = window.get("window") or [0.0, 0.0]
    n = frames_for(span[0], span[1], 1.0, cap=MAX_PLACE_FRAMES)
    chunk = window.get("chunk")
    frames = ([None] * n if frames_of is None else
              frames_of("conflict-%d" % chunk, span[0], span[1], 1.0, n))
    report = next((r for r in reports
                   if "W%d" % r["chunk"] == committed["unit"]), None)
    prompt = CONFLICT_PROMPT.format(
        draft=row.get("draft") or "(none)", unit=committed["unit"],
        option=committed["option"], question=question,
        memory=render_memory_block(session, question, memory),
        ledger=render_ledger([report] if report else []))
    reply = call(prompt, frames, "conflict")
    log.append({"role": "conflict", "prompt": prompt, "reply": reply,
                "frames": len(list(frames))})
    parsed = parse_conflict(reply)
    if not parsed:
        return True, "unparsable-kept", ("the re-look reply was unparsable; the "
                                        "gate-verified commit stands")
    if not parsed["keep"]:
        return False, "dropped", parsed["why"] or "the frames do not support it"
    cite = {"unit": committed["unit"], "option": committed["option"],
            "quote": parsed["quote"]}
    ok, why = gate_commit(cite, reports, memory=memory,
                          about=session.get("topic") or "", options=options)
    return (True, "kept", why) if ok else (False, "dropped", why)


def _read_density(reports, chunk_id):
    """The densest rate this question has read a chunk at, or 0 when unread."""
    rates = [r["fps"] for r in reports if r["chunk"] == chunk_id]
    return max(rates) if rates else 0.0


def escalation_pool(chunks, reports):
    """What one escalation round may look at, in priority order."""
    fresh = [cid for cid, _, _ in chunks if _read_density(reports, cid) == 0.0]
    upgrade = [cid for cid, _, _ in chunks
               if 0.0 < _read_density(reports, cid) < ALL_FPS]
    return {"fresh": fresh, "upgrade": upgrade}


def re_read_eligible(reports, chunk_id, density):
    """May a chunk be read again at this density?

    * never read at this density -> yes (this is a first look at that rate);
    * a read that **reported** (a parsable reply) -> no: a second look at the same
      density adds nothing;
    * a read whose reader never reported -> **one** second chance, because the batch
      was empty rather than informative (question 1601: five of six readers replied
      junk). A second unparsable read exhausts it, so nothing can loop.
    """
    at = [r for r in reports if r["chunk"] == chunk_id]
    if any(r.get("parsable") and float(r["fps"]) >= float(density) for r in at):
        return False
    same = [r for r in at if float(r["fps"]) == float(density)]
    if not same:
        return True
    return len(same) < 2


def re_read_notes(reports, picks, density=SELECTIVE_FPS):
    """The note a second chance carries, for whichever picks need it."""
    notes = {}
    for chunk_id in picks:
        earlier = [r for r in reports if r["chunk"] == chunk_id
                   and float(r["fps"]) == float(density)]
        if earlier and not any(r.get("parsable") for r in earlier):
            notes[chunk_id] = ("Your last reply about this chunk was not readable. "
                               "Reply with the FRAMES and CAN_ANSWER lines.")
    return notes


def still_readable(chunks, reports, density=SELECTIVE_FPS):
    """The chunks a selection at `density` could still open - **one definition**, used
    by the menu's marks and by the nudge's list, so the prompt cannot say a chunk is
    read while the filter would accept it."""
    return [cid for cid, _, _ in chunks if re_read_eligible(reports, cid, density)]


def nearest_unread(chunks, reports, named, limit=MAX_PICK):
    """The unread chunks nearest the ones O named - the executor's repair."""
    read = {r["chunk"] for r in reports or ()}
    anchors = [int(cid) for cid in named or () if str(cid).isdigit()] or sorted(read)
    if not anchors:
        anchors = [1]
    ordered = sorted(chunks, key=lambda c: (min(abs(c[0] - a) for a in anchors), c[0]))
    out = []
    for chunk_id, _, _ in ordered:
        if chunk_id not in read:
            out.append(chunk_id)
        if len(out) >= limit:
            break
    return out


def _remaining(budget, key, log, start, frames=False):
    """How much of the question's budget is left, for the selection's state."""
    total = (budget or {}).get(key)
    if total is None:
        return None
    spent = (sum(entry.get("frames") or 0 for entry in log[start:]) if frames
             else len(log) - start)
    return max(0, int(total) - int(spent))


def _action_label(action):
    if action["action"] == "read":
        return "READ %s" % ", ".join(str(cid) for cid in action["ids"])
    if action["action"] in ("look", "recall", "memory"):
        return "%s %s" % (action["action"].upper(), action["handle"])
    return "ALL"


def _select(call, session, chunks, question, reports, why, frames_of, log, row,
            duration=0.0, options=(), question_number=None, round_no=1,
            attempts=(), calls_left=None, frames_left=None):
    """O's one recovery, **executed**. Returns `{name, reports, recall_read, why}`.

    v3's adaptive selection: O decides what to look at next, from the memory, this
    question's failure history and the remaining budget. The menu is exclusive and
    ordered - `READ <ids>`, `MEMORY`, `RECALL M<k>`, `ALL` - and a reply that names
    none of them becomes the all-chunks read rather than a decision.

    **Anti-stall**: chunks already read at that density (or denser) are filtered out;
    an empty surviving set spends the round and is recorded as a stall. `RECALL` is
    spent once per question. Returns the round's record, with the state O saw, the
    action taken, its cost, and the menu items it did not take.
    """
    read_at = {cid: _read_density(reports, cid) for cid, _, _ in chunks}
    shown = shown_memories(session, question)
    # **The candidate line a policy needs**: how often the chunk was read (so "read
    # twice, empty both times" is visible), at what density, and whether it claimed.
    state = {"candidates": [{"chunk": cid, "first": first, "last": last,
                             "density": read_at[cid],
                             "reads": len([r for r in reports if r["chunk"] == cid]),
                             "claimed": any(r["claimed"] for r in reports
                                            if r["chunk"] == cid)}
                            for cid, first, last in chunks],
             "failure_history": list(attempts),
             "memory_handles": [entry["handle"] for entry in shown],
             "rounds_left": MAX_SELECTION_ROUNDS - round_no,
             "calls_left": calls_left, "frames_left": frames_left}
    prompt = SELECT_PROMPT.format(
        why=why, dur=int(duration), k=len(chunks),
        menu=read_marked_menu(chunks, reports, question=question),
        question=question, memory=render_memory_block(session, question),
        citable=render_citable(reports, shown),
        looked=render_looks(row),
        history=render_history(attempts), rounds_left=state["rounds_left"],
        calls_left="unknown" if calls_left is None else calls_left,
        frames_left="unknown" if frames_left is None else frames_left, pick=MAX_PICK)
    role = "select" if round_no == 1 else "select%d" % round_no
    reply = call(prompt, (), role)
    log.append({"role": role, "prompt": prompt, "reply": reply, "frames": 0})
    action = parse_select(reply)
    record = {"round": round_no, "state": state, "action": _action_label(action),
              "reply": reply, "reports": [], "recall_read": None, "stall": False}
    valid = {cid for cid, _, _ in chunks}

    def fresh_picks(ids, density=SELECTIVE_FPS):
        return [cid for cid in dict.fromkeys(ids)
                if cid in valid and re_read_eligible(reports, cid, density)]

    # **One nudge when every named chunk is already read** - the same pattern the
    # commit gate uses for a refused citation. The re-ask is a **re-rendered** prompt,
    # not the same one plus a sentence: it marks the read state and names what is left.
    if action["action"] == "read":
        if not fresh_picks(action["ids"]):
            unread = still_readable(chunks, reports, SELECTIVE_FPS)
            retry_prompt = (read_marked_menu(chunks, reports, question=question) + "\n" + prompt)
            retry_prompt += (
                "\nYour last answer named chunks that have already been read this "
                "question, or none at all. The chunks not yet read are: %s. Name some "
                "of those, or choose ALL, or answer from memory."
                % (", ".join(str(cid) for cid in unread) or "none"))
            retry = call(retry_prompt, (), role + "-retry")
            log.append({"role": role + "-retry", "prompt": retry_prompt,
                        "reply": retry, "frames": 0})
            action = parse_select(retry)
            reply = retry
            record["reply"] = retry
            record["action"] = _action_label(action)
    menu = [{"action": "READ <chunk ids>", "price": "0.5 fps per chunk"},
            {"action": "MEMORY M<k>", "price": "one read of the cited chunk"},
            {"action": "RECALL M<k>", "price": "one read of that chunk"},
            {"action": "ALL", "price": "%d reads at 1 fps" % len(chunks)}]

    def finish(out, keep):
        out["not_taken"] = [item for item in menu
                            if not item["action"].startswith(keep)]
        return out

    if action["action"] == "read":
        picks = fresh_picks(action["ids"])
        if not picks:
            filled = nearest_unread(chunks, reports, action["ids"])
            record["repaired"] = {"named": list(action["ids"]), "filled": filled}
            picks = filled
            if not picks:
                record["stall"] = True
                record["why"] = "every chunk has been read at that density"
                return finish(record, "READ")
        picks, clamped = clamp_reads(
            picks, MAX_ESCALATION_READS,
            "the escalation is bounded at %d reads" % MAX_ESCALATION_READS)
        if clamped:
            record["clamped"] = clamped
        # A chunk whose earlier read never reported gets the note that breaks the
        # same-input determinism; a first look gets nothing.
        notes = re_read_notes(reports, picks, SELECTIVE_FPS)
        record["reports"] = _read_chunks(call, session, chunks, picks, question,
                                         SELECTIVE_FPS, "replan", log, frames_of,
                                         options, question_number, notes=notes)
        return finish(record, "READ")
    # **The fetch**: `LOOK M<k>` returns that entry's own words into the next call and
    # nothing else - no answer, no read, and one more round to act on what it said. The
    # filter is the store's (`matching_memories`), so a retracted, unverified, provisional
    # or method entry is never served, and a look naming something O was not shown returns
    # nothing and spends the round.
    if action["action"] == "look":
        record["look_requested"] = action["handle"]
        entry = memory_by_handle(shown, action["handle"])
        if entry is not None and not row.get("looks"):
            row.setdefault("looks", []).append({
                "handle": entry["handle"], "chunk": entry["chunk"],
                "address": entry.get("address"),
                "text": ("%s %s" % (entry.get("description") or "",
                                    entry.get("update") or "")).strip()})
            record["served_look"] = entry["handle"]
            record["why"] = "the entry's own words were returned"
            return finish(record, "LOOK")
        if row.get("looks"):
            record["stall"] = True
            record["why"] = "the look for this question is already spent"
            return finish(record, "LOOK")
        record["action"] = "ALL (%s was not among the entries O was shown)" % action["handle"]
    if action["action"] in ("recall", "memory"):
        entry = memory_by_handle(shown, action["handle"])
        if entry is not None and not row.get("recall"):
            chunk_id = entry["chunk"]
            # **A recall re-reads its chunk on purpose**: O is asking for a fresh look
            # at the place a stored conclusion came from, not for coverage. It is
            # bounded by the once-per-question rule and the selection rounds.
            record["reports"] = _read_chunks(
                call, session, chunks, [chunk_id], question, SELECTIVE_FPS,
                "recall", log, frames_of, options, question_number,
                notes=re_read_notes(reports, [chunk_id], SELECTIVE_FPS))
            record["recall_read"] = {
                "handle": action["handle"], "chunk": chunk_id,
                "frames": sum(r["frames"] for r in record["reports"]),
                "executed": True, "option": action.get("option", "")}
            row["memory_answer"] = action["action"] == "memory"
            return finish(record, action["action"].upper())
        if row.get("recall"):
            record["stall"] = True
            record["why"] = "the recall for this question is already spent"
            return finish(record, action["action"].upper())
        why = "%s was not among the memory entries O was shown" % action["handle"]
        record["action"] = "ALL (%s)" % why
    pool = escalation_pool(chunks, reports)
    picks = spread(pool["fresh"] or pool["upgrade"], MAX_ESCALATION_READS)
    asked = len(pool["fresh"]) if pool["fresh"] else len(pool["upgrade"])
    if asked > len(picks):
        record["clamped"] = {"asked": asked, "read": len(picks),
                             "why": "the escalation is bounded at %d reads"
                                    % MAX_ESCALATION_READS}
    if not picks:
        record["stall"] = True
        record["why"] = record.get("why") or "every chunk is already read at 1 fps"
        return finish(record, "ALL")
    record["reports"] = _read_chunks(call, session, chunks, picks, question, ALL_FPS,
                                     "all", log, frames_of, options, question_number)
    return finish(record, "ALL")


CITABLE_LINES = 6             # the units the commit prompt lists for citation


def render_citable(reports, memories, limit=CITABLE_LINES):
    """**What a final answer may cite, listed**."""
    lines = []
    total = len(reports or []) + len(memories or [])
    for report in list(reports or []):
        if len(lines) >= limit:
            break
        chunk = report.get("chunk")
        option = _letter_of(report.get("answer") or "")
        place = str(report.get("place") or "").strip()
        bits = ["chunk %s" % chunk]
        if place and place.upper() != "NONE":
            bits.append("read %s" % place)          # the read's own place, not only its chunk
        if option:
            bits.append("its answer %s" % option)
        bits.append("claimed" if report.get("claimed")
                    else "no claim - REVIEW grounds nothing, read again")
        lines.append("  W%s (%s)" % (chunk, "; ".join(bits)))
    for item in list(memories or []):
        if len(lines) >= limit:
            break
        # **The door says how it opens** (the same run): an unverified entry is cited with its own
        # words or not at all, and a `VERIFY` is what makes it a `checked-cited` citation - the
        # listing says which, so the agent is not left guessing at the gate.
        lines.append("  %s %s%s \"%s\"  (%s)"
                     % (item.get("handle"), _span_of(item.get("address")),
                        " %s" % item.get("status") if item.get("status") else "",
                        _few_words(item.get("description") or item.get("update") or "",
                                   CUE_WORDS),
                        "verified - citable"
                        if str(item.get("status") or "") == "verified"
                        else "unverified - REVIEW it to cite it"))
    if total > len(lines):
        lines.append("  (+%d more)" % (total - len(lines)))
    return "\n".join(lines) if lines else "  (nothing yet: read a chunk, or check a memory entry)"


def option_text(letter, options):
    """The option as the record writes it: `(B) a red hat` when its text is known."""
    at = ord(str(letter or "A").upper()[:1] or "A") - ord("A")
    options = list(options or ())
    if 0 <= at < len(options):
        return "(%s) %s" % (chr(ord("A") + at), options[at])
    return str(letter or "")


LOOP_RESUME_VERSION = "loop-resume-1"
LOOP_RESUME_GOLD_KEYS = frozenset((
    "gold", "gold_window", "gold_span", "gold_bearing", "target_hit", "training_only",
    "terminal_correct", "terminal_credit", "credit", "reward_components",
    "advantage", "sample_weight", "candidate_overlaps", "label_status",
    "label_source", "excluded_reason",
))
LOOP_RESUME_FIELDS = frozenset((
    "resume_version", "question", "duration", "family", "question_number",
    "max_turns", "row", "question_reports", "turns", "standing", "seen",
    "last_survey", "last_learned", "last_scan", "turn_no", "free_refusals",
    "refusals", "refused_keys", "reviewed_units", "harness_reviewed",
    "read_rereads", "turn_mask_refusals", "row_errors", "log", "call_start",
    "budget", "loop_options",
))
LOOP_OPTION_FIELDS = frozenset((
    "force_read", "stage_gate", "grounded_submit", "check_coin", "coin_seed",
    "check_to_finalize", "survey_first", "free_refusals_max", "structural_merge",
    "review_first", "clock_read",
))


def run_question(ask, session, question, duration, log=None, frames_of=None,
                 budget=None, row=None, oracle_chunks=None, family=""):
    """One question end to end. Returns the row's fields.

    `frames_of` is required: it is the only way frames reach a prompt, the coarse overview
    included, so the runner has to say what it shows on every call.

    The row is filled as the flow goes. A caller that passes its own dict gets the partial
    trace back even when a `StopRun` aborts the question."""
    if frames_of is None:
        raise ValueError(
            "run_question needs frames_of(label, first, last, rate, n): every "
            "prompt's frames come from the runner, O's coarse overview included")
    log = log if log is not None else []
    start = len(log)
    row_errors = []

    def call(prompt, images, role):
        """**A model error is a row, not a crash**. The rest of
        the harness writes an error row and carries on; this does the same, and the
        empty reply makes the parsers degrade instead of raising."""
        try:
            reply = ask(prompt, images, role)
            call.last_finish_reason = getattr(ask, "last_finish_reason", None)
            call.last_usage = dict(getattr(ask, "last_usage", {}) or {})
            call.last_request_id = getattr(ask, "last_request_id", "")
            return reply
        except Exception as error:                       # noqa: BLE001
            row_errors.append("%s: %s" % (type(error).__name__, error))
            log.append({"role": role, "prompt": prompt, "reply": "", "frames":
                        len(list(images)), "error": row_errors[-1]})
            return ""

    chunks = chunk_list(duration)
    degenerate = duration <= DEGENERATE_BELOW
    # The options are needed by the reads (the write gate refuses an option echo) and
    # by the check, so they are read once here.
    options = options_of(question)
    question_number = len(session["question_history"]) + 1
    session["_active_question_number"] = question_number
    session["family"] = family or ""
    row = {} if row is None else row
    row.update({"question": question, "chunks": len(chunks), "degenerate": degenerate,
                "decide": None, "plan_source": None, "picks": [],
                "selective_reads": [], "all_reads": [], "memory_answer": False,
                "memory_candidate": None,
                "looks": [],
                "verification_reads": [], "commit": None, "commit_ok": False,
                "commit_why": "", "answer": "", "fallback_answer": "", "cited": False,
                "memory_handles": [], "calls": 0, "frames": 0,
                "draft": row.get("draft", "")})
    # **This question's own reads**, in the order they happened - the ledger the
    # commit may cite. It is built here rather than filtered out of the session by
    # chunk id: the session holds every question's reads, so a chunk-id filter let
    # an earlier question's report stand in as this question's evidence (both
    # reviewers, round 2).
    question_reports = []
    frozen_memory = None

    warm = bool(session["question_history"])
    row["consult"] = ("memory-off" if session.get("memory_disabled")
                      else ("cold-skipped" if not warm else ""))
    row["consult_outcome"] = ("memory-off" if session.get("memory_disabled")
                              else ("cold-skipped" if not warm else ""))
    if warm and not session.get("memory_disabled"):
        memory = render_memory_block(session, question, consult=True)
        row["memory_block"] = memory
        actions, action_note = decide_menu(session.get("memory_view"))
        prompt = DECIDE_PROMPT.format(question=question, memory=memory,
                                      menu=chunk_menu(chunks), actions=actions,
                                      action_note=action_note)
        prompt = append_method_memory(prompt, session, ["plan"])
        row["consult_menu"] = ("plan_only" if session.get("memory_view") == "index"
                               else "answer_allowed")
        reply = call(prompt, (), "decide")
        log.append({"role": "decide", "prompt": prompt, "reply": reply, "frames": 0})
        decision = parse_decide(reply)
        row["consult_rereads"] = 0
        if decision is None:
            sharp = (prompt + "\nYour last reply used none of the lines above. Reply with "
                     "one line only, beginning with COARSE_PLAN, PLAN or NO_USABLE.")
            retry = call(sharp, (), "decide-retry")
            log.append({"role": "decide-retry", "prompt": sharp, "reply": retry,
                        "frames": 0})
            row["consult_rereads"] = 1
            second = parse_decide(retry)
            if second is not None:
                decision, reply = second, retry
        row["decide"] = decision["action"] if decision else "unparsed"
        row["consult"] = row["decide"] or "unparsed"
        row["consult_outcome"] = {
            "answer": "answer_unverifiable", "coarse_plan": "coarse_plan",
            "plan": "plan", "no_usable": "no_usable",
        }.get(row["decide"], "unparsed")
        if decision and decision["action"] == "no_usable":
            shown_now = matching_memories(session, question)
            if session.get("memory_view") == "index":
                rendered = set(rendered_handles(memory))
                shown_now = [item for item in shown_now
                             if item["handle"] in rendered]
            ok, why = validate_no_usable(decision.get("reason") or "", shown_now)
            row["consult_handles"] = [entry["handle"] for entry in shown_now]
            row["falsified_declaration"] = not ok
            row["consult_why"] = why
            row["decide"] = "no_usable" if ok else "no_usable_incomplete"
            row["consult_outcome"] = ("no_usable" if ok
                                      else "falsified_no_usable")
        if decision and decision["action"] == "answer":
            shown = shown_memories(session, question)
            entry = memory_by_handle(shown, decision["handle"])
            quote = (decision["quote"] or "").lower()
            option = (decision["option"] or "").strip()
            usable = (entry is not None and len(quote.split()) >= 2
                      and option and not option.upper().startswith(
                          ("NONE", "N/A", "ABSTAIN"))
                      and quote in (entry["description"] + " "
                                    + entry["update"]).lower())
            if usable:
                row["memory_answer"] = True
                row["consult_outcome"] = "lead_read"
                row["memory_candidate"] = {"unit": entry["handle"], "option": option,
                                           "quote": decision["quote"]}
                # **The memory is frozen here**, so the handle O named means the same
                # entry when it commits, after the read has written to memory.
                frozen_memory = shown
                question_reports += _read_chunks(
                    call, session, chunks, [entry["chunk"]], question, SELECTIVE_FPS,
                    "verify", log, frames_of, options, question_number)
                row["verification_reads"] = [r["chunk"] for r in question_reports]
                # v2: a memory answer is **not** an answer until the check grounds the
                # evidence it pays for, so the flow falls through to the check.
            else:
                row["decide"] = "unverifiable_answer"

    if row["decide"] != "answer" or not session.get("memory_steering", True):
        plan = None
        # **The coarse read happens unless it would be degenerate or is not asked
        # for** - the `PLAN` branch of a warm question plans from memory alone, and a
        # degenerate video (the coarse read no longer sparser than a reader's pass)
        # skips planning altogether and goes straight to the all-chunks read.
        coarse_n = (0 if degenerate or (row["decide"] == "plan"
                                        and session.get("memory_steering", True))
                    else frames_for(0.0, duration, 1.0, cap=64))
        coarse = []
        if coarse_n:
            coarse = ([None] * coarse_n if frames_of is None else
                      frames_of("coarse", 0.0, duration,
                                coarse_n / duration if duration > 0 else 1.0,
                                coarse_n))
        coarse_note = (COARSE_WITHOUT_FRAMES if not coarse
                       else COARSE_WITH_FRAMES.format(n=len(coarse)))
        if not degenerate:
            plan_memory = render_memory_block(session, question)
            row.setdefault("memory_chars", {})["plan"] = len(plan_memory)
            prompt = PLAN_PROMPT.format(
                dur=int(duration), k=len(chunks), menu=chunk_menu(chunks),
                coarse=coarse_note, n=len(coarse), question=question,
                memory=plan_memory, pick=MAX_PICK)
            prompt = append_method_memory(prompt, session, ["plan"])
            reply = call(prompt, coarse, "coarse-plan" if coarse else "plan")
            log.append({"role": "coarse-plan" if coarse else "plan", "prompt": prompt,
                        "reply": reply, "frames": len(coarse)})
            plan = parse_plan(reply, chunks)
            if plan and coarse:
                remember_global(session, plan["about"])
        if plan is None:
            plan = {"think": "", "about": "", "draft": "", "plan_source": "degenerate"
                    if degenerate else "partition-clamped",
                    "picks": spread([cid for cid, _, _ in chunks], MAX_PICK)}
        if oracle_chunks is not None:
            plan = dict(plan, picks=list(oracle_chunks), plan_source="oracle")
        session["plan_log"].append({"question": question, **{k: plan[k] for k in
                                    ("think", "about", "picks", "draft", "plan_source")}})
        row["plan_source"] = plan["plan_source"]
        row["picks"] = list(plan["picks"])
        row["draft"] = plan["draft"]

    # **A degenerate video skips the selective pass**: its coarse read was already
    # at least as dense, so the team goes straight to the all-chunks read.
    selective_reports = []
    if not degenerate:
        selective_reports = _read_chunks(
            call, session, chunks, row["picks"], question, SELECTIVE_FPS,
            "selective", log, frames_of, options, question_number)
        row["selective_reads"] = [r["chunk"] for r in selective_reports]
    question_reports += selective_reports
    if degenerate:
        # **A degenerate video reads everything once**: its coarse look would not be
        # sparser than a reader's pass, so the plan is skipped and every chunk is read
        # at the all-pass density.
        all_reports = _read_chunks(
            call, session, chunks, [cid for cid, _, _ in chunks], question,
            ALL_FPS, "all", log, frames_of, options, question_number)
        row["all_reads"] = [r["chunk"] for r in all_reports]
        question_reports += all_reports

    # ---- BiFocal v3: O checks, then selects what to look at next ------------
    # **No mandatory whole-video read.** The measured cost of that rule was 47 extra
    # reads and 538 s on one question, so the selection is O's: after a check that
    # grounds nothing, O chooses more chunks (bounded), a memory answer, one named
    # chunk to recall, or ALL - at most MAX_SELECTION_ROUNDS times.
    row.update({"check": None, "check_ok": False, "check_why": "", "check_round": 0,
                "answer_source": "", "decide_reply": "", "decide_fault": "",
                "decide_why": "", "abstained": False, "abstain_why": "",
                "decide_named_unit": "", "decide_named_quote": "",
                "submitted_unit": "", "executor_overrode": False, "refusals": [],
                "conflict": None, "re_look_outcome": "", "re_look_why": "",
                "place_window": [], "recovery": None, "recall": [], "selections": [],
                "selection_reads": [], "selection_stall": False, "rounds_used": 0,
                "coverage": 0.0, "verified": False, "verified_bucket": "",
                "unverified_final": False})
    check, grounded, why, reads = None, False, "no reader claimed anything to ground", []
    attempts = []
    extra_rounds = 0
    for round_no in range(1, MAX_SELECTION_ROUNDS + 3):
        check, grounded, why, reads = _check_round(
            call, session, question, question_reports, options, frames_of, log,
            round_no, frozen_memory)
        row["check_round"] = round_no
        row["check"], row["check_ok"], row["check_why"] = check, bool(grounded), why
        row["place_window"] += reads
        row["check_rereads"] = row.get("check_rereads", 0) + sum(
            1 for read in reads if read.get("reread"))
        attempts.append({
            "round": round_no,
            "read": [(r["chunk"], r["fps"]) for r in question_reports
                     if r["pass"] != "coarse"],
            "claimed": [(r["chunk"], r["answer"]) for r in question_reports
                        if r["claimed"]],
            "check": why})
        if grounded or round_no > MAX_SELECTION_ROUNDS + extra_rounds:
            break
        selection = _select(call, session, chunks, question, question_reports, why,
                            frames_of, log, row, duration=duration, options=options,
                            question_number=question_number, round_no=round_no,
                            attempts=attempts,
                            calls_left=_remaining(budget, "calls", log, start),
                            frames_left=_remaining(budget, "frames", log, start,
                                                   frames=True))
        row["selections"].append({key: value for key, value in selection.items()
                                  if key != "reports"})
        if selection.get("served_look"):
            extra_rounds = 1
        row["recovery"] = selection["action"]
        if selection["recall_read"]:
            row["recall"].append(selection["recall_read"])
        if selection["action"].startswith("ALL"):
            row["all_reads"] = sorted(set(row["all_reads"])
                                      | {r["chunk"] for r in selection["reports"]})
        row["selection_reads"] += [r["chunk"] for r in selection["reports"]]
        question_reports += selection["reports"]
        if selection["stall"]:
            row["selection_stall"] = True
            break

    row["rounds_used"] = len(row["selections"])
    row["coverage"] = round(
        len({r["chunk"] for r in question_reports}) / float(len(chunks)), 4)
    re_reads, waiting = [], set()
    for report in question_reports:
        if float(report["fps"]) != SELECTIVE_FPS:
            continue
        if report["chunk"] in waiting:
            re_reads.append({"chunk": report["chunk"],
                             "parsable": bool(report.get("parsable"))})
            waiting.discard(report["chunk"])
        if not report.get("parsable"):
            waiting.add(report["chunk"])
    row["re_reads"] = re_reads
    row["records_written"] = sum(1 for report in question_reports if report.get("typed"))
    refused = [report for report in question_reports
               if report.get("typed") is None and report.get("typed_why")]
    row["records_refused"] = len(refused)
    row["record_refusal_whys"] = [report["typed_why"] for report in refused][:5]
    counts = collections.Counter(refusal_kind(report["typed_why"])
                                 for report in refused)
    row["record_refusal_kinds"] = dict(counts)
    supply = supply_counts(question_reports, session.get("family") or "")
    row["supply"] = supply
    row["reads_claiming"] = supply["claiming"]
    row["reads_unmapped_family"] = supply["unmapped_reads"]
    row["addressed_claim_rate"] = supply["addressed_claim_rate"]
    # **How much of the pass the reader re-ask bought** - the same reading the check's
    # re-ask already carries.
    row["reader_rereads"] = sum(1 for report in question_reports
                                if report.get("reread_field"))
    row["reader_reread_flips"] = sum(1 for report in question_reports
                                     if report.get("reread_flip"))
    row["place_read_as"] = dict(collections.Counter(
        place_read_as(report) for report in question_reports))
    row["cross_question_reads"] = sum(1 for report in question_reports
                                      if report.get("earlier_reads"))
    row["verified"] = bool(grounded)
    row["verified_bucket"] = verified_bucket(check) if grounded else ""
    cited, window = None, None
    committed = None
    if grounded:
        unit = str(check.get("unit") or "")
        cited = next((r for r in question_reports
                      if "W%d" % r["chunk"] == unit), None)
        window = next((w for w in row["place_window"]
                       if w["chunk"] == (cited or {}).get("chunk")), None)
        # **O's own option, never the reader's**: v2 fell back to the reader's `ANSWER`
        # when the support named none, which let the option echo back in.
        option = (check.get("option") or "").strip()
        row["commit"] = {"unit": unit, "option": option, "quote": check["quote"]}
        row["commit_ok"] = True
        row["cited"] = True
        row["commit_why"] = "the check grounded %s" % unit
        committed = {"unit": unit, "option": option, "quote": check["quote"]}
    else:
        row["unverified_final"] = True
        row["commit_why"] = "no check grounded the evidence: %s" % why

    # **The conflict re-look**: the one place the draft may override a passing gate.
    # It fires only when a grounded commit disagrees with O's draft, and it decides
    # on the same place frames the check saw - a passing gate the re-look confirms
    # cannot be vetoed by the decision call afterwards.
    if committed and window and _letter_of(committed["option"]) and \
            _letter_of(row.get("draft") or "") and \
            _letter_of(committed["option"]) != _letter_of(row["draft"]):
        row["conflict"] = {"committed": committed["option"],
                           "draft": row.get("draft") or ""}
        kept, outcome, detail = _conflict_relook(
            call, log, frames_of, session, question, options, committed, row, window,
            frozen_memory, question_reports)
        row["re_look_outcome"] = outcome
        row["re_look_why"] = detail
        if not kept:
            committed = None
            row["commit_ok"] = False
            row["cited"] = False
            row["verified"] = False
            row["verified_bucket"] = ""
            row["commit_why"] = "the conflict re-look dropped the commit: %s" % detail

    decide_prompt = SUBMIT_PROMPT.format(
        question=question, memory=render_memory_block(session, question,
                                                      frozen_memory),
        looked=render_looks(row),
        draft=row.get("draft") or "(none)",
        reports=render_ledger([r for r in question_reports if r["pass"] != "coarse"])
        or "(no reader reported)",
        verdicts=render_history(attempts),
        grounded=(committed["unit"] if committed else "none"))
    decide_prompt = append_method_memory(decide_prompt, session, ["commit"])
    decide_reply = call(decide_prompt, (), "decide-submit")
    log.append({"role": "decide-submit", "prompt": decide_prompt,
                "reply": decide_reply, "frames": 0})
    row["decide_reply"] = decide_reply
    resolved = resolve_submission(decide_reply, row.get("draft"), committed,
                                  question_reports, options, frozen_memory,
                                  session.get("topic") or "",
                                  commit_why=row.get("commit_why") or "")
    row["answer_source"] = resolved["answer_source"]
    row["decide_fault"] = resolved["fault"]
    row["decide_why"] = resolved["why"]
    # what O *named*, kept beside the citation the gate actually verified
    row["decide_named_unit"] = resolved.get("named_unit", "")
    row["decide_named_quote"] = resolved.get("named_quote", "")
    row["answer"] = resolved["answer"]
    # **A grounded commit stays on the record even when O submits something else**,
    # because it is what the verification claim is about: `commit_survives` says the
    # gate grounded a commit the re-look did not drop, and `commit_refused` says O
    # chose a draft or an abstention over it. Both are reported.
    row["commit_survives"] = bool(committed)
    row["commit_refused"] = bool(resolved.get("commit_refused"))
    row["abstain_over_draft"] = bool(resolved.get("abstain_over_draft"))
    row["submitted_unit"] = committed["unit"] if (
        committed and _letter_of(row["answer"]) == _letter_of(committed["option"])) else ""
    row["executor_overrode"] = bool(row["decide_named_unit"]) and (
        row["decide_named_unit"] != row["submitted_unit"])
    row["refusals"] = [{"unit": "W%d" % entry["chunk"], "why": entry.get("why") or ""}
                       for entry in row["place_window"] if not entry.get("ok")]
    # **`verified` is about the submitted answer, not about the reply's wording**: a
    # grounded commit submitted through the default path is still a verified row, and
    # a row that submitted something else is not - however well-formed its decision.
    row["verified"] = bool(committed) and (
        _letter_of(row["answer"]) == _letter_of(committed["option"]))
    if not row["verified"]:
        # the bucket describes a *submitted* commit, so a refused one is not in it
        row["verified_bucket"] = ""
    row["abstained"] = resolved["answer_source"] == "abstain"
    if row["abstained"]:
        row["abstain_why"] = resolved["why"]
    # what the checks refused is kept as evidence, never as the answer
    claimed = [r for r in question_reports if r["claimed"]]
    row["fallback_answer"] = claimed[0]["answer"] if claimed else ""
    if not grounded:
        row["unverified_final"] = resolved["answer_source"] != "commit"
    remember_qa(session, question, row["answer"] or "",
                (row["commit"] or {}).get("unit", ""),
                (row["commit"] or {}).get("quote", ""),
                (cited or {}).get("chunk"), (window or {}).get("window"),
                verified=row["verified"],
                support=(check or {}).get("quote", ""),
                support_second=(check or {}).get("second"),
                rejected=[a["check"] for a in attempts[:-1] if a.get("check")],
                answer_side=(check or {}).get("option", "") or (row["answer"] or ""))
    # and the question's provisional writes settle, so they are citable from now on
    row["settled"] = settle_question(session, question_number,
                                     verified=row["verified"])
    promotion = (BS.promote_question_claims(
        session, row, question_number=question_number,
        next_question=question_number + 1,
        chunk_spans={chunk: [first, last] for chunk, first, last in chunks})
        if session.get("memory_protocol") == "two_level"
        else {"events": []})
    mark_promoted_entries_verified(session, promotion)
    row["bifocal_promotions"] = promotion["events"]
    row["errors"] = list(row_errors)
    _remember_question_marker(session, question, question_number, duration)
    row["memory_terminal"], row["memory_positive"] = memory_terminal(row, session)
    row["memory_kind_counts"] = memory_kind_counts(session, row)
    session["question_history"].append(row["question"])
    row["calls"] = len(log) - start
    row["frames"] = sum(entry["frames"] for entry in log[start:])
    return row

