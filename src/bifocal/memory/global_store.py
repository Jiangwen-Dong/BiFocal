#!/usr/bin/env python3
"""**The Time Tree Memory's store**: the directory is the truth.

The tree body is the timeline and the branches are the events. Per video:

    memory/<video>/tree.json                  the root: the video and its [0, duration]
    memory/<video>/<chunk:03d>/_folder.json   a folder: the chunk's span and its one-line summary
    memory/<video>/<chunk:03d>/<t0>-<t1>.e<id>.json   a leaf: one event, with its sighting,
                                             status, provenance, and its whole history

**One writer.** `remember_chunk` (the typed gate) is the only function that creates a leaf; the
operations below only rewrite a leaf and append to its `history`. The session view
(`chunks`/`notes`/`chunk_map`) is a **rebuild** of this directory, so the in-session consumers -
the audit, the probes, `memory_after`, replay - keep reading the shape they already read, and
there is one truth rather than two."""

import copy
import json
import os

CHUNK_SECONDS = 64
FRAME_STORE_PX = 256        # the stored frame's long side - coarse on purpose
FRAME_STORE_QUALITY = 75
MAX_REVIEW_FRAMES = 12       # a review looks at the memory's own pixels: enough to see, not all 32
DEAD_LINE = "(no supported claim)"


def leaf_path(root, chunk, span, entry_id):
    """`memory/<video>/<chunk:03d>/<t0>-<t1>.e<id>.json` - the address is the time.

    The folder is zero-padded so a plain string sort of a directory listing is the timeline's
    order (chunk 7 before chunk 70), because the listing *is* the index.
    """
    return "%s/%03d/%d-%d.e%d.json" % (root.rstrip("/"), int(chunk),
                                       int(span[0]), int(span[1]), int(entry_id))


def chunk_span(chunk, duration=None, chunk_seconds=CHUNK_SECONDS):
    """The folder's own span: `[ (chunk-1) * 64, chunk * 64 )`, clipped to the video."""
    if int(chunk) == 0:
        return [0.0, float(duration) if duration else 0.0]
    first = (int(chunk) - 1) * chunk_seconds
    last = first + chunk_seconds
    if duration:
        last = min(last, float(duration))
    return [float(first), float(last)]


def frames_dir(root, video, chunk):
    """`memory/<video>/<chunk:03d>/frames` - the pixels of one chunk, beside its leaves."""
    return "%s/%s/%03d/frames" % (str(root).rstrip("/"), str(video), int(chunk))


def save_frames(root, video, chunk, span, images, max_side=FRAME_STORE_PX,
                quality=FRAME_STORE_QUALITY, name=None):
    """**Store the frames a read saw**, at a declared coarse size; return the leaf's own record.

    One JPEG per frame, under the chunk's `frames/` directory. `None` entries are skipped (the tests
    and the ablation pass placeholder lists), and a read with no real image returns `None` - so a leaf
    **never claims frames it does not have**, and a re-look can tell from the leaf alone whether it may
    skip the video decode."""
    real = [image for image in (images or ()) if image is not None
            and getattr(image, "size", None) is not None]
    if not real:
        return None
    from PIL import Image                       # local: the store is text-only except for this
    out = frames_dir(root, video, chunk)
    if name:
        out = os.path.join(out, str(name))
    os.makedirs(out, exist_ok=True)
    stamp = str(name or ("r%d-%d" % (int(span[0]), int(span[1]))))
    written = 0
    for at, image in enumerate(images or ()):
        if image is None or getattr(image, "size", None) is None:
            continue
        size = image.size
        scale = min(1.0, float(max_side) / max(size))
        if scale < 1.0:
            image = image.resize((max(1, int(size[0] * scale)), max(1, int(size[1] * scale))))
        path = "%s/%s.%02d.jpg" % (out, stamp, at)
        try:
            image.convert("RGB").save(path, "JPEG", quality=int(quality))
        except OSError:
            continue                            # a frame that will not encode is not stored
        written += 1
    if not written:
        return None
    return {"dir": out, "n": written, "px": int(max_side), "read": [float(span[0]), float(span[1])]}


def load_frames(record, limit=MAX_REVIEW_FRAMES, around=None, span=None):
    """**The frames come back out of the store**.

    A review is O looking at the memory a reader left - its words and the pixels it saw - so the
    images are read from `record["dir"]`, in the order they were stored, and the video is never
    touched. A leaf with no frames (or a directory that is gone) gives an empty list rather than an
    error: "this memory has no pixels" is a fact to report, not a crash."""
    if not record:
        return []
    out_dir = str(record.get("dir") or "")
    if not out_dir or not os.path.isdir(out_dir):
        return []
    names = sorted(name for name in os.listdir(out_dir) if name.endswith(".jpg"))
    total = len(names)
    keep = list(range(total))
    if limit and total > int(limit):
        want = int(limit)
        if around and span and float(span[1]) > float(span[0]):
            fraction = (float(around) - float(span[0])) / (float(span[1]) - float(span[0]))
            centre = max(0.0, min(1.0, fraction)) * (total - 1)
            start = int(round(centre - (want - 1) / 2.0))
            start = max(0, min(total - want, start))
            keep = list(range(start, start + want))
        else:
            step = (total - 1) / float(want - 1) if want > 1 else 0.0
            keep = sorted({int(round(at * step)) for at in range(want)})
    from PIL import Image
    images, shown = [], []
    for index in keep:
        try:
            images.append(Image.open(os.path.join(out_dir, names[index])).convert("RGB"))
            shown.append(index)
        except OSError:
            continue
    return _ShownFrames(images, shown, total)


class _ShownFrames(list):
    """The images a review was shown, plus **which of the read's frames they were**."""

    def __init__(self, images, indices, total):
        super().__init__(images)
        self.indices = list(indices)
        self.total = int(total)


def write_json(path, payload):
    """**Atomic**: a temp file beside the target, then `os.replace`, so a reader never sees a
    half-written leaf and a killed writer leaves no partial file in the tree."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(temp, path)


def save_leaf(root, video, chunk, entry, span, duration=None, chunk_seconds=CHUNK_SECONDS,
              assigned=None, frames=None):
    """Create a leaf under its chunk folder - **the only way a leaf comes into being**.

    The root and the folder are written beside it, because a leaf with no folder above it is not
    in a tree. `parent_span` is the folder the leaf is stored in; the leaf's own `span` may be
    wider than it (an event that straddles the grid), and that is what the containment invariant
    is for.
    """
    video_dir = os.path.join(str(root).rstrip("/"), str(video))
    write_json(os.path.join(video_dir, "tree.json"),
               {"level": "video", "video": str(video),
                "span": [0.0, float(duration) if duration else float(span[1])]})
    # **The folder is the grid, not the leaf** (the first cut inferred the duration from the
    # leaf's own end, which shortened every folder to its child and made containment vacuous).
    # An unknown duration leaves the folder unclipped, which is what the grid means.
    folder = chunk_span(chunk, duration, chunk_seconds)
    write_json(os.path.join(video_dir, "%03d" % int(chunk), "_folder.json"),
               {"level": "chunk", "chunk": int(chunk), "span": folder,
                # **The folder keeps today's four-field note as well as the leaf list**, so the
                # session view is a pure read of the directory: `summary` is the one line the
                # push view renders, and `note` is what `session["notes"]` carries.
                "summary": str(entry.get("sighting") or ""),
                "note": {"chunk": int(chunk), "sighting": str(entry.get("sighting") or ""),
                         "span": [float(span[0]), float(span[1])],
                         "family": str(entry.get("family") or ""),
                         "question_number": entry.get("question_number")}})
    leaf = {"id": int(entry["id"]), "level": "event",
            "span": [float(span[0]), float(span[1])],
            "parent": int(chunk), "parent_span": folder,
            "sighting": str(entry.get("sighting") or ""),
            "kind": str(entry.get("kind") or ""), "family": str(entry.get("family") or ""),
            "assigned": ([float(assigned[0]), float(assigned[1])] if assigned is not None else None),
            "frames": (dict(frames) if frames else
                       (dict(entry.get("frames")) if entry.get("frames") else None)),
            "status": "verified" if entry.get("verified") else "unverified",
            "citable": bool(entry.get("citable", True)),
            # **Support is earned by review, not guessed from a raw answer.** A reader's
            # answer-bearing observation exists before O reviews it, so both flags start false
            # unless the caller is replaying an already-reviewed record.
            "answer_support": bool(entry.get("answer_support", False)),
            "citation_allowed": bool(entry.get("citation_allowed", False)),
            "global": bool(entry.get("global", False)),
            "review_status": str(entry.get("review_status") or "unreviewed"),
            "raw_reply": str(entry.get("raw_reply") or ""),
            "quote": str(entry.get("quote") or ""),
            "answer": str(entry.get("answer") or ""),
            "claim": copy.deepcopy(entry.get("claim")),
            "provisional": bool(entry.get("provisional", False)),
            "verified": entry.get("verified"),
            "answer_side": str(entry.get("answer_side") or ""),
            "history": [],
            "provenance": {"author": str(entry.get("author") or ""),
                           "question": str(entry.get("question") or ""),
                           "question_number": entry.get("question_number"),
                           "cost": entry.get("cost") or {}}}
    path = leaf_path(video_dir, chunk, span, entry["id"])
    write_json(path, leaf)
    return path


def _find(video_dir, entry_id):
    for path, leaf in _leaf_files(video_dir):
        if int(leaf["id"]) == int(entry_id):
            return path, leaf
    return None, None


def mark_leaf(root, video, entry_id, status=None, span=None, by="", why="", fields=None):
    """**The one writer of a leaf's state**.

    A verdict, a retraction, a re-confirmation or a reader's finer place all arrive here, so the
    tree is told by the same door and the leaf's `history` keeps what it used to say. Nothing is
    deleted; the folder's `parent_span` never moves, so containment stays true while the leaf's own
    span becomes the event it resolved."""
    video_dir = os.path.join(str(root).rstrip("/"), str(video))
    path, leaf = _find(video_dir, entry_id)
    if leaf is None:
        raise KeyError("no leaf %s in %s" % (entry_id, video))
    note = {"by": str(by), "why": str(why)}
    updates = dict(fields or {})
    changed = {key: copy.deepcopy(leaf.get(key)) for key in updates
               if leaf.get(key) != updates[key]}
    if changed:
        note["prior_fields"] = changed
        leaf.update(copy.deepcopy(updates))
    if status is not None and str(status) != str(leaf.get("status")):
        note["prior_status"] = leaf.get("status")
        leaf["status"] = str(status)
    if span is not None:
        span = [float(span[0]), float(span[1])]
        if span != leaf.get("span"):
            note["prior_span"] = leaf.get("span")
            leaf["span"] = span
    leaf.setdefault("history", []).append(note)
    write_json(path, leaf)
    return leaf


def _leaf_files(video_dir):
    """`[(path, leaf), ...]` - every leaf on disk, whatever its folder is called."""
    out = []
    for name in sorted(os.listdir(video_dir)) if os.path.isdir(video_dir) else []:
        folder = os.path.join(video_dir, name)
        if not os.path.isdir(folder):
            continue
        for leaf_name in sorted(os.listdir(folder)):
            if leaf_name.endswith(".json") and leaf_name != "_folder.json":
                with open(os.path.join(folder, leaf_name)) as handle:
                    out.append((os.path.join(folder, leaf_name), json.load(handle)))
    return out


def census(root, video):
    """Summarise the structure of a video's memory tree.

    `spans_not_the_grid` counts leaves whose span is not their whole folder: an event level
    only exists where a read resolved a span finer than the 64-second chunk. `generic_share` is
    1 - distinct sightings / leaves. `merge_opportunities` counts pairs of leaves in one folder
    whose spans overlap and whose sightings are identical."""
    video_dir = os.path.join(str(root).rstrip("/"), str(video))
    leaves = [leaf for _path, leaf in _leaf_files(video_dir)]
    sightings = [str(leaf.get("sighting") or "").strip().lower() for leaf in leaves]
    statuses = {}
    for leaf in leaves:
        statuses[leaf.get("status") or "unverified"] = statuses.get(leaf.get("status")
                                                                   or "unverified", 0) + 1
    not_the_grid = 0
    for leaf in leaves:
        parent = leaf.get("parent_span") or []
        span = leaf.get("span") or []
        if len(span) < 2 or len(parent) < 2 or abs(span[0] - parent[0]) > 1e-9 \
                or abs(span[1] - parent[1]) > 1e-9:
            not_the_grid += 1
    by_folder = {}
    for leaf in leaves:
        by_folder.setdefault(leaf.get("parent"), []).append(leaf)
    merges = 0
    contradictions = 0
    for folder_leaves in by_folder.values():
        for index, first in enumerate(folder_leaves):
            for second in folder_leaves[index + 1:]:
                same = (str(first.get("sighting") or "").strip().lower()
                        == str(second.get("sighting") or "").strip().lower())
                a, b = first.get("span") or [], second.get("span") or []
                if same and len(a) >= 2 and len(b) >= 2 and min(a[1], b[1]) > max(a[0], b[0]):
                    merges += 1
                elif not same and len(a) >= 2 and len(b) >= 2 and min(a[1], b[1]) > max(a[0], b[0]):
                    contradictions += 1
    return {"video": str(video), "leaves": len(leaves), "folders": len(by_folder),
            "statuses": statuses, "spans_not_the_grid": not_the_grid,
            "merge_opportunities": merges,
            "merge_note": ("identical sightings on overlapping spans - the generic duplicates, "
                           "where folding is a no-op until the write side improves"),
            "contradiction_pairs": contradictions,
            "generic_share": (1 - len(set(sightings)) / len(sightings)) if sightings else None,
            "distinct_sightings": len(set(sightings))}


def tree_lines(root, video, handles=None, limit=24, cues=None, offers=None):
    """**The tree as O is shown it**: one line per folder, paths and statuses and counts."""
    video_dir = os.path.join(str(root).rstrip("/"), str(video))
    folders = {}
    statuses = {}
    for _path, leaf in _leaf_files(video_dir):
        folders.setdefault(leaf.get("parent"), []).append(leaf)
        statuses[leaf.get("status")] = statuses.get(leaf.get("status"), 0) + 1
    if not folders:
        return []
    counts = ", ".join("%d %s" % (value, key) for key, value in sorted(statuses.items()))
    head = "tree: %d folder(s), %d leaf/leaves" % (
        len(folders), sum(len(leaves) for leaves in folders.values()))
    if counts:
        room = 64 - len(head) - 3
        head += " (%s%s)" % (counts[:max(0, room)], "..." if len(counts) > max(0, room) else "")
    lines = [head]
    if offers is not None:
        for offer in list(offers)[:max(0, limit - 1)]:
            lines.append(offer_line(offer))
        return lines
    for chunk in sorted(folders):
        leaves = sorted(folders[chunk], key=lambda leaf: (leaf["span"][0], leaf["id"]))
        for leaf in leaves:
            handle = (handles or {}).get(int(leaf["id"]))
            cue = str((cues or {}).get(int(leaf["id"])) or "")
            lines.append(offer_line({"handle": handle, "chunk": chunk,
                                     "address": [leaf["span"][0], leaf["span"][1]],
                                     "status": leaf.get("status") or "unverified",
                                     "cue": cue}))
            if len(lines) >= limit:
                return lines
    return lines


def offer_line(offer):
    """**The one rendered leaf line** - the same string `chunked_team.tree_offer_line` builds.

    Duplicated rather than imported (the import would be circular), and `test_tree_wiring` holds the
    two to each other: a rendered form the audit does not know is how a run voids for words its own
    renderer was told to show.
    """
    span = list(offer.get("address") or [])
    chunk = offer.get("chunk") or 0
    handle = offer.get("handle")
    cue = str(offer.get("cue") or "")
    return "%s%s %d-%d %s%s" % (
        ("%s " % handle) if handle else "", chunk,
        int(span[0]) if span else 64 * (chunk - 1),
        int(span[1]) if span else 64 * chunk,
        offer.get("status") or "unverified",
        (" \"%s\"" % cue) if cue else "")
