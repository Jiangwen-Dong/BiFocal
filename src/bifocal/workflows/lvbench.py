"""Load a pinned LVBench selection into runtime items."""

from __future__ import annotations

import hashlib
import json
import os


SCHEMA_VERSION = "lvbench-custom-o-selection-v1"
SPLITS = ("train", "dev", "test", "feedback", "reserve")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate(block):
    if block.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported LVBench selection schema")
    stored = str(block.get("canonical_sha256") or "")
    copy = dict(block)
    copy.pop("canonical_sha256", None)
    if stored != _canonical_sha256(copy):
        raise ValueError("LVBench selection canonical hash changed")
    splits = block.get("splits") or {}
    if set(splits) != set(SPLITS):
        raise ValueError("LVBench selection must contain all five splits")
    videos = block.get("videos") or []
    ids = [str(row.get("video") or "") for row in videos]
    if not videos or "" in ids or len(ids) != len(set(ids)):
        raise ValueError("LVBench selection videos are missing or duplicated")
    split_members = {name: set(str(v) for v in splits.get(name) or ()) for name in SPLITS}
    union = set().union(*split_members.values())
    if union != set(ids):
        raise ValueError("LVBench split membership differs from the video list")
    for index, name in enumerate(SPLITS):
        for left in SPLITS[:index]:
            if split_members[left] & split_members[name]:
                raise ValueError("LVBench splits overlap")
    if block.get("split_counts") != {name: len(split_members[name]) for name in SPLITS}:
        raise ValueError("LVBench split video counts changed")
    items = block.get("items") or []
    uids = [str(row.get("uid") or "") for row in items]
    if "" in uids or len(uids) != len(set(uids)):
        raise ValueError("LVBench item UIDs are missing or duplicated")
    for item in items:
        if str(item.get("video")) not in union:
            raise ValueError("LVBench item names an unselected video")
    expected_items = {
        name: sum(str(item.get("video")) in split_members[name] for item in items)
        for name in SPLITS
    }
    if block.get("split_items") != expected_items:
        raise ValueError("LVBench split item counts changed")
    expected_stamped = {
        name: sum(str(item.get("video")) in split_members[name]
                  and bool(item.get("stamped")) for item in items)
        for name in SPLITS
    }
    if block.get("split_stamped") != expected_stamped:
        raise ValueError("LVBench split stamped counts changed")
    if block.get("split_unstamped") != {
            name: expected_items[name] - expected_stamped[name] for name in SPLITS}:
        raise ValueError("LVBench split unstamped counts changed")
    return block, split_members, {row["video"]: row for row in videos}


def load_items(data_root, selection_path, split, *, expected_sha256=None,
               require_pinned=False):
    if split not in SPLITS[:-1]:
        raise ValueError("unknown or sealed LVBench split: %s" % split)
    actual_sha = _sha256_file(selection_path)
    if require_pinned and not expected_sha256:
        raise ValueError("a pinned LVBench selection requires its file SHA256")
    if expected_sha256 and actual_sha != expected_sha256:
        raise ValueError("LVBench selection file SHA256 differs")
    with open(selection_path, encoding="utf-8") as handle:
        block = json.load(handle)
    block, split_members, video_by_id = _validate(block)
    selected_videos = sorted(split_members[split])
    selected_ids = set(selected_videos)

    if require_pinned:
        metadata_path = os.path.join(data_root, "lvbench-meta.jsonl")
        if _sha256_file(metadata_path) != block["source"]["metadata_sha256"]:
            raise ValueError("LVBench metadata changed")
        for video in selected_videos:
            row = video_by_id[video]
            local = os.path.join(data_root, row["local_path"])
            if not os.path.isfile(local):
                raise ValueError("LVBench video is missing: %s" % local)
            if os.path.getsize(local) != int(row["size"]):
                raise ValueError("LVBench video size changed: %s" % video)
            if _sha256_file(local) != row["sha256"]:
                raise ValueError("LVBench video SHA256 changed: %s" % video)

    items = []
    positions = {}
    for row in block["items"]:
        video = str(row["video"])
        if video not in selected_ids:
            continue
        meta = video_by_id[video]
        position = positions.get(video, 0)
        positions[video] = position + 1
        items.append({
            "uid": row["uid"],
            "source_uid": row["source_uid"],
            "video": video,
            "video_path": os.path.join(data_root, meta["local_path"]),
            "frame_dir": os.path.join(data_root, "frames", video),
            "meta_duration": float(row["meta_duration"]),
            "question": row["question"],
            "answer": row["answer"],
            "types": list(row["types"]),
            "question_type": row["types"][0],
            "time_reference": row["time_reference"],
            "window": list(row["window"]),
            "stamped": bool(row["stamped"]),
            "direct_o_split": split,
            "session_position": position,
            "session_length": sum(
                1 for candidate in block["items"]
                if str(candidate.get("video")) == video),
        })
    expected = int((block.get("split_items") or {}).get(split) or 0)
    if len(items) != expected:
        raise ValueError("LVBench runtime item count changed for %s" % split)
    report = {
        "schema_version": SCHEMA_VERSION,
        "data_format": "lvbench-selection",
        "selection_sha256": actual_sha,
        "canonical_sha256": block["canonical_sha256"],
        "split": split,
        "selected_items": len(items),
        "selected_videos": len(selected_videos),
        "selected_stamped": sum(item["stamped"] for item in items),
        "selected_unstamped": sum(not item["stamped"] for item in items),
    }
    return items, report

