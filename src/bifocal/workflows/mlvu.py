"""Load MLVU multiple-choice annotations into chunked-team session items.

The loader is read-only. It indexes the local MLVU videos, reads the seven
letter-scorable multiple-choice task files, groups questions by video, and refuses
the generative tasks that the current letter-based loop cannot score.

Expected layout under the data root::

    <data-root>/mlvu/MLVU/json/<task>.json
    <data-root>/mlvu/MLVU/video/<task-folder>/<video>.mp4
"""

from __future__ import annotations

import hashlib
import json
import os


MULTIPLE_CHOICE_TASKS = (
    "1_plotQA", "2_needle", "3_ego", "4_count", "5_order",
    "6_anomaly_reco", "7_topic_reasoning",
)
# Generative tasks are not scored by the letter-based loop and are never loaded.
EXCLUDED_TASKS = ("8_sub_scene", "9_summary")


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_key(path):
    return "mlvu-" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _option_text(candidates):
    return "\n".join("(%s) %s" % (chr(65 + index), text)
                     for index, text in enumerate(candidates))


def _all_video_files(data_root):
    """Index MLVU MP4 files by basename and relative path."""
    root = os.path.join(data_root, "mlvu", "MLVU", "video")
    if not os.path.isdir(root):
        raise ValueError("MLVU video directory is missing: %s" % root)
    by_name = {}
    for folder, _dirs, files in os.walk(root):
        for name in files:
            if not name.lower().endswith(".mp4"):
                continue
            path = os.path.join(folder, name)
            by_name.setdefault(name, []).append(path)
    return by_name


def load_all_mcq_items(data_root, require_files=True, require_duration=True):
    """Load every scorable MLVU multiple-choice item from the task JSON files.

    The return value is grouped by video and keeps annotation order inside each
    task. The report records every rejected annotation instead of silently
    dropping it.
    """
    videos = _all_video_files(data_root)
    rejected = {
        "invalid_options_or_answer": 0,
        "missing_video": 0,
        "duplicate_video_name": 0,
        "missing_duration": 0,
    }
    grouped = {}
    annotation_sha256 = {}
    for task in MULTIPLE_CHOICE_TASKS:
        path = os.path.join(data_root, "mlvu", "MLVU", "json", task + ".json")
        if not os.path.isfile(path):
            raise ValueError("MLVU annotation is missing: %s" % task)
        annotation_sha256[task] = _file_sha256(path)
        with open(path, encoding="utf-8") as handle:
            rows = json.load(handle)
        for position, row in enumerate(rows):
            options = [str(value) for value in row.get("candidates") or ()]
            answer = str(row.get("answer") or "")
            if not options or answer not in options:
                rejected["invalid_options_or_answer"] += 1
                continue
            basename = os.path.basename(str(row.get("video") or ""))
            matches = videos.get(basename) or []
            if not matches:
                rejected["missing_video"] += 1
                if require_files:
                    raise ValueError("MLVU video is missing: %s" % basename)
                continue
            if len(matches) != 1:
                rejected["duplicate_video_name"] += 1
                raise ValueError("MLVU video name is ambiguous: %s" % basename)
            video_path = matches[0]
            relative = os.path.relpath(video_path, os.path.join(data_root, "mlvu"))
            duration = float(row.get("duration") or 0.0)
            if duration <= 0.0:
                rejected["missing_duration"] += 1
                if require_duration:
                    raise ValueError("MLVU item has no duration: %s" % basename)
            video_key = _video_key(relative)
            grouped.setdefault(video_key, []).append({
                "uid": "mlvu-all:%s:%s:%d" % (task, video_key, position),
                "video": video_key,
                "video_path": video_path,
                "frame_dir": os.path.join(data_root, "mlvu", "frames", video_key),
                "meta_duration": duration,
                "question": "%s\n%s" % (row.get("question") or "",
                                         _option_text(options)),
                "candidates": options,
                "answer": chr(65 + options.index(answer)),
                "gold_answer_text": answer,
                "types": [task],
                "question_type": task,
                "task_file": task,
                "source_item_position": position,
                "source_video": relative,
            })
    items = []
    for video_key in sorted(grouped):
        rows = sorted(
            grouped[video_key],
            key=lambda row: (
                MULTIPLE_CHOICE_TASKS.index(str(row["task_file"])),
                int(row["source_item_position"]),
            ),
        )
        for position, row in enumerate(rows):
            item = dict(row)
            item["session_position"] = position
            item["session_length"] = len(rows)
            item["question_number"] = position + 1
            items.append(item)
    if not items:
        raise ValueError("MLVU annotations contain no scorable items")
    report = {
        "data_format": "mlvu-all-mcq",
        "schema_version": "mlvu-all-mcq-v1",
        "data_root": os.path.abspath(data_root),
        "annotation_sha256": annotation_sha256,
        "selected_items": len(items),
        "selected_videos": len(grouped),
        "rejected": rejected,
    }
    return items, report

