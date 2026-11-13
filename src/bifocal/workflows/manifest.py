#!/usr/bin/env python3
"""Build three video-disjoint MLVU online-RL round manifests.

Each round contains the same number of two-question video groups. Each
video stays in one round and its two questions keep source task order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from ..harness import actions as A
from ..memory import artifacts as AMA
from . import mlvu as MLVU


SCHEMA_VERSION = "mlvu-online-rl-manifest-v1"
LAYOUT_VERSION = "mlvu-online-sequence-layout-v1"
PLAN_SCHEMA = "mlvu-online-rl-plan-v1"
VIDEO_GROUP_QUESTIONS = 2
MANIFEST_NAME_FORMAT = "manifest-round-%d.json"
PLAN_NAME = "plan.json"


def _stable_hash(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _by_video(items):
    grouped = {}
    for item in items:
        grouped.setdefault(str(item["video"]), []).append(dict(item))
    for rows in grouped.values():
        rows.sort(key=lambda row: (
            int(row.get("session_position") or 0), str(row.get("uid") or "")))
        if not rows:
            raise ValueError("MLVU video has no questions")
        if len({str(row["video_path"]) for row in rows}) != 1:
            raise ValueError("MLVU video has multiple paths")
    return grouped


def partition_rounds(items, rounds=3, questions_per_round=600,
                     video_group_questions=VIDEO_GROUP_QUESTIONS,
                     order_salt="mlvu-bifocal-rl-v1"):
    """Select equal two-question video groups for three disjoint rounds."""
    rounds = int(rounds)
    target = int(questions_per_round)
    per_video = int(video_group_questions)
    if rounds != 3:
        raise ValueError("the MLVU campaign requires exactly three rounds")
    if target <= 0 or per_video <= 0 or target % per_video:
        raise ValueError(
            "questions per round must be divisible by video group size")
    videos_per_round = target // per_video
    required_videos = rounds * videos_per_round
    grouped = _by_video(items)
    total = sum(len(rows) for rows in grouped.values())
    eligible = {
        video: rows[:per_video]
        for video, rows in grouped.items() if len(rows) >= per_video
    }
    short_videos = len(grouped) - len(eligible)
    ordered = sorted(
        eligible,
        key=lambda video: _stable_hash("%s:video:%s" % (order_salt, video)),
    )
    if len(ordered) < required_videos:
        raise ValueError(
            "MLVU has %d videos with at least %d local questions; "
            "three rounds need %d videos"
            % (len(eligible), per_video, required_videos))
    selected = ordered[:required_videos]
    out = []
    for round_index in range(rounds):
        chunk = selected[
            round_index * videos_per_round:(round_index + 1) * videos_per_round]
        chunk = sorted(chunk, key=lambda video: _stable_hash(
            "%s:round:%d:%s" % (order_salt, round_index + 1, video)))
        out.append(chunk)
    seen = [video for values in out for video in values]
    if len(seen) != len(set(seen)):
        raise ValueError("round partitioning repeated a video")
    loads = [target for _ in range(rounds)]
    return (
        out, grouped, eligible, loads, total, len(grouped) - required_videos,
        short_videos,
    )

def _initial_tree_sha256(video, duration):
    tree = A.AtomicTimeTree(str(video), float(duration))
    return AMA.canonical_sha256(tree.to_dict())


def _states_for_video(video, items):
    rows = sorted(items, key=lambda row: int(row["question_number"]))
    duration = float(rows[0].get("meta_duration") or 0.0)
    if duration <= 0.0:
        raise ValueError("MLVU video has no duration: %s" % video)
    if any(abs(float(row.get("meta_duration") or 0.0) - duration) > 1e-9
           for row in rows):
        raise ValueError("MLVU video duration differs across items: %s" % video)
    initial = _initial_tree_sha256(video, duration)
    states = []
    for number, item in enumerate(rows, start=1):
        states.append({
            "uid": str(item["uid"]),
            "video": str(video),
            "video_path": os.path.abspath(str(item["video_path"])),
            "frame_dir": os.path.abspath(str(item["frame_dir"])),
            "meta_duration": duration,
            "question": str(item["question"]),
            "candidates": list(item.get("candidates") or ()),
            "answer": str(item["answer"]),
            "types": list(item.get("types") or ()),
            "question_type": str(item.get("question_type") or ""),
            "task_file": str(item.get("task_file") or ""),
            "question_number": number,
            "source_item_position": int(item.get("source_item_position") or 0),
            "session_position": number - 1,
            "session_length": len(rows),
            "split": "dev",
            "initial_state": "empty",
            "initial_tree_sha256": initial,
        })
    sequence = {
        "video": str(video),
        "uids": [row["uid"] for row in states],
        "question_numbers": list(range(1, len(states) + 1)),
        "source_item_positions": [
            row["source_item_position"] for row in states],
        "start_question": 1,
        "initial_state": "empty",
        "initial_tree_sha256": initial,
    }
    return states, sequence


def build_rounds(items, report, rounds=3, questions_per_round=600,
                 video_group_questions=VIDEO_GROUP_QUESTIONS,
                 order_salt="mlvu-bifocal-rl-v1", data_root=""):
    (videos_by_round, grouped, eligible, loads, total, unused_videos,
     short_videos) = partition_rounds(
        items, rounds=rounds, questions_per_round=questions_per_round,
        video_group_questions=video_group_questions,
        order_salt=order_salt)
    manifests = []
    for index, videos in enumerate(videos_by_round, start=1):
        states, sequences = [], []
        for video in videos:
            video_states, sequence = _states_for_video(video, eligible[video])
            states.extend(video_states)
            sequences.append(sequence)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "layout_version": LAYOUT_VERSION,
            "benchmark": "mlvu",
            "training_allowed": True,
            "training_scope": "development_split",
            "data_split": "dev",
            "round_index": index,
            "round_count": int(rounds),
            "target_questions": int(questions_per_round),
            "video_group_questions": int(video_group_questions),
            "order_salt": str(order_salt),
            "source": {
                "data_root": os.path.abspath(str(data_root or report.get(
                    "data_root") or "")),
                "annotation_sha256": dict(report["annotation_sha256"]),
            },
            "selection_rule": (
                "Select local videos with at least two scorable MLVU "
                "multiple-choice questions. Take the first two questions in "
                "source task order. Keep both in one round; each round has "
                "the same number of two-question video groups."),
            "questions_per_sequence": int(video_group_questions),
            "train_states": states,
            "train_sequences": sequences,
            "counts": {
                "round_questions": len(states),
                "round_videos": len(sequences),
                "available_questions": int(total),
                "available_videos": len(grouped),
                "eligible_video_groups": len(eligible),
                "skipped_short_videos": int(short_videos),
                "unused_questions": int(total - sum(loads)),
                "unused_videos": int(unused_videos),
                "selected_questions": int(sum(loads)),
                "selected_videos": int(sum(len(values) for values in videos_by_round)),
                "video_group_questions": int(video_group_questions),
            },
        }
        manifest["manifest_sha256"] = AMA.canonical_sha256(manifest)
        manifests.append(manifest)
    return manifests, {
        "round_items": [int(value) for value in loads],
        "round_videos": [len(values) for values in videos_by_round],
        "available_items": int(total),
        "available_videos": len(grouped),
        "eligible_video_groups": len(eligible),
        "skipped_short_videos": int(short_videos),
        "selected_items": int(sum(loads)),
        "selected_videos": int(sum(len(values) for values in videos_by_round)),
        "unused_items": int(total - sum(loads)),
        "unused_videos": int(unused_videos),
        "video_group_questions": int(video_group_questions),
    }


def build(data_root, rounds=3, questions_per_round=600,
          video_group_questions=VIDEO_GROUP_QUESTIONS,
          order_salt="mlvu-bifocal-rl-v1", require_files=False):
    items, report = MLVU.load_all_mcq_items(
        data_root, require_files=require_files, require_duration=True)
    manifests, counts = build_rounds(
        items, report, rounds=rounds,
        questions_per_round=questions_per_round,
        video_group_questions=video_group_questions,
        order_salt=order_salt, data_root=data_root)
    return manifests, report, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--questions-per-round", type=int, default=600)
    parser.add_argument("--video-group-questions", type=int, default=2)
    parser.add_argument("--order-salt", default="mlvu-bifocal-rl-v1")
    parser.add_argument(
        "--require-all-videos", action="store_true",
        help="fail if any annotation video is absent locally; default skips it")
    args = parser.parse_args(argv)
    try:
        if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
            raise ValueError("output directory is not empty")
        manifests, report, counts = build(
            args.data, rounds=args.rounds,
            questions_per_round=args.questions_per_round,
            video_group_questions=args.video_group_questions,
            order_salt=args.order_salt,
            require_files=bool(args.require_all_videos))
        os.makedirs(args.out_dir, exist_ok=True)
        paths = []
        for index, manifest in enumerate(manifests, start=1):
            path = os.path.join(args.out_dir, MANIFEST_NAME_FORMAT % index)
            _write_json(path, manifest)
            paths.append({
                "round": index,
                "path": os.path.abspath(path),
                "sha256": _file_sha256(path),
                "questions": manifest["counts"]["round_questions"],
                "videos": manifest["counts"]["round_videos"],
            })
        plan = {
            "schema_version": PLAN_SCHEMA,
            "data_root": os.path.abspath(args.data),
            "rounds": int(args.rounds),
            "target_questions": int(args.questions_per_round),
            "video_group_questions": int(args.video_group_questions),
            "order_salt": str(args.order_salt),
            "annotation_sha256": report["annotation_sha256"],
            "counts": counts,
            "rejected_annotations": report["rejected"],
            "manifests": paths,
        }
        plan_path = os.path.join(args.out_dir, PLAN_NAME)
        _write_json(plan_path, plan)
    except (OSError, ValueError, KeyError) as error:
        print("MLVU manifest build refused: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
