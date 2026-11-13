#!/usr/bin/env python3
"""Paired, video-clustered comparison of two evaluation runs.

Reads `question-ledger.jsonl` and `eval-only.jsonl` from a baseline run directory and a
candidate run directory, pairs the same question and rollout in both, and writes the paired
accuracy difference with a percentile bootstrap interval that resamples videos. A pair
counts only when both rows are clean and no earlier question of the same video and rollout
was unclean in either arm. Every pair left out is counted by reason. Wrong answers,
abstentions and protocol failures score zero. Gold answers are read only from
`eval-only.jsonl`.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

SCHEMA = "paired-compare-v1"


def _rows(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_arm(run_dir):
    gold = {row["row_id"]: row for row in _rows(
        os.path.join(run_dir, "eval-only.jsonl")) if row.get("row_id")}
    arm = {}
    for row in _rows(os.path.join(run_dir, "question-ledger.jsonl")):
        key = (row["video_id"], row.get("question_id"), int(row["rollout_index"]))
        if key in arm:
            raise ValueError("duplicate ledger row for %s" % (key,))
        evaluation = gold.get(row.get("row_id")) or {}
        cost = row.get("cost") or {}
        arm[key] = {
            "question_number": int(row["question_number"]),
            "correct": bool(evaluation.get("correct")),
            "clean": (bool(row.get("clean")) and bool(evaluation)
                      and row.get("disposition", "kept") != "rolled_back"),
            "calls": cost.get("calls_total"),
            "frames": cost.get("frames_selected"),
        }
    return arm


def pair_arms(baseline, candidate):
    """Classify every pair as primary, unclean, cascade or missing_pair."""
    first_bad = {}
    for arm in (baseline, candidate):
        for (video, _question, rollout), row in arm.items():
            if not row["clean"]:
                slot = (video, rollout)
                first_bad[slot] = min(first_bad.get(slot, 10 ** 9),
                                      row["question_number"])
    pairs = []
    for key in sorted(set(baseline) | set(candidate), key=str):
        video, _question, rollout = key
        left, right = baseline.get(key), candidate.get(key)
        if left is None or right is None:
            status = "missing_pair"
        elif not (left["clean"] and right["clean"]):
            status = "unclean"
        elif left["question_number"] > first_bad.get((video, rollout), 10 ** 9):
            status = "cascade"
        else:
            status = "primary"
        pairs.append({"video": video, "status": status, "left": left, "right": right})
    return pairs


def _score(row):
    return 1.0 if row and row["correct"] and row["clean"] else 0.0


def cluster_totals(pairs):
    """Per video: pair count and summed (candidate - baseline) correctness."""
    totals = {}
    for pair in pairs:
        if pair["status"] != "primary":
            continue
        slot = totals.setdefault(pair["video"], [0, 0.0, 0.0, 0.0])
        slot[0] += 1
        slot[1] += _score(pair["right"]) - _score(pair["left"])
        slot[2] += _score(pair["left"])
        slot[3] += _score(pair["right"])
    return totals


def bootstrap(totals, iterations=10000, seed=0):
    """Percentile interval of the paired difference, resampling whole videos."""
    videos = sorted(totals)
    count = sum(totals[video][0] for video in videos)
    if not videos or not count:
        return {"delta": None, "interval": [None, None], "pairs": 0, "videos": 0}
    delta = sum(totals[video][1] for video in videos) / count
    rng = random.Random(seed)
    draws = []
    for _ in range(int(iterations)):
        picked = [totals[videos[rng.randrange(len(videos))]] for _ in videos]
        size = sum(slot[0] for slot in picked)
        draws.append(sum(slot[1] for slot in picked) / size if size else 0.0)
    draws.sort()
    low = draws[int(0.025 * (len(draws) - 1))]
    high = draws[int(round(0.975 * (len(draws) - 1)))]
    return {"delta": round(delta, 6), "interval": [round(low, 6), round(high, 6)],
            "pairs": count, "videos": len(videos)}


def compare(baseline_dir, candidate_dir, iterations=10000, seed=0):
    pairs = pair_arms(load_arm(baseline_dir), load_arm(candidate_dir))
    totals = cluster_totals(pairs)
    primary = [pair for pair in pairs if pair["status"] == "primary"]

    def mean(side, key):
        values = [pair[side][key] for pair in primary
                  if pair[side].get(key) is not None]
        return round(sum(values) / len(values), 4) if values else None

    result = bootstrap(totals, iterations, seed)
    result.update({
        "schema": SCHEMA,
        "baseline_accuracy": round(sum(v[2] for v in totals.values())
                                   / max(1, result["pairs"]), 4),
        "candidate_accuracy": round(sum(v[3] for v in totals.values())
                                    / max(1, result["pairs"]), 4),
        "baseline_calls": mean("left", "calls"),
        "candidate_calls": mean("right", "calls"),
        "baseline_frames": mean("left", "frames"),
        "candidate_frames": mean("right", "frames"),
        "excluded": dict(collections.Counter(
            pair["status"] for pair in pairs if pair["status"] != "primary")),
        "bootstrap_iterations": int(iterations),
        "bootstrap_seed": int(seed),
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    result = compare(args.baseline, args.candidate, args.iterations, args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
