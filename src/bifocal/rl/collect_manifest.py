#!/usr/bin/env python3
"""Freeze, collect, and evaluate LVBench online-RL sequences."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
import shlex
import sys

from ..harness import actions as A
from ..memory import artifacts as AMA
from ..memory import branches as B
from ..harness import protocol as C
from ..serving import client as LC
from ..harness import frames as K
from ..workflows import lvbench as LVS
from . import episode as E
from ..serving import lanes as RS
from ..harness import calls as RC


MANIFEST_SCHEMA = "lvbench-online-rl-manifest-v1"
DEV_TREND_ITEMS = 28
ROLLOUT_SEEDS = (20261111, 20261112, 20261113, 20261114)
SELECTION_SHA256 = (
    "0000000000000000000000000000000000000000000000000000000000000000")
SOURCE_SEED = 20261105


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _snapshot_sha256(payload):
    """Match the snapshot writer's canonical hash with integer question keys."""
    value = copy.deepcopy(payload)
    tree = value.get("tree") or {}
    for name in ("questions", "branches"):
        if isinstance(tree.get(name), dict):
            tree[name] = {
                int(key): item for key, item in tree[name].items()}
    return AMA.canonical_sha256(value)


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _selection_items(data, selection, split):
    return LVS.load_items(
        data, selection, split, expected_sha256=SELECTION_SHA256,
        require_pinned=True)[0]


def _load_snapshots(directory):
    found = {}
    for name in sorted(os.listdir(directory)):
        if not name.startswith("question-start-") or not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        if record.get("schema_version") != "atomic-question-start-v1":
            raise ValueError("unexpected snapshot schema: %s" % path)
        payload = record.get("payload") or {}
        actual = _snapshot_sha256(payload)
        tree = payload.get("tree") or {}
        key = (str(tree.get("video") or ""),
               int(payload.get("question_number") or 0),
               str(payload.get("question") or ""))
        if key in found:
            raise ValueError("duplicate question-start snapshot: %s" % (key,))
        found[key] = {
            "path": os.path.abspath(path), "sha256": actual,
            "declared_sha256": str(record.get("snapshot_sha256") or ""),
            "declared_hash_matches_payload": (
                record.get("snapshot_sha256") == actual),
            "payload": payload,
        }
    return found


def _source_identity(train_run, dev_run, selection, verify_content=True):
    """Check that train and dev frozen states came from one source policy setup."""
    with open(os.path.join(dev_run, "out", "RUN.json"), encoding="utf-8") as handle:
        dev = json.load(handle)
    selection = os.path.abspath(selection)
    expected = {
        "selection": selection,
        "data_split": "dev",
        "harness_mode": "atomic",
        "legality_only": True,
        "adapter_roles": ["turn"],
    }
    for name, value in expected.items():
        if dev.get(name) != value:
            raise ValueError("dev source identity differs at %s" % name)
    for name in ("model", "model_sha256", "adapter", "adapter_sha256",
                 "code_hash", "run_id"):
        if not str(dev.get(name) or "").strip():
            raise ValueError("dev source identity is missing %s" % name)
    selection_report = dev.get("lvbench_selection") or {}
    if selection_report.get("selection_sha256") != SELECTION_SHA256:
        raise ValueError("dev source selection report has a different SHA")
    if selection_report.get("split") != "dev":
        raise ValueError("dev source selection report is not the dev split")
    if int(dev.get("seed") or 0) != SOURCE_SEED \
            or int(dev.get("policy_seed") or 0) != SOURCE_SEED:
        raise ValueError("dev source seed is not the frozen source seed")
    if verify_content:
        adapter_actual = _sha256_file(os.path.join(
            str(dev["adapter"]), "adapter_model.safetensors"))
        if adapter_actual != dev["adapter_sha256"]:
            raise ValueError("source adapter content differs from the dev run")
        model_actual = RS.model_content_sha256(str(dev["model"]))
        if model_actual != dev["model_sha256"]:
            raise ValueError("source model content differs from the dev run")
    launch_path = os.path.join(train_run, "launch.sh")
    with open(launch_path, encoding="utf-8") as handle:
        launch = handle.read()
    tokens = shlex.split(launch)
    required = {
        "--selection": selection,
        "--selection-sha256": SELECTION_SHA256,
        "--data-split": "train",
        "--model": os.path.abspath(str(dev["model"])),
        "--adapter": os.path.abspath(str(dev["adapter"])),
        "--adapter-roles": "turn",
        "--harness-mode": "atomic",
        "--policy-temperature": str(dev.get("policy_temperature")),
        "--policy-top-p": str(dev.get("policy_top_p")),
        "--policy-seed": str(dev.get("policy_seed")),
        "--seed": str(SOURCE_SEED),
        "--max-calls": str(dev.get("max_calls")),
        "--frame-cap": str(dev.get("frame_cap")),
        "--max-minutes": str(dev.get("max_minutes")),
        "--max-new-tokens": str(dev.get("max_new_tokens")),
    }
    numeric_flags = {
        "--policy-temperature", "--policy-top-p", "--policy-seed",
        "--max-calls", "--frame-cap", "--max-minutes", "--max-new-tokens",
    }

    def same_value(flag, left, right):
        if flag in numeric_flags:
            return abs(float(left) - float(right)) <= 1e-9
        return left == right

    missing = ["%s %s" % (name, value) for name, value in required.items()
               if not any(tokens[index] == name
                          and index + 1 < len(tokens)
                          and same_value(name, tokens[index + 1], value)
                          for index in range(len(tokens)))]
    if "--legality-only" not in tokens:
        missing.append("--legality-only")
    if missing:
        raise ValueError("train source launch lacks identity: %s" % missing)
    code_hashes = set()
    for root in (train_run, dev_run):
        for row in _jsonl(os.path.join(root, "out", "rows.jsonl")):
            if row.get("complete"):
                code_hashes.add(str(row.get("code_hash") or ""))
    if code_hashes != {str(dev["code_hash"])}:
        raise ValueError("source rows do not share the dev-run code hash")
    train_report_path = os.path.join(train_run, "out", "RUN.json")
    train_report_checked = False
    if os.path.exists(train_report_path):
        with open(train_report_path, encoding="utf-8") as handle:
            train = json.load(handle)
        for name in ("model_sha256", "adapter_sha256", "code_hash",
                     "policy_seed", "seed"):
            if train.get(name) != dev.get(name):
                raise ValueError("train source identity differs at %s" % name)
        train_report_checked = True
    return {
        "dev_run_id": str(dev["run_id"]),
        "train_launch": os.path.abspath(launch_path),
        "selection": selection,
        "code_hash": str(dev["code_hash"]),
        "model": os.path.abspath(str(dev["model"])),
        "model_sha256": str(dev["model_sha256"]),
        "adapter": os.path.abspath(str(dev["adapter"])),
        "adapter_sha256": str(dev["adapter_sha256"]),
        "policy_temperature": float(dev.get("policy_temperature") or 0.0),
        "policy_top_p": float(dev.get("policy_top_p") or 0.0),
        "policy_seed": int(dev.get("policy_seed") or 0),
        "max_calls": int(dev.get("max_calls") or 0),
        "frame_cap": int(dev.get("frame_cap") or 0),
        "max_minutes": float(dev.get("max_minutes") or 0.0),
        "max_new_tokens": int(dev.get("max_new_tokens") or 0),
        "selection_report_sha256": SELECTION_SHA256,
        "adapter_content_checked": bool(verify_content),
        "model_content_checked": bool(verify_content),
        "train_run_report_checked": train_report_checked,
    }


def _stable_order(values, salt):
    return sorted(values, key=lambda value: _sha256_text(
        "%s:%s" % (salt, value["uid"])))


def _round_robin_videos(candidates, count):
    buckets = collections.defaultdict(list)
    for value in _stable_order(candidates, "lvbench-online-train-v1"):
        buckets[str(value["video"])].append(value)
    videos = sorted(buckets, key=lambda video: _sha256_text(
        "lvbench-online-video-v1:%s" % video))
    chosen = []
    offset = 0
    while len(chosen) < int(count):
        progressed = False
        for video in videos:
            if offset < len(buckets[video]) and len(chosen) < int(count):
                chosen.append(buckets[video][offset])
                progressed = True
        if not progressed:
            break
        offset += 1
    return chosen


def freeze_manifest(args):
    if _sha256_file(args.selection) != SELECTION_SHA256:
        raise ValueError("selection SHA differs from the frozen LVBench block")
    if int(args.train_states) < int(args.train_turns) * int(args.groups_per_turn):
        raise ValueError("train state count does not cover every turn pool")
    source_identity = _source_identity(
        args.source_run, args.dev_source_run, args.selection)
    train_items = _selection_items(args.data, args.selection, "train")
    dev_items = _selection_items(args.data, args.selection, "dev")
    completed_uids = {
        str(row["uid"]) for row in _jsonl(os.path.join(
            args.source_run, "out", "rows.jsonl")) if row.get("complete")}
    snapshots = _load_snapshots(os.path.join(args.source_run, "question-start"))
    dev_rows = {str(row["uid"]): row for row in _jsonl(os.path.join(
        args.dev_source_run, "out", "rows.jsonl")) if row.get("complete")}
    dev_snapshots = _load_snapshots(os.path.join(
        args.dev_source_run, "question-start"))
    train_by_uid = {str(item["uid"]): item for item in train_items}
    candidates = []
    for key, snapshot in snapshots.items():
        video, question_number, question = key
        matches = [item for item in train_items
                   if str(item["video"]) == video
                   and str(item["question"]) == question
                   and not item.get("stamped")]
        if len(matches) != 1:
            continue
        item = matches[0]
        if item["uid"] not in completed_uids:
            continue
        tree = B.tree_from_dict(snapshot["payload"]["tree"])
        if tree.prior_memory_candidate(question_number) is None:
            continue
        candidates.append({
            "uid": str(item["uid"]),
            "video": str(item["video"]),
            "question_number": int(question_number),
            "question": str(question),
            "answer": str(item.get("answer") or ""),
            "window": item.get("window"),
            "video_path": item["video_path"],
            "frame_dir": item["frame_dir"],
            "meta_duration": float(item["meta_duration"]),
            "snapshot": snapshot["path"],
            "snapshot_sha256": snapshot["sha256"],
            "declared_snapshot_sha256": snapshot["declared_sha256"],
            "declared_snapshot_hash_matches_payload": snapshot[
                "declared_hash_matches_payload"],
            "stamped": False,
        })
    candidates = [row for row in candidates if row["uid"] in train_by_uid]
    selected = _round_robin_videos(candidates, args.train_states)
    if len(selected) != int(args.train_states):
        raise ValueError("only %d eligible train states, need %d"
                         % (len(selected), int(args.train_states)))
    videos = {row["video"] for row in selected}
    if len(videos) < 10:
        raise ValueError("train states cover %d videos, need at least 10"
                         % len(videos))
    def enrich_dev(item):
        row = dev_rows.get(str(item["uid"]))
        if row is None:
            raise ValueError("dev item is missing from the source rows: %s"
                             % item["uid"])
        key = (str(item["video"]), int(row["question_number"]),
               str(item["question"]))
        snapshot = dev_snapshots.get(key)
        if snapshot is None:
            raise ValueError("dev item is missing a question-start snapshot: %s"
                             % item["uid"])
        return dict(item, question_number=int(row["question_number"]),
                    snapshot=snapshot["path"],
                    snapshot_sha256=snapshot["sha256"],
                    declared_snapshot_sha256=snapshot["declared_sha256"],
                    declared_snapshot_hash_matches_payload=snapshot[
                        "declared_hash_matches_payload"])

    unstamped = [enrich_dev(item) for item in dev_items
                 if not item.get("stamped")]
    by_video = collections.defaultdict(list)
    for item in _stable_order(unstamped, "lvbench-online-dev-v1"):
        by_video[str(item["video"])].append(item)
    dev_selected = []
    for video in sorted(by_video):
        if len(by_video[video]) < 4:
            raise ValueError("dev video %s has fewer than four unstamped items"
                             % video)
        dev_selected.extend(by_video[video][:4])
    dev_selected = {str(item["uid"]): item for item in dev_selected}
    if len(dev_selected) != int(args.dev_items):
        raise ValueError("dev trend has %d items, need %d"
                         % (len(dev_selected), int(args.dev_items)))
    turns = []
    for turn in range(int(args.train_turns)):
        start = turn * int(args.groups_per_turn)
        turns.append([row["uid"] for row in selected[
            start:start + int(args.groups_per_turn)]])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "selection": os.path.abspath(args.selection),
        "selection_sha256": SELECTION_SHA256,
        "source_run": os.path.abspath(args.source_run),
        "dev_source_run": os.path.abspath(args.dev_source_run),
        "source_identity": source_identity,
        "train_states": selected,
        "turn_states": turns,
        "dev_items": [dev_selected[uid] for uid in sorted(dev_selected)],
        "dev_all_items": [item for item in _stable_order(
            unstamped, "lvbench-online-dev-full-v1")],
        "counts": {
            "train_states": len(selected),
            "train_videos": len(videos),
            "train_turns": len(turns),
            "groups_per_turn": int(args.groups_per_turn),
            "target_mixed_groups": int(args.target_mixed_groups),
            "dev_items": len(dev_selected),
            "dev_all_items": len(unstamped),
            "dev_videos": len({str(item["video"]) for item in dev_selected.values()}),
        },
    }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest["counts"], indent=2, sort_keys=True))
    return 0


def _manifest_sha256(manifest):
    if manifest.get("layout_version") != "memory-menu-order-v1":
        return AMA.canonical_sha256(manifest)
    path_keys = {"selection", "candidate_plan", "snapshot", "source_snapshot",
                 "video_path", "frame_dir"}

    def strip(value):
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items()
                    if key not in path_keys and key != "manifest_sha256"}
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    return AMA.canonical_sha256(strip(manifest))


def _read_manifest(path):
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    actual = manifest.pop("manifest_sha256", "")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unexpected online-RL manifest schema")
    if actual != _manifest_sha256(manifest):
        raise ValueError("online-RL manifest hash differs")
    manifest["manifest_sha256"] = actual
    return manifest


def _answer_letter(value):
    match = re.search(r"\b([A-H])\b", str(value or "").upper())
    return match.group(1) if match else ""


def _empty_tree(state):
    return A.AtomicTimeTree(str(state["video"]), float(state["meta_duration"]))


def _snapshot_tree(state):
    with open(state["snapshot"], encoding="utf-8") as handle:
        record = json.load(handle)
    payload = record.get("payload") or {}
    actual = _snapshot_sha256(payload)
    if actual != state["snapshot_sha256"]:
        raise ValueError("train snapshot no longer matches its manifest hash")
    if str((payload.get("tree") or {}).get("video") or "") != str(state["video"]):
        raise ValueError("train snapshot video differs from the manifest")
    if str(payload.get("question") or "") != str(state["question"]):
        raise ValueError("train snapshot question differs from the manifest")
    if int(payload.get("question_number") or 0) != int(state["question_number"]):
        raise ValueError("train snapshot question number differs from the manifest")
    return B.tree_from_dict(payload["tree"])


def _local_memory_visible(args):
    value = getattr(args, "local_memory_visible", None)
    return bool(getattr(args, "memory_visible", True) if value is None else value)


def _read_limit(value):
    """Interpret zero as the ordinary call/frame budget, not zero READs."""
    return int(value or 0) or None


def _rollout_plan(args):
    if args.mode == "collect":
        count = int(getattr(args, "collect_rollout_count", 4) or 4)
        if count not in (1, 4):
            raise ValueError("collect rollout count must be 1 or 4")
        return [(index, seed, "train")
                for index, seed in enumerate(ROLLOUT_SEEDS[:count])]
    count = int(getattr(args, "rollout_count", 1) or 1)
    if count == 1:
        return [(0, int(args.policy_seed), "dev")]
    if count == 4:
        return [(index, seed, "dev")
                for index, seed in enumerate(ROLLOUT_SEEDS)]
    raise ValueError("evaluation rollout count must be 1 or 4")


def _wall_time_policy(args):
    return {
        "reader_min_runtime_seconds": float(getattr(
            args, "reader_min_runtime_seconds",
            A.READER_MIN_RUNTIME_SECONDS)),
        "policy_turn_reserve_seconds": float(getattr(
            args, "policy_turn_reserve_seconds",
            A.POLICY_TURN_RESERVE_SECONDS)),
        "final_answer_reserve_seconds": float(getattr(
            args, "final_answer_reserve_seconds",
            A.FINAL_ANSWER_RESERVE_SECONDS)),
        "min_check_to_final_seconds": float(getattr(
            args, "min_check_to_final_seconds",
            A.MIN_CHECK_TO_FINAL_SECONDS)),
    }


def _settings(args, split, item, snapshot_sha=""):
    return {
        "max_calls": args.max_calls,
        "frame_cap": args.frame_cap,
        "max_minutes": args.max_minutes,
        "max_new_tokens": args.max_new_tokens,
        "model": {"name": os.path.abspath(args.model)},
        "model_sha256": str(args.model_sha256),
        "serving": {"model_revision": "local:" + args.model_sha256[:16]},
        "adapter": os.path.abspath(args.adapter) if args.adapter else "",
        "adapter_sha256": str(args.adapter_sha256),
        "o_adapter": os.path.abspath(args.adapter) if args.adapter else "",
        "o_adapter_sha256": str(args.adapter_sha256),
        "reader_adapter": (os.path.abspath(args.reader_adapter)
                           if args.reader_policy else ""),
        "reader_adapter_sha256": (
            str(args.reader_adapter_sha256) if args.reader_policy else ""),
        "adapter_roles": (["turn", "loopread"] if args.reader_policy
                          else ["turn"]),
        "reader_policy_training": bool(args.reader_policy),
        "reader_schema_version": int(args.reader_schema_version),
        "memory_visible": bool(getattr(args, "memory_visible", True)),
        "local_memory_visible": _local_memory_visible(args),
        "memory_scope": "prior_question",
        "global_memory": bool(getattr(args, "global_memory", False)),
        "menu_layout": str(getattr(args, "menu_layout", "grouped")),
        "memory_menu_order": str(getattr(
            args, "memory_menu_order", "use_first")),
        "memory_decision_timing": str(getattr(
            args, "memory_decision_timing", "pre_read")),
        "card_list_visible": bool(getattr(
            args, "card_list_visible", False)),
        "card_local_visible": bool(getattr(
            args, "card_local_visible", True)),
        "card_global_visible": bool(getattr(
            args, "card_global_visible", True)),
        "cards_per_scope": int(getattr(args, "cards_per_scope", 4)),
        "force_card_check": bool(getattr(
            args, "force_card_check", False)),
        "budget_answer": bool(getattr(args, "budget_answer", False)),
        **_wall_time_policy(args),
        "max_reads_per_question": _read_limit(getattr(
            args, "max_reads_per_question", 0)),
        "diagnostic_only": bool(getattr(args, "diagnostic_only", False)),
        "data_split": split,
        "training_gold_window": item.get("window"),
        "training_gold_windows": (
            item.get("clue_intervals")
            or ([item.get("window")] if item.get("window") else [])),
        "snapshot_sha256": snapshot_sha,
    }


def _run_one(args, client, state, out_dir, rollout_index, seed, split):
    tree = _snapshot_tree(state) if state.get("snapshot") else _empty_tree(state)
    run_name = "%s-%s-r%d" % (args.run_id, _sha256_text(state["uid"])[:8],
                              int(rollout_index))
    tree.root = os.path.join(out_dir, "memory", run_name)
    os.makedirs(tree.root, exist_ok=True)
    calls = []
    calls_path = os.path.join(out_dir, "calls.jsonl")
    on_read_precommit = None
    if getattr(args, "memory_precommit_dir", None):
        def on_read_precommit(payload):
            AMA.write_precommit_snapshot(args.memory_precommit_dir, payload)
    ask = RC.make_ask(
        client, calls, state, {
            "calls": args.max_calls, "frames": args.frame_cap,
            "minutes": args.max_minutes,
        }, calls_path, max_new_tokens=args.max_new_tokens,
        turn_max_new_tokens=args.turn_max_new_tokens,
        run_id=run_name,
        adapter_roles=(("turn", "loopread") if args.reader_policy else ("turn",)),
        turn_parser=A.action_log, policy_top_p=1.0,
        reader_policy=bool(args.reader_policy),
        role_adapters=({"turn": "o", "loopread": "reader"}
                       if args.reader_policy else None))
    try:
        row = A.run_question(
            ask, tree, str(state["question"]),
            C.chunk_list(float(state["meta_duration"])),
            RC.frames_of_factory(K.Frames(
                state["video_path"], state["frame_dir"])),
            max_turns=A.turn_cap(len(C.chunk_list(
                float(state["meta_duration"])))),
            frame_cap=args.frame_cap, root=tree.root,
            question_number=int(state.get("question_number") or 1),
            call_cap=args.max_calls,
            max_reads_per_question=_read_limit(
                getattr(args, "max_reads_per_question", 0)),
            memory_visible=bool(args.memory_visible),
            local_memory_visible=_local_memory_visible(args),
            policy_temperature=(0.0 if args.reader_policy
                                else float(args.temperature)),
            policy_seed=(int(args.policy_seed)
                         if args.reader_policy else int(seed)),
            reader_temperature=(float(args.reader_temperature)
                                if args.reader_policy else 0.0),
            reader_seed=(int(seed) if args.reader_policy else None),
            reader_policy=bool(args.reader_policy),
            reader_schema_version=int(args.reader_schema_version),
            review_temperature=0.0,
            allow_read_window=False, legality_only=True, memory_use=True,
            menu_mode=True,
            off_menu_probability=float(args.off_menu_probability),
            global_memory=bool(args.global_memory),
            menu_layout=str(getattr(args, "menu_layout", "grouped")),
            memory_menu_order=str(getattr(
                args, "memory_menu_order", "use_first")),
            memory_decision_timing=str(getattr(
                args, "memory_decision_timing", "pre_read")),
            card_list_visible=bool(getattr(
                args, "card_list_visible", False)),
            card_local_visible=bool(getattr(
                args, "card_local_visible", True)),
            card_global_visible=bool(getattr(
                args, "card_global_visible", True)),
            cards_per_scope=int(getattr(args, "cards_per_scope", 4)),
            force_card_check=bool(getattr(
                args, "force_card_check", False)),
            budget_answer=bool(getattr(args, "budget_answer", False)),
            reader_min_runtime_seconds=float(getattr(
                args, "reader_min_runtime_seconds",
                A.READER_MIN_RUNTIME_SECONDS)),
            policy_turn_reserve_seconds=float(getattr(
                args, "policy_turn_reserve_seconds",
                A.POLICY_TURN_RESERVE_SECONDS)),
            final_answer_reserve_seconds=float(getattr(
                args, "final_answer_reserve_seconds",
                A.FINAL_ANSWER_RESERVE_SECONDS)),
            min_check_to_final_seconds=float(getattr(
                args, "min_check_to_final_seconds",
                A.MIN_CHECK_TO_FINAL_SECONDS)),
            on_read_precommit=on_read_precommit)
    except RC.StopRun as stop:
        if stop.kind not in ("capped", "backend_cut"):
            raise
        row = copy.deepcopy(getattr(stop, "atomic_row", {}))
        row["capped"] = stop.kind in ("capped", "backend_cut")
        row["truncated"] = stop.kind == "backend_cut"
        row["fault"] = "%s: %s" % (stop.kind, stop.detail)
        row["stopped"] = True
    row["calls"] = int(ask.state["n"])
    row["frames"] = int(ask.state["frames"])
    answer_letter = _answer_letter(row.get("answer"))
    gold = _answer_letter(state.get("answer"))
    row.update({
        "uid": state["uid"], "video": state["video"],
        "run_id": run_name, "gold": gold,
        "window": state.get("window"),
        "num_options": len(C.options_of(str(state["question"]))),
        "credit": 1.0 if answer_letter and answer_letter == gold else 0.0,
    })
    settings = _settings(
        args, split, state,
        snapshot_sha=str(state.get("snapshot_sha256") or ""))
    episode = E.from_row(
        row, calls, settings, group_id=str(state["uid"]),
        rollout_index=int(rollout_index),
        source_ref=str(state.get("snapshot") or "fresh-dev-start"))
    episode["memory_fork_from"] = {
        "snapshot": str(state.get("snapshot") or ""),
        "snapshot_sha256": str(state.get("snapshot_sha256") or ""),
        "isolated_root": os.path.abspath(tree.root),
    }
    problems = E.validate_episode(episode)
    if problems:
        raise ValueError("online episode schema problems: %s" % problems)
    _append_jsonl(os.path.join(out_dir, "rows.jsonl"), row)
    _append_jsonl(os.path.join(out_dir, "episodes.jsonl"), episode)
    return episode


def _limit_video_states(states, video_offset, video_limit, video_ids=()):
    wanted = {str(value) for value in video_ids or () if str(value)}
    if not video_offset and not video_limit and not wanted:
        return list(states)
    if wanted:
        return [state for state in states if str(state["video"]) in wanted]
    videos = []
    seen = set()
    for state in states:
        video = str(state["video"])
        if video not in seen:
            seen.add(video)
            videos.append(video)
    start = int(video_offset)
    stop = (start + int(video_limit)) if int(video_limit) else None
    selected = set(videos[start:stop])
    return [state for state in states if str(state["video"]) in selected]


def _selected_states(manifest, args):
    if args.mode == "collect":
        by_uid = {row["uid"]: row for row in manifest["train_states"]}
        turn = int(args.turn)
        if turn:
            if not 1 <= turn <= len(manifest["turn_states"]):
                raise ValueError("turn is outside the frozen manifest")
            return [by_uid[uid] for uid in manifest["turn_states"][turn - 1]]
        selected = _limit_video_states(
            manifest["train_states"], getattr(args, "video_offset", 0),
            getattr(args, "video_limit", 0),
            getattr(args, "video_id", ()))
        uids = {str(value) for value in getattr(args, "uid", ()) if str(value)}
        return [state for state in selected if str(state["uid"]) in uids] if uids else selected
    states = (list(manifest["dev_all_items"])
              if getattr(args, "item_set", "trend") == "full"
              else list(manifest["dev_items"]))
    wanted = {str(value) for value in getattr(args, "video_id", ()) if str(value)}
    return [state for state in states if str(state["video"]) in wanted] \
        if wanted else states


def _select_run_states(manifest, args):
    states = _selected_states(manifest, args)
    if (args.mode == "collect" and getattr(args, "uid", ())
            and not states):
        raise ValueError("requested UID selection is empty")
    if getattr(args, "video_id", ()) and not states:
        raise ValueError("requested video selection is empty")
    return states


def _clean_episode(episode):
    return bool(
        episode.get("training_only", {}).get("training_ready")
        and episode.get("outcome", {}).get("infrastructure_valid")
        and not episode.get("outcome", {}).get("protocol_fallback")
        and not episode.get("outcome", {}).get("harness_recovery")
        and episode.get("outcome", {}).get("terminal") != "protocol_abstain")


def _hybrid_episode(episode):
    return bool(
        episode.get("training_only", {}).get("hybrid_ready")
        and episode.get("outcome", {}).get("infrastructure_valid")
        and not episode.get("outcome", {}).get("protocol_fallback")
        and not episode.get("outcome", {}).get("harness_recovery")
        and episode.get("outcome", {}).get("terminal") != "protocol_abstain")


def _clean_mixed_group(rows):
    return bool(
        len(rows) == 4
        and all(_clean_episode(row) for row in rows)
        and len({bool(row["outcome"]["correct"]) for row in rows}) > 1)


def run_rollouts(args):
    manifest = _read_manifest(args.manifest)
    model_hash = RS.model_content_sha256(args.model)
    if model_hash != args.model_sha256:
        raise ValueError("base model content hash differs from the launch record")
    adapter_sha = _sha256_file(os.path.join(args.adapter,
                                            "adapter_model.safetensors"))
    if adapter_sha != args.adapter_sha256:
        raise ValueError("adapter hash differs from the launch record")
    reader_sha = ""
    if args.reader_policy:
        if not args.reader_adapter or not args.reader_adapter_sha256:
            raise ValueError("Reader policy needs an adapter and SHA-256")
        reader_sha = _sha256_file(os.path.join(
            args.reader_adapter, "adapter_model.safetensors"))
        if reader_sha != args.reader_adapter_sha256:
            raise ValueError("Reader adapter hash differs from the launch record")
    states = _select_run_states(manifest, args)
    if int(getattr(args, "limit", 0) or 0):
        states = states[:int(args.limit)]
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "episodes.jsonl")):
        raise ValueError("output already has episodes; use a new directory")
    client = LC.LocalClient(
        model_dir=args.model, adapter=args.adapter,
        reader_adapter=(args.reader_adapter if args.reader_policy else None))
    records = []
    rollouts = _rollout_plan(args)
    collected_groups = 0
    clean_mixed_groups = 0
    for state in states:
        group_rows = []
        for rollout_index, seed, split in rollouts:
            episode = _run_one(
                args, client, state, args.out, rollout_index, seed, split)
            records.append(episode)
            group_rows.append(episode)
        collected_groups += 1
        if args.mode == "collect" and _clean_mixed_group(group_rows):
            clean_mixed_groups += 1
    action_mix = collections.Counter()
    stage_mix = collections.Counter()
    menu_mix = collections.Counter()
    decision_mix = collections.Counter()
    reward_components = collections.defaultdict(list)
    for episode in records:
        for name, value in episode["training_only"]["reward_components"].items():
            if name != "step_returns":
                reward_components[name].append(float(value))
        for step in episode["steps"]:
            if step.get("actor") != "o" or step.get("recovered_action"):
                continue
            action_mix[str((step.get("action") or {}).get("verb") or "unparsed")] += 1
            stage_mix[str(step.get("stage") or "unknown")] += 1
            menu_mix[str(step.get("menu_class") or "unknown")] += 1
            decision_mix[str(step.get("menu_decision_mode") or "unknown")] += 1
    total_actions = sum(action_mix.values())
    eligible_actions = sum(
        bool(step.get("other_legal_action_verbs"))
        and step.get("valid")
        for row in records for step in row["steps"]
        if step.get("actor") == "o" and not step.get("recovered_action"))
    off_menu_actions = sum(
        step.get("menu_class") == "off_menu"
        and bool(step.get("other_legal_action_verbs"))
        and step.get("valid")
        for row in records for step in row["steps"]
        if step.get("actor") == "o" and not step.get("recovered_action"))
    preferred_target_matches = sum(
        bool(step.get("preferred_target_match"))
        for row in records for step in row["steps"]
        if step.get("actor") == "o" and not step.get("recovered_action"))
    grouped_records = collections.defaultdict(list)
    for episode in records:
        grouped_records[episode["group_id"]].append(episode)
    summary = {
        "schema_version": "lvbench-online-run-v1",
        "mode": args.mode,
        "turn": int(args.turn) if args.turn else None,
        "groups": collected_groups,
        "episodes": len(records),
        "model": os.path.abspath(args.model),
        "model_sha256": model_hash,
        "adapter": os.path.abspath(args.adapter),
        "adapter_sha256": adapter_sha,
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "off_menu_probability": float(args.off_menu_probability),
        "menu_layout": str(getattr(args, "menu_layout", "grouped")),
        "memory_menu_order": str(getattr(
            args, "memory_menu_order", "use_first")),
        "memory_decision_timing": str(getattr(
            args, "memory_decision_timing", "pre_read")),
        "rollout_seeds": [int(seed) for _index, seed, _split in rollouts],
        "memory_visible": bool(args.memory_visible),
        "local_memory_visible": _local_memory_visible(args),
        "memory_scope": "prior_question",
        "global_memory": bool(args.global_memory),
        "temperature": float(args.temperature),
        "budget": {"calls": args.max_calls, "frames": args.frame_cap,
                   "minutes": args.max_minutes,
                   "new_tokens": args.max_new_tokens},
        "wall_time_policy": _wall_time_policy(args),
        "correct": sum(bool(row["outcome"]["correct"]) for row in records),
        "protocol_fallbacks": sum(
            bool(row["outcome"].get("protocol_fallback")) for row in records),
        "harness_recoveries": sum(
            bool(row["outcome"].get("harness_recovery")) for row in records),
        "clean_groups": sum(
            len(rows) == 4 and all(_clean_episode(row) for row in rows)
            for rows in grouped_records.values()),
        "clean_mixed_groups": clean_mixed_groups,
        "target_mixed_groups": int(getattr(
            args, "target_mixed_groups", 0) or 0),
        "in_menu_actions": sum(step.get("menu_class") == "in_menu"
                               for row in records for step in row["steps"]
                               if step.get("actor") == "o"
                               and not step.get("recovered_action")),
        "off_menu_actions": off_menu_actions,
        "off_menu_eligible_actions": eligible_actions,
        "off_menu_rate": (off_menu_actions / float(eligible_actions or 1)),
        "preferred_target_rate": (
            preferred_target_matches / float(total_actions or 1)),
        "explore_mode_rate": (decision_mix["explore"]
                              / float(total_actions or 1)),
        "action_mix": dict(sorted(action_mix.items())),
        "stage_mix": dict(sorted(stage_mix.items())),
        "menu_mix": dict(sorted(menu_mix.items())),
        "menu_decision_mix": dict(sorted(decision_mix.items())),
        "reward_component_mean": {
            name: sum(values) / float(len(values) or 1)
            for name, values in sorted(reward_components.items())},
        "reward_mean": (sum(float(row["training_only"]["reward"])
                            for row in records) / float(len(records) or 1)),
    }
    with open(os.path.join(args.out, "RUN.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def filter_training_episodes(args):
    episodes = _jsonl(args.episodes)
    groups = collections.defaultdict(list)
    for episode in episodes:
        problems = E.validate_episode(episode)
        if problems:
            raise ValueError("episode schema problems: %s" % problems)
        groups[str(episode["group_id"])].append(episode)
    for group, rows in groups.items():
        rows.sort(key=lambda value: int(value["rollout_index"]))
        if [int(row["rollout_index"]) for row in rows] != [0, 1, 2, 3]:
            raise ValueError("group %s does not have four rollouts" % group)
    mode = str(getattr(args, "filter_mode", "mixed"))
    if mode == "mixed":
        eligible = [group for group, rows in groups.items()
                    if _clean_mixed_group(rows)]
        eligible_label = "clean_mixed"
    elif mode == "clean":
        eligible = [group for group, rows in groups.items()
                    if len(rows) == 4 and all(_clean_episode(row) for row in rows)]
        eligible_label = "clean"
    elif mode == "hybrid":
        eligible = [group for group, rows in groups.items()
                    if len(rows) == 4 and all(_hybrid_episode(row) for row in rows)]
        eligible_label = "hybrid"
    else:
        raise ValueError("unknown filter mode: %s" % mode)
    target = int(args.target_groups)
    if target > 0 and len(eligible) < target:
        raise ValueError("only %d %s groups, need %d"
                         % (len(eligible), eligible_label, target))
    ordered = sorted(eligible, key=lambda group: _sha256_text(
        "%s:%s" % (args.order_salt, group)))
    selected = ordered[:target] if target > 0 else ordered
    if os.path.exists(args.out):
        raise ValueError("training episode output already exists")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for group in selected:
            for episode in groups[group]:
                handle.write(json.dumps(episode, sort_keys=True) + "\n")
    report = {
        "schema_version": "lvbench-online-rl-filter-v1",
        "input_episodes": len(episodes),
        "input_groups": len(groups),
        "clean_mixed_groups": sum(
            _clean_mixed_group(rows) for rows in groups.values()),
        "clean_groups": sum(
            len(rows) == 4 and all(_clean_episode(row) for row in rows)
            for rows in groups.values()),
        "hybrid_groups": sum(
            len(rows) == 4 and all(_hybrid_episode(row) for row in rows)
            for rows in groups.values()),
        "eligible_groups": len(eligible),
        "eligible_kind": eligible_label,
        "filter_mode": mode,
        "target_groups": target,
        "selected_groups": len(selected),
        "selected_group_ids": selected,
        "order_salt": args.order_salt,
        "training_episodes": os.path.abspath(args.out),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--data", required=True)
    freeze.add_argument("--selection", required=True)
    freeze.add_argument("--source-run", required=True)
    freeze.add_argument("--dev-source-run", required=True)
    freeze.add_argument("--train-states", type=int, default=120)
    freeze.add_argument("--train-turns", type=int, default=3)
    freeze.add_argument("--groups-per-turn", type=int, default=40)
    freeze.add_argument("--target-mixed-groups", type=int, default=16)
    freeze.add_argument("--dev-items", type=int, default=28)
    freeze.add_argument("--out", required=True)
    freeze.set_defaults(func=freeze_manifest)

    for name in ("collect", "evaluate"):
        run = sub.add_parser(name)
        run.add_argument("--manifest", required=True)
        run.add_argument("--mode", default=name)
        run.add_argument("--turn", type=int, default=0)
        run.add_argument("--model", required=True)
        run.add_argument("--model-sha256", required=True)
        run.add_argument("--adapter", required=True)
        run.add_argument("--adapter-sha256", required=True)
        run.add_argument("--reader-adapter", default="")
        run.add_argument("--reader-adapter-sha256", default="")
        run.add_argument("--reader-policy", action="store_true")
        run.add_argument("--reader-temperature", type=float, default=0.8)
        run.add_argument("--reader-schema-version", type=int,
                         choices=(1, 2, 3), default=2)
        run.add_argument("--video-offset", type=int, default=0)
        run.add_argument("--video-limit", type=int, default=0)
        run.add_argument("--video-id", action="append", default=[])
        run.add_argument("--uid", action="append", default=[])
        run.add_argument("--out", required=True)
        run.add_argument("--run-id", required=True)
        run.add_argument("--max-calls", type=int, default=20)
        run.add_argument("--collect-rollout-count", type=int, choices=(1, 4),
                         default=4)
        run.add_argument("--frame-cap", type=int, default=256)
        run.add_argument("--max-minutes", type=float, default=10.0)
        run.add_argument("--max-new-tokens", type=int, default=512)
        run.add_argument("--turn-max-new-tokens", type=int, default=512)
        run.add_argument("--max-reads-per-question", type=int, default=0)
        run.add_argument("--budget-answer", action=argparse.BooleanOptionalAction,
                         default=False,
                         help="allow bare SUBMIT from seen evidence when the budget ends")
        run.add_argument(
            "--reader-min-runtime-seconds", type=float,
            default=A.READER_MIN_RUNTIME_SECONDS,
            help="minimum remaining Reader runtime used by the wall-time gate")
        run.add_argument(
            "--policy-turn-reserve-seconds", type=float,
            default=A.POLICY_TURN_RESERVE_SECONDS,
            help="O-turn reserve added to the Reader menu threshold")
        run.add_argument(
            "--final-answer-reserve-seconds", type=float,
            default=A.FINAL_ANSWER_RESERVE_SECONDS,
            help="wall time reserved after a Reader for the final answer")
        run.add_argument(
            "--min-check-to-final-seconds", type=float,
            default=A.MIN_CHECK_TO_FINAL_SECONDS,
            help="minimum wall time for CHECK followed by the final answer")
        run.add_argument("--policy-seed", type=int, default=20261111)
        run.add_argument("--temperature", type=float, default=0.8)
        run.add_argument("--off-menu-probability", type=float, default=0.1)
        run.add_argument("--menu-layout", choices=("grouped", "flat"),
                         default="grouped")
        run.add_argument("--memory-menu-order",
                         choices=("use_first", "defer_first"),
                         default="use_first")
        run.add_argument("--memory-decision-timing",
                         choices=("pre_read", "after_read", "off"),
                         default="pre_read")
        run.add_argument("--rollout-count", type=int, default=1,
                         help="evaluation rollouts: 1 or 4; collection always uses 4")
        run.add_argument("--memory-visible", action=argparse.BooleanOptionalAction,
                         default=True,
                         help="show prior local and global memory to the policy")
        run.add_argument("--local-memory-visible", action=argparse.BooleanOptionalAction,
                         default=None,
                         help="show prior local memory to the policy")
        run.add_argument("--global-memory", action=argparse.BooleanOptionalAction,
                         default=False,
                         help="enable global-memory actions and claims")
        run.add_argument("--memory-precommit-dir", default="",
                         help="capture each pre-commit Reader read state")
        run.add_argument("--limit", type=int, default=0,
                         help="smoke-only item limit; zero runs the full selection")
        run.add_argument("--target-mixed-groups", type=int, default=0,
                         help="recorded collection target for the fixed state pool")
        run.add_argument("--item-set", choices=("trend", "full"),
                         default="trend")
        run.set_defaults(func=run_rollouts)
    filtered = sub.add_parser("filter-training")
    filtered.add_argument("--episodes", required=True)
    filtered.add_argument("--target-groups", type=int, default=16)
    filtered.add_argument("--filter-mode", choices=("mixed", "clean", "hybrid"),
                          default="mixed")
    filtered.add_argument("--order-salt", default="lvbench-online-rl-canary-v1")
    filtered.add_argument("--out", required=True)
    filtered.add_argument("--report", required=True)
    filtered.set_defaults(func=filter_training_episodes)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, KeyError) as error:
        print("online LVBench runner refused: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
