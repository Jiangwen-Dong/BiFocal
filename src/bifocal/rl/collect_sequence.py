#!/usr/bin/env python3
"""Collect and gate ordered video sequences for online O and Reader RL."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

from ..harness import actions as A
from ..memory import artifacts as AMA
from ..memory import cards as AMC
from ..harness import protocol as C
from ..harness import frames as K
from ..serving import client as LC
from . import episode as E
from . import sequence as S
from . import question_ledger as QL
from ..serving import lanes as RS
from . import scripted_client as SSC
from ..harness import calls as RC
from . import collect_manifest as ROL


SCHEMA_VERSION = "online-sequence-run-v2"
TRAINING_SEQUENCE_FILE = "training-sequences.jsonl"


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_manifest(manifest):
    declared = str(manifest.get("manifest_sha256") or "")
    if not declared:
        raise ValueError("manifest has no declared SHA-256")
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    actual = _canonical_sha256(payload)
    if actual != declared:
        raise ValueError("manifest SHA-256 differs from its payload")
    benchmark = str(manifest.get("benchmark") or "lvbench")
    allowed_schemas = {
        "lvbench": ROL.MANIFEST_SCHEMA,
        "mlvu": "mlvu-online-rl-manifest-v1",
    }
    if manifest.get("schema_version") != allowed_schemas.get(benchmark):
        raise ValueError("unexpected online-sequence manifest schema")
    if benchmark == "mlvu":
        if manifest.get("training_allowed") is not True:
            raise ValueError("MLVU manifest does not permit training")
        if manifest.get("training_scope") != "development_split":
            raise ValueError("MLVU training scope is not the development split")
        source = manifest.get("source") or {}
        data_root = os.path.abspath(str(source.get("data_root") or ""))
        if not data_root or not os.path.isdir(data_root):
            raise ValueError("MLVU data root is missing")
        annotations = source.get("annotation_sha256") or {}
        if not annotations:
            raise ValueError("MLVU annotation hashes are missing")
        for task, expected in annotations.items():
            path = os.path.join(
                data_root, "mlvu", "MLVU", "json", str(task) + ".json")
            if not os.path.isfile(path) or _sha256_file(path) != str(expected):
                raise ValueError("MLVU annotation SHA differs: %s" % task)
    return benchmark


def _serving(args):
    if not getattr(args, "vllm_runtime", ""):
        return {"model_revision": "local:" + args.model_sha256[:16]}
    runtime = _vllm_runtime(args)
    return {**RS.public_runtime(runtime),
            "model_sha256": runtime.get("model_content_sha256"),
            "evaluation_only": True, "excluded_from_ppo": True,
            # vLLM serves the base model. The identity adapter is zero-initialized
            # and only recorded by hash; no LoRA is loaded by the server.
            "adapter_served": False}


def _settings(args, state, snapshot_sha, split):
    return {
        "max_calls": args.max_calls,
        "frame_cap": args.frame_cap,
        "max_minutes": args.max_minutes,
        "max_tokens": args.max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "model": {"name": os.path.abspath(args.model)},
        "serving": _serving(args),
        "model_quantization": str(getattr(
            args, "model_quantization",
            "4bit_nf4" if getattr(args, "load_in_4bit", False) else "bf16")),
        "adapter": os.path.abspath(args.adapter),
        "adapter_sha256": str(args.adapter_sha256),
        "o_adapter": os.path.abspath(args.adapter),
        "o_adapter_sha256": str(args.adapter_sha256),
        "reader_adapter": (os.path.abspath(args.reader_adapter)
                           if args.reader_policy else ""),
        "reader_adapter_sha256": (
            str(args.reader_adapter_sha256) if args.reader_policy else ""),
        "adapter_roles": (["turn", "loopread"] if args.reader_policy
                          else ["turn"]),
        "reader_policy_training": bool(args.reader_policy),
        "reader_schema_version": int(getattr(
            args, "reader_schema_version", 1)),
        "replay_forward": (
            "teacher_forced_full_sequence" if args.mode == "collect"
            else "generation_only"),
        "card_list_visible": bool(getattr(
            args, "card_list_visible", False)),
        "card_local_visible": bool(getattr(
            args, "card_local_visible", True)),
        "card_global_visible": bool(getattr(
            args, "card_global_visible", True)),
        "cards_per_scope": int(getattr(args, "cards_per_scope", 4)),
        "force_card_check": bool(getattr(
            args, "force_card_check", False)),
        "global_source": str(getattr(args, "global_source", "timeline")),
        "read_order": str(getattr(args, "read_order", "positional")),
        "facts_block": bool(getattr(args, "facts_block", False)),
        "budget_answer": bool(getattr(args, "budget_answer", False)),
        "fresh_memory_each_question": bool(getattr(
            args, "fresh_memory_each_question", False)),
        "diagnostic_only": bool(getattr(args, "diagnostic_only", False)),
        "data_split": split,
        "initial_state": str(state.get("initial_state") or ""),
        "initial_tree_sha256": str(state.get("initial_tree_sha256") or ""),
        "training_gold_window": state.get("window"),
        "training_gold_windows": (
            state.get("clue_intervals")
            or ([state.get("window")] if state.get("window") else [])),
        "snapshot_sha256": snapshot_sha,
        "global_memory": True,
        "memory_precommit_dir": (os.path.abspath(args.memory_precommit_dir)
                                 if getattr(args, "memory_precommit_dir", "")
                                 else ""),
        "smoke_use_cache": bool(getattr(args, "smoke_use_cache", False)),
        "scripted_smoke_client": bool(getattr(
            args, "scripted_smoke_client", False)),
    }


def _diagnostic_question_ready(result):
    outcome = result.get("outcome") or {}
    return bool(
        not result.get("forced_failure")
        and outcome.get("complete")
        and outcome.get("infrastructure_valid")
        and not outcome.get("protocol_fallback")
        and not outcome.get("harness_recovery"))


_RUNTIMES = {}


def _vllm_runtime(args):
    """The served runtime for an evaluation-only run, validated once."""
    path = os.path.abspath(args.vllm_runtime)
    if path not in _RUNTIMES:
        _RUNTIMES[path] = RS.load_runtime(path)
    return _RUNTIMES[path]


def _make_client(args):
    if getattr(args, "scripted_smoke_client", False):
        return SSC.ScriptedSmokeClient()
    if getattr(args, "vllm_runtime", ""):
        from ..serving import vllm as VC
        return VC.RoleClient(_vllm_runtime(args))
    return LC.LocalClient(
        model_dir=args.model, adapter=args.adapter,
        reader_adapter=(args.reader_adapter if args.reader_policy else None),
        load_in_4bit=bool(getattr(args, "load_in_4bit", False)),
        teacher_forced_logprobs=(args.mode == "collect"))


def _chunk_windows(state):
    return C.chunk_list(float(state["meta_duration"]))


def _answer_letter(value):
    import re
    match = re.search(r"\b([A-H])\b", str(value or "").upper())
    return match.group(1) if match else ""


def _forced_failure(value):
    if not value:
        return None
    try:
        rollout_text, question_text = value.split(":", 1)
        rollout = int(rollout_text)
        question = None if question_text.lower() == "auto" else int(question_text)
    except ValueError as error:
        raise ValueError("--force-fail must be ROLLOUT:QUESTION") from error
    return rollout, question


def _split_rows(manifest, split):
    name = "train_states" if split == "train" else "%s_items" % split
    value = manifest.get(name)
    return value if isinstance(value, list) else []


def _assert_disjoint_splits(manifest):
    seen_videos = {}
    seen_uids = {}
    for split in ("train", "dev", "test"):
        rows = _split_rows(manifest, split)
        videos = {str(row.get("video") or "") for row in rows}
        uids = {str(row.get("uid") or "") for row in rows}
        if "" in videos or "" in uids:
            raise ValueError("%s split has an empty video or uid" % split)
        for video in videos:
            if video in seen_videos and seen_videos[video] != split:
                raise ValueError("video appears in multiple splits: %s" % video)
            seen_videos[video] = split
        for uid in uids:
            if uid in seen_uids and seen_uids[uid] != split:
                raise ValueError("item appears in multiple splits: %s" % uid)
            seen_uids[uid] = split


def _initial_tree(state, benchmark):
    if state.get("snapshot"):
        return ROL._snapshot_tree(state)
    if benchmark in ("lvbench", "mlvu") and state.get("initial_state") == "empty":
        tree = ROL._empty_tree(state)
        expected = str(state.get("initial_tree_sha256") or "")
        actual = AMA.canonical_sha256(tree.to_dict())
        if not expected or actual != expected:
            raise ValueError("empty initial tree hash differs from the manifest")
        return tree
    raise ValueError("%s sequence state has no frozen initial tree" % benchmark)


def _validate_sequence_layout(manifest, mode, expected_questions,
                             evaluation_split="dev"):
    """Require the frozen per-video order, not an inferred hash order."""
    if mode == "collect":
        split = "train"
    elif mode == "evaluate":
        split = str(evaluation_split)
    else:
        raise ValueError("unknown sequence mode")
    if split not in ("train", "dev", "test"):
        raise ValueError("evaluation split must be dev or test")
    benchmark = str(manifest.get("benchmark") or "lvbench")
    expected_layout = {
        "lvbench": "lvbench-online-sequence-layout-v1",
        "mlvu": "mlvu-online-sequence-layout-v1",
    }.get(benchmark)
    if expected_layout is None:
        raise ValueError("manifest names an unsupported benchmark")
    if manifest.get("layout_version") != expected_layout:
        raise ValueError("manifest layout version differs from its benchmark")
    _assert_disjoint_splits(manifest)
    name = "%s_sequences" % split
    states = manifest.get("%s_states" % split if split == "train"
                          else "%s_items" % split)
    entries = manifest.get(name)
    if not isinstance(entries, list) or not isinstance(states, list):
        raise ValueError("manifest has no frozen sequence lists")
    by_uid = {str(row["uid"]): row for row in states}
    used = set()
    for entry in entries:
        rows = [by_uid.get(str(uid)) for uid in entry.get("uids") or []]
        if any(row is None for row in rows):
            raise ValueError("sequence layout names an unknown state")
        if benchmark in ("lvbench", "mlvu"):
            first = rows[0]
            snapshot_ready = bool(
                first.get("snapshot") and first.get("snapshot_sha256"))
            empty_ready = bool(
                first.get("initial_state") == "empty"
                and first.get("initial_tree_sha256"))
            if not snapshot_ready and not empty_ready:
                raise ValueError(
                    "%s sequence has no frozen initial state: %s"
                    % (benchmark, entry.get("video")))
            if empty_ready and entry.get("initial_state") != "empty":
                raise ValueError("sequence entry disagrees on empty initial state")
            if empty_ready and str(entry.get("initial_tree_sha256") or "") != str(
                    first.get("initial_tree_sha256") or ""):
                raise ValueError("sequence initial tree hash differs from state")
        expected = int(expected_questions)
        entry_numbers = [int(value) for value in entry.get(
            "question_numbers") or []]
        if expected and len(rows) != expected:
            raise ValueError("sequence layout has a different question count")
        if len(entry_numbers) != len(rows):
            raise ValueError("sequence layout length differs from its entry")
        numbers = [int(row["question_number"]) for row in rows]
        if numbers != entry_numbers:
            raise ValueError("sequence layout question order differs")
        if any(right != left + 1 for left, right in zip(
                numbers, numbers[1:])):
            raise ValueError("sequence layout is not contiguous")
        start = int(entry.get("start_question") or 0)
        if numbers != list(range(start, start + len(rows))):
            raise ValueError("sequence layout start question differs")
        if mode == "collect" and start != 1:
            raise ValueError("train sequence layout must start at question one")
        if len({str(row["video"]) for row in rows}) != 1:
            raise ValueError("sequence layout spans multiple videos")
        used.update(str(row["uid"]) for row in rows)
    if used != set(by_uid) or sum(len(entry.get("uids") or [])
                                  for entry in entries) != len(states):
        raise ValueError("sequence layout does not cover every frozen state")
    return entries


def _run_sequences(args, states, expected_questions, split, benchmark):
    groups = S.group_states(states, questions_per_sequence=expected_questions)
    if args.video_stride > 1:
        groups = groups[int(args.video_offset)::int(args.video_stride)]
    if args.limit_videos:
        groups = groups[:int(args.limit_videos)]
    limit_questions = int(getattr(args, "limit_questions", 0) or 0)
    if limit_questions:
        groups = [dict(group, states=group["states"][:limit_questions])
                  for group in groups]
    if args.mode == "collect":
        seeds = _parse_rollout_seeds(getattr(args, "rollout_seeds", ""))
    else:
        seeds = [int(args.policy_seed)]
    forced = _forced_failure(args.force_fail)
    client = _make_client(args)
    adapter_roles = ("turn", "loopread") if args.reader_policy else ("turn",)
    role_adapters = ({"turn": "o", "loopread": "reader"}
                     if args.reader_policy else None)
    calls = []
    calls_path = os.path.join(args.out, "calls.jsonl")
    progress_path = os.path.join(args.out, "progress.json")
    os.makedirs(args.out, exist_ok=True)
    progress = {
        "schema_version": "online-sequence-progress-v1",
        "started_unix": time.time(),
        "completed_videos": 0,
        "expected_videos": len(groups),
        "completed_rollout_questions": 0,
        "expected_rollout_questions": (
            sum(len(group["states"]) for group in groups)
            * len(seeds)),
        "last_event": {},
    }
    def write_progress(event):
        progress["last_event"] = dict(event)
        progress["updated_unix"] = time.time()
        progress["elapsed_seconds"] = round(
            progress["updated_unix"] - progress["started_unix"], 3)
        progress["calls"] = len(calls)
        progress["frames"] = sum(int(row.get("frames") or 0)
                                 for row in calls)
        progress["last_call"] = ({
            name: calls[-1].get(name)
            for name in ("role", "uid", "request_id", "finish_reason",
                         "completion_tokens", "seconds", "use_cache",
                         "max_seconds")
        } if calls else {})
        _write_json(progress_path, progress)
    write_progress({"phase": "start"})
    ledger = QL.Ledger(args.out, {
        "run_id": args.run_id,
        "code_commit": str(getattr(args, "code_commit", "") or ""),
        "prompt_revision": QL.prompt_revision(
            [A.__file__, AMC.__file__], C.PROMPT_SCHEMA),
        "adapter_id": os.path.basename(os.path.abspath(args.adapter)),
        "adapter_sha256": str(args.adapter_sha256),
        "model_sha256": str(args.model_sha256),
        "round": str(getattr(args, "round_id", "") or ""),
        "benchmark": benchmark, "split": split,
        "global_source": str(getattr(args, "global_source", "timeline")),
        "read_order": str(getattr(args, "read_order", "positional")),
        "facts_block": bool(getattr(args, "facts_block", False)),
        "budget_answer": bool(getattr(args, "budget_answer", False)),
        "fresh_memory_each_question": bool(getattr(
            args, "fresh_memory_each_question", False)),
        "budget": {"calls": args.max_calls, "frames": args.frame_cap,
                   "minutes": args.max_minutes,
                   "tokens": args.max_tokens,
                   "new_tokens": args.max_new_tokens,
                   "level": "%dc-%df-%gm-%dk" % (
                       args.max_calls, args.frame_cap, args.max_minutes,
                       args.max_tokens // 1024)},
    })
    sequence_results = []
    summaries = []
    training_rows = []
    question_episodes = []
    capture_read_precommit = None
    if getattr(args, "memory_precommit_dir", ""):
        def capture_read_precommit(payload):
            AMA.write_precommit_snapshot(args.memory_precommit_dir, payload)
    for group_index, group in enumerate(groups):
        video = group["video"]
        question_states = group["states"]
        first = question_states[0]
        start_tree = _initial_tree(first, benchmark)
        memory_root = os.path.join(args.out, "memory", video)
        snapshot_sha = str(first.get("snapshot_sha256") or "")
        auto_forced_used = [False]

        def run_question(state, tree, rollout_index, seed):
            question_number = int(state["question_number"])
            question_started = time.time()
            pre_question_sha = S.memory_sha256(tree.to_dict())
            pre_question_full_sha = S.checkpoint_sha256(tree.to_dict())
            run_name = "%s-%s-q%d-r%d" % (
                args.run_id, video, question_number, rollout_index)
            ask = RC.make_ask(
                client, calls, state,
                {"calls": args.max_calls, "frames": args.frame_cap,
                 "minutes": args.max_minutes, "tokens": args.max_tokens},
                calls_path, max_new_tokens=args.max_new_tokens,
                turn_max_new_tokens=args.turn_max_new_tokens,
                run_id=run_name, adapter_roles=adapter_roles,
                turn_parser=A.action_log, policy_top_p=1.0,
                reader_policy=bool(args.reader_policy),
                role_adapters=role_adapters,
                reader_use_cache=(True if getattr(
                    args, "smoke_use_cache", False) else None))
            row = A.run_question(
                ask, tree, str(state["question"]),
                _chunk_windows(state),
                RC.frames_of_factory(K.Frames(
                    state["video_path"], state["frame_dir"])),
                max_turns=A.turn_cap(len(_chunk_windows(state))),
                frame_cap=args.frame_cap, root=tree.root,
                question_number=question_number, call_cap=args.max_calls,
                memory_visible=True,
                policy_temperature=(0.0 if args.reader_policy
                                    else float(args.temperature)),
                policy_seed=(int(args.policy_seed) if args.reader_policy
                             else int(seed)),
                reader_temperature=(float(args.reader_temperature)
                                    if args.reader_policy else 0.0),
                reader_seed=(int(seed) if args.reader_policy else None),
                reader_policy=bool(args.reader_policy),
                reader_schema_version=int(args.reader_schema_version),
                review_temperature=0.0, allow_read_window=False,
                legality_only=True, memory_use=True, menu_mode=True,
                off_menu_probability=float(args.off_menu_probability),
                global_memory=True,
                card_list_visible=bool(getattr(
                    args, "card_list_visible", False)),
                card_local_visible=bool(getattr(
                    args, "card_local_visible", True)),
                card_global_visible=bool(getattr(
                    args, "card_global_visible", True)),
                cards_per_scope=int(getattr(args, "cards_per_scope", 4)),
                global_source=str(getattr(args, "global_source", "timeline")),
                read_order=str(getattr(args, "read_order", "positional")),
                facts_block=bool(getattr(args, "facts_block", False)),
                budget_answer=bool(getattr(args, "budget_answer", False)),
                force_card_check=bool(getattr(
                    args, "force_card_check", False)),
                on_read_precommit=capture_read_precommit)
            row["calls"] = int(ask.state["n"])
            row["frames"] = int(ask.state["frames"])
            answer = _answer_letter(row.get("answer"))
            gold = _answer_letter(state.get("answer"))
            row.update({
                "uid": state["uid"], "video": video, "run_id": run_name,
                "gold": gold, "window": state.get("window"),
                "num_options": len(C.options_of(str(state["question"]))),
                "credit": 1.0 if answer and answer == gold else 0.0,
            })
            settings = _settings(args, state, snapshot_sha, split)
            episode = E.from_row(
                row, calls, settings,
                group_id="%s:q%d" % (video, question_number),
                rollout_index=rollout_index,
                source_ref="%s:q%d:r%d" % (memory_root, question_number,
                                            rollout_index))
            episode["memory_fork_from"] = {
                "video": video,
                "question_number": question_number,
                "checkpoint_sha256": pre_question_sha,
                "checkpoint_full_sha256": pre_question_full_sha,
                "isolated_root": os.path.abspath(tree.root),
            }
            problems = E.validate_episode(episode)
            if problems:
                raise ValueError("online episode schema problems: %s" % problems)
            hybrid_ready = bool(episode["training_only"].get("hybrid_ready"))
            forced_after_global = False
            if forced and forced[0] == rollout_index:
                if forced[1] is None:
                    forced_after_global = bool(
                        tree.active_global_claims()) and not auto_forced_used[0]
                    forced_failure = forced_after_global
                    auto_forced_used[0] = auto_forced_used[0] or forced_failure
                else:
                    forced_failure = forced[1] == question_number
                    forced_after_global = forced_failure and bool(
                        tree.active_global_claims())
            else:
                forced_failure = False
            if forced_failure:
                hybrid_ready = False
            ledger_key = {
                "video": video, "question_number": question_number,
                "uid": state["uid"], "rollout_index": rollout_index,
                "seed": int(seed),
                "position": [int(s["question_number"])
                             for s in question_states].index(question_number)
                + 1}
            try:
                ledger.stage(ledger_key, QL.build_bundle(
                    ledger.ctx, ledger_key, episode,
                    [c for c in calls if c.get("run_id") == run_name],
                    tree.to_dict(), refusals=row.get("refusals"),
                    wall_seconds=time.time() - question_started,
                    forced=forced_failure, gold=gold,
                    windows=settings.get("training_gold_windows")))
            except Exception as error:
                ledger.errors += 1
                ledger.failure(ledger_key, error, kind="ledger_error")
            result = {
                "episode_id": episode["episode_id"],
                "hybrid_ready": hybrid_ready,
                "training_ready": bool(
                    episode["training_only"].get("training_ready")),
                "reader_training_ready": bool(
                    episode["training_only"].get("reader_training_ready")),
                "forced_failure": forced_failure,
                "forced_failure_after_global": forced_after_global,
                "outcome": episode["outcome"],
                "episode": episode,
            }
            question_episodes.append(episode)
            return result

        def on_progress(event):
            progress["completed_rollout_questions"] += 1
            write_progress(event)

        def on_question(question_number, kept, blockers):
            state = next(s for s in question_states
                         if int(s["question_number"]) == question_number)
            for index, seed in enumerate(seeds):
                if (video, question_number, index) not in ledger.pending:
                    ledger.failure({
                        "video": video, "question_number": question_number,
                        "uid": state["uid"], "rollout_index": index,
                        "seed": int(seed)}, "no episode row was produced")
            ledger.finalize(video, question_number, kept, blockers)

        if getattr(args, "scripted_smoke_client", False) \
                or getattr(args, "vllm_runtime", ""):
            question_ready = _diagnostic_question_ready
        elif args.reader_policy:
            question_ready = lambda result: bool(
                result.get("reader_training_ready"))
        else:
            question_ready = None
        sequence = S.run_video_sequence(
            video, question_states, seeds, start_tree, run_question,
            memory_root, question_ready=question_ready,
            progress_callback=on_progress, question_callback=on_question,
            fresh_memory_each_question=bool(getattr(
                args, "fresh_memory_each_question", False)))
        sequence_results.append(sequence)
        if args.mode == "collect":
            minimum_common = 2 if benchmark == "mlvu" else S.MIN_COMMON_QUESTIONS
            summary, rows = (
                S.admit_reader_sequence(
                    sequence, require_variance=not args.smoke,
                    min_common_questions=minimum_common)
                if args.reader_policy else
                S.admit_sequence(
                    sequence, require_variance=not args.smoke,
                    min_common_questions=minimum_common))
            if forced is not None:
                summary["smoke_forced_failure"] = list(forced)
            summary["smoke_mode"] = bool(args.smoke)
            summaries.append(summary)
            training_rows.extend(rows)
            _append_jsonl(
                os.path.join(args.out, "sequence-diagnostics.jsonl"), summary)
        else:
            common = S.common_questions(sequence["rollouts"])
            rollouts = [S.aggregate_rollout(sequence, run)
                        for run in sequence["rollouts"]]
            summaries.append({
                "video": video,
                "questions": sequence["questions"],
                "common_questions": common,
                "common_count": len(common),
                "rollout_seeds": [int(run["seed"])
                                  for run in sequence["rollouts"]],
                "memory_roots": [str(run["memory_root"])
                                 for run in sequence["rollouts"]],
                "rows": [row for row in rollouts if row is not None],
                "rollback_checks": S.sequence_rollback_checks(sequence),
            })
        progress["completed_videos"] = group_index + 1
        progress["admitted_so_far"] = (
            sum(row["admitted"] for row in summaries)
            if args.mode == "collect" else None)
        write_progress({
            "phase": "video_complete",
            "video": video,
            "completed_videos": group_index + 1,
        })

    episodes_path = os.path.join(args.out, "question-episodes.jsonl")
    for episode in question_episodes:
        _append_jsonl(episodes_path, episode)
    summaries_path = os.path.join(args.out, "sequences.jsonl")
    for summary in summaries:
        _append_jsonl(summaries_path, summary)
    training_path = os.path.join(args.out, TRAINING_SEQUENCE_FILE)
    for episode in training_rows:
        _append_jsonl(training_path, episode)

    admitted = sum(bool(row["admitted"]) for row in summaries) \
        if args.mode == "collect" else 0
    rollback_checks = [check for row in summaries
                       for check in row.get("rollback_checks") or []]
    problems = collections.Counter(
        problem for row in summaries for problem in row.get("problems") or [])
    global_ops = collections.Counter(
        op.get("operation")
        for episode in question_episodes
        for step in episode.get("steps") or []
        for op in step.get("memory_ops") or []
        if op.get("op") == "evolve_global")
    global_valid = sum(
        bool(op.get("accepted"))
        for episode in question_episodes
        for step in episode.get("steps") or []
        for op in step.get("memory_ops") or []
        if op.get("op") == "evolve_global")
    forced_after_global_count = sum(
        bool(result.get("forced_failure_after_global"))
        for sequence in sequence_results
        for run in sequence["rollouts"]
        for question in run["skipped_questions"]
        for result in question.get("rollouts") or [])
    kept_questions = {
        (str(row["video"]), int(question))
        for row in summaries
        for question in row.get("common_questions") or []}
    kept_global_ops = collections.Counter(
        op.get("operation")
        for episode in question_episodes
        if (str(episode.get("video_id")),
            int(episode.get("question_number") or 0)) in kept_questions
        for step in episode.get("steps") or []
        for op in step.get("memory_ops") or []
        if op.get("op") == "evolve_global")
    kept_global_valid = sum(
        bool(op.get("accepted"))
        for episode in question_episodes
        if (str(episode.get("video_id")),
            int(episode.get("question_number") or 0)) in kept_questions
        for step in episode.get("steps") or []
        for op in step.get("memory_ops") or []
        if op.get("op") == "evolve_global")
    required = int(args.required_admitted)
    sequence_audit = (S.audit_sequence_bundle(
        sequence_results, summaries, training_rows,
        required_groups=required) if args.mode == "collect" else None)
    clean_groups = _clean_group_count(training_rows)
    required_clean = int(getattr(args, "require_clean_groups", 0))
    clean_gate_passes = clean_groups >= required_clean
    gate_passes = bool(
        (sequence_audit["passes"] if sequence_audit is not None
         else not any(not check.get("restored")
                      for check in rollback_checks))
        and clean_gate_passes and ledger.errors == 0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "benchmark": benchmark,
        "evaluation_split": (str(args.evaluation_split)
                             if args.mode == "evaluate" else "train"),
        "run_id": args.run_id,
        "manifest": os.path.abspath(args.manifest),
        "manifest_sha256": _sha256_file(args.manifest),
        "model": os.path.abspath(args.model),
        "model_sha256": args.model_sha256,
        "adapter": os.path.abspath(args.adapter),
        "adapter_sha256": args.adapter_sha256,
        "questions_per_sequence": int(expected_questions),
        "limit_questions": int(getattr(args, "limit_questions", 0) or 0),
        "videos": len(groups),
        "video_offset": int(args.video_offset),
        "video_stride": int(args.video_stride),
        "video_filter": str(args.video_id or ""),
        "question_episodes": len(question_episodes),
        "calls": sum(int((row.get("outcome") or {}).get("calls") or 0)
                     for row in question_episodes),
        "frames": sum(int((row.get("outcome") or {}).get("frames") or 0)
                      for row in question_episodes),
        "kept_calls": sum(
            int((row.get("outcome") or {}).get("calls") or 0)
            for row in question_episodes
            if (str(row.get("video_id")),
                int(row.get("question_number") or 0)) in kept_questions),
        "kept_frames": sum(
            int((row.get("outcome") or {}).get("frames") or 0)
            for row in question_episodes
            if (str(row.get("video_id")),
                int(row.get("question_number") or 0)) in kept_questions),
        "training_sequences": len(training_rows) // len(seeds),
        "training_ready_rows": sum(
            bool((row.get("training_only") or {}).get("training_ready"))
            for row in training_rows),
        "hybrid_ready_rows": sum(
            bool((row.get("training_only") or {}).get("hybrid_ready"))
            for row in training_rows),
        "diagnostic_only_rows": sum(
            bool((row.get("memory_protocol") or {}).get("diagnostic_only"))
            for row in training_rows),
        "admitted_sequences": admitted,
        "required_admitted_sequences": required,
        "admission_problem_counts": dict(sorted(problems.items())),
        "rollback_check_count": len(rollback_checks),
        "rollback_failures": sum(
            not check.get("restored") for check in rollback_checks),
        "global_operation_counts": dict(sorted(global_ops.items())),
        "global_operations_valid": global_valid,
        "kept_global_operation_counts": dict(sorted(kept_global_ops.items())),
        "kept_global_operations_valid": kept_global_valid,
        "budget": {"calls": args.max_calls, "frames": args.frame_cap,
                   "minutes": args.max_minutes,
                   "tokens": args.max_tokens,
                   "new_tokens": args.max_new_tokens,
                   "turn_new_tokens": args.turn_max_new_tokens,
                   "total_timeout_seconds": args.total_timeout_seconds,
                   "smoke_use_cache": bool(getattr(
                       args, "smoke_use_cache", False)),
                   "scripted_smoke_client": bool(getattr(
                       args, "scripted_smoke_client", False))},
        "temperature": float(args.temperature),
        "policy_seed": int(args.policy_seed),
        **_run_metadata(args, seeds),
        "off_menu_probability": float(args.off_menu_probability),
        "memory_protocol": {
            "card_list_visible": bool(getattr(
                args, "card_list_visible", False)),
            "card_local_visible": bool(getattr(
                args, "card_local_visible", True)),
            "card_global_visible": bool(getattr(
                args, "card_global_visible", True)),
            "cards_per_scope": int(getattr(args, "cards_per_scope", 4)),
            "force_card_check": bool(getattr(
                args, "force_card_check", False)),
        },
        "groups_with_four_clean_rows": clean_groups,
        "required_clean_groups": required_clean,
        "global_source": str(getattr(args, "global_source", "timeline")),
        "read_order": str(getattr(args, "read_order", "positional")),
        "facts_block": bool(getattr(args, "facts_block", False)),
        "budget_answer": bool(getattr(args, "budget_answer", False)),
        "fresh_memory_each_question": bool(getattr(
            args, "fresh_memory_each_question", False)),
        "vllm_runtime": (os.path.abspath(args.vllm_runtime)
                         if getattr(args, "vllm_runtime", "") else ""),
        "vllm_runtime_sha256": (
            _sha256_file(args.vllm_runtime)
            if getattr(args, "vllm_runtime", "") else ""),
        "ledger_clean": ledger.errors == 0,
        "forced_failure": list(forced) if forced else None,
        "forced_failure_after_global_count": forced_after_global_count,
        "smoke_mode": bool(args.smoke),
        "scripted_smoke_client": bool(getattr(
            args, "scripted_smoke_client", False)),
        "sequence_audit": sequence_audit,
        "ledger": {"schema": QL.SCHEMA, "question_rows": ledger.rows,
                   "write_errors": ledger.errors,
                   "code_commit": ledger.ctx["code_commit"],
                   "prompt_revision": ledger.ctx["prompt_revision"],
                   "round": ledger.ctx["round"]},
        "gate_passes": bool(gate_passes),
    }
    _write_json(os.path.join(args.out, "RUN.json"), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_gate and not gate_passes:
        return 3
    return 0


def run(args):
    if _sha256_file(os.path.join(args.adapter,
                                 "adapter_model.safetensors")) \
            != args.adapter_sha256:
        raise ValueError("adapter hash differs from the launch record")
    if args.reader_policy:
        if not args.reader_adapter or not args.reader_adapter_sha256:
            raise ValueError("Reader policy needs an adapter and SHA-256")
        if _sha256_file(os.path.join(
                args.reader_adapter, "adapter_model.safetensors")) \
                != args.reader_adapter_sha256:
            raise ValueError("Reader adapter hash differs from the launch record")
    if RS.model_content_sha256(args.model) != args.model_sha256:
        raise ValueError("base model hash differs from the launch record")
    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    benchmark = _verify_manifest(manifest)
    _validate_sequence_layout(
        manifest, args.mode, args.questions_per_sequence,
        evaluation_split=args.evaluation_split)
    split = "train" if args.mode == "collect" else args.evaluation_split
    if args.mode == "collect":
        states = manifest.get("train_states") or []
    else:
        states = manifest.get("%s_items" % args.evaluation_split) or []
    states = [dict(state, split=split) for state in states]
    if args.video_id:
        states = [row for row in states if str(row["video"]) == args.video_id]
        if not states:
            raise ValueError("video filter matches no frozen state")
    expected = int(args.questions_per_sequence)
    if args.limit_videos and args.video_stride == 1:
        keep = {row["video"] for row in states}
        keep = sorted(keep)[:int(args.limit_videos)]
        states = [row for row in states if row["video"] in set(keep)]
    if os.path.exists(os.path.join(args.out, "RUN.json")):
        raise ValueError("output already has RUN.json; use a new directory")
    missing_videos = sorted({
        os.path.abspath(str(row.get("video_path") or ""))
        for row in states
        if not os.path.isfile(str(row.get("video_path") or ""))
    })
    if missing_videos and not args.smoke:
        raise ValueError("selected video files are missing: %s"
                         % ", ".join(missing_videos[:3]))
    return _run_sequences(args, states, expected, split, benchmark)


def _parse_rollout_seeds(value):
    if not str(value or "").strip():
        return list(ROL.ROLLOUT_SEEDS)
    try:
        seeds = [int(part.strip()) for part in str(value).split(",")
                 if part.strip()]
    except ValueError as error:
        raise ValueError("rollout seeds must be comma-separated integers") from error
    if len(seeds) != 4:
        raise ValueError("rollout seed override needs exactly four seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("rollout seed override seeds must be distinct")
    return seeds


def _clean_group_count(training_rows):
    groups = collections.defaultdict(list)
    for row in training_rows or ():
        group = str(row.get("group_id") or row.get("video_id") or "")
        groups[group].append(bool(
            (row.get("training_only") or {}).get("training_ready")))
    return sum(
        int(len(values) == 4 and all(values))
        for values in groups.values())


def _run_metadata(args, seeds):
    return {
        "rollout_seeds": [int(seed) for seed in seeds],
        "rollout_seed_source": (
            "override" if str(getattr(args, "rollout_seeds", "")).strip()
            else "frozen"),
        "diagnostic_only": bool(getattr(args, "diagnostic_only", False)),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--mode", choices=("collect", "evaluate"),
                        default="collect")
    parser.add_argument("--evaluation-split", choices=("dev", "test"),
                        default="dev")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--reader-adapter", default="")
    parser.add_argument("--reader-adapter-sha256", default="")
    parser.add_argument("--reader-policy", action="store_true")
    parser.add_argument("--reader-schema-version", type=int,
                        choices=(1, 2, 3), default=1)
    parser.add_argument("--reader-temperature", type=float, default=0.8)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--card-list-visible",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--card-local-visible",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--card-global-visible",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cards-per-scope", type=int, default=4)
    parser.add_argument("--force-card-check", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--questions-per-sequence", type=int, default=6,
        help="positive for fixed sequences; zero for variable MLVU sequences")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--max-minutes", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="per-question cap on prompt plus generated tokens")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--turn-max-new-tokens", type=int, default=512)
    parser.add_argument("--total-timeout-seconds", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=20261111)
    parser.add_argument("--rollout-seeds", default="")
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--require-clean-groups", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--off-menu-probability", type=float, default=0.1)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--video-offset", type=int, default=0)
    parser.add_argument("--video-stride", type=int, default=1)
    parser.add_argument("--video-id", default="")
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--required-admitted", type=int,
                        default=S.MIN_ADMITTED_SEQUENCES)
    parser.add_argument("--global-source", choices=("timeline",),
                        default="timeline",
                        help="global memory is the chronological timeline the "
                             "harness builds from stored local observations")
    parser.add_argument("--read-order", choices=("positional", "steered"),
                        default="positional")
    parser.add_argument("--facts-block", action="store_true")
    parser.add_argument("--fresh-memory-each-question", action="store_true",
                        help="start every question from the empty tree, so no "
                             "memory from an earlier question is visible")
    parser.add_argument("--budget-answer", action="store_true",
                        help="when the budget is nearly spent, require a final "
                             "answer from the evidence read so far instead of "
                             "leaving ABSTAIN as the only legal action")
    parser.add_argument("--vllm-runtime", default="",
                        help="evaluation only: serve every role from a running "
                             "vLLM lane described by this runtime file")
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--round-id", default="")
    parser.add_argument("--force-fail", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-use-cache", action="store_true")
    parser.add_argument("--scripted-smoke-client", action="store_true")
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument("--memory-precommit-dir", default="")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.video_offset < 0 or args.video_stride < 1 \
            or args.video_offset >= args.video_stride:
        parser.error("video offset must be in [0, video stride)")
    if args.smoke and not args.limit_videos:
        parser.error("smoke mode requires --limit-videos")
    if args.smoke_use_cache and not args.smoke:
        parser.error("--smoke-use-cache requires --smoke")
    if args.scripted_smoke_client and not args.smoke:
        parser.error("--scripted-smoke-client requires --smoke")
    if args.scripted_smoke_client and not args.diagnostic_only:
        parser.error("--scripted-smoke-client requires --diagnostic-only")
    if args.vllm_runtime:
        if args.mode != "evaluate":
            parser.error("--vllm-runtime is for evaluate mode only")
        if not args.diagnostic_only:
            parser.error("--vllm-runtime requires --diagnostic-only: vLLM "
                         "traces are never PPO evidence")
        if args.scripted_smoke_client or args.reader_policy:
            parser.error("--vllm-runtime excludes the scripted client and "
                         "a Reader policy")
    if args.questions_per_sequence < 0:
        parser.error("--questions-per-sequence must be non-negative")
    if args.limit_questions < 0:
        parser.error("--limit-questions must be non-negative")
    if args.limit_questions and not args.smoke:
        parser.error("--limit-questions requires --smoke")
    if args.questions_per_sequence and (
            args.limit_questions > args.questions_per_sequence):
        parser.error("--limit-questions exceeds the frozen sequence length")
    if args.cards_per_scope < 1:
        parser.error("--cards-per-scope must be positive")
    if args.force_card_check and not args.card_list_visible:
        parser.error("--force-card-check requires --card-list-visible")
    try:
        _parse_rollout_seeds(args.rollout_seeds)
    except ValueError as error:
        parser.error(str(error))
    if args.mode != "collect" and args.rollout_seeds:
        parser.error("rollout seed override is only valid in collect mode")
    if args.require_clean_groups < 0:
        parser.error("required clean groups must be non-negative")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
