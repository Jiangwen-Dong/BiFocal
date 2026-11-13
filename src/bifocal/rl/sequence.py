"""Ordered online-RL sequences with persistent memory and common masking."""

from __future__ import annotations

import copy
import hashlib
import json
import math

from ..memory import branches as B


SEQUENCE_SCHEMA = "online-memory-sequence-v1"
QUESTIONS_PER_SEQUENCE = 6
MIN_COMMON_QUESTIONS = 3
MIN_ADMITTED_SEQUENCES = 300


def group_states(states, questions_per_sequence=QUESTIONS_PER_SEQUENCE):
    """Group frozen states into ordered video sequences.

    A positive count requires equal-length sequences. Zero keeps every question
    for each video and therefore supports MLVU videos with different counts.
    """
    grouped = {}
    for state in states:
        grouped.setdefault(str(state["video"]), []).append(copy.deepcopy(state))
    sequences = []
    expected = int(questions_per_sequence)
    if expected < 0:
        raise ValueError("questions per sequence must be non-negative")
    for video in sorted(grouped):
        rows = sorted(grouped[video], key=lambda row: int(row["question_number"]))
        if expected and len(rows) != expected:
            raise ValueError(
                "video %s has %d questions, need %d"
                % (video, len(rows), expected))
        if not rows:
            raise ValueError("video %s has no questions" % video)
        numbers = [int(row["question_number"]) for row in rows]
        if len(set(numbers)) != len(numbers):
            raise ValueError("video %s repeats a question number" % video)
        if any(right != left + 1 for left, right in zip(numbers, numbers[1:])):
            raise ValueError("video %s sequence is not contiguous" % video)
        sequences.append({"video": video, "states": rows})
    if not sequences:
        raise ValueError("no video sequences")
    return sequences


def checkpoint(tree):
    return copy.deepcopy(tree.to_dict())


def checkpoint_sha256(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def memory_sha256(value):
    """Hash memory content but not allocator counters changed by rollback."""
    clean = copy.deepcopy(value)
    for name in ("next_revision", "next_event", "next_review",
                 "next_memory_use", "next_global_claim",
                 "next_global_operation"):
        clean.pop(name, None)
    return checkpoint_sha256(clean)


def restore(checkpoint_value, root):
    return B.tree_from_dict(checkpoint_value, root=root)


def sequence_rollback_checks(sequence):
    """Did every rolled-back question leave the next question's start tree
    equal to its own? Same rule the admission code applies in collect mode."""
    checks = []
    questions = [int(value) for value in sequence["questions"]]
    for run in sequence["rollouts"]:
        starts = {int(row["question_number"]): str(row["sha256"])
                  for row in run.get("question_start_sha256") or []}
        skipped = {int(row["question_number"])
                   for row in run["skipped_questions"]}
        for index, question in enumerate(questions[:-1]):
            if question in skipped:
                checks.append({
                    "rollout_index": int(run["rollout_index"]),
                    "question_number": question,
                    "next_question_number": questions[index + 1],
                    "restored": starts.get(question) == starts.get(
                        questions[index + 1]),
                })
    return checks


def common_questions(rollout_runs):
    if not rollout_runs:
        return []
    sets = [set(int(value) for value in run["kept_questions"])
            for run in rollout_runs]
    common = set.intersection(*sets)
    return sorted(common)


def run_video_sequence(video, states, rollout_seeds, start_tree,
                       run_question, memory_root, question_ready=None,
                       progress_callback=None, question_callback=None,
                       fresh_memory_each_question=False):
    """Run rollouts in question lockstep and roll back any failed question.

    With fresh_memory_each_question every question starts from start_tree, so no
    memory written for an earlier question is visible (the no-persistent-memory
    arm). Revision counters still advance through high_water, so ids stay unique.
    """
    if len(set(int(value) for value in rollout_seeds)) != len(rollout_seeds):
        raise ValueError("rollout seeds must be distinct")
    rollouts = []
    for index, seed in enumerate(rollout_seeds):
        root = "%s/rollout-%d" % (memory_root.rstrip("/"), index)
        rollouts.append({
            "rollout_index": index,
            "seed": int(seed),
            "memory_root": root,
            "tree": restore(checkpoint(start_tree), root + "/q0"),
            "kept_questions": [],
            "question_episodes": [],
            "skipped_questions": [],
            "question_start_sha256": [],
            "high_water": {name: int(getattr(
                restore(checkpoint(start_tree), root + "/q0"), name))
                for name in ("next_revision", "next_event", "next_review",
                             "next_memory_use", "next_global_claim",
                             "next_global_operation")},
        })
    for state in states:
        question_number = int(state["question_number"])
        before = [checkpoint(start_tree if fresh_memory_each_question
                             else run["tree"]) for run in rollouts]
        for run, value in zip(rollouts, before):
            run["question_start_sha256"].append({
                "question_number": question_number,
                "sha256": memory_sha256(value),
            })
        candidate_trees = []
        results = []
        for run in rollouts:
            root = "%s/q%d" % (run["memory_root"], question_number)
            tree = restore(before[run["rollout_index"]], root)
            for name, value in run["high_water"].items():
                setattr(tree, name, max(int(getattr(tree, name)), int(value)))
            try:
                result = run_question(
                    state, tree, run["rollout_index"], run["seed"])
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                result = {
                    "hybrid_ready": False,
                    "training_ready": False,
                    "episode_id": "",
                    "outcome": {},
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "atomic_row_present": hasattr(error, "atomic_row"),
                        "atomic_row": copy.deepcopy(
                            getattr(error, "atomic_row", None)),
                    },
                }
            candidate_trees.append(tree)
            results.append(copy.deepcopy(result))
            if progress_callback is not None:
                progress_callback({
                    "phase": "rollout_complete",
                    "video": video,
                    "question_number": question_number,
                    "rollout_index": int(run["rollout_index"]),
                    "seed": int(run["seed"]),
                    "hybrid_ready": bool(result.get("hybrid_ready")),
                    "training_ready": bool(result.get("training_ready")),
                    "episode_id": str(result.get("episode_id") or ""),
                    "failure": copy.deepcopy(result.get("failure")),
                })
        ready = question_ready or (
            lambda result: bool(result.get("hybrid_ready")))
        if question_callback is not None:
            flags = [bool(ready(result)) for result in results]
            question_callback(question_number, all(flags), [{
                "rollout_index": index, "ready": flags[index],
                "forced_failure": bool(value.get("forced_failure")),
                "forced_failure_after_global": bool(
                    value.get("forced_failure_after_global")),
                "failure": (value.get("failure") or {}).get("type"),
            } for index, value in enumerate(results) if not flags[index]])
        if all(ready(result) for result in results):
            for run, tree, result in zip(rollouts, candidate_trees, results):
                run["tree"] = tree
                run["kept_questions"].append(question_number)
                run["question_episodes"].append(result.get("episode") or result)
        else:
            for run, result, candidate in zip(
                    rollouts, results, candidate_trees):
                for name in run["high_water"]:
                    run["high_water"][name] = max(
                        int(run["high_water"][name]),
                        int(getattr(candidate, name)))
                run["tree"] = restore(
                    before[run["rollout_index"]],
                    run["memory_root"] + "/restored")
                for name, value in run["high_water"].items():
                    setattr(run["tree"], name,
                            max(int(getattr(run["tree"], name)), int(value)))
                run["skipped_questions"].append({
                    "question_number": question_number,
                    "rollback": True,
                    "rollouts": [{
                        "rollout_index": index,
                        "hybrid_ready": bool(value.get("hybrid_ready")),
                        "episode_id": value.get("episode_id", ""),
                        "terminal": (value.get("outcome") or {}).get("terminal"),
                        "correct": bool((value.get("outcome") or {}).get("correct")),
                        "protocol_fallback": bool(
                            (value.get("outcome") or {}).get("protocol_fallback")),
                        "harness_recovery": bool(
                            (value.get("outcome") or {}).get("harness_recovery")),
                        "quality": ((value.get("episode") or {}).get(
                            "training_only") or {}).get("quality"),
                        "forced_failure": bool(value.get("forced_failure")),
                        "forced_failure_after_global": bool(
                            value.get("forced_failure_after_global")),
                        "failure": copy.deepcopy(value.get("failure")),
                    } for index, value in enumerate(results)],
                })
    common = common_questions(rollouts)
    for run in rollouts:
        if sorted(run["kept_questions"]) != common:
            raise ValueError("rollouts retained different common question sets")
    return {
        "schema_version": SEQUENCE_SCHEMA,
        "video": video,
        "questions": [int(state["question_number"]) for state in states],
        "common_questions": common,
        "rollouts": rollouts,
    }


def _sum_components(episodes, key):
    totals = {}
    for episode in episodes:
        value = (episode.get("training_only") or {}).get(key) or {}
        for name, item in value.items():
            if name in ("step_returns", "total") or not isinstance(
                    item, (int, float)):
                continue
            totals[name] = totals.get(name, 0.0) + float(item)
    return {name: round(value, 6) for name, value in sorted(totals.items())}


def aggregate_rollout(sequence, run):
    episodes = list(run["question_episodes"])
    if not episodes:
        return None
    steps = []
    for question_index, episode in enumerate(episodes):
        for step in copy.deepcopy(episode.get("steps") or []):
            step["t"] = len(steps)
            step["parent_step"] = len(steps) - 1 if steps else None
            step["sequence_question_index"] = question_index
            step["question_number"] = int(episode.get("question_number") or 0)
            steps.append(step)
    reward_total_components = _sum_components(episodes, "reward_components")
    optimization_total_components = _sum_components(
        episodes, "optimization_reward_components")
    reader_total_components = _sum_components(
        episodes, "reader_reward_components")
    count = float(len(episodes))
    reward_components = {name: round(value / count, 6)
                         for name, value in reward_total_components.items()}
    optimization = {name: round(value / count, 6)
                    for name, value in optimization_total_components.items()}
    reader_components = {name: round(value / count, 6)
                         for name, value in reader_total_components.items()}
    reward = sum(float(value) for value in reward_components.values())
    optimization_reward = sum(float(value) for value in optimization.values())
    reader_optimization_reward = sum(
        float(value) for value in reader_components.values())
    first = episodes[0]
    outcomes = [episode.get("outcome") or {} for episode in episodes]
    quality = ("clean" if all(
        (episode.get("training_only") or {}).get("quality") == "clean"
        for episode in episodes) else "repair_masked")
    question_rows = [{
        "episode_id": episode.get("episode_id"),
        "uid": episode.get("uid"),
        "question_number": int(episode.get("question_number") or 0),
        "num_options": (episode.get("training_only") or {}).get("num_options"),
        "outcome": copy.deepcopy(episode.get("outcome") or {}),
        "reward_components": copy.deepcopy(
            (episode.get("training_only") or {}).get("reward_components") or {}),
        "global_operations": [
            op for step in episode.get("steps") or []
            for op in step.get("memory_ops") or []
            if op.get("op") == "evolve_global"],
    } for episode in episodes]
    return {
        "schema_version": first["schema_version"],
        "episode_id": "%s:%d" % (sequence["video"], run["rollout_index"]),
        "group_id": str(sequence["video"]),
        "rollout_index": int(run["rollout_index"]),
        "split": first.get("split"),
        "uid": str(sequence["video"]),
        "video_id": str(sequence["video"]),
        "question_number": 0,
        "budget": copy.deepcopy(first.get("budget") or {}),
        "sampling": copy.deepcopy(first.get("sampling") or {}),
        "policy_versions": copy.deepcopy(first.get("policy_versions") or {}),
        "memory_root_ref": str(run["memory_root"]),
        "memory_fork_from": copy.deepcopy(first.get("memory_fork_from")),
        "memory_budget": copy.deepcopy(first.get("memory_budget") or {}),
        "memory_protocol": copy.deepcopy(first.get("memory_protocol") or {}),
        "steps": steps,
        "outcome": {
            "terminal": ("agent_submit" if any(value.get("answered")
                                                    for value in outcomes)
                         else "agent_abstain"),
            "answer": "; ".join(str(value.get("answer") or "") for value in outcomes),
            "answered": all(bool(value.get("answered")) for value in outcomes),
            "complete": True,
            "correct": sum(bool(value.get("correct")) for value in outcomes) > 0,
            "correct_count": sum(bool(value.get("correct")) for value in outcomes),
            "question_count": len(outcomes),
            "calls": sum(int(value.get("calls") or 0) for value in outcomes),
            "frames": sum(int(value.get("frames") or 0) for value in outcomes),
            "fault": "",
            "protocol_fallback": any(
                bool(value.get("protocol_fallback")) for value in outcomes),
            "harness_recovery": any(
                bool(value.get("harness_recovery")) for value in outcomes),
            "infrastructure_valid": all(
                bool(value.get("infrastructure_valid")) for value in outcomes),
        },
        "training_only": {
            "gold": [episode.get("training_only", {}).get("gold")
                     for episode in episodes],
            "correct": sum(bool(value.get("correct")) for value in outcomes) > 0,
            "correct_count": sum(bool(value.get("correct")) for value in outcomes),
            "reward_components": reward_components,
            "reward_components_total": reward_total_components,
            "reward": round(reward, 6),
            "reward_total": round(sum(reward_total_components.values()), 6),
            "step_returns": [copy.deepcopy(value)
                             for episode in episodes
                             for value in (episode.get("training_only") or {}
                                           ).get("step_returns") or []],
            "memory_rewards": _sum_components(episodes, "memory_rewards"),
            "memory_carrier": None,
            "quality": quality,
            "hybrid_ready": all(
                bool((episode.get("training_only") or {}).get("hybrid_ready"))
                for episode in episodes),
            "optimization_reward_components": optimization,
            "optimization_reward_components_total": optimization_total_components,
            "optimization_reward": round(optimization_reward, 6),
            "optimization_reward_total": round(
                sum(optimization_total_components.values()), 6),
            "reader_reward_components": reader_components,
            "reader_reward_components_total": reader_total_components,
            "reader_reward": round(sum(reader_components.values()), 6),
            "reader_optimization_reward": round(
                reader_optimization_reward, 6),
            "reader_execution_clean": all(
                bool((episode.get("training_only") or {}).get(
                    "reader_execution_clean")) for episode in episodes),
            "reader_reward_clean": all(
                bool((episode.get("training_only") or {}).get(
                    "reader_reward_clean")) for episode in episodes),
            "reader_training_ready": all(
                bool((episode.get("training_only") or {}).get(
                    "reader_training_ready")) for episode in episodes),
            "reader_memory_valid": all(
                bool((episode.get("training_only") or {}).get(
                    "reader_memory_valid", True)) for episode in episodes),
            "reader_provenance_valid": all(
                bool((episode.get("training_only") or {}).get(
                    "reader_provenance_valid", True)) for episode in episodes),
            "reader_event_records": [
                copy.deepcopy(value)
                for episode in episodes
                for value in (episode.get("training_only") or {}).get(
                    "reader_event_records") or []],
            "training_ready": all(
                bool((episode.get("training_only") or {}).get("training_ready"))
                for episode in episodes),
            "pairing_errors": [],
            "missing": [],
        },
        "sequence": {
            "questions": list(sequence["questions"]),
            "common_questions": list(sequence["common_questions"]),
            "kept_questions": list(run["kept_questions"]),
            "skipped_questions": copy.deepcopy(run["skipped_questions"]),
            "seed": int(run["seed"]),
            "memory_root": str(run["memory_root"]),
            "question_start_sha256": copy.deepcopy(
                run.get("question_start_sha256") or []),
            "question_episodes": question_rows,
        },
    }


def permute_within_group(values, seed, group):
    """Deterministically permute one rollout group's values (shuffled-reward control).

    The four rollouts keep their sampled tokens but receive another rollout's
    advantage, so the update carries no reward signal. `seed` and `group` fix the
    permutation, which makes a control run repeatable.
    """
    import random
    order = list(range(len(values)))
    random.Random("%s:%s" % (seed, group)).shuffle(order)
    return [values[index] for index in order]


def attach_sequence_advantages(episodes, require_variance=True):
    """Normalize full sequence returns within one four-rollout group."""
    if len(episodes) != 4:
        raise ValueError("a trainable sequence needs four rollouts")
    rewards = [float(row["training_only"]["optimization_reward"])
               for row in episodes]
    mean = sum(rewards) / 4.0
    variance = sum((value - mean) ** 2 for value in rewards) / 4.0
    std = math.sqrt(variance)
    if std <= 1e-9 and require_variance:
        raise ValueError("sequence reward variance is zero")
    for row, reward in zip(episodes, rewards):
        row["training_only"]["sequence_return"] = round(reward, 6)
        row["training_only"]["advantage"] = round(
            (reward - mean) / std if std > 1e-9 else 0.0, 12)
        row["training_only"]["task_advantage"] = row["training_only"]["advantage"]
        row["training_only"]["cost_advantage"] = 0.0
    return {
        "returns": rewards,
        "mean": mean,
        "std": std,
        "advantages": [row["training_only"]["advantage"] for row in episodes],
    }


def admit_sequence(sequence, require_variance=True,
                   min_common_questions=MIN_COMMON_QUESTIONS):
    rollouts = sequence["rollouts"]
    problems = []
    if len(rollouts) != 4:
        problems.append("rollout_count")
    if len({int(row["seed"]) for row in rollouts}) != 4:
        problems.append("duplicate_seed")
    if len({str(row["memory_root"]) for row in rollouts}) != 4:
        problems.append("shared_memory_root")
    common = common_questions(rollouts)
    minimum_common = int(min_common_questions)
    if minimum_common < 1:
        raise ValueError("minimum common questions must be positive")
    if len(common) < minimum_common:
        name = {1: "one", 2: "two", 3: "three", 4: "four"}.get(
            minimum_common, str(minimum_common))
        problems.append("fewer_than_%s_common_questions" % name)
    if any(sorted(row["kept_questions"]) != common for row in rollouts):
        problems.append("common_question_set_differs")
    episodes = [aggregate_rollout(sequence, run) for run in rollouts]
    question_diagnostics = []
    for question_number in sequence["questions"]:
        rollout_rows = []
        for run in rollouts:
            episode = next((
                row for row in run["question_episodes"]
                if int(row.get("question_number") or 0)
                == int(question_number)), None)
            if episode is None:
                rollout_rows.append({
                    "rollout_index": int(run["rollout_index"]), "kept": False})
                continue
            outcome = episode.get("outcome") or {}
            training = episode.get("training_only") or {}
            rollout_rows.append({
                "rollout_index": int(run["rollout_index"]),
                "kept": True,
                "hybrid_ready": bool(training.get("hybrid_ready")),
                "training_ready": bool(training.get("training_ready")),
                "quality": training.get("quality"),
                "terminal": outcome.get("terminal"),
                "correct": bool(outcome.get("correct")),
                "calls": int(outcome.get("calls") or 0),
                "frames": int(outcome.get("frames") or 0),
                "protocol_fallback": bool(outcome.get("protocol_fallback")),
                "harness_recovery": bool(outcome.get("harness_recovery")),
            })
        question_diagnostics.append({
            "question_number": int(question_number),
            "rollouts": rollout_rows,
        })
    rollback_checks = []
    for run in rollouts:
        starts = {int(row["question_number"]): str(row["sha256"])
                  for row in run.get("question_start_sha256") or []}
        questions = [int(value) for value in sequence["questions"]]
        for index, question in enumerate(questions[:-1]):
            skipped = {int(row["question_number"])
                       for row in run["skipped_questions"]}
            if question in skipped:
                rollback_checks.append({
                    "rollout_index": int(run["rollout_index"]),
                    "question_number": question,
                    "next_question_number": questions[index + 1],
                    "restored": starts.get(question) == starts.get(
                        questions[index + 1]),
                })
    if any(not row["restored"] for row in rollback_checks):
        problems.append("rollback_not_restored")
    advantage = None
    if episodes and all(value is not None for value in episodes):
        try:
            advantage = attach_sequence_advantages(
                episodes, require_variance=require_variance)
        except ValueError:
            problems.append("zero_sequence_reward_variance")
    else:
        problems.append("missing_rollout_episode")
    summary = {
        "video": sequence["video"],
        "questions": sequence["questions"],
        "common_questions": common,
        "common_count": len(common),
        "skipped_questions": [
            int(row["question_number"])
            for row in rollouts[0]["skipped_questions"]],
        "rollout_seeds": [int(row["seed"]) for row in rollouts],
        "memory_roots": [str(row["memory_root"]) for row in rollouts],
        "returns": advantage["returns"] if advantage else [],
        "question_diagnostics": question_diagnostics,
        "skipped_diagnostics": copy.deepcopy(
            rollouts[0]["skipped_questions"]),
        "rollback_checks": rollback_checks,
        "admitted": not problems,
        "problems": problems,
    }
    return summary, (episodes if not problems else [])


def attach_reader_advantages(episodes, require_variance=True):
    """Normalize Reader full returns within one four-rollout video group."""
    if len(episodes) != 4:
        raise ValueError("a trainable Reader sequence needs four rollouts")
    rewards = [float(row["training_only"]["reader_optimization_reward"])
               for row in episodes]
    mean = sum(rewards) / 4.0
    std = math.sqrt(sum((value - mean) ** 2 for value in rewards) / 4.0)
    if std <= 1e-9 and require_variance:
        raise ValueError("Reader sequence reward variance is zero")
    for row, reward in zip(episodes, rewards):
        row["training_only"]["reader_sequence_return"] = round(reward, 6)
        row["training_only"]["reader_advantage"] = round(
            (reward - mean) / std if std > 1e-9 else 0.0, 12)
    return {
        "returns": rewards,
        "mean": mean,
        "std": std,
        "advantages": [row["training_only"]["reader_advantage"]
                       for row in episodes],
    }


def admit_reader_sequence(sequence, require_variance=True,
                          min_common_questions=MIN_COMMON_QUESTIONS):
    """Strict Reader-clean admission; repaired Reader calls never enter PPO."""
    rollouts = sequence["rollouts"]
    problems = []
    if len(rollouts) != 4:
        problems.append("rollout_count")
    if len({int(row["seed"]) for row in rollouts}) != 4:
        problems.append("duplicate_reader_seed")
    if len({str(row["memory_root"]) for row in rollouts}) != 4:
        problems.append("shared_memory_root")
    common = common_questions(rollouts)
    minimum_common = int(min_common_questions)
    if minimum_common < 1:
        raise ValueError("minimum common questions must be positive")
    if len(common) < minimum_common:
        name = {1: "one", 2: "two", 3: "three", 4: "four"}.get(
            minimum_common, str(minimum_common))
        problems.append("fewer_than_%s_common_questions" % name)
    if any(sorted(row["kept_questions"]) != common for row in rollouts):
        problems.append("common_question_set_differs")
    episodes = [aggregate_rollout(sequence, run) for run in rollouts]
    question_diagnostics = []
    for question_number in sequence["questions"]:
        rollout_rows = []
        for run in rollouts:
            episode = next((
                row for row in run["question_episodes"]
                if int(row.get("question_number") or 0)
                == int(question_number)), None)
            if episode is None:
                rollout_rows.append({
                    "rollout_index": int(run["rollout_index"]), "kept": False})
                continue
            outcome = episode.get("outcome") or {}
            training = episode.get("training_only") or {}
            rollout_rows.append({
                "rollout_index": int(run["rollout_index"]),
                "kept": True,
                "reader_execution_clean": bool(
                    training.get("reader_execution_clean")),
                "reader_reward_clean": bool(
                    training.get("reader_reward_clean")),
                "reader_training_ready": bool(
                    training.get("reader_training_ready")),
                "reader_memory_valid": bool(
                    training.get("reader_memory_valid")),
                "terminal": outcome.get("terminal"),
                "correct": bool(outcome.get("correct")),
                "calls": int(outcome.get("calls") or 0),
                "frames": int(outcome.get("frames") or 0),
                "protocol_fallback": bool(outcome.get("protocol_fallback")),
                "harness_recovery": bool(outcome.get("harness_recovery")),
            })
        question_diagnostics.append({
            "question_number": int(question_number),
            "rollouts": rollout_rows,
        })
    if any(not row.get("reader_execution_clean")
           for question in question_diagnostics
           for row in question["rollouts"] if row.get("kept")):
        problems.append("reader_not_execution_clean")
    if any(not row.get("reader_reward_clean")
           for question in question_diagnostics
           for row in question["rollouts"] if row.get("kept")):
        problems.append("reader_not_reward_clean")
    if any(episode is None for episode in episodes):
        problems.append("missing_rollout_episode")
    rollback_checks = []
    for run in rollouts:
        starts = {int(row["question_number"]): str(row["sha256"])
                  for row in run.get("question_start_sha256") or []}
        questions = [int(value) for value in sequence["questions"]]
        for index, question in enumerate(questions[:-1]):
            skipped = {int(row["question_number"])
                       for row in run["skipped_questions"]}
            if question in skipped:
                rollback_checks.append({
                    "rollout_index": int(run["rollout_index"]),
                    "question_number": question,
                    "next_question_number": questions[index + 1],
                    "restored": starts.get(question) == starts.get(
                        questions[index + 1]),
                })
    if any(not row["restored"] for row in rollback_checks):
        problems.append("rollback_not_restored")
    advantage = None
    if episodes and all(value is not None for value in episodes):
        correct = [bool(value["outcome"].get("correct"))
                   for value in episodes]
        if len(set(correct)) <= 1:
            problems.append("frozen_o_correctness_constant")
        try:
            advantage = attach_reader_advantages(
                episodes, require_variance=require_variance)
        except ValueError:
            problems.append("zero_reader_reward_variance")
    summary = {
        "video": sequence["video"],
        "questions": sequence["questions"],
        "common_questions": common,
        "common_count": len(common),
        "skipped_questions": [
            int(row["question_number"])
            for row in rollouts[0]["skipped_questions"]],
        "rollout_seeds": [int(row["seed"]) for row in rollouts],
        "memory_roots": [str(row["memory_root"]) for row in rollouts],
        "returns": advantage["returns"] if advantage else [],
        "reader_correct_bits": [
            bool(value["outcome"].get("correct")) for value in episodes
            if value is not None],
        "question_diagnostics": question_diagnostics,
        "skipped_diagnostics": copy.deepcopy(
            rollouts[0]["skipped_questions"]),
        "rollback_checks": rollback_checks,
        "admitted": not problems,
        "problems": problems,
    }
    return summary, (episodes if not problems else [])


def audit_sequence_bundle(sequence_results, summaries, training_rows,
                          required_groups=MIN_ADMITTED_SEQUENCES):
    """Audit the sequence masks, roots, seeds and rollback evidence."""
    summary_by_video = {row["video"]: row for row in summaries}
    rows_by_video = {}
    for row in training_rows:
        rows_by_video.setdefault(str(row["group_id"]), []).append(row)
    checks = {
        "all_videos_have_one_summary": len(summary_by_video) == len(
            sequence_results),
        "four_distinct_seeds": True,
        "four_distinct_roots": True,
        "first_question_hashes_equal": True,
        "common_sets_equal": True,
        "rollback_checks_restored": True,
        "training_groups_have_four_rows": True,
        "masked_questions_absent_from_training": True,
        "training_returns_finite": True,
        "global_memory_enabled": True,
        "admitted_groups_meet_gate": len(rows_by_video) >= int(required_groups),
    }
    for sequence in sequence_results:
        rollouts = sequence["rollouts"]
        checks["four_distinct_seeds"] &= (
            len({int(row["seed"]) for row in rollouts}) == 4)
        checks["four_distinct_roots"] &= (
            len({str(row["memory_root"]) for row in rollouts}) == 4)
        starts = [str((run.get("question_start_sha256") or [{}])[0].get(
            "sha256") or "") for run in rollouts]
        checks["first_question_hashes_equal"] &= (
            len(set(starts)) == 1 and bool(starts[0]))
        common = sorted(common_questions(rollouts))
        checks["common_sets_equal"] &= all(
            sorted(run["kept_questions"]) == common for run in rollouts)
        summary = summary_by_video.get(sequence["video"]) or {}
        checks["rollback_checks_restored"] &= all(
            check.get("restored") for check in summary.get("rollback_checks") or [])
        rows = rows_by_video.get(sequence["video"], [])
        if summary.get("admitted"):
            checks["training_groups_have_four_rows"] &= len(rows) == 4
            for row in rows:
                checks["training_returns_finite"] &= math.isfinite(
                    float(row["training_only"]["optimization_reward"]))
                checks["global_memory_enabled"] &= bool(
                    (row.get("memory_protocol") or {}).get(
                        "global_memory_enabled"))
                checks["masked_questions_absent_from_training"] &= all(
                    int(step.get("question_number") or 0) in common
                    for step in row.get("steps") or [])
    return {"passes": all(checks.values()), "checks": checks}
