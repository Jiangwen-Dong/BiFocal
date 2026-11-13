#!/usr/bin/env python3
"""One grouped policy update for O with Reader and REVIEW frozen."""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import random
import re
import sys
import time

from . import credit as MD
from . import action_signature as ACTION
from ..serving import client as LC
from . import episode as E
from .sequence import permute_within_group
from ..serving import lanes as RS
from ..paths import project_root

SCHEMA_VERSION = "online-o-v1"
REVIEW_TOKEN_WEIGHT = 0.5
REPLAY_FORWARD = "teacher_forced_full_sequence"
MIN_SEQUENCE_QUESTIONS = 3
AUTO_MODEL_SHA256 = "AUTO"
ADVANTAGE_CLIP = 2.0
# ``full_reward`` is the legacy name for the same task-plus-cost objective.
TASK_COST_MODES = ("task_cost", "full_reward")


def resolve_model_sha256(expected, model_path):
    """Resolve the configured model hash against the selected local snapshot."""
    if str(expected) == AUTO_MODEL_SHA256:
        return RS.model_content_sha256(model_path)
    return str(expected)


def clip_advantage(value):
    return max(-ADVANTAGE_CLIP, min(ADVANTAGE_CLIP, float(value)))


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported online-O smoke schema")
    collection = config.get("collection") or {}
    training = config.get("training") or {}
    advantage_mode = str(collection.get("advantage_mode") or "task_variation")
    if advantage_mode not in ("task_variation",) + TASK_COST_MODES:
        raise ValueError("unsupported advantage_mode")
    admission_mode = str(collection.get("admission_mode") or "strict")
    if admission_mode not in ("strict", "hybrid"):
        raise ValueError("unsupported admission_mode")
    guess_incentive_scope = str(collection.get(
        "guess_incentive_scope") or "batch")
    if guess_incentive_scope not in ("batch", "per_group"):
        raise ValueError("unsupported guess_incentive_scope")
    if "sequence_level" in collection \
            and not isinstance(collection["sequence_level"], bool):
        raise ValueError("sequence_level must be boolean")
    if "streaming_step" in collection \
            and not isinstance(collection["streaming_step"], bool):
        raise ValueError("streaming_step must be boolean")
    streaming_step = bool(collection.get("streaming_step"))
    if int(collection.get("group_size") or 0) != 4:
        raise ValueError("online-O smoke requires four rollouts per group")
    if float(collection.get("top_p") or 0.0) != 1.0:
        raise ValueError("online-O mechanics smoke requires top_p=1")
    if streaming_step and int(training.get("world_size") or 0) != 1:
        raise ValueError("streaming PPO requires one training rank")
    if not streaming_step and int(training.get("world_size") or 0) not in (1, 2):
        raise ValueError("batched online-O requires one or two ranks")
    if int(training.get("micro_batch_per_gpu") or 0) != 1:
        raise ValueError("online-O smoke requires micro batch one")
    if int(training.get("grad_accum") or 0) <= 0:
        raise ValueError("gradient accumulation must be positive")
    if int(training.get("effective_groups") or 0) != (
            int(training["world_size"]) * int(training["grad_accum"])):
        raise ValueError("effective groups must equal world size times accumulation")
    if streaming_step:
        if not collection.get("sequence_level"):
            raise ValueError("streaming PPO requires sequence_level")
        if advantage_mode not in TASK_COST_MODES \
                or admission_mode != "hybrid":
            raise ValueError(
                "streaming PPO requires task_cost and hybrid admission")
        expected = int(collection.get("expected_groups") or 0)
        if not 1 <= expected <= 4:
            raise ValueError("streaming PPO update must contain one to four groups")
        expected_fields = {
            "batch_size": expected,
            "expected_batches": 1,
            "min_task_variable_groups": expected,
            "min_task_variable_per_batch": expected,
            "max_task_variable_per_batch": expected,
        }
        for name, wanted in expected_fields.items():
            if int(collection.get(name) or 0) != wanted:
                raise ValueError("streaming PPO %s must equal %d" % (name, wanted))
        if int(training.get("update_batches") or 0) != 1:
            raise ValueError("streaming PPO requires one update batch")
        if int(training["grad_accum"]) != expected:
            raise ValueError("streaming PPO accumulation must equal group count")
    if list(training.get("target_modules") or ()) != [
            "q_proj", "k_proj", "v_proj", "o_proj"]:
        raise ValueError("unexpected language LoRA target modules")
    if training.get("vision_trainable") is not False:
        raise ValueError("vision must remain frozen")
    if str(training.get("precision")) != "bf16":
        raise ValueError("training precision must be bf16")
    if training.get("gradient_checkpointing") is not True:
        raise ValueError("gradient_checkpointing must be enabled")
    if str(training.get("replay_forward") or REPLAY_FORWARD) != REPLAY_FORWARD:
        raise ValueError("online-O training requires teacher-forced replay")
    if str(training.get("attention", "sdpa")) not in (
            "sdpa", "flash_attention_2"):
        raise ValueError("attention must be sdpa or flash_attention_2")
    quantization = training.get("quantization")
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise ValueError("quantization must be an object")
        if quantization.get("load_in_4bit") is not True:
            raise ValueError("quantization must declare load_in_4bit=true")
        if quantization.get("quant_type") != "nf4":
            raise ValueError("quantization must use nf4")
        if quantization.get("compute_dtype") != "bfloat16":
            raise ValueError("quantized compute dtype must be bfloat16")
        if not isinstance(quantization.get("double_quant"), bool):
            raise ValueError("quantization double_quant must be boolean")
        if int(training.get("world_size") or 0) != 1:
            raise ValueError("4-bit training requires one GPU rank")
        if str(training.get("precision")) != "bf16":
            raise ValueError("4-bit training requires bf16 compute precision")
        if str(collection.get("model_quantization")) != "4bit_nf4":
            raise ValueError(
                "4-bit training requires 4bit_nf4 rollout collection")
    elif str(collection.get("model_quantization") or "bf16") != "bf16":
        raise ValueError("unquantized training requires bf16 rollouts")
    if int(training.get("epochs") or 1) <= 0:
        raise ValueError("training epochs must be positive")
    entropy_coef = training.get("entropy_coef", 0.0)
    if not _finite(entropy_coef) or float(entropy_coef) < 0.0:
        raise ValueError("entropy_coef must be a finite non-negative number")
    ratio_tolerance = training.get("max_initial_ratio_log_abs_diff", 0.0001)
    if not _finite(ratio_tolerance) or float(ratio_tolerance) <= 0.0:
        raise ValueError(
            "max_initial_ratio_log_abs_diff must be finite and positive")
    token_cap = training.get("max_replay_token_forwards")
    if token_cap is not None and (
            not _finite(token_cap) or int(token_cap) <= 0):
        raise ValueError("max_replay_token_forwards must be positive")
    sequence_cap = training.get("max_replay_sequence_forwards")
    if sequence_cap is not None and (
            not _finite(sequence_cap) or int(sequence_cap) <= 0):
        raise ValueError("max_replay_sequence_forwards must be positive")
    review_weight = training.get("review_token_weight")
    if review_weight is not None and not 0.0 < float(review_weight) <= 1.0:
        raise ValueError("review token weight must be in (0, 1]")
    if "include_reviewer_tokens" in training \
            and not isinstance(training["include_reviewer_tokens"], bool):
        raise ValueError("include_reviewer_tokens must be boolean")
    batch_size = collection.get("batch_size")
    expected_batches = collection.get("expected_batches")
    min_active = collection.get("min_task_variable_per_batch")
    max_active = collection.get("max_task_variable_per_batch")
    order_salt = collection.get("batch_order_salt")
    expected_group_count = int(collection.get("expected_groups") or 0)
    max_completion_tokens = collection.get("max_completion_tokens")
    if max_completion_tokens is not None and int(max_completion_tokens) <= 0:
        raise ValueError("max_completion_tokens must be positive")
    min_mixed = int(collection.get("min_mixed_action_groups") or 0)
    if min_mixed < 0 or min_mixed > expected_group_count:
        raise ValueError(
            "min_mixed_action_groups must be within the expected group count")
    if any(value is not None for value in (
            batch_size, expected_batches, min_active, max_active)):
        if not all(value is not None for value in (
                batch_size, expected_batches, min_active, max_active)):
            raise ValueError("batch size, count, and active range must be declared together")
        if int(batch_size) <= 0 or int(expected_batches) <= 0:
            raise ValueError("update batches must be positive")
        if math.ceil(expected_group_count / float(batch_size)) != int(
                expected_batches):
            raise ValueError("fixed batches do not cover every expected group")
        if int(min_active) <= 0 or int(min_active) > int(batch_size):
            raise ValueError("task-variable batch floor is outside the batch")
        if int(max_active) < int(min_active) or int(max_active) > int(batch_size):
            raise ValueError("task-variable batch ceiling is outside the batch")
        if int(max_active) > (
                int(training["world_size"]) * int(training["grad_accum"])):
            raise ValueError(
                "task-variable batch ceiling exceeds effective capacity")
        if int(collection.get("min_task_variable_groups") or 0) < (
                int(expected_batches) * int(min_active)):
            raise ValueError(
                "global task-variable floor must cover every batch floor")
        if int(training.get("update_batches") or 0) != int(expected_batches):
            raise ValueError("training update-batch count differs from collection")
        if not isinstance(order_salt, str) or not order_salt.strip():
            raise ValueError("batched runs require a non-empty batch order salt")
    return config


def resolve_run_path(value):
    """Resolve repository-relative run paths from any current directory."""
    value = str(value or "")
    if not value or os.path.isabs(value):
        return value
    return os.path.normpath(os.path.join(project_root(), value))


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_adapter_config(path, training):
    config_path = os.path.join(path, "adapter_config.json")
    with open(config_path, encoding="utf-8") as handle:
        adapter = json.load(handle)
    expected = {
        "r": int(training["rank"]),
        "lora_alpha": int(training["alpha"]),
        "lora_dropout": float(training["dropout"]),
        "target_modules": list(training["target_modules"]),
    }
    actual = {
        "r": int(adapter.get("r") or 0),
        "lora_alpha": int(adapter.get("lora_alpha") or 0),
        "lora_dropout": float(adapter.get("lora_dropout") or 0.0),
        "target_modules": sorted(adapter.get("target_modules") or ()),
    }
    expected["target_modules"] = sorted(expected["target_modules"])
    if actual != expected:
        raise ValueError("adapter config differs from the training config")
    return actual


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(float(value))


def load_episode_groups(paths, group_size=4, admission_mode="strict",
                        allow_diagnostic_only=False):
    """Read complete, clean episode groups from one or more rollout files."""
    groups = {}
    seen = set()
    for path in paths:
        for episode in _read(path):
            problems = E.validate_episode(episode)
            if problems:
                raise ValueError("episode schema problems: %s" % problems)
            training = episode.get("training_only") or {}
            outcome = episode.get("outcome") or {}
            if ((episode.get("memory_protocol") or {}).get("diagnostic_only")
                    and not allow_diagnostic_only):
                raise ValueError("episode is diagnostic-only: %s"
                                 % episode.get("episode_id"))
            ready_key = ("hybrid_ready" if admission_mode == "hybrid"
                         else "training_ready")
            if not training.get(ready_key):
                raise ValueError("episode is not training-ready: %s"
                                 % episode.get("episode_id"))
            if not outcome.get("infrastructure_valid"):
                raise ValueError("episode has an infrastructure fault: %s"
                                 % episode.get("episode_id"))
            if admission_mode == "hybrid" and (
                    outcome.get("protocol_fallback")
                    or outcome.get("harness_recovery")
                    or outcome.get("terminal") == "protocol_abstain"):
                raise ValueError("hybrid episode has a protocol fallback: %s"
                                 % episode.get("episode_id"))
            if not _finite(training.get("reward")):
                raise ValueError("episode reward is not finite")
            group = str(episode["group_id"])
            rollout = int(episode["rollout_index"])
            key = (group, rollout)
            if key in seen:
                raise ValueError("duplicate group rollout: %s/%d" % key)
            seen.add(key)
            groups.setdefault(group, []).append(episode)

    for group, rows in groups.items():
        rows.sort(key=lambda value: int(value["rollout_index"]))
        indices = [int(value["rollout_index"]) for value in rows]
        if len(rows) != int(group_size) or indices != list(range(int(group_size))):
            raise ValueError("group size is not %d for %s: %s"
                             % (group_size, group, indices))
        versions = {json.dumps(value.get("policy_versions") or {}, sort_keys=True)
                    for value in rows}
        if len(versions) != 1:
            raise ValueError("policy versions differ inside group %s" % group)
        videos = {value.get("video_id") for value in rows}
        if len(videos) != 1:
            raise ValueError("question group spans multiple videos: %s" % group)
    protocols = [value.get("memory_protocol") or {}
                 for rows in groups.values() for value in rows]
    differing = sorted({
        key for protocol in protocols for key in set(protocol) | set(protocols[0])
        if protocol.get(key) != protocols[0].get(key)})
    if differing:
        # One prompt shape per corpus: the same code, flags and card contract
        # must have produced every action the policy is trained on.
        raise ValueError("memory protocol differs inside the corpus: %s"
                         % ", ".join(differing))
    if not groups:
        raise ValueError("no episode groups")
    return {group: groups[group] for group in sorted(groups)}


def _validate_policy_roles(value):
    steps = value.get("steps") or []
    o_steps = [step for step in steps if step.get("actor") == "o"]
    if not o_steps:
        raise ValueError("episode has no O action: %s"
                         % value.get("episode_id"))
    if any(step.get("adapter") != "o" for step in o_steps):
        raise ValueError("O step was not sampled from the O adapter")
    if any(step.get("adapter") != "base" for step in steps
           if step.get("actor") == "reader"):
        raise ValueError("Reader adapter moved in O-only training")
    if any(step.get("adapter") not in ("base", "o") for step in steps
           if step.get("actor") == "reviewer"):
        raise ValueError("REVIEW has an unknown adapter")


def load_training_groups(paths, group_size=4, admission_mode="strict"):
    groups = load_episode_groups(
        paths, group_size=group_size, admission_mode=admission_mode)
    for rows in groups.values():
        for value in rows:
            o_steps = [step for step in value.get("steps") or []
                       if step.get("actor") == "o"]
            if not o_steps:
                raise ValueError("episode has no O action: %s"
                                 % value.get("episode_id"))
            _validate_policy_roles(value)
    return groups


def validate_sequence_layout(
        groups, group_size=4, min_common_questions=MIN_SEQUENCE_QUESTIONS):
    """Check the shared sequence shape, question overlap, and isolation."""
    for group, rows in groups.items():
        if len(rows) != int(group_size):
            raise ValueError("sequence group size differs: %s" % group)
        sequences = [row.get("sequence") or {} for row in rows]
        common = [tuple(int(value) for value in row.get("common_questions") or ())
                  for row in sequences]
        if any(not value for value in common) or len(set(common)) != 1:
            raise ValueError("sequence common questions differ: %s" % group)
        if len(common[0]) < int(min_common_questions):
            raise ValueError(
                "sequence has fewer than %d common questions"
                % int(min_common_questions))
        seeds = [int(row.get("seed") or 0) for row in sequences]
        roots = [str(row.get("memory_root") or "") for row in sequences]
        if len(set(seeds)) != len(rows) or len(set(roots)) != len(rows):
            raise ValueError("sequence seeds or roots are not isolated")
    return True


def validate_sequence_training_groups(
        groups, group_size=4, min_common_questions=MIN_SEQUENCE_QUESTIONS):
    """Refuse invalid layouts or groups without reward variance."""
    validate_sequence_layout(
        groups, group_size=group_size,
        min_common_questions=min_common_questions)
    for group, rows in groups.items():
        returns = [float(row["training_only"]["optimization_reward"])
                   for row in rows]
        mean = sum(returns) / float(len(returns))
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        if variance <= 1e-12:
            raise ValueError("sequence reward variance is zero: %s" % group)
    return True


def attach_group_advantages(groups, advantage_mode="task_variation",
                            shuffle_seed=None):
    """Attach full, task, and non-task advantages to every rollout."""
    if advantage_mode not in ("task_variation",) + TASK_COST_MODES:
        raise ValueError("unsupported advantage mode")
    summary = {"groups": {}, "active_groups": 0, "task_variable_groups": 0,
               "cost_only_groups": 0, "reward_shaped_inactive_groups": 0}
    for group, rows in groups.items():
        rewards = [float(row["training_only"].get(
            "optimization_reward", row["training_only"]["reward"]))
            for row in rows]
        components = [row["training_only"].get(
            "optimization_reward_components",
            row["training_only"]["reward_components"]) for row in rows]
        task = [float(value.get("task") or 0.0) for value in components]
        non_task = [float(value.get("cost") or 0.0)
                    + float(value.get("protocol") or 0.0)
                    + float(value.get("evidence_gain") or 0.0)
                    + float(value.get("grounding") or 0.0)
                    + float(value.get("memory_reuse") or 0.0)
                    + float(value.get("progress") or 0.0)
                    + float(value.get("global_memory") or 0.0)
                    for value in components]
        correct = [bool(row["outcome"]["correct"]) for row in rows]
        mean = sum(rewards) / len(rewards)
        std = math.sqrt(sum((value - mean) ** 2 for value in rewards)
                        / len(rewards))
        task_mean = sum(task) / len(task)
        task_std = math.sqrt(sum((value - task_mean) ** 2 for value in task)
                             / len(task))
        non_task_mean = sum(non_task) / len(non_task)
        non_task_std = math.sqrt(sum(
            (value - non_task_mean) ** 2 for value in non_task)
            / len(non_task))
        task_variable = len({round(value, 12) for value in task}) > 1
        active = (task_variable if advantage_mode == "task_variation"
                  else std > 1e-9 or task_std > 1e-9)
        task_advantages = [
            ((value - task_mean) / task_std if task_std > 1e-9 else 0.0)
            for value in task]
        cost_advantages = [
            ((value - non_task_mean) / non_task_std
             if non_task_std > 1e-9 else 0.0)
            for value in non_task]
        advantages = (
            [((value - mean) / std if std > 1e-9 else 0.0)
             for value in rewards]
            if std > 1e-9 else list(task_advantages))
        advantages = [clip_advantage(value) for value in advantages]
        if shuffle_seed is not None:
            task_advantages = permute_within_group(
                task_advantages, shuffle_seed, group)
            cost_advantages = permute_within_group(
                cost_advantages, shuffle_seed, group)
            advantages = permute_within_group(advantages, shuffle_seed, group)
        for row, advantage, task_advantage, cost_advantage in zip(
                rows, advantages, task_advantages, cost_advantages):
            row["training_only"]["advantage"] = round(advantage, 12)
            row["training_only"]["task_advantage"] = round(task_advantage, 12)
            row["training_only"]["cost_advantage"] = round(cost_advantage, 12)
        summary["groups"][group] = {
            "rewards": rewards,
            "task": task,
            "non_task": non_task,
            "correct": correct,
            "reward_mean": mean,
            "reward_std": std,
            "advantage_clip": ADVANTAGE_CLIP,
            "advantages": advantages,
            "task_advantages": task_advantages,
            "cost_advantages": cost_advantages,
            "active": active,
            "task_variable": task_variable,
        }
        summary["active_groups"] += int(active)
        summary["task_variable_groups"] += int(task_variable)
        summary["cost_only_groups"] += int(
            task_std <= 1e-9 and non_task_std > 1e-9)
        # Kept for older report compatibility; task-varying groups are active
        # in both advantage modes and should make this count zero.
        summary["reward_shaped_inactive_groups"] += int(
            not active and task_std > 1e-9)
    return summary


def _policy_step(step, episode_id):
    prompt = step.get("prompt_token_ids")
    tokens = step.get("completion_token_ids")
    old = step.get("completion_logprobs")
    raw_old = step.get("completion_raw_logprobs")
    if not E._sampling_ok(step):
        raise ValueError("O step has invalid sampling data: %s" % episode_id)
    if step.get("logprob_kind") != "sampling_post_warper":
        raise ValueError("O step does not carry sampling logprobs")
    if step.get("raw_logprob_kind") not in (
            "raw_pre_warper", "teacher_forced_raw_pre_warper"):
        raise ValueError("O step does not carry raw pre-warper logprobs")
    if not isinstance(old, list) or not isinstance(raw_old, list):
        raise ValueError("O step is missing logprob arrays")
    if not all(_finite(value) for value in old + raw_old):
        raise ValueError("O step has a non-finite old logprob: %s" % episode_id)
    temperature = float(step.get("temperature") or 0.0)
    top_p = float(step.get("top_p") or 0.0)
    if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("O step has invalid sampling parameters")
    if int(step.get("top_k") or 0) != 0:
        raise ValueError("O step must disable top-k sampling")
    if step.get("use_cache") is not False:
        raise ValueError("O step was generated with KV cache")
    if step.get("generation_constraint") != "action_menu_prefix_v1":
        raise ValueError("O step was not generated under the action constraint")
    if any(name in step for name in (
            "images", "pixel_values", "image_grid_thw")):
        raise ValueError("O policy replay must remain text-only")
    allowed_token_ids = step.get("allowed_token_ids")
    if (not isinstance(allowed_token_ids, list)
            or len(allowed_token_ids) != len(tokens)
            or any(not isinstance(row, list) or not row
                   or any(not isinstance(value, int)
                          or isinstance(value, bool) for value in row)
                   for row in allowed_token_ids)):
        raise ValueError("O step has invalid constrained token menu")
    loss_token_mask = step.get("memory_decision_token_mask")
    if loss_token_mask is None:
        loss_token_mask = [True] * len(tokens)
    if (not isinstance(loss_token_mask, list)
            or len(loss_token_mask) != len(tokens)
            or not any(loss_token_mask)
            or any(not isinstance(value, bool) for value in loss_token_mask)):
        raise ValueError("O step has invalid loss-token mask")
    return {
        "episode_id": episode_id,
        "prompt_token_ids": list(prompt),
        "completion_token_ids": list(tokens),
        "old_logprobs": [float(value) for value in old],
        "old_raw_logprobs": [float(value) for value in raw_old],
        "allowed_token_ids": [list(row) for row in allowed_token_ids],
        "loss_token_mask": list(loss_token_mask),
        "temperature": temperature,
        "top_p": top_p,
    }


def build_policy_samples(groups, group_size=4, world_size=1,
                         include_reviewer=True,
                         review_token_weight=REVIEW_TOKEN_WEIGHT,
                         active_groups=None, decision_token_only=False):
    """Flatten active groups into weighted O action-token samples."""
    if active_groups is None:
        active = []
        for group, rows in groups.items():
            if len({bool(row["outcome"]["correct"]) for row in rows}) > 1:
                active.append(group)
    else:
        active = [group for group in groups if group in set(active_groups)]
    if not active:
        return []
    samples = []
    for group in active:
        for episode in groups[group]:
            if decision_token_only:
                decision = MD.decision_step(episode)
                trainable_steps = [decision["step"]]
                o_steps = trainable_steps
                review_steps = []
            else:
                o_steps = [step for step in episode["steps"]
                           if step.get("actor") == "o"
                           and not step.get("scripted_turn")
                           and not step.get("repair")
                           and not step.get("normalized")
                           and not step.get("recovered_action")
                           and not step.get("recovered_turn_cut")
                           and step.get("valid")]
                review_steps = ([step for step in episode["steps"]
                                 if step.get("actor") == "reviewer"
                                 and step.get("adapter") == "o"
                                 and step.get("valid")]
                                if include_reviewer else [])
                trainable_steps = o_steps + review_steps
            for step in trainable_steps:
                count = len(o_steps) if step.get("actor") == "o" else len(review_steps)
                factor = (1.0 if step.get("actor") == "o"
                          else float(review_token_weight))
                weight = (factor * float(world_size)
                          / (len(active) * int(group_size) * count))
                sample = _policy_step(step, episode["episode_id"])
                override = step.get("training_advantage")
                if override is None:
                    task_advantage = float(
                        episode["training_only"].get("task_advantage") or 0.0)
                    cost_advantage = float(
                        episode["training_only"].get("cost_advantage") or 0.0)
                    reward_advantage = float(
                        episode["training_only"].get("advantage") or 0.0)
                else:
                    reward_advantage = float(override)
                    task_advantage = reward_advantage
                    cost_advantage = 0.0
                if not decision_token_only:
                    sample["loss_token_mask"] = [
                        True] * len(sample["completion_token_ids"])
                sample.update({
                    "actor": step.get("actor"),
                    "group_id": group,
                    "rollout_index": int(episode["rollout_index"]),
                    "advantage": clip_advantage(reward_advantage),
                    "credit_override": override is not None,
                    "source_advantage": (
                        float(override) if override is not None else None),
                    "task_advantage": task_advantage,
                    "cost_advantage": cost_advantage,
                    "reward_advantage": reward_advantage,
                    "weight": weight,
                })
                samples.append(sample)
    return samples


def select_batch_samples(samples, active_groups, group_size=4, world_size=1,
                         review_token_weight=REVIEW_TOKEN_WEIGHT):
    """Keep one fixed batch's active groups and normalize their weights."""
    active = list(active_groups)
    if not active:
        raise ValueError("update batch has no task-variable groups")
    selected = [dict(sample) for sample in samples
                if sample["group_id"] in set(active)]
    if not selected:
        raise ValueError("update batch has no policy samples")
    rollout_steps = {}
    rollouts = {}
    for sample in selected:
        key = (sample["group_id"], int(sample["rollout_index"]))
        actor = sample["actor"]
        rollout_steps.setdefault(key, {"o": 0, "reviewer": 0})[actor] += 1
        rollouts.setdefault(sample["group_id"], set()).add(key[1])
    for group in active:
        if rollouts.get(group) != set(range(int(group_size))):
            raise ValueError("update batch is missing a rollout for %s" % group)
    for sample in selected:
        key = (sample["group_id"], int(sample["rollout_index"]))
        counts = rollout_steps[key]
        actor = sample["actor"]
        factor = 1.0 if actor == "o" else float(review_token_weight)
        sample["weight"] = (factor * float(world_size)
                            / (len(active) * int(group_size) * counts[actor]))
    return selected


def validate_sample_lengths(samples, max_len, max_completion_tokens=None):
    """Refuse a sampled turn that exceeds the declared sequence budget."""
    longest = 0
    for sample in samples:
        length = len(sample["prompt_token_ids"]) + len(
            sample["completion_token_ids"])
        longest = max(longest, length)
        if length > int(max_len):
            raise ValueError(
                "sequence length %d exceeds max_len %d for %s"
                % (length, int(max_len), sample.get("episode_id", "")))
        completion = len(sample["completion_token_ids"])
        if max_completion_tokens is not None and completion > int(
                max_completion_tokens):
            raise ValueError(
                "completion length %d exceeds max_completion_tokens %d for %s"
                % (completion, int(max_completion_tokens),
                   sample.get("episode_id", "")))
    return longest


def validate_model_quantization(groups, expected):
    if not expected:
        return
    found = {
        str((row.get("policy_versions") or {}).get("quantization") or "bf16")
        for rows in groups.values() for row in rows}
    if found != {str(expected)}:
        raise ValueError(
            "rollout quantization %s differs from training %s"
            % (sorted(found), expected))


def declared_group_order(group_advantages, order_salt=None):
    """Order groups by a predeclared hash, never by outcome."""
    groups = list(group_advantages)
    if not order_salt:
        return groups
    return sorted(groups, key=lambda group: hashlib.sha256(
        ("%s:%s" % (order_salt, group)).encode("utf-8")).hexdigest())


def plan_update_batches(group_advantages, batch_size,
                        expected_batches=None, min_task_variable_per_batch=None,
                        max_task_variable_per_batch=None, order_salt=None):
    """Split fixed group order into audited update batches."""
    ordered = declared_group_order(group_advantages, order_salt=order_salt)
    batches = []
    for index in range(0, len(ordered), int(batch_size)):
        groups = ordered[index:index + int(batch_size)]
        active = [group for group in groups
                  if group_advantages[group]["active"]]
        task_variable = [
            group for group in groups
            if group_advantages[group].get(
                "task_variable", group_advantages[group]["active"])]
        batches.append({
            "index": len(batches) + 1,
            "groups": groups,
            "active_groups": active,
            "task_variable_groups": task_variable,
            "order_salt": order_salt,
        })
    if expected_batches is not None and len(batches) != int(expected_batches):
        raise ValueError("expected %d update batches, found %d"
                         % (int(expected_batches), len(batches)))
    if min_task_variable_per_batch is not None:
        for batch in batches:
            if len(batch["task_variable_groups"]) < int(
                    min_task_variable_per_batch):
                raise ValueError(
                    "batch %d has %d task-variable groups, need %d"
                    % (batch["index"], len(batch["task_variable_groups"]),
                       int(min_task_variable_per_batch)))
    if max_task_variable_per_batch is not None:
        for batch in batches:
            if len(batch["task_variable_groups"]) > int(
                    max_task_variable_per_batch):
                raise ValueError(
                    "batch %d has %d task-variable groups, maximum %d"
                    % (batch["index"], len(batch["task_variable_groups"]),
                       int(max_task_variable_per_batch)))
    return batches


def _gold_letter(value):
    match = re.search(r"\b([A-H])\b", str(value or "").upper())
    return match.group(1) if match else ""


def _gold_letters(value):
    if isinstance(value, (list, tuple)):
        return [letter for item in value for letter in _gold_letters(item)]
    letter = _gold_letter(value)
    return [letter] if letter else []


def _normalized_abs_mass(values):
    mean = sum(values) / float(len(values) or 1)
    std = math.sqrt(sum((value - mean) ** 2 for value in values)
                    / float(len(values) or 1))
    if std <= 1e-9:
        return 0.0
    return sum(abs((value - mean) / std) for value in values)


def _blind_chance(groups):
    """Mean chance level over unique questions with declared option counts."""
    counts = []
    for rows in groups.values():
        first = rows[0] if rows else {}
        questions = (first.get("sequence") or {}).get("question_episodes") or []
        if questions:
            values = [question.get("num_options")
                      or (question.get("outcome") or {}).get("num_options")
                      for question in questions]
        else:
            values = [(first.get("training_only") or {}).get("num_options")]
        counts.extend(int(value) for value in values
                      if value is not None and int(value) > 1)
    if not counts:
        return None
    return round(sum(1.0 / value for value in counts) / len(counts), 6)


def _terminal_task_values(rows):
    """Return terminal task rewards separately for each group."""
    values = {}
    for row in rows:
        questions = (row.get("sequence") or {}).get("question_episodes")
        if questions:
            for question in questions:
                terminal = str((question.get("outcome") or {}).get(
                    "terminal") or "")
                if terminal in ("agent_submit", "agent_abstain"):
                    values.setdefault(terminal, []).append(float(
                        (question.get("reward_components") or {}).get(
                            "task") or 0.0))
        else:
            terminal = str((row.get("outcome") or {}).get("terminal") or "")
            if terminal in ("agent_submit", "agent_abstain"):
                values.setdefault(terminal, []).append(float(
                    (row.get("training_only") or {}).get(
                        "reward_components", {}).get("task") or 0.0))
    return values


def _reward_audit(groups, report, allow_guess_incentive=False,
                  guess_incentive_scope="batch"):
    """Describe task signal, cost signal, and blind-submit baselines."""
    if guess_incentive_scope not in ("batch", "per_group"):
        raise ValueError("unsupported guess_incentive_scope")
    gold = [letter for rows in groups.values()
            for letter in _gold_letters(rows[0]["training_only"].get("gold"))]
    group_count = len(gold)
    blind = {
        letter: (sum(value == letter for value in gold) / float(group_count))
        for letter in "ABCDEFGH"
    }
    episodes = sum(len(rows) for rows in groups.values())
    sequence_rows = [row for rows in groups.values() for row in rows
                     if row.get("sequence")]
    if sequence_rows:
        total_questions = sum(len(
            row.get("sequence", {}).get("question_episodes") or [])
            for row in sequence_rows)
        correct = sum(sum(
            bool((question.get("outcome") or {}).get("correct"))
            for question in row.get("sequence", {}).get(
                "question_episodes") or [])
            for row in sequence_rows)
    else:
        total_questions = episodes
        correct = sum(bool(row["outcome"]["correct"])
                      for rows in groups.values() for row in rows)
    task_all = [_normalized_abs_mass(values["task"])
                for values in report["group_advantages"].values()]
    cost_all = [_normalized_abs_mass(values["non_task"])
                for values in report["group_advantages"].values()]
    active = [values["active"]
              for values in report["group_advantages"].values()]
    task_active = sum(value for value, flag in zip(task_all, active) if flag)
    cost_active = sum(value for value, flag in zip(cost_all, active) if flag)
    task_inactive = sum(value for value, flag in zip(task_all, active) if not flag)
    cost_inactive = sum(value for value, flag in zip(cost_all, active) if not flag)

    group_terminals = {
        group: _terminal_task_values(rows) for group, rows in groups.items()
    }
    wrong_submit_values = []
    abstain_values = []
    guess_incentive_groups = []
    for group, terminal_values in group_terminals.items():
        wrong = [value for value in terminal_values.get(
            "agent_submit") or () if value <= 0.0]
        abstain = list(terminal_values.get("agent_abstain") or ())
        wrong_submit_values.extend(wrong)
        abstain_values.extend(abstain)
        if wrong and abstain and min(wrong) > min(abstain):
            guess_incentive_groups.append(group)
    legacy_wrong_submit = min(wrong_submit_values or [0.0])
    legacy_abstain = min(abstain_values or [0.0])
    batch_guess_incentive = legacy_wrong_submit > legacy_abstain
    guess_incentive = (
        bool(guess_incentive_groups)
        if guess_incentive_scope == "per_group" else batch_guess_incentive)

    correct_anchor_groups = [
        group for group, rows in groups.items()
        if any(bool(row["outcome"]["correct"]) for row in rows)]
    task_reward_variance_groups = [
        group for group, values in report["group_advantages"].items()
        if len(set(values["task"])) > 1]
    task_signal_groups = [
        group for group in task_reward_variance_groups
        if group in set(correct_anchor_groups)]
    cost_protocol_only_groups = [
        group for group, values in report["group_advantages"].items()
        if group not in set(task_signal_groups)
        and len(set(values["task"])) <= 1
        and len(set(values["non_task"])) > 1]
    all_wrong_groups = [
        group for group, rows in groups.items()
        if group not in set(correct_anchor_groups)]
    return {
        "groups": group_count,
        "episodes": episodes,
        "task_variable_groups": [
            group for group, values in report["group_advantages"].items()
            if values["task_variable"]],
        "task_variable_group_count": sum(
            values["task_variable"]
            for values in report["group_advantages"].values()),
        "correct_anchor_groups": correct_anchor_groups,
        "correct_anchor_group_count": len(correct_anchor_groups),
        "task_reward_variance_groups": task_reward_variance_groups,
        "task_reward_variance_group_count": len(task_reward_variance_groups),
        "task_signal_groups": task_signal_groups,
        "task_signal_group_count": len(task_signal_groups),
        "cost_protocol_only_groups": cost_protocol_only_groups,
        "cost_protocol_only_group_count": len(cost_protocol_only_groups),
        "cost_only_groups": cost_protocol_only_groups,
        "all_wrong_groups": all_wrong_groups,
        "all_wrong_group_count": len(all_wrong_groups),
        "reward_shaped_inactive_groups": [
            group for group, values in report["group_advantages"].items()
            if not values["active"] and len(set(values["task"])) > 1],
        "episode_correct_count": correct,
        "episode_correct_rate": correct / float(total_questions or 1),
        "blind_option_accuracy": blind,
        "blind_best_accuracy": max(blind.values() or [0.0]),
        "blind_chance": _blind_chance(groups),
        "task_advantage_mass_all": round(sum(task_all), 6),
        "cost_advantage_mass_all": round(sum(cost_all), 6),
        "task_advantage_mass_active": round(task_active, 6),
        "cost_advantage_mass_active": round(cost_active, 6),
        "task_advantage_mass_inactive": round(task_inactive, 6),
        "cost_advantage_mass_inactive": round(cost_inactive, 6),
        "wrong_submit_task_reward": (
            min(wrong_submit_values) if wrong_submit_values else None),
        "abstain_task_reward": min(abstain_values) if abstain_values else None,
        "wrong_submit_terminal_count": len(wrong_submit_values),
        "abstain_terminal_count": len(abstain_values),
        "guess_incentive_scope": guess_incentive_scope,
        "guess_incentive_groups": guess_incentive_groups,
        "guess_incentive": guess_incentive,
        "allow_guess_incentive": bool(allow_guess_incentive),
    }

def prepare_training(paths, group_size=4, expected_groups=None,
                     min_task_variable_groups=2, expected_temperature=None,
                     expected_top_p=None, expected_top_k=None, world_size=1,
                     allow_guess_incentive=False,
                     min_mixed_action_groups=0,
                     expected_model_quantization=None, batch_size=None,
                     expected_batches=None,
                     min_task_variable_per_batch=None,
                     max_task_variable_per_batch=None,
                     batch_order_salt=None, include_reviewer=True,
                     review_token_weight=REVIEW_TOKEN_WEIGHT,
                     advantage_mode="task_variation",
                     admission_mode="strict", sequence_level=False,
                     guess_incentive_scope="batch", decision_token_only=False,
                     min_common_questions=MIN_SEQUENCE_QUESTIONS,
                     shuffle_rewards_seed=None):
    if batch_size is not None and (
            not isinstance(batch_order_salt, str)
            or not batch_order_salt.strip()):
        raise ValueError("batched runs require a non-empty batch order salt")
    if batch_size is not None and (
            min_task_variable_per_batch is None
            or max_task_variable_per_batch is None):
        raise ValueError(
            "batched runs require an active-group floor and ceiling")
    groups = load_training_groups(
        paths, group_size=group_size, admission_mode=admission_mode)
    if sequence_level:
        validate_sequence_training_groups(
            groups, group_size=group_size,
            min_common_questions=min_common_questions)
    validate_model_quantization(groups, expected_model_quantization)
    if expected_groups is not None and len(groups) != int(expected_groups):
        raise ValueError("expected %d groups, found %d"
                         % (int(expected_groups), len(groups)))
    for rows in groups.values():
        for episode in rows:
            sampling = episode.get("sampling") or {}
            if expected_temperature is not None \
                    and abs(float(sampling.get("temperature") or 0.0)
                            - float(expected_temperature)) > 1e-9:
                raise ValueError("sampling temperature differs from the config")
            if expected_top_p is not None \
                    and sampling.get("top_p") is None:
                raise ValueError("sampling top_p is missing")
            if expected_top_p is not None \
                    and abs(float(sampling["top_p"]) - float(expected_top_p)) > 1e-9:
                raise ValueError("sampling top_p differs from the config")
            if expected_top_k is not None \
                    and int(sampling.get("top_k") or 0) != int(expected_top_k):
                raise ValueError("sampling top_k differs from the config")
    advantage_summary = attach_group_advantages(
        groups, advantage_mode=advantage_mode, shuffle_seed=shuffle_rewards_seed)
    active_groups = [group for group, values in
                     advantage_summary["groups"].items() if values["active"]]
    task_variable_groups = [
        group for group, values in advantage_summary["groups"].items()
        if values["task_variable"]]
    samples = build_policy_samples(
        groups, group_size=group_size, world_size=world_size,
        include_reviewer=include_reviewer,
        review_token_weight=review_token_weight,
        active_groups=task_variable_groups,
        decision_token_only=decision_token_only)
    report = {
        "groups": len(groups),
        "episodes": sum(len(rows) for rows in groups.values()),
        "active_groups": advantage_summary["active_groups"],
        "task_variable_groups": advantage_summary["task_variable_groups"],
        "mixed_action_groups": (
            sum(ACTION.group_has_mixed_actions(rows)
                for rows in groups.values())
            if int(min_mixed_action_groups) else None),
        "task_variable_group_ids": task_variable_groups,
        "constant_groups": len(groups) - advantage_summary["active_groups"],
        "cost_only_groups": advantage_summary["cost_only_groups"],
        "reward_shaped_inactive_groups": advantage_summary[
            "reward_shaped_inactive_groups"],
        "policy_samples": len(samples),
        "advantage_mode": advantage_mode,
        "admission_mode": admission_mode,
        "decision_token_only": bool(decision_token_only),
        "sequence_level": bool(sequence_level),
        "group_advantages": advantage_summary["groups"],
        "shuffle_rewards_seed": shuffle_rewards_seed,
    }
    report["reward_audit"] = _reward_audit(
        groups, report, allow_guess_incentive=allow_guess_incentive,
        guess_incentive_scope=guess_incentive_scope)
    report["reward_audit"]["minimum_task_variable_groups"] = int(
        min_task_variable_groups)
    report["reward_audit"]["minimum_mixed_action_groups"] = int(
        min_mixed_action_groups)
    report["reward_audit"]["passes"] = bool(
        report["task_variable_groups"] >= int(min_task_variable_groups)
        and (not int(min_mixed_action_groups)
             or report["mixed_action_groups"] >= int(
                 min_mixed_action_groups))
        and (allow_guess_incentive
             or not report["reward_audit"]["guess_incentive"]))
    report["training_batches"] = []
    if batch_size is not None:
        report["training_batches"] = plan_update_batches(
            report["group_advantages"], batch_size=batch_size,
            expected_batches=expected_batches,
            min_task_variable_per_batch=min_task_variable_per_batch,
            max_task_variable_per_batch=max_task_variable_per_batch,
            order_salt=batch_order_salt)
        report["reward_audit"]["batches"] = [{
            "groups": batch["groups"],
            "active_group_ids": batch["active_groups"],
            "active_groups": len(batch["active_groups"]),
            "task_variable_group_ids": batch["task_variable_groups"],
            "task_variable_groups": len(batch["task_variable_groups"]),
        } for batch in report["training_batches"]]
        report["reward_audit"]["batches_pass"] = True
        report["reward_audit"]["minimum_task_variable_per_batch"] = (
            int(min_task_variable_per_batch)
            if min_task_variable_per_batch is not None else None)
    return report, samples


def replay_gate_passes(raw_diff, behavior_diff, limit):
    """Both served/recomputed differences must be strictly below the limit."""
    return float(raw_diff) < float(limit) and float(behavior_diff) < float(limit)


def initial_ratio_gate_passes(log_abs_diff, tolerance):
    """The policy and served probabilities must agree at the first step."""
    return abs(float(log_abs_diff)) <= float(tolerance)


def _parameter_movement_valid(movement):
    return bool(
        math.isfinite(float(movement.get("max_abs", float("nan"))))
        and math.isfinite(float(movement.get("relative_l2", float("nan"))))
        and float(movement["max_abs"]) > 0.0
        and float(movement["relative_l2"]) > 0.0)


def _parameter_delta(before, parameters):
    max_abs = 0.0
    squared = 0.0
    norm = 0.0
    for name, parameter in parameters:
        prior = before[name]
        current = parameter.detach().float()
        difference = current - prior.float()
        max_abs = max(max_abs, float(difference.abs().max().item()))
        squared += float(difference.pow(2).sum().item())
        norm += float(prior.float().pow(2).sum().item())
    return {
        "max_abs": max_abs,
        "relative_l2": math.sqrt(squared) / max(math.sqrt(norm), 1e-12),
    }


def projected_sequence_forwards(samples, sweeps=1):
    """Count teacher-forced sequence forwards across replay sweeps."""
    if int(sweeps) <= 0:
        raise ValueError("sequence-forward sweep count must be positive")
    return int(sweeps) * len(samples)


def projected_token_forwards(samples, sweeps=1):
    """Legacy completion-token count retained in run reports."""
    if int(sweeps) <= 0:
        raise ValueError("token-forward sweep count must be positive")
    return int(sweeps) * sum(
        len(sample.get("completion_token_ids") or ())
        for sample in samples)


def planned_token_forward_sweeps(epochs, reference_reused):
    """Behavior plus optional reference plus one policy sweep per epoch."""
    return 2 + int(epochs) - int(bool(reference_reused))


def balanced_group_partition(active_groups, samples, world_size):
    """Assign groups by token cost so DDP ranks finish close together."""
    world_size = int(world_size)
    if world_size <= 0:
        raise ValueError("world size must be positive")
    active = [str(group) for group in active_groups]
    if len(set(active)) != len(active):
        raise ValueError("active groups contain a duplicate")
    costs = {group: 0 for group in active}
    for sample in samples:
        group = str(sample.get("group_id") or "")
        if group in costs:
            costs[group] += len(sample.get("completion_token_ids") or ())
    loads = [0] * world_size
    partitions = [[] for _ in range(world_size)]
    for group in sorted(active, key=lambda value: (-costs[value], value)):
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        partitions[rank].append(group)
        loads[rank] += costs[group]
    return partitions


def ppo_token_loss_torch(policy_logprobs, old_logprobs, reference_logprobs,
                         advantage, weight, clip=0.2, kl_beta=0.001,
                         policy_raw_logprobs=None, entropy=None,
                         entropy_coef=0.0):
    """One O-turn PPO-clipped loss with a frozen-adapter KL penalty."""
    import torch
    old = torch.as_tensor(old_logprobs, dtype=torch.float32,
                          device=policy_logprobs.device)
    ref = torch.as_tensor(reference_logprobs, dtype=torch.float32,
                          device=policy_logprobs.device)
    ratio = torch.exp((policy_logprobs.float() - old).clamp(-20.0, 20.0))
    clipped = torch.clamp(ratio, 1.0 - float(clip), 1.0 + float(clip))
    surrogate = torch.minimum(ratio * float(advantage),
                              clipped * float(advantage))
    policy_for_kl = (policy_logprobs if policy_raw_logprobs is None
                     else policy_raw_logprobs)
    log_ratio = ref - policy_for_kl.float()
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    loss = (-surrogate + float(kl_beta) * kl).mean()
    if entropy is not None and float(entropy_coef):
        loss = loss - float(entropy_coef) * entropy.float().mean()
    return loss * float(weight)


def ppo_token_stats_torch(policy_logprobs, old_logprobs, reference_logprobs,
                          clip=0.2, policy_raw_logprobs=None, entropy=None):
    """Detached per-token numbers for the round log. Never used in the loss."""
    import torch
    with torch.no_grad():
        old = torch.as_tensor(old_logprobs, dtype=torch.float32,
                              device=policy_logprobs.device)
        ref = torch.as_tensor(reference_logprobs, dtype=torch.float32,
                              device=policy_logprobs.device)
        ratio = torch.exp((policy_logprobs.float() - old).clamp(-20.0, 20.0))
        base = (policy_logprobs if policy_raw_logprobs is None
                else policy_raw_logprobs)
        log_ratio = ref - base.float()
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        return {
            "tokens": int(ratio.numel()),
            "kl_sum": float(kl.sum().item()),
            "ratio_dev_sum": float((ratio - 1.0).abs().sum().item()),
            "clipped": int(((ratio < 1.0 - float(clip))
                            | (ratio > 1.0 + float(clip))).sum().item()),
            "entropy_sum": (float(entropy.float().sum().item())
                            if entropy is not None else 0.0),
        }


def merge_token_stats(total, stats):
    for key, value in stats.items():
        total[key] = total.get(key, 0) + value
    return total


def summarize_token_stats(total):
    tokens = int(total.get("tokens") or 0)
    if not tokens:
        return {"tokens": 0, "kl_mean": None, "ratio_abs_dev_mean": None,
                "clip_fraction": None, "entropy_mean": None}
    return {"tokens": tokens,
            "kl_mean": total["kl_sum"] / tokens,
            "ratio_abs_dev_mean": total["ratio_dev_sum"] / tokens,
            "clip_fraction": total["clipped"] / tokens,
            "entropy_mean": total["entropy_sum"] / tokens}


def _completion_logits(model, prompt_ids, completion_ids, device):
    import torch
    ids = list(prompt_ids) + list(completion_ids)
    prompt_len = len(prompt_ids)
    keep = len(completion_ids) + 1
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    output = model(
        input_ids=input_ids, attention_mask=attention,
        logits_to_keep=keep, use_cache=False)
    logits = output.logits[:, :-1, :].float()
    if int(logits.shape[1]) != len(completion_ids):
        raise RuntimeError("teacher-forced logits differ from completion length")
    return logits[0]


def teacher_forced_completion_logprobs(model, sample, device):
    """Return raw, sampling and entropy tensors from one full-sequence pass."""
    logits = _completion_logits(
        model, sample["prompt_token_ids"],
        sample["completion_token_ids"], device)
    allowed = sample.get("allowed_token_ids")
    raw = _selected_logprobs(
        logits, sample["completion_token_ids"], device)
    sampling = _sampling_logprobs(
        logits, sample["completion_token_ids"],
        sample["temperature"], sample["top_p"], device,
        allowed_token_ids=allowed)
    entropy = _sampling_entropy(
        logits, sample["temperature"], sample["top_p"],
        allowed_token_ids=allowed)
    return raw, sampling, entropy


def _constrain_logits(logits, allowed_token_ids):
    """Mask logits to the exact legal token menu for each position."""
    import torch
    if allowed_token_ids is None:
        return logits
    if len(allowed_token_ids) != int(logits.shape[0]):
        raise ValueError("constrained token menu length differs from logits")
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    for index, token_ids in enumerate(allowed_token_ids):
        if not token_ids:
            raise ValueError("constrained token menu is empty")
        allowed[index, torch.tensor(
            token_ids, dtype=torch.long, device=logits.device)] = True
    return logits.masked_fill(~allowed, -float("inf"))


def _selected_logprobs(logits, completion_ids, device,
                       allowed_token_ids=None):
    import torch
    logits = _constrain_logits(logits, allowed_token_ids)
    chosen = torch.tensor(completion_ids, dtype=torch.long, device=device)
    values = torch.log_softmax(logits, dim=-1).gather(
        -1, chosen.unsqueeze(-1)).squeeze(-1)
    if int(values.numel()) != len(completion_ids):
        raise RuntimeError("teacher-forced logprob length mismatch")
    return values


def _sampling_logprobs(logits, completion_ids, temperature, top_p, device,
                       allowed_token_ids=None):
    """Selected-token logprobs after the declared temperature and nucleus cut."""
    import torch
    scaled = _constrain_logits(logits.float(), allowed_token_ids) \
        / float(temperature)
    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        scaled = torch.full_like(scaled, -float("inf")).scatter(
            1, sorted_indices, sorted_logits)
    return _selected_logprobs(scaled, completion_ids, device)


def _sampling_entropy(logits, temperature, top_p, allowed_token_ids=None):
    """Mean token entropy after the declared temperature and nucleus cut."""
    import torch
    scaled = _constrain_logits(logits.float(), allowed_token_ids) \
        / float(temperature)
    if float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        scaled = torch.full_like(scaled, -float("inf")).scatter(
            1, sorted_indices, sorted_logits)
    log_probs = torch.log_softmax(scaled, dim=-1)
    safe_log_probs = torch.where(
        torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
    terms = torch.exp(safe_log_probs) * safe_log_probs
    entropy = -terms.sum(dim=-1)
    return entropy


def _null_context():
    class Null:
        def __enter__(self):
            return None

        def __exit__(self, *_info):
            return False
    return Null()


def _pad_samples(samples, count):
    """Repeat the last sample with zero weight to align DDP backward calls."""
    if not samples:
        raise ValueError("cannot pad an empty sample list")
    output = list(samples)
    while len(output) < int(count):
        pad = dict(output[-1])
        pad["weight"] = 0.0
        pad["advantage"] = 0.0
        pad["padded"] = True
        output.append(pad)
    return output


def train(args, report, samples):
    started_unix = time.time()
    import torch
    import torch.distributed as dist
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    no_save = bool(getattr(args, "no_save", False))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(
            "nccl", device_id=local_rank,
            timeout=datetime.timedelta(hours=2))
    adapter_path = os.path.join(args.init_adapter, "adapter_model.safetensors")
    adapter_actual = _sha256(adapter_path)
    if adapter_actual != args.init_adapter_sha256:
        raise RuntimeError("initial adapter hash differs from the config")
    reference_path = os.path.join(
        args.reference_adapter, "adapter_model.safetensors")
    reference_actual = _sha256(reference_path)
    if reference_actual != args.reference_adapter_sha256:
        raise RuntimeError("reference adapter hash differs from the config")
    adapter_settings = validate_adapter_config(args.init_adapter, {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
    })
    validate_adapter_config(args.reference_adapter, {
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
    })
    if rank == 0:
        model_actual = RS.model_content_sha256(args.model)
        if model_actual != args.model_sha256:
            raise RuntimeError("base model content hash differs from the config")
    torch.manual_seed(int(args.seed) + rank)
    random.seed(int(args.seed) + rank)

    quantization = getattr(args, "quantization", None)
    if str(getattr(args, "attention", "sdpa")) == "sdpa":
        LC.configure_sdpa(torch)
    model_kwargs = {
        "dtype": torch.bfloat16,
        "attn_implementation": str(getattr(args, "attention", "sdpa")),
    }
    if quantization:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=str(quantization["quant_type"]),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=bool(quantization["double_quant"]),
        )
        model_kwargs["device_map"] = {"": local_rank}
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, **model_kwargs)
    if not quantization:
        model.to(local_rank)
    model.config.use_cache = False
    if quantization:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    policy = PeftModel.from_pretrained(
        model, args.init_adapter, adapter_name="default", is_trainable=True)
    policy.load_adapter(
        args.init_adapter, adapter_name="behavior", is_trainable=False)
    policy.load_adapter(
        args.reference_adapter, adapter_name="ref", is_trainable=False)
    policy.set_adapter("default")
    trainable_parameters = [
        (name, parameter) for name, parameter in policy.named_parameters()
        if parameter.requires_grad]
    trainable = [name for name, _parameter in trainable_parameters]
    parameters_before = (
        {name: parameter.detach().clone()
         for name, parameter in trainable_parameters} if no_save else {})
    if any("ref" in name or "behavior" in name for name in trainable):
        raise RuntimeError("reference or behavior adapter is trainable")
    if any(mark in name.lower() for name in trainable
           for mark in ("visual", "vision", "image_encoder")):
        raise RuntimeError("O-only training would move vision parameters")
    inner = policy
    if world_size > 1:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(args.lr), weight_decay=0.01)

    batches = report.get("training_batches") or [{
        "index": 1,
        "groups": list(report["group_advantages"]),
        "active_groups": [group for group in report["group_advantages"]
                          if report["group_advantages"][group]["active"]],
        "task_variable_groups": [
            group for group in report["group_advantages"]
            if report["group_advantages"][group]["task_variable"]],
    }]
    limit = world_size * int(args.grad_accum)
    inner.eval()
    checkpoint_records = []
    gradient_norms = []
    seen_adapter_hashes = {str(args.init_adapter_sha256)}
    for batch in batches:
        active_groups = list(batch.get(
            "task_variable_groups", batch["active_groups"]))
        if not active_groups:
            raise RuntimeError("update batch has no task-variable groups")
        if len(active_groups) > limit:
            raise RuntimeError("update batch exceeds %d active groups" % limit)
        local_groups = balanced_group_partition(
            active_groups, samples, world_size)[rank]
        local = select_batch_samples(
            samples, active_groups, group_size=int(getattr(args, "group_size", 4)),
            world_size=world_size,
            review_token_weight=float(getattr(
                args, "review_token_weight", REVIEW_TOKEN_WEIGHT)))
        local = [sample for sample in local
                 if sample["group_id"] in set(local_groups)]
        if not local:
            raise RuntimeError("rank %d has no policy samples" % rank)

        optimizer.zero_grad(set_to_none=True)
        max_local = len(local)
        if world_size > 1:
            count = torch.tensor([len(local)], dtype=torch.long,
                                 device=local_rank)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            max_local = int(count.item())
        max_raw_diff = 0.0
        max_behavior_diff = 0.0
        max_initial_ratio_diff = 0.0
        references = []
        old_behavior = []
        with torch.no_grad():
            for sample in local:
                inner.set_adapter("behavior")
                behavior_raw, behavior, _behavior_entropy = (
                    teacher_forced_completion_logprobs(
                        inner, sample, local_rank))
                initial_policy = behavior
                if reference_actual == adapter_actual:
                    raw_reference = behavior_raw
                else:
                    inner.set_adapter("ref")
                    raw_reference, _reference_sampling, _reference_entropy = (
                        teacher_forced_completion_logprobs(
                            inner, sample, local_rank))
                raw_old = torch.tensor(
                    sample["old_raw_logprobs"], dtype=torch.float32,
                    device=local_rank)
                behavior_old = torch.tensor(
                    sample["old_logprobs"], dtype=torch.float32,
                    device=local_rank)
                max_raw_diff = max(
                    max_raw_diff,
                    float((behavior_raw - raw_old).abs().max().item()))
                max_behavior_diff = max(
                    max_behavior_diff,
                    float((behavior - behavior_old).abs().max().item()))
                max_initial_ratio_diff = max(
                    max_initial_ratio_diff,
                    float((initial_policy - behavior_old).abs().max().item()))
                references.append(raw_reference.detach())
                old_behavior.append(behavior_old.detach())
        if world_size > 1:
            stats = torch.tensor([
                max_raw_diff, max_behavior_diff, max_initial_ratio_diff,
            ],
                                 dtype=torch.float32, device=local_rank)
            dist.all_reduce(stats, op=dist.ReduceOp.MAX)
            max_raw_diff, max_behavior_diff, max_initial_ratio_diff = [
                float(value) for value in stats]
        replay_limit = float(args.max_served_recompute_abs_diff)
        if not replay_gate_passes(max_raw_diff, max_behavior_diff, replay_limit):
            raise RuntimeError(
                "initial policy does not reproduce rollouts: raw=%s behavior=%s"
                % (max_raw_diff, max_behavior_diff))
        ratio_limit = float(args.max_initial_ratio_log_abs_diff)
        if not initial_ratio_gate_passes(
                max_initial_ratio_diff, ratio_limit):
            raise RuntimeError(
                "initial policy log-ratio differs by %s, tolerance %s"
                % (max_initial_ratio_diff, ratio_limit))

        losses = []
        epoch_losses = []
        batch_stats = {}
        local = _pad_samples(local, max_local)
        references = _pad_samples(
            [{"value": value} for value in references], max_local)
        old_behavior = _pad_samples(
            [{"value": value} for value in old_behavior], max_local)
        for _epoch in range(int(getattr(args, "epochs", 1))):
            optimizer.zero_grad(set_to_none=True)
            epoch_values = []
            for index, (sample, reference_logprobs, old_logprobs) in enumerate(zip(
                    local, references, old_behavior)):
                inner.set_adapter("default")
                completion = list(sample["completion_token_ids"])
                loss_mask = list(sample.get("loss_token_mask") or (
                    [True] * len(completion)))
                selected = [index for index, keep in enumerate(loss_mask) if keep]
                if not selected:
                    raise RuntimeError("policy sample has no trainable token")
                policy_raw_tokens, policy_tokens, entropy_tokens = (
                    teacher_forced_completion_logprobs(
                        policy, sample, local_rank))
                selected_tensor = torch.tensor(
                    selected, dtype=torch.long, device=local_rank)
                policy_logprobs = policy_tokens.index_select(
                    0, selected_tensor)
                policy_raw = policy_raw_tokens.index_select(
                    0, selected_tensor)
                entropy = entropy_tokens.index_select(0, selected_tensor)
                old_selected = old_logprobs["value"].index_select(
                    0, selected_tensor)
                reference_selected = reference_logprobs["value"].index_select(
                    0, selected_tensor)
                merge_token_stats(batch_stats, ppo_token_stats_torch(
                    policy_logprobs, old_selected, reference_selected,
                    args.clip, policy_raw_logprobs=policy_raw,
                    entropy=entropy))
                sample_loss_tensor = ppo_token_loss_torch(
                    policy_logprobs, old_selected, reference_selected,
                    sample["advantage"], float(sample["weight"]),
                    args.clip, args.kl_beta,
                    policy_raw_logprobs=policy_raw, entropy=entropy,
                    entropy_coef=getattr(args, "entropy_coef", 0.0))
                if not torch.isfinite(sample_loss_tensor):
                    raise RuntimeError("non-finite policy loss")
                sync_here = index == len(local) - 1
                with policy.no_sync() if (
                        not sync_here and hasattr(policy, "no_sync")) \
                        else _null_context():
                    sample_loss_tensor.backward()
                sample_loss = float(sample_loss_tensor.detach().item())
                losses.append(sample_loss)
                epoch_values.append(sample_loss)
                torch.cuda.empty_cache()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                float(getattr(args, "max_grad_norm", 1.0)))
            if rank == 0:
                gradient_norms.append(float(grad_norm))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            epoch_losses.append(epoch_values)

        adapter_sha = ""
        if rank == 0:
            checkpoint = os.path.join(args.out, "batch-%02d" % batch["index"])
            os.makedirs(checkpoint, exist_ok=True)
            if not no_save:
                inner.set_adapter("default")
                inner.save_pretrained(checkpoint, selected_adapters=["default"])
                adapter_path = os.path.join(
                    checkpoint, "adapter_model.safetensors")
                adapter_sha = _sha256(adapter_path)
            record = {
                "batch": int(batch["index"]),
                "groups": list(batch["groups"]),
                "active_groups": active_groups,
                "policy_samples": len(select_batch_samples(
                    samples, active_groups,
                    group_size=int(getattr(args, "group_size", 4)),
                    world_size=world_size,
                    review_token_weight=float(getattr(
                        args, "review_token_weight", REVIEW_TOKEN_WEIGHT)))),
                "optimizer_steps": int(getattr(args, "epochs", 1)),
                "losses_rank0": losses,
                "losses_note": "per_sample_sum_of_token_mean_losses",
                "epoch_losses_rank0": epoch_losses,
                "max_raw_old_abs_diff": max_raw_diff,
                "max_behavior_old_abs_diff": max_behavior_diff,
                "max_initial_policy_ratio_log_abs_diff": (
                    max_initial_ratio_diff),
                "max_initial_policy_ratio_log_tolerance": ratio_limit,
                "replay_forward": REPLAY_FORWARD,
                "token_forward_sweeps": int(args.token_forward_sweeps),
                "projected_token_forwards": int(
                    args.projected_token_forward_count),
                "initial_policy_reused": True,
                "reference_adapter_reused": (
                    args.reference_adapter_sha256 == args.init_adapter_sha256),
                "behavior_old_abs_diff_gate": "hard",
                "adapter_sha256": adapter_sha,
                "checkpoint": checkpoint,
                "no_save": no_save,
                "token_stats": summarize_token_stats(batch_stats),
            }
            checkpoint_records.append(record)
            with open(os.path.join(checkpoint, "ONLINE_O_BATCH.json"), "w",
                      encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
        if not no_save:
            duplicate = torch.tensor(
                [int(rank == 0 and adapter_sha in seen_adapter_hashes)],
                dtype=torch.long, device=local_rank)
            if world_size > 1:
                dist.all_reduce(duplicate, op=dist.ReduceOp.MAX)
            if int(duplicate.item()):
                raise RuntimeError("update batch did not change the adapter")
            if rank == 0:
                seen_adapter_hashes.add(adapter_sha)

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        finished_unix = time.time()
        payload = {
            "started_unix": round(started_unix, 6),
            "finished_unix": round(finished_unix, 6),
            "elapsed_seconds": round(finished_unix - started_unix, 6),
            "groups": report["groups"],
            "episodes": report["episodes"],
            "active_groups": report["active_groups"],
            "task_variable_groups": report["task_variable_groups"],
            "constant_groups": report["constant_groups"],
            "policy_samples": report["policy_samples"],
            "local_optimizer_steps": (
                len(checkpoint_records) * int(getattr(args, "epochs", 1))),
            "gradient_norms": gradient_norms,
            "losses_note": "per_sample_sum_of_token_mean_losses",
            "expected_optimizer_steps": (
                len(batches) * int(getattr(args, "epochs", 1))),
            "world_size": world_size,
            "epochs": int(getattr(args, "epochs", 1)),
            "grad_accum": int(args.grad_accum),
            "advantage_mode": str(report.get("advantage_mode") or ""),
            "optimization_reward": "task plus O cost",
            "lr": float(args.lr),
            "clip": float(args.clip),
            "kl_beta": float(args.kl_beta),
            "entropy_coef": float(getattr(args, "entropy_coef", 0.0)),
            "max_len": int(args.max_len),
            "max_prompt_completion_tokens": int(args.max_prompt_completion_tokens),
            "seed": int(args.seed),
            "max_raw_old_abs_diff": max_raw_diff,
            "max_behavior_old_abs_diff": max_behavior_diff,
            "max_served_recompute_abs_diff": float(
                args.max_served_recompute_abs_diff),
            "behavior_old_abs_diff_gate": "hard",
            "replay_forward": REPLAY_FORWARD,
            "max_initial_policy_ratio_log_abs_diff": (
                max_initial_ratio_diff),
            "max_initial_policy_ratio_log_tolerance": ratio_limit,
            "token_forward_sweeps": int(args.token_forward_sweeps),
            "projected_token_forwards": int(
                args.projected_token_forward_count),
            "projected_sequence_forwards": int(
                args.projected_sequence_forward_count),
            "max_replay_sequence_forwards": (
                int(args.max_replay_sequence_forwards)
                if args.max_replay_sequence_forwards is not None else None),
            "max_replay_token_forwards": (
                int(args.max_replay_token_forwards)
                if args.max_replay_token_forwards is not None else None),
            "initial_policy_reused": True,
            "reference_adapter_reused": (
                args.reference_adapter_sha256 == args.init_adapter_sha256),
            "group_partition": {
                str(index): list(values)
                for index, values in enumerate(
                    balanced_group_partition(active_groups, samples, world_size))
            },
            "old_logprobs": "served_sampling_post_warper",
            "kl_reference": "frozen_campaign_reference_adapter",
            "base_model": args.model,
            "base_model_sha256": args.model_sha256,
            "quantization": copy.deepcopy(getattr(args, "quantization", None)),
            "attention": str(getattr(args, "attention", "sdpa")),
            "init_adapter": args.init_adapter,
            "init_adapter_sha256": args.init_adapter_sha256,
            "behavior_adapter": args.init_adapter,
            "behavior_adapter_sha256": args.init_adapter_sha256,
            "reference_adapter": args.reference_adapter,
            "reference_adapter_sha256": args.reference_adapter_sha256,
            "adapter_sha256": checkpoint_records[-1]["adapter_sha256"],
            "trainable_parameters": int(sum(
                parameter.numel() for _name, parameter in trainable_parameters)),
            "gpu_count": int(world_size),
            "gpu_hours": round(
                (finished_unix - started_unix) * int(world_size) / 3600.0, 6),
            "code_sha256": _sha256(__file__),
            "group_advantages": report["group_advantages"],
            "reward_audit": report["reward_audit"],
            "training_batches": report.get("training_batches") or batches,
            "checkpoints": checkpoint_records,
            "no_save": no_save,
        }
        with open(os.path.join(args.out, "ONLINE_O_TRAINING.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    parameter_movement = (
        _parameter_delta(parameters_before, trainable_parameters)
        if no_save else {"max_abs": None, "relative_l2": None})
    if no_save and not _parameter_movement_valid(parameter_movement):
        raise RuntimeError("O update did not change trainable parameters")
    if rank == 0:
        payload["parameter_delta"] = parameter_movement
        with open(os.path.join(args.out, "ONLINE_O_TRAINING.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--decision-token-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        collection = config["collection"]
        training = config["training"]
        report, samples = prepare_training(
            args.episodes, group_size=collection["group_size"],
            expected_groups=collection["expected_groups"],
            min_task_variable_groups=collection["min_task_variable_groups"],
            expected_temperature=collection["temperature"],
            expected_top_p=collection["top_p"],
            expected_top_k=collection["top_k"],
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            allow_guess_incentive=bool(
                collection.get("allow_guess_incentive")),
            min_mixed_action_groups=int(collection.get(
                "min_mixed_action_groups") or 0),
            expected_model_quantization=str(collection.get(
                "model_quantization") or "bf16"),
            batch_size=collection.get("batch_size"),
            expected_batches=collection.get("expected_batches"),
            min_task_variable_per_batch=collection.get(
                "min_task_variable_per_batch"),
            max_task_variable_per_batch=collection.get(
                "max_task_variable_per_batch"),
            batch_order_salt=collection.get("batch_order_salt"),
            include_reviewer=bool(training.get("include_reviewer_tokens", False)),
            review_token_weight=float(training.get(
                "review_token_weight", REVIEW_TOKEN_WEIGHT)),
            advantage_mode=str(collection.get(
                "advantage_mode", "task_variation")),
            admission_mode=str(collection.get(
                "admission_mode", "strict")),
            sequence_level=bool(collection.get("sequence_level")),
            min_common_questions=int(
                collection.get("min_common_questions") or
                MIN_SEQUENCE_QUESTIONS),
            guess_incentive_scope=str(collection.get(
                "guess_incentive_scope", "batch")),
            decision_token_only=bool(args.decision_token_only),
            shuffle_rewards_seed=training.get("shuffle_rewards_seed"))
    except (OSError, ValueError, KeyError) as error:
        print("online-O preparation refused: %s" % error, file=sys.stderr)
        return 2
    args.reference_adapter = resolve_run_path(str(training.get(
        "reference_adapter") or args.init_adapter))
    args.reference_adapter_sha256 = str(training.get(
        "reference_adapter_sha256") or args.init_adapter_sha256)
    args.max_len = int(training["max_len"])
    args.max_prompt_completion_tokens = validate_sample_lengths(
        samples, args.max_len,
        (int(collection["max_completion_tokens"])
         if collection.get("max_completion_tokens") is not None else None))
    epochs = int(training.get("epochs", 1))
    reference_reused = (
        str(training.get("reference_adapter_sha256")
            or training["init_adapter_sha256"])
        == str(training["init_adapter_sha256"]))
    token_forward_sweeps = planned_token_forward_sweeps(
        epochs, reference_reused=reference_reused)
    token_forwards = projected_token_forwards(
        samples, sweeps=token_forward_sweeps)
    sequence_forwards = projected_sequence_forwards(
        samples, sweeps=token_forward_sweeps)
    report["replay_forward"] = REPLAY_FORWARD
    report["token_forward_sweeps"] = token_forward_sweeps
    report["projected_token_forwards"] = token_forwards
    report["projected_sequence_forwards"] = sequence_forwards
    report["initial_policy_reused"] = True
    report["reference_adapter_reused"] = reference_reused
    max_token_forwards = training.get("max_replay_token_forwards")
    if max_token_forwards is not None and token_forwards > int(
            max_token_forwards):
        print("replay refused: projected %d token forwards exceed %d"
              % (token_forwards, int(max_token_forwards)), file=sys.stderr)
        return 3
    max_sequence_forwards = training.get("max_replay_sequence_forwards")
    if max_sequence_forwards is not None and sequence_forwards > int(
            max_sequence_forwards):
        print("replay refused: projected %d sequence forwards exceed %d"
              % (sequence_forwards, int(max_sequence_forwards)), file=sys.stderr)
        return 3
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["reward_audit"]["passes"]:
        audit = report["reward_audit"]
        reasons = []
        if report["task_variable_groups"] < int(
                audit["minimum_task_variable_groups"]):
            reasons.append("task_variable_groups=%d/%d" % (
                report["task_variable_groups"],
                audit["minimum_task_variable_groups"]))
        if int(audit["minimum_mixed_action_groups"]) and (
                report["mixed_action_groups"] < int(
                    audit["minimum_mixed_action_groups"])):
            reasons.append("mixed_action_groups=%d/%d" % (
                report["mixed_action_groups"],
                audit["minimum_mixed_action_groups"]))
        if audit.get("guess_incentive"):
            reasons.append("guess_incentive_groups=%s" % ",".join(
                audit.get("guess_incentive_groups") or []))
        if audit.get("batches_pass") is False:
            reasons.append("batch_signal_gate=false")
        print("reward audit refused: %s" % "; ".join(reasons),
              file=sys.stderr)
        return 3
    if args.dry_run:
        return 0
    args.model_sha256 = resolve_model_sha256(
        training["base_model_sha256"], args.model)
    args.init_adapter_sha256 = training["init_adapter_sha256"]
    args.lr = training["lr"]
    args.grad_accum = training["grad_accum"]
    args.clip = training["clip"]
    args.kl_beta = training["kl_beta"]
    args.entropy_coef = float(training.get("entropy_coef", 0.0))
    args.epochs = epochs
    args.token_forward_sweeps = token_forward_sweeps
    args.projected_token_forward_count = token_forwards
    args.projected_sequence_forward_count = sequence_forwards
    args.max_served_recompute_abs_diff = training[
        "max_served_recompute_abs_diff"]
    args.max_initial_ratio_log_abs_diff = float(training.get(
        "max_initial_ratio_log_abs_diff", 0.0001))
    args.seed = training["seed"]
    args.quantization = copy.deepcopy(training.get("quantization"))
    args.attention = str(training.get("attention", "sdpa"))
    args.max_grad_norm = float(training["max_grad_norm"])
    args.max_replay_sequence_forwards = training.get("max_replay_sequence_forwards")
    args.max_replay_token_forwards = training.get("max_replay_token_forwards")
    args.rank = int(training["rank"])
    args.alpha = int(training["alpha"])
    args.dropout = float(training["dropout"])
    args.target_modules = list(training["target_modules"])
    args.gradient_checkpointing = bool(training["gradient_checkpointing"])
    args.group_size = int(collection["group_size"])
    args.review_token_weight = float(training.get(
        "review_token_weight", REVIEW_TOKEN_WEIGHT))
    return train(args, report, samples)


if __name__ == "__main__":
    raise SystemExit(main())
