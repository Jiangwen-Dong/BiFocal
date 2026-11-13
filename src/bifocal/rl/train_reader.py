#!/usr/bin/env python3
"""One grouped Reader-token PPO update with O and the vision encoder frozen."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

from ..serving import client as LC
from ..harness import frames as K
from ..serving import lanes as RS
from . import train_o as T
from .sequence import permute_within_group

SCHEMA_VERSION = "online-reader-v1"
READER_ROLES = ("reader",)
REPLAY_CACHE_SCHEMA = "online-reader-replay-cache-v1"
REPLAY_FORWARD = "teacher_forced_full_sequence"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replay_cache_key(episodes_path, model_sha, behavior_sha, reference_sha,
                      torch_version, transformers_version):
    rl_dir = os.path.dirname(os.path.abspath(__file__))
    bifocal_dir = os.path.dirname(rl_dir)
    code_hashes = {
        "train_reader": _sha256(os.path.join(rl_dir, "train_reader.py")),
        "train_o": _sha256(os.path.join(rl_dir, "train_o.py")),
        "serving_client": _sha256(os.path.join(
            bifocal_dir, "serving", "client.py")),
        "harness_actions": _sha256(os.path.join(
            bifocal_dir, "harness", "actions.py")),
    }
    payload = {
        "schema": REPLAY_CACHE_SCHEMA,
        "episodes_sha256": _sha256(episodes_path),
        "model_sha256": str(model_sha),
        "behavior_adapter_sha256": str(behavior_sha),
        "reference_adapter_sha256": str(reference_sha),
        "torch_version": str(torch_version),
        "transformers_version": str(transformers_version),
        "replay_forward": REPLAY_FORWARD,
        "code_hashes": code_hashes,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest(), payload


def _replay_cache_checksum(value):
    payload = {name: item for name, item in value.items()
               if name != "checksum"}
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_replay_cache(path, key, binding, sample_count, device=0):
    import torch
    try:
        with open(path, encoding="utf-8") as handle:
            cache = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if (cache.get("schema") != REPLAY_CACHE_SCHEMA
            or cache.get("key") != key
            or cache.get("binding") != binding
            or cache.get("checksum") != _replay_cache_checksum(cache)):
        return None
    try:
        references = [torch.tensor(value, dtype=torch.float32, device=device)
                      for value in cache["references"]]
        old_behavior = [torch.tensor(value, dtype=torch.float32, device=device)
                        for value in cache["old_behavior"]]
        raw = float(cache["max_raw_diff"])
        behavior = float(cache["max_behavior_diff"])
    except (KeyError, TypeError, ValueError):
        return None
    if (len(references) != int(sample_count)
            or len(old_behavior) != int(sample_count)):
        return None
    return references, old_behavior, raw, behavior


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported online Reader schema")
    collection = config.get("collection") or {}
    training = config.get("training") or {}
    advantage_mode = str(collection.get("advantage_mode") or "task_cost")
    if advantage_mode not in T.TASK_COST_MODES:
        raise ValueError("Reader PPO requires task_cost advantage")
    if int(collection.get("group_size") or 0) != 4:
        raise ValueError("Reader PPO needs four rollouts per group")
    if not collection.get("sequence_level"):
        raise ValueError("Reader PPO requires sequence groups")
    if not collection.get("streaming_step"):
        raise ValueError("Reader PPO requires streaming_step")
    if int(training.get("world_size") or 0) != 1:
        raise ValueError("Reader PPO requires one training rank")
    expected = int(collection.get("expected_groups") or 0)
    if expected < 1:
        raise ValueError("Reader PPO needs at least one group")
    batch_size = int(collection.get("batch_size") or expected)
    if not 1 <= batch_size <= expected:
        raise ValueError("Reader batch size must be within the group count")
    batch_count = int(math.ceil(expected / float(batch_size)))
    layout = [min(batch_size, expected - start)
              for start in range(0, expected, batch_size)]
    expected_fields = {
        "batch_size": batch_size,
        "expected_batches": batch_count,
        "min_task_variable_groups": expected,
        "min_task_variable_per_batch": min(layout),
        "max_task_variable_per_batch": max(layout),
    }
    for name, wanted in expected_fields.items():
        if int(collection.get(name) or 0) != wanted:
            raise ValueError("Reader PPO %s must equal %d" % (name, wanted))
    if int(training.get("grad_accum") or 0) != batch_size:
        raise ValueError("Reader gradient accumulation must equal batch size")
    if int(training.get("update_batches") or 0) != batch_count:
        raise ValueError("Reader PPO update-batch count differs from its layout")
    if int(training.get("epochs") or 1) <= 0:
        raise ValueError("Reader training epochs must be positive")
    sequence_cap = training.get("max_replay_sequence_forwards")
    if sequence_cap is not None and (
            not isinstance(sequence_cap, (int, float))
            or isinstance(sequence_cap, bool) or int(sequence_cap) <= 0):
        raise ValueError("max_replay_sequence_forwards must be positive")
    if list(training.get("target_modules") or ()) != [
            "q_proj", "k_proj", "v_proj", "o_proj"]:
        raise ValueError("unexpected Reader LoRA target modules")
    if training.get("vision_trainable") is not False:
        raise ValueError("Reader training must keep vision frozen")
    if str(training.get("replay_forward") or REPLAY_FORWARD) != REPLAY_FORWARD:
        raise ValueError("Reader training requires teacher-forced replay")
    return config


def _load_groups(path, group_size, allow_diagnostic=False):
    groups = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if ((row.get("memory_protocol") or {}).get("diagnostic_only")
                    and not allow_diagnostic):
                raise ValueError("Reader episode is diagnostic-only: %s"
                                 % row.get("episode_id"))
            groups.setdefault(str(row["group_id"]), []).append(row)
    for group, rows in groups.items():
        rows.sort(key=lambda row: int(row.get("rollout_index") or 0))
        if len(rows) != int(group_size):
            raise ValueError("Reader group %s has %d rollouts, need %d" % (
                group, len(rows), int(group_size)))
        if len({int(row.get("rollout_index") or 0) for row in rows}) != len(rows):
            raise ValueError("Reader group %s has duplicate rollouts" % group)
    return groups


def _reader_policy_step(step, episode_id):
    prompt = step.get("prompt_token_ids")
    tokens = step.get("completion_token_ids")
    old = step.get("completion_logprobs")
    raw_old = step.get("completion_raw_logprobs")
    if not (isinstance(prompt, list) and prompt
            and isinstance(tokens, list) and tokens
            and isinstance(old, list) and len(old) == len(tokens)
            and isinstance(raw_old, list) and len(raw_old) == len(tokens)):
        raise ValueError("Reader step has missing token or logprob data: %s"
                         % episode_id)
    temperature = float(step.get("temperature") or 0.0)
    top_p = float(step.get("top_p") or 0.0)
    if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
        raise ValueError("Reader step has invalid sampling parameters: %s"
                         % episode_id)
    if int(step.get("top_k") or 0) != 0:
        raise ValueError("Reader step must disable top-k sampling: %s"
                         % episode_id)
    no_repeat_ngram_size = step.get("no_repeat_ngram_size")
    no_repeat_ngram_size = (
        6 if no_repeat_ngram_size is None else int(no_repeat_ngram_size))
    if no_repeat_ngram_size < 0:
        raise ValueError("Reader step has invalid no-repeat size: %s" % episode_id)
    return {
        "episode_id": episode_id,
        "prompt": str(step.get("prompt") or ""),
        "video_path": str(step.get("video_path") or ""),
        "frame_dir": str(step.get("frame_dir") or ""),
        "frame_manifest": step.get("frame_manifest"),
        "tensor_dump": str(step.get("tensor_dump") or ""),
        "prompt_token_ids": list(prompt),
        "completion_token_ids": list(tokens),
        "old_logprobs": [float(value) for value in old],
        "old_raw_logprobs": [float(value) for value in raw_old],
        "temperature": temperature,
        "top_p": top_p,
        "no_repeat_ngram_size": no_repeat_ngram_size,
    }


def _reader_images(sample):
    manifest = sample.get("frame_manifest") or {}
    frames = manifest.get("frames") or []
    video_path = sample.get("video_path") or ""
    frame_dir = sample.get("frame_dir") or ""
    if not frames or not video_path or not frame_dir:
        return []
    window = manifest.get("window") or (0.0, 0.0)
    triples = K.Frames(video_path, frame_dir).window(
        float(window[0]), float(window[1]))
    by_md5 = {str(md5): image for image, _second, md5 in triples}
    images = []
    for frame in frames:
        md5 = str(frame.get("md5") or "")
        if not md5:
            raise ValueError("saved Reader frame has no md5")
        image = by_md5.get(md5)
        if image is None:
            raise ValueError(
                "saved Reader frame md5 is missing from the frame cache: %s" % md5)
        width, height = int(frame.get("width") or 0), int(frame.get("height") or 0)
        if width and height and image.size != (width, height):
            image = image.resize((width, height))
        images.append(image)
    return images


def _reader_processor_prompt(processor, prompt):
    parts = str(prompt or "").split("<image>")
    content = []
    for index, piece in enumerate(parts):
        if piece:
            content.append({"type": "text", "text": piece})
        if index < len(parts) - 1:
            content.append({"type": "image"})
    return processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)


def _reader_inputs(processor, sample, processor_kwargs=None):
    import torch
    dump = str(sample.get("tensor_dump") or "")
    if dump and os.path.exists(dump):
        return torch.load(dump, map_location="cpu", weights_only=True)
    images = _reader_images(sample)
    if not images:
        raise ValueError("Reader sample has no replay images")
    return processor(
        text=[_reader_processor_prompt(processor, sample["prompt"])],
        images=images, return_tensors="pt",
        **(processor_kwargs or {}))


def _reader_teacher_forced_completion_logits(
        policy, processor, sample, completion_ids, device,
        processor_kwargs=None):
    """Return all completion logits from one full teacher-forced pass."""
    import torch
    inputs = _reader_inputs(processor, sample, processor_kwargs)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_attention = inputs.get("attention_mask")
    if prompt_attention is None:
        prompt_attention = torch.ones_like(prompt_ids)
    prompt_attention = prompt_attention.to(device)
    completion = torch.tensor(
        [list(completion_ids)], dtype=torch.long, device=device)
    input_ids = torch.cat([prompt_ids, completion], dim=1)
    attention = torch.cat([
        prompt_attention, torch.ones_like(completion)], dim=1)
    vision = {
        name: value.to(device) for name, value in inputs.items()
        if name not in ("input_ids", "attention_mask")}
    output = policy(
        input_ids=input_ids, attention_mask=attention,
        logits_to_keep=len(completion_ids) + 1,
        use_cache=False, **vision)
    logits = output.logits[:, :-1, :].float()
    if int(logits.shape[1]) != len(completion_ids):
        raise RuntimeError(
            "Reader teacher-forced logits differ from completion length")
    return logits[0]

def _reader_train_sample_teacher_forced(policy, processor, sample,
                                        reference, old, training,
                                        processor_kwargs, stats=None):
    """Backpropagate one full-sequence teacher-forced Reader sample."""
    import torch
    completion_ids = list(sample["completion_token_ids"])
    logits = _reader_teacher_forced_completion_logits(
        policy, processor, sample, completion_ids, 0, processor_kwargs)
    sampling_logits = _reader_sampling_logits(
        logits, sample["prompt_token_ids"], completion_ids,
        sample["no_repeat_ngram_size"],
        tokenizer=processor.tokenizer,
        eos_ids=LC.eos_token_ids(processor.tokenizer, policy.get_base_model()))
    policy_logprobs = T._sampling_logprobs(
        sampling_logits, completion_ids, sample["temperature"],
        sample["top_p"], 0)
    policy_raw = T._selected_logprobs(logits, completion_ids, 0)
    entropy = T._sampling_entropy(
        sampling_logits, sample["temperature"], sample["top_p"])
    if stats is not None:
        T.merge_token_stats(stats, T.ppo_token_stats_torch(
            policy_logprobs, old, reference, float(training["clip"]),
            policy_raw_logprobs=policy_raw, entropy=entropy))
    loss = T.ppo_token_loss_torch(
        policy_logprobs, old, reference, sample["advantage"],
        float(sample["weight"]),
        float(training["clip"]), float(training["kl_beta"]),
        policy_raw_logprobs=policy_raw, entropy=entropy,
        entropy_coef=float(training.get("entropy_coef") or 0.0))
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite Reader policy loss")
    loss.backward()
    return float(loss.detach().item())

def _forbidden_no_repeat_tokens(token_ids, ngram_size):
    """Return token IDs banned by the no-repeat rule at the next position."""
    prefix = list(token_ids)
    ngram_size = int(ngram_size)
    if ngram_size < 2 or len(prefix) < ngram_size:
        return set()
    tail = prefix[-(ngram_size - 1):]
    banned = set()
    for start in range(len(prefix) - ngram_size + 1):
        if prefix[start:start + ngram_size - 1] == tail:
            banned.add(prefix[start + ngram_size - 1])
    return banned


def _reader_sampling_logits(logits, prompt_ids, completion_ids, ngram_size,
                            tokenizer=None, eos_ids=None):
    """Apply the Reader's generation-time masks to full logits."""
    import torch
    raw = logits.float()
    mask = torch.zeros_like(raw, dtype=torch.bool)
    prefix = list(prompt_ids)
    for index, token_id in enumerate(completion_ids):
        banned = _forbidden_no_repeat_tokens(prefix, ngram_size)
        if banned:
            mask[index, torch.tensor(
                sorted(banned), dtype=torch.long, device=raw.device)] = True
        if tokenizer is not None:
            completion_prefix = completion_ids[:index]
            text = tokenizer.decode(
                completion_prefix, skip_special_tokens=True)
            if not LC.reader_report_fields_complete(
                text, LC.READER_POLICY_REQUIRED_FIELDS):
                stop_ids = sorted(
                    eos_ids if eos_ids is not None
                    else LC.eos_token_ids(tokenizer))
                if stop_ids:
                    mask[index, torch.tensor(
                        stop_ids, dtype=torch.long, device=raw.device)] = True
        prefix.append(int(token_id))
    return raw.masked_fill(mask, -float("inf"))


def _reader_samples(groups, group_size=4):
    samples = []
    active = list(groups)
    for group in active:
        for episode in groups[group]:
            if not bool((episode.get("training_only") or {}).get(
                    "reader_reward_clean")):
                raise ValueError("Reader group contains a non-reward-clean episode")
            steps = [step for step in episode.get("steps") or []
                     if step.get("actor") == "reader"
                     and step.get("adapter") == "reader"
                     and step.get("valid")
                     and not step.get("reader_retry")
                     and not step.get("reader_retry_cut")
                     and not step.get("recovered_action")
                     and not step.get("repair")
                     and not step.get("normalized")]
            if not steps:
                raise ValueError("Reader episode has no trainable Reader tokens")
            for step in steps:
                sample = _reader_policy_step(step, episode.get("episode_id") or "")
                sample.update({
                    "actor": "reader",
                    "group_id": group,
                    "rollout_index": int(episode.get("rollout_index") or 0),
                    "advantage": float((episode.get("training_only") or {}).get(
                        "reader_advantage") or 0.0),
                    "weight": 1.0 / float(len(active) * int(group_size)
                                          * len(steps)),
                })
                samples.append(sample)
    if not samples:
        raise ValueError("Reader PPO has no policy samples")
    return samples


def plan_reader_batches(samples, batch_size, expected_batches,
                        min_groups_per_batch, max_groups_per_batch,
                        order_salt):
    """Order Reader groups once and split them into fixed update batches."""
    group_ids = sorted({str(sample["group_id"]) for sample in samples})
    if not group_ids:
        raise ValueError("Reader PPO has no groups")
    ordered = sorted(group_ids, key=lambda group: hashlib.sha256(
        ("%s:%s" % (order_salt, group)).encode("utf-8")).hexdigest())
    batches = [ordered[index:index + int(batch_size)]
               for index in range(0, len(ordered), int(batch_size))]
    if len(batches) != int(expected_batches):
        raise ValueError("Reader batch count differs from its config")
    if any(len(batch) < int(min_groups_per_batch)
           or len(batch) > int(max_groups_per_batch) for batch in batches):
        raise ValueError("Reader batch size is outside its config")
    return batches


def weight_reader_batch(samples, group_ids, group_size=4):
    """Normalize one update batch so every group contributes equal weight."""
    selected = set(str(group) for group in group_ids)
    rows = [sample for sample in samples
            if str(sample["group_id"]) in selected]
    counts = collections.Counter(str(sample["group_id"]) for sample in rows)
    if set(counts) != selected:
        raise ValueError("Reader batch is missing an admitted group")
    steps_per_group = {}
    for group, count in counts.items():
        if int(count) % int(group_size):
            raise ValueError("Reader group sample count is not a rollout multiple")
        steps_per_group[group] = int(count) // int(group_size)
    for sample in rows:
        group = str(sample["group_id"])
        sample["weight"] = 1.0 / float(
            len(selected) * int(group_size) * steps_per_group[group])
    return rows


def _parameter_delta(before, parameters):
    """Return a small, deterministic summary of trainable parameter movement."""
    max_abs = 0.0
    squared_delta = 0.0
    squared_before = 0.0
    for name, parameter in parameters:
        old = before.get(name)
        if old is None:
            continue
        delta = parameter.detach() - old
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_delta += float((delta.float() ** 2).sum().item())
        squared_before += float((old.float() ** 2).sum().item())
    relative_l2 = (
        (squared_delta ** 0.5) / (squared_before ** 0.5)
        if squared_before else 0.0)
    return {"max_abs": max_abs, "relative_l2": relative_l2}


def _enable_decoder_layer_checkpointing(policy, checkpoint_fn=None):
    """Checkpoint each language decoder layer explicitly.

    The Qwen3-VL build used by the study exposes the usual checkpointing flag but
    its decoder loop does not consume it. Wrapping each layer keeps the frozen
    vision path and LoRA adaptation unchanged while reducing training memory.
    """
    get_base_model = getattr(policy, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else policy
    language_model = getattr(getattr(base, "model", None),
                             "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("cannot find Qwen3-VL decoder layers for checkpointing")
    if checkpoint_fn is None:
        from torch.utils.checkpoint import checkpoint as checkpoint_fn

    wrapped = 0
    for layer in layers:
        if getattr(layer, "_bifocal_checkpointed", False):
            continue
        original = layer.forward

        def checkpointed(*args, _original=original, **kwargs):
            return checkpoint_fn(
                _original, *args, use_reentrant=False, **kwargs)

        layer.forward = checkpointed
        layer._bifocal_checkpointed = True
        wrapped += 1
    return wrapped


def prepare(episodes_path, config, allow_diagnostic=False):
    collection = config["collection"]
    groups = _load_groups(
        episodes_path, group_size=int(collection["group_size"]),
        allow_diagnostic=allow_diagnostic)
    if len(groups) != int(collection["expected_groups"]):
        raise ValueError("expected %d Reader groups, found %d" % (
            int(collection["expected_groups"]), len(groups)))
    if collection.get("sequence_level"):
        T.validate_sequence_layout(
            groups, group_size=int(collection["group_size"]),
            min_common_questions=int(
                collection.get("min_common_questions") or
                T.MIN_SEQUENCE_QUESTIONS))
    shuffle_seed = config["training"].get("shuffle_rewards_seed")
    returns = []
    for group, rows in groups.items():
        if len(rows) != 4:
            raise ValueError("Reader group %s does not have four rollouts" % group)
        values = [float((row.get("training_only") or {}).get(
            "reader_optimization_reward") or 0.0) for row in rows]
        mean = sum(values) / 4.0
        std = math.sqrt(sum((value - mean) ** 2 for value in values) / 4.0)
        if std <= 1e-9:
            raise ValueError("Reader group %s has zero reward variance" % group)
        correct = [bool((row.get("outcome") or {}).get("correct"))
                   for row in rows]
        if len(set(correct)) <= 1:
            raise ValueError("Reader group %s has constant frozen-O correctness"
                             % group)
        advantages = [(value - mean) / std for value in values]
        if shuffle_seed is not None:
            advantages = permute_within_group(advantages, shuffle_seed, group)
        for row, advantage in zip(rows, advantages):
            row["training_only"]["reader_advantage"] = round(advantage, 12)
        returns.append({
            "group": group,
            "returns": values,
            "mean": mean,
            "std": std,
            "correct_bits": correct,
            "reader_ready": all(bool((row.get("training_only") or {}).get(
                "reader_reward_clean")) for row in rows),
        })
    samples = _reader_samples(groups)
    report = {
        "groups": len(groups),
        "episodes": sum(len(rows) for rows in groups.values()),
        "policy_samples": len(samples),
        "returns": returns,
        "shuffle_rewards_seed": shuffle_seed,
        "group_advantages": {
            group: [float((row.get("training_only") or {}).get(
                "reader_advantage") or 0.0) for row in rows]
            for group, rows in groups.items()},
    }
    return report, samples


def train(config, episodes_path, model_path, init_adapter, out, no_save=False,
          allow_diagnostic=False):
    if allow_diagnostic and not no_save:
        raise ValueError("diagnostic Reader data requires --no-save")
    import torch
    from peft import PeftModel
    import transformers
    from transformers import AutoProcessor

    training = config["training"]
    actual_model = RS.model_content_sha256(model_path)
    expected_model = T.resolve_model_sha256(
        training["base_model_sha256"], model_path)
    if actual_model != expected_model:
        raise ValueError("base model hash differs from the config")
    training["base_model_sha256"] = actual_model
    actual_adapter = _sha256(
        os.path.join(init_adapter, "adapter_model.safetensors"))
    if actual_adapter != training["init_adapter_sha256"]:
        raise ValueError("initial Reader adapter hash differs from the config")
    reference_adapter = str(
        training.get("reference_adapter") or init_adapter)
    reference_sha = str(
        training.get("reference_adapter_sha256") or actual_adapter)
    actual_reference = _sha256(
        os.path.join(reference_adapter, "adapter_model.safetensors"))
    if actual_reference != reference_sha:
        raise ValueError("reference Reader adapter hash differs from the config")
    report, samples = prepare(
        episodes_path, config, allow_diagnostic=allow_diagnostic)
    diagnostic_only_groups = sum(
        any(bool((row.get("memory_protocol") or {}).get("diagnostic_only"))
            for row in rows)
        for rows in _load_groups(
            episodes_path, int(config["collection"]["group_size"]),
            allow_diagnostic=True).values())
    max_len = int(training["max_len"])
    max_tokens = T.validate_sample_lengths(samples, max_len)

    model_class = LC.model_class(model_path)
    model = model_class.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=LC.attention_implementation())
    processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
    processor_kwargs = (
        {"max_pixels": LC.CAP} if LC.accepts_max_pixels(processor) else {})
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    policy = PeftModel.from_pretrained(
        model, init_adapter, adapter_name="default", is_trainable=True)
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
    if hasattr(policy, "enable_input_require_grads"):
        policy.enable_input_require_grads()
    _enable_decoder_layer_checkpointing(policy)
    policy.load_adapter(
        init_adapter, adapter_name="behavior", is_trainable=False)
    policy.load_adapter(
        reference_adapter, adapter_name="ref", is_trainable=False)
    policy.set_adapter("default")
    trainable = [name for name, parameter in policy.named_parameters()
                 if parameter.requires_grad]
    if any("ref" in name or "behavior" in name for name in trainable):
        raise RuntimeError("Reader reference or behavior adapter is trainable")
    if any(mark in name.lower() for name in trainable
           for mark in ("visual", "vision", "image_encoder")):
        raise RuntimeError("Reader training would move vision parameters")
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=float(training["lr"]), weight_decay=0.01)
    policy.eval()
    parameters = [(name, parameter) for name, parameter in
                  policy.named_parameters() if parameter.requires_grad]
    parameters_before = (
        {name: parameter.detach().clone()
         for name, parameter in parameters} if no_save else {})

    local = list(samples)
    projected_sequence_forwards = (
        len(local) * (1 + int(training["epochs"])))
    sequence_cap = training.get("max_replay_sequence_forwards")
    if sequence_cap is not None and projected_sequence_forwards > int(sequence_cap):
        raise ValueError(
            "Reader replay needs %d sequence forwards, cap is %d"
            % (projected_sequence_forwards, int(sequence_cap)))
    os.makedirs(out, exist_ok=True)
    cache_key, cache_binding = _replay_cache_key(
        episodes_path, actual_model, actual_adapter, actual_reference,
        torch.__version__, transformers.__version__)
    cache_path = os.path.join(out, "REPLAY_CACHE.json")
    cached = _load_replay_cache(
        cache_path, cache_key, cache_binding, len(local))
    replay_cache_hit = cached is not None
    if cached is not None:
        references, old_behavior, max_raw_diff, max_behavior_diff = cached
        print("Reader replay cache hit %s" % cache_key, flush=True)
    else:
        references, old_behavior = [], []
        max_raw_diff = 0.0
        max_behavior_diff = 0.0
        with torch.no_grad():
            for sample_index, sample in enumerate(local, 1):
                print("Reader replay sample %d/%d" % (
                    sample_index, len(local)), flush=True)
                policy.set_adapter("behavior")
                behavior_logits = _reader_teacher_forced_completion_logits(
                    policy, processor, sample,
                    sample["completion_token_ids"], 0, processor_kwargs)
                behavior_raw = T._selected_logprobs(
                    behavior_logits, sample["completion_token_ids"], 0)
                behavior_sampling_logits = _reader_sampling_logits(
                    behavior_logits, sample["prompt_token_ids"],
                    sample["completion_token_ids"],
                    sample["no_repeat_ngram_size"],
                    tokenizer=processor.tokenizer,
                    eos_ids=LC.eos_token_ids(processor.tokenizer, model))
                behavior = T._sampling_logprobs(
                    behavior_sampling_logits, sample["completion_token_ids"],
                    sample["temperature"], sample["top_p"], 0)
                # Identical adapter files imply identical reference logits.
                if actual_reference == actual_adapter:
                    raw_reference = behavior_raw.detach()
                else:
                    policy.set_adapter("ref")
                    reference_logits = _reader_teacher_forced_completion_logits(
                        policy, processor, sample,
                        sample["completion_token_ids"], 0, processor_kwargs)
                    raw_reference = T._selected_logprobs(
                        reference_logits, sample["completion_token_ids"], 0)
                raw_old = torch.tensor(
                    sample["old_raw_logprobs"], dtype=torch.float32, device=0)
                behavior_old = torch.tensor(
                    sample["old_logprobs"], dtype=torch.float32, device=0)
                max_raw_diff = max(
                    max_raw_diff,
                    float((behavior_raw - raw_old).abs().max().item()))
                max_behavior_diff = max(
                    max_behavior_diff,
                    float((behavior - behavior_old).abs().max().item()))
                references.append(raw_reference.detach())
                old_behavior.append(behavior_old.detach())
        print("Reader replay gate raw=%s behavior=%s" % (
            max_raw_diff, max_behavior_diff), flush=True)
    replay_limit = float(training["max_served_recompute_abs_diff"])
    if not T.replay_gate_passes(max_raw_diff, max_behavior_diff, replay_limit):
        raise RuntimeError(
            "Reader adapter does not reproduce rollouts: raw=%s behavior=%s"
            % (max_raw_diff, max_behavior_diff))
    if cached is None:
        cache = {
            "schema": REPLAY_CACHE_SCHEMA,
            "key": cache_key,
            "binding": cache_binding,
            "max_raw_diff": max_raw_diff,
            "max_behavior_diff": max_behavior_diff,
            "references": [value.detach().cpu().tolist()
                           for value in references],
            "old_behavior": [value.detach().cpu().tolist()
                             for value in old_behavior],
        }
        cache["checksum"] = _replay_cache_checksum(cache)
        temporary = cache_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, cache_path)

    losses, epoch_losses, batch_records = [], [], []
    optimizer_steps = 0
    token_totals, gradient_norms = {}, []
    collection = config["collection"]
    batch_layout = plan_reader_batches(
        local, int(collection["batch_size"]),
        int(collection["expected_batches"]),
        int(collection["min_task_variable_per_batch"]),
        int(collection["max_task_variable_per_batch"]),
        str(collection["batch_order_salt"]))
    started_unix = time.time()
    for epoch_index in range(int(training["epochs"])):
        epoch_values = []
        for batch_index, group_ids in enumerate(batch_layout, start=1):
            optimizer.zero_grad(set_to_none=True)
            selected_groups = set(group_ids)
            batch_pairs = [
                (sample, reference, old)
                for sample, reference, old in zip(local, references, old_behavior)
                if str(sample["group_id"]) in selected_groups
            ]
            weight_reader_batch(
                local, group_ids, group_size=int(collection["group_size"]))
            batch_stats = {}
            batch_values = []
            for sample_index, (sample, reference, old) in enumerate(
                    batch_pairs, 1):
                print("Reader train epoch %d/%d batch %d/%d sample %d/%d" % (
                    epoch_index + 1, int(training["epochs"]),
                    batch_index, len(batch_layout),
                    sample_index, len(batch_pairs)), flush=True)
                policy.set_adapter("default")
                loss_value = _reader_train_sample_teacher_forced(
                    policy, processor, sample, reference, old, training,
                    processor_kwargs, stats=batch_stats)
                losses.append(loss_value)
                epoch_values.append(loss_value)
                batch_values.append(loss_value)
                torch.cuda.empty_cache()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                float(training.get("max_grad_norm") or 1.0)))
            gradient_norms.append(grad_norm)
            optimizer.step()
            optimizer_steps += 1
            T.merge_token_stats(token_totals, batch_stats)
            batch_records.append({
                "epoch": epoch_index + 1,
                "batch": batch_index,
                "groups": list(group_ids),
                "samples": len(batch_pairs),
                "loss_mean": (
                    sum(batch_values) / len(batch_values)
                    if batch_values else None),
                "gradient_norm": grad_norm,
                "token_stats": T.summarize_token_stats(batch_stats),
            })
        epoch_losses.append(epoch_values)

    os.makedirs(out, exist_ok=True)
    parameter_movement = (
        _parameter_delta(parameters_before, parameters)
        if no_save else {"max_abs": None, "relative_l2": None})
    if no_save:
        if not T._parameter_movement_valid(parameter_movement):
            raise RuntimeError("Reader update did not change trainable parameters")
        checkpoint = ""
        adapter_sha = ""
    else:
        checkpoint = os.path.join(out, "batch-01")
        os.makedirs(checkpoint, exist_ok=True)
        policy.set_adapter("default")
        policy.save_pretrained(checkpoint, selected_adapters=["default"])
        adapter_sha = _sha256(os.path.join(checkpoint, "adapter_model.safetensors"))
        if adapter_sha == actual_adapter:
            raise RuntimeError("Reader update did not change the adapter")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "role": "reader",
        "advantage_mode": str(
            config["collection"].get("advantage_mode") or "task_cost"),
        "optimization_reward": "task plus reader cost",
        "report": report,
        "model": os.path.abspath(model_path),
        "model_sha256": actual_model,
        "init_adapter": os.path.abspath(init_adapter),
        "init_adapter_sha256": actual_adapter,
        "behavior_adapter": os.path.abspath(init_adapter),
        "behavior_adapter_sha256": actual_adapter,
        "reference_adapter": os.path.abspath(reference_adapter),
        "reference_adapter_sha256": actual_reference,
        "adapter_sha256": adapter_sha,
        "checkpoint": checkpoint,
        "no_save": bool(no_save),
        "allow_diagnostic": bool(allow_diagnostic),
        "diagnostic_only_groups": diagnostic_only_groups,
        "parameter_delta": parameter_movement,
        "max_len": max_len,
        "max_prompt_completion_tokens": max_tokens,
        "epochs": int(training["epochs"]),
        "optimizer_steps": optimizer_steps,
        "local_optimizer_steps": optimizer_steps,
        "expected_optimizer_steps": (
            len(batch_layout) * int(training["epochs"])),
        "batch_layout": batch_layout,
        "training_batches": batch_records,
        "token_stats": T.summarize_token_stats(token_totals),
        "gradient_norms": gradient_norms,
        "trainable_parameters": int(sum(
            parameter.numel() for parameter in policy.parameters()
            if parameter.requires_grad)),
        "gpu_count": 1,
        "elapsed_seconds": round(time.time() - started_unix, 6),
        "gpu_hours": round((time.time() - started_unix) / 3600.0, 6),
        "group_advantages": (report or {}).get("group_advantages") or {},
        "losses": losses,
        "epoch_losses": epoch_losses,
        "max_raw_old_abs_diff": max_raw_diff,
        "max_behavior_old_abs_diff": max_behavior_diff,
        "behavior_old_abs_diff_gate": "hard",
        "replay_forward": REPLAY_FORWARD,
        "replay_cache_hit": bool(replay_cache_hit),
        "replay_cache_key": cache_key,
        "replay_cache_binding": cache_binding,
        "replay_cache_sha256": (
            _sha256(cache_path) if os.path.exists(cache_path) else ""),
        "projected_sequence_forwards": projected_sequence_forwards,
        "max_replay_sequence_forwards": (
            int(sequence_cap) if sequence_cap is not None else None),
        "old_logprobs": "served_sampling_post_warper",
        "vision_trainable": False,
    }
    with open(os.path.join(out, "ONLINE_READER_TRAINING.json"), "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.allow_diagnostic and not args.no_save:
            raise ValueError("diagnostic Reader data requires --no-save")
        report, samples = prepare(
            args.episodes, config,
            allow_diagnostic=args.allow_diagnostic)
    except (OSError, ValueError, KeyError) as error:
        print("online Reader preparation refused: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    train(config, args.episodes, args.model, args.init_adapter, args.out,
          no_save=args.no_save, allow_diagnostic=args.allow_diagnostic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
