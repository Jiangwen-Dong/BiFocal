#!/usr/bin/env python3
"""Plan or run three MLVU rounds, with O trained before Reader each round."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from ..serving import lanes as RS
from ..workflows.manifest import VIDEO_GROUP_QUESTIONS


SCHEMA_VERSION = "mlvu-bifocal-rl-campaign-v1"
DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])
DEFAULT_O_TEMPLATE = "configs/orchestrator.json"
DEFAULT_READER_TEMPLATE = "configs/reader.json"
QUESTIONS_PER_ROUND = 600
VIDEO_GROUPS_PER_ROUND = QUESTIONS_PER_ROUND // VIDEO_GROUP_QUESTIONS


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_sha(path):
    return _sha256(os.path.join(path, "adapter_model.safetensors"))


def _tree_sha256(root):
    digest = hashlib.sha256()
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(name for name in dirs if name != "__pycache__")
        for name in sorted(files):
            if name.endswith((".pyc", ".pyo")):
                continue
            path = os.path.join(folder, name)
            relative = os.path.relpath(path, root)
            digest.update(relative.encode("utf-8") + b"\0")
            digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _divisor_batch_size(groups, requested):
    groups = int(groups)
    requested = int(requested)
    for size in range(min(groups, requested), 0, -1):
        if groups % size == 0:
            return size
    return 1


def _json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, value):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _python(args, items):
    """Prefix a command with the interpreter chosen by `--python`."""
    return [str(args.python), *map(str, items)]


def _collect_command(args, manifest, out, run_id, round_index, phase,
                     o_adapter, o_sha, reader_adapter="", reader_sha=""):
    command = _python(args, [
        "-m", "bifocal.rl.collect_sequence",
        "--manifest", manifest,
        "--mode", "collect",
        "--model", args.model,
        "--model-sha256", args.model_sha256,
        "--adapter", o_adapter,
        "--adapter-sha256", o_sha,
        "--out", out,
        "--run-id", run_id,
        "--questions-per-sequence", str(VIDEO_GROUP_QUESTIONS),
        "--max-calls", args.max_calls,
        "--frame-cap", args.frame_cap,
        "--max-minutes", args.max_minutes,
        "--max-tokens", args.max_tokens,
        "--max-new-tokens", args.max_new_tokens,
        "--turn-max-new-tokens", args.turn_new_tokens,
        "--temperature", args.temperature,
        "--off-menu-probability", args.off_menu_probability,
        "--required-admitted", args.min_groups,
        "--require-gate",
        "--code-commit", args.code_commit,
        "--round-id", "mlvu-round-%d-%s" % (round_index, phase),
        "--card-list-visible",
        "--card-local-visible",
        "--card-global-visible",
        "--global-source", "timeline",
        "--cards-per-scope", "2",
        "--force-card-check",
        "--budget-answer",
    ])
    if phase == "reader":
        command.extend([
            "--reader-policy",
            "--reader-adapter", reader_adapter,
            "--reader-adapter-sha256", reader_sha,
            "--reader-temperature", args.temperature,
        ])
    return command


def _prepare_command(args, template, adapter, groups, out, batch_size):
    return _python(args, [
        "-m", "bifocal.workflows.prepare",
        "--template", template,
        "--init-adapter", adapter,
        "--reference-adapter", adapter,
        "--groups", groups,
        "--batch-size", batch_size,
        "--out", out,
    ])


def _train_command(args, trainer, config, episodes, model, adapter, out,
                   dry_run=False):
    command = _python(args, [
        "-m", ("bifocal.rl.train_reader" if 'reader' in trainer else "bifocal.rl.train_o"),
        "--config", config,
        "--episodes", episodes,
        "--model", model,
        "--init-adapter", adapter,
        "--out", out,
    ])
    if dry_run:
        command.append("--dry-run")
    return command


def _distributed_command(args, trainer, config, episodes, adapter, out):
    training = _json(config)["training"]
    world_size = int(training["world_size"])
    direct = _train_command(
        args, trainer, config, episodes, args.model, adapter, out)
    if world_size == 1:
        return direct
    return _python(args, [
        "-m", "torch.distributed.run", "--standalone",
        "--nproc_per_node", world_size,
        *direct[1:],
    ])


def _run(command, log_path, cwd):
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    env = os.environ.copy()
    pythonpath = [os.path.join(cwd, "src")]
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        pythonpath + ([current] if current else []))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as handle:
        handle.write("$ " + shlex.join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(command, cwd=cwd, stdout=handle,
                                stderr=subprocess.STDOUT, env=env)
    return {
        "command": command,
        "returncode": int(result.returncode),
        "log": os.path.abspath(log_path),
        "elapsed_seconds": round(time.time() - started, 3),
    }


def _require_success(result, phase):
    if result["returncode"]:
        raise RuntimeError("%s failed with exit %d; see %s" % (
            phase, result["returncode"], result["log"]))


def plan_campaign(args):
    rounds = []
    for index in range(1, int(args.rounds) + 1):
        base = os.path.join(args.run_root, "round-%d" % index)
        rounds.append({
            "round": index,
            "manifest": os.path.join(args.plan_dir, "manifest-round-%d.json" % index),
            "outputs": {
                "o_collect": os.path.join(base, "o", "collect"),
                "o_config": os.path.join(base, "o", "training-config.json"),
                "o_dry_run": os.path.join(base, "o", "training-dry-run"),
                "o_train": os.path.join(base, "o", "training"),
                "reader_collect": os.path.join(base, "reader", "collect"),
                "reader_config": os.path.join(base, "reader", "training-config.json"),
                "reader_train": os.path.join(base, "reader", "training"),
                "record": os.path.join(base, "ROUND.json"),
            },
            "order": [
                "o_collect", "o_config", "o_dry_run", "o_train",
                "reader_collect", "reader_config", "reader_train", "record",
            ],
        })
    return rounds


def _plan_fingerprint(args, o_sha, reader_sha):
    files = [os.path.join(args.plan_dir, "manifest-round-%d.json" % index)
             for index in range(1, int(args.rounds) + 1)]
    payload = {
        "schema": SCHEMA_VERSION,
        "model": os.path.abspath(args.model),
        "model_sha256": args.model_sha256,
        "o_init_adapter_sha256": o_sha,
        "reader_init_adapter_sha256": reader_sha,
        "o_template_sha256": _sha256(args.o_template),
        "reader_template_sha256": _sha256(args.reader_template),
        "manifests_sha256": [_sha256(path) for path in files],
        "code_commit": str(args.code_commit),
        "code_tree_sha256": _tree_sha256(
            os.path.join(args.code_root, "src")),
        "rounds": int(args.rounds),
        "batch_size": int(args.batch_size),
        "min_groups": int(args.min_groups),
        "temperature": float(args.temperature),
        "off_menu_probability": float(args.off_menu_probability),
        "budget": {
            "calls": int(args.max_calls),
            "frames": int(args.frame_cap),
            "minutes": float(args.max_minutes),
            "tokens": int(args.max_tokens),
            "new_tokens": int(args.max_new_tokens),
            "turn_new_tokens": int(args.turn_new_tokens),
        },
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest(), payload


def _validate_plan(plan_dir, rounds):
    plan_path = os.path.join(plan_dir, "plan.json")
    if not os.path.isfile(plan_path):
        raise ValueError("campaign plan is missing: %s" % plan_path)
    plan = _json(plan_path)
    if plan.get("schema_version") != "mlvu-online-rl-plan-v1":
        raise ValueError("unexpected MLVU manifest plan schema")
    if int(plan.get("rounds") or 0) != int(rounds):
        raise ValueError("MLVU plan round count differs from the campaign")
    if int(plan.get("video_group_questions") or 0) != VIDEO_GROUP_QUESTIONS:
        raise ValueError(
            "MLVU plan does not use %d-question video groups"
            % VIDEO_GROUP_QUESTIONS)
    if int(plan.get("target_questions") or 0) != QUESTIONS_PER_ROUND:
        raise ValueError(
            "MLVU campaign requires %d questions per round" % QUESTIONS_PER_ROUND)
    declared = plan.get("manifests") or []
    if len(declared) != int(rounds):
        raise ValueError("MLVU plan does not list every round manifest")
    all_videos = set()
    all_uids = set()
    for index, row in enumerate(declared, start=1):
        path = os.path.join(plan_dir, "manifest-round-%d.json" % index)
        if not os.path.isfile(path):
            raise ValueError("round manifest is missing: %s" % path)
        if os.path.abspath(str(row.get("path") or "")) != os.path.abspath(path):
            raise ValueError("round manifest path differs from the plan")
        if str(row.get("sha256") or "") != _sha256(path):
            raise ValueError("round manifest hash differs from the plan")
        questions = int(row.get("questions") or 0)
        videos = int(row.get("videos") or 0)
        if questions != QUESTIONS_PER_ROUND:
            raise ValueError(
                "round manifest does not contain %d questions" % QUESTIONS_PER_ROUND)
        if videos != VIDEO_GROUPS_PER_ROUND:
            raise ValueError(
                "round manifest does not contain %d video groups"
                % VIDEO_GROUPS_PER_ROUND)
        manifest = _json(path)
        if int(manifest.get("video_group_questions") or 0) != VIDEO_GROUP_QUESTIONS:
            raise ValueError("round manifest video group size differs")
        if int(manifest.get("questions_per_sequence") or 0) != VIDEO_GROUP_QUESTIONS:
            raise ValueError("round manifest question sequence size differs")
        states = manifest.get("train_states") or []
        sequences = manifest.get("train_sequences") or []
        state_videos = [str(state.get("video") or "") for state in states]
        state_uids = [str(state.get("uid") or "") for state in states]
        if "" in state_videos or "" in state_uids:
            raise ValueError("round manifest has an empty video or uid")
        if len(states) != QUESTIONS_PER_ROUND:
            raise ValueError(
                "round manifest state count differs from %d" % QUESTIONS_PER_ROUND)
        if len(set(state_videos)) != VIDEO_GROUPS_PER_ROUND:
            raise ValueError(
                "round manifest does not have %d unique videos"
                % VIDEO_GROUPS_PER_ROUND)
        if len(sequences) != VIDEO_GROUPS_PER_ROUND:
            raise ValueError(
                "round manifest sequence count differs from %d"
                % VIDEO_GROUPS_PER_ROUND)
        if any(len(sequence.get("uids") or ()) != VIDEO_GROUP_QUESTIONS
               for sequence in sequences):
            raise ValueError("round manifest has a wrong-sized video group")
        videos_in_round = set(state_videos)
        uids_in_round = set(state_uids)
        if all_videos & videos_in_round:
            raise ValueError("the same video appears in more than one round")
        if all_uids & uids_in_round:
            raise ValueError("the same question appears in more than one round")
        all_videos.update(videos_in_round)
        all_uids.update(uids_in_round)
    return plan


def _validate_inputs(args):
    if int(args.rounds) != 3:
        raise ValueError("the campaign requires exactly three rounds")
    _validate_plan(args.plan_dir, args.rounds)
    if not os.path.isdir(args.model):
        raise ValueError("model directory is missing: %s" % args.model)
    actual_model = RS.model_content_sha256(args.model)
    if actual_model != args.model_sha256:
        raise ValueError("model content hash differs from --model-sha256")
    for name in ("o_init_adapter", "reader_init_adapter"):
        path = str(getattr(args, name))
        if not os.path.isdir(path):
            raise ValueError("%s is missing: %s" % (name, path))
    if _adapter_sha(args.o_init_adapter) != args.o_init_sha256:
        raise ValueError("O adapter hash differs from --o-init-sha256")
    if _adapter_sha(args.reader_init_adapter) != args.reader_init_sha256:
        raise ValueError("Reader adapter hash differs from --reader-init-sha256")
    floors = []
    for template in (args.o_template, args.reader_template):
        collection = _json(template).get("collection") or {}
        floors.extend([
            int(collection.get("min_task_variable_groups") or 0),
            int(collection.get("min_mixed_action_groups") or 0),
        ])
    required_groups = max(floors or [0])
    if int(args.min_groups) != VIDEO_GROUPS_PER_ROUND:
        raise ValueError(
            "--min-groups must equal the declared %d video groups"
            % VIDEO_GROUPS_PER_ROUND)
    if int(args.min_groups) < required_groups:
        raise ValueError(
            "--min-groups is below the template floor of %d" % required_groups)


def execute_campaign(args):
    _validate_inputs(args)
    o_adapter = os.path.abspath(args.o_init_adapter)
    o_sha = args.o_init_sha256
    reader_adapter = os.path.abspath(args.reader_init_adapter)
    reader_sha = args.reader_init_sha256
    fingerprint, fingerprint_payload = _plan_fingerprint(args, o_sha, reader_sha)
    state_path = os.path.join(args.run_root, "STATE.json")
    if os.path.exists(state_path):
        if not args.resume:
            raise ValueError("run root already has STATE.json; pass --resume")
        state = _json(state_path)
        if state.get("plan_fingerprint") != fingerprint:
            raise ValueError("existing campaign state has a different plan")
        completed = {int(row["round"]) for row in state.get("rounds") or []
                     if row.get("complete")}
        if completed and max(completed) != len(completed):
            raise ValueError("completed rounds are not contiguous")
        o_adapter = str(state["current_o_adapter"])
        o_sha = str(state["current_o_adapter_sha256"])
        reader_adapter = str(state["current_reader_adapter"])
        reader_sha = str(state["current_reader_adapter_sha256"])
        if _adapter_sha(o_adapter) != o_sha:
            raise ValueError("resumed O adapter content differs from STATE.json")
        if _adapter_sha(reader_adapter) != reader_sha:
            raise ValueError("resumed Reader adapter content differs from STATE.json")
    else:
        completed = set()
        state = {
            "schema_version": SCHEMA_VERSION,
            "plan_fingerprint": fingerprint,
            "plan": fingerprint_payload,
            "started_unix": time.time(),
            "current_o_adapter": o_adapter,
            "current_o_adapter_sha256": o_sha,
            "current_reader_adapter": reader_adapter,
            "current_reader_adapter_sha256": reader_sha,
            "rounds": [],
        }
        _write_json(state_path, state)
    plan = plan_campaign(args)
    for round_plan in plan:
        index = int(round_plan["round"])
        if index in completed:
            print("round %d already complete" % index, flush=True)
            continue
        outputs = {name: os.path.abspath(value)
                   for name, value in round_plan["outputs"].items()}
        if any(os.path.exists(path) for path in outputs.values()):
            raise ValueError(
                "round %d has partial outputs; move them away before resume" % index)
        manifest = os.path.abspath(round_plan["manifest"])
        events = []

        o_collect = _collect_command(
            args, manifest, outputs["o_collect"],
            "mlvu-round-%d-o-collect" % index, index, "o",
            o_adapter, o_sha)
        result = _run(o_collect, outputs["o_collect"] + ".log", args.code_root)
        events.append({"phase": "o_collect", **result})
        _require_success(result, "O collection")
        o_report = _json(os.path.join(outputs["o_collect"], "RUN.json"))
        o_groups = int(o_report.get("admitted_sequences") or 0)
        if o_groups != VIDEO_GROUPS_PER_ROUND:
            raise RuntimeError(
                "O collection admitted %d groups; exactly %d are required"
                % (o_groups, VIDEO_GROUPS_PER_ROUND))

        o_batch_size = _divisor_batch_size(o_groups, args.batch_size)
        prepare = _prepare_command(
            args, os.path.abspath(args.o_template), o_adapter,
            o_groups, outputs["o_config"], batch_size=o_batch_size)
        result = _run(prepare, outputs["o_config"] + ".log", args.code_root)
        events.append({"phase": "o_config", **result})
        _require_success(result, "O config")

        dry = _distributed_command(
            args, "train_o", outputs["o_config"],
            os.path.join(outputs["o_collect"], "training-sequences.jsonl"),
            o_adapter, outputs["o_dry_run"])
        dry.append("--dry-run")
        result = _run(dry, outputs["o_dry_run"] + ".log", args.code_root)
        events.append({"phase": "o_dry_run", **result})
        _require_success(result, "O dry-run")

        o_train = _distributed_command(
            args, "train_o", outputs["o_config"],
            os.path.join(outputs["o_collect"], "training-sequences.jsonl"),
            o_adapter, outputs["o_train"])
        result = _run(o_train, outputs["o_train"] + ".log", args.code_root)
        events.append({"phase": "o_train", **result})
        _require_success(result, "O training")
        o_training = _json(os.path.join(
            outputs["o_train"], "ONLINE_O_TRAINING.json"))
        o_adapter = os.path.abspath(o_training["checkpoints"][-1]["checkpoint"])
        o_sha = str(o_training["adapter_sha256"])
        if _adapter_sha(o_adapter) != o_sha:
            raise RuntimeError("O trainer adapter hash differs from its record")

        reader_collect = _collect_command(
            args, manifest, outputs["reader_collect"],
            "mlvu-round-%d-reader-collect" % index, index, "reader",
            o_adapter, o_sha, reader_adapter, reader_sha)
        result = _run(reader_collect, outputs["reader_collect"] + ".log", args.code_root)
        events.append({"phase": "reader_collect", **result})
        _require_success(result, "Reader collection")
        reader_report = _json(os.path.join(outputs["reader_collect"], "RUN.json"))
        reader_groups = int(reader_report.get("admitted_sequences") or 0)
        if reader_groups != VIDEO_GROUPS_PER_ROUND:
            raise RuntimeError(
                "Reader collection admitted %d groups; exactly %d are required"
                % (reader_groups, VIDEO_GROUPS_PER_ROUND))

        reader_batch_size = _divisor_batch_size(
            reader_groups, args.batch_size)
        prepare = _prepare_command(
            args, os.path.abspath(args.reader_template), reader_adapter,
            reader_groups, outputs["reader_config"],
            batch_size=reader_batch_size)
        result = _run(prepare, outputs["reader_config"] + ".log", args.code_root)
        events.append({"phase": "reader_config", **result})
        _require_success(result, "Reader config")

        reader_train = _distributed_command(
            args, "train_reader", outputs["reader_config"],
            os.path.join(outputs["reader_collect"], "training-sequences.jsonl"),
            reader_adapter, outputs["reader_train"])
        result = _run(reader_train, outputs["reader_train"] + ".log", args.code_root)
        events.append({"phase": "reader_train", **result})
        _require_success(result, "Reader training")
        reader_training = _json(os.path.join(
            outputs["reader_train"], "ONLINE_READER_TRAINING.json"))
        reader_adapter = os.path.abspath(reader_training["checkpoint"])
        reader_sha = str(reader_training["adapter_sha256"])
        if _adapter_sha(reader_adapter) != reader_sha:
            raise RuntimeError("Reader trainer adapter hash differs from its record")

        record = {
            "schema_version": "mlvu-bifocal-round-result-v1",
            "round": index,
            "manifest": manifest,
            "manifest_sha256": _sha256(manifest),
            "o_groups": o_groups,
            "o_batch_size": o_batch_size,
            "reader_groups": reader_groups,
            "reader_batch_size": reader_batch_size,
            "o_adapter": o_adapter,
            "o_adapter_sha256": o_sha,
            "reader_adapter": reader_adapter,
            "reader_adapter_sha256": reader_sha,
            "o_collect": os.path.abspath(outputs["o_collect"]),
            "reader_collect": os.path.abspath(outputs["reader_collect"]),
            "o_training": os.path.abspath(outputs["o_train"]),
            "reader_training": os.path.abspath(outputs["reader_train"]),
            "events": events,
            "complete": True,
        }
        _write_json(outputs["record"], record)
        state.update({
            "current_o_adapter": o_adapter,
            "current_o_adapter_sha256": o_sha,
            "current_reader_adapter": reader_adapter,
            "current_reader_adapter_sha256": reader_sha,
        })
        state["rounds"] = [row for row in state.get("rounds") or []
                           if int(row["round"]) != index]
        state["rounds"].append(record)
        state["rounds"].sort(key=lambda row: int(row["round"]))
        _write_json(state_path, state)
        print("round %d complete O=%s Reader=%s" % (
            index, o_sha[:12], reader_sha[:12]), flush=True)
    state["complete"] = True
    state["finished_unix"] = time.time()
    _write_json(state_path, state)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.environ.get(
        "BIFOCAL_ROOT", DEFAULT_ROOT))
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--o-init-adapter", required=True)
    parser.add_argument("--o-init-sha256", required=True)
    parser.add_argument("--reader-init-adapter", required=True)
    parser.add_argument("--reader-init-sha256", required=True)
    parser.add_argument("--o-template", default=DEFAULT_O_TEMPLATE)
    parser.add_argument("--reader-template", default=DEFAULT_READER_TEMPLATE)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-groups", type=int, default=VIDEO_GROUPS_PER_ROUND)
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--frame-cap", type=int, default=256)
    parser.add_argument("--max-minutes", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="per-question cap on prompt plus generated tokens")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--turn-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--off-menu-probability", type=float, default=0.1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--code-commit", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.root = os.path.abspath(args.root)
    args.code_root = args.root
    args.plan_dir = os.path.abspath(args.plan_dir)
    args.run_root = os.path.abspath(args.run_root)
    args.model = os.path.abspath(args.model)
    args.o_init_adapter = os.path.abspath(args.o_init_adapter)
    args.reader_init_adapter = os.path.abspath(args.reader_init_adapter)
    args.o_template = (args.o_template if os.path.isabs(args.o_template)
                       else os.path.join(args.root, args.o_template))
    args.reader_template = (args.reader_template
                            if os.path.isabs(args.reader_template)
                            else os.path.join(args.root, args.reader_template))
    try:
        if args.resume and not args.execute:
            raise ValueError("--resume requires --execute")
        if args.batch_size < 1 or args.min_groups < 1:
            raise ValueError("batch size and minimum groups must be positive")
        if args.execute:
            return execute_campaign(args)
        plan = plan_campaign(args)
        print(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "dry_run": True,
            "rounds": plan,
            "phase_order": (
                "O collect -> O config -> O dry-run -> O train -> "
                "Reader collect -> Reader config -> Reader train -> round record"),
        }, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("MLVU BiFocal RL campaign refused: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
