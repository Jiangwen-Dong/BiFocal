#!/usr/bin/env python3
"""Bind a training config template to the adapters used by a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--init-adapter", required=True)
    parser.add_argument("--groups", type=int, default=0,
                        help="admitted groups in the collection; default keeps "
                             "the template's 300")
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="groups per optimizer update; default keeps the template's 4")
    parser.add_argument("--reference-adapter", default="")
    parser.add_argument(
        "--shuffle-rewards-seed", type=int, default=None,
        help="shuffled-reward control: permute advantages within each rollout group")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        with open(args.template, encoding="utf-8") as handle:
            config = json.load(handle)
        adapter_file = os.path.join(args.init_adapter, "adapter_model.safetensors")
        digest = sha256_file(adapter_file)
        training = config.setdefault("training", {})
        training["init_adapter"] = os.path.abspath(args.init_adapter)
        training["init_adapter_sha256"] = digest
        reference = args.reference_adapter or args.init_adapter
        reference_file = os.path.join(
            reference, "adapter_model.safetensors")
        reference_digest = sha256_file(reference_file)
        training["reference_adapter"] = os.path.abspath(reference)
        training["reference_adapter_sha256"] = reference_digest
        collection = config.setdefault("collection", {})
        training = config.setdefault("training", {})
        groups = int(args.groups or collection.get("expected_groups") or 0)
        batch_size = int(args.batch_size or collection.get("batch_size") or 0)
        if groups < 1 or batch_size < 1:
            raise ValueError("group and batch sizes must be positive")
        floor = max(int(collection.get("min_task_variable_groups") or 0),
                    int(collection.get("min_mixed_action_groups") or 0))
        if groups < floor:
            raise ValueError(
                "admitted groups %d are below the template floor %d"
                % (groups, floor))
        if groups % batch_size:
            raise ValueError("admitted groups must be divisible by batch size")
        batch_size = min(batch_size, groups)
        layout = [batch_size] * (groups // batch_size)
        collection["expected_groups"] = groups
        collection["batch_size"] = batch_size
        collection["expected_batches"] = len(layout)
        collection["min_task_variable_groups"] = groups
        collection["min_task_variable_per_batch"] = batch_size
        collection["max_task_variable_per_batch"] = batch_size
        training["update_batches"] = len(layout)
        world_size = int(training.get("world_size") or 1)
        if world_size < 1:
            raise ValueError("world size must be positive")
        training["grad_accum"] = max(1, -(-batch_size // world_size))
        training["effective_groups"] = world_size * int(training["grad_accum"])
        if args.shuffle_rewards_seed is not None:
            config.setdefault("training", {})["shuffle_rewards_seed"] = int(
                args.shuffle_rewards_seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(json.dumps({"config": os.path.abspath(args.out),
                          "init_adapter": config["training"]["init_adapter"],
                          "init_adapter_sha256": digest,
                          "reference_adapter": training["reference_adapter"],
                          "reference_adapter_sha256": reference_digest,
                          "group_count": groups,
                          "batch_layout": layout}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print("online config preparation refused: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
