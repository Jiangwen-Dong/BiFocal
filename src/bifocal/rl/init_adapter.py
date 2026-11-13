#!/usr/bin/env python3
"""Create a zero-initialized LoRA adapter for one role on the frozen base model.

A zero-initialized adapter leaves the base model unchanged, so it is the starting
policy of training and the fixed prompt policy used as the untrained reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from ..serving import lanes as RS

ROLES = ("orchestrator", "reader")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_SEEDS = {"orchestrator": 20261103, "reader": 20261028}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapter_record(args, adapter_sha256):
    if args.role not in ROLES:
        raise ValueError("unknown adapter role: %s" % args.role)
    return {
        "schema_version": "role-adapter-init-v1",
        "role": str(args.role),
        "base_model": os.path.abspath(args.model),
        "base_model_sha256": str(args.model_sha256),
        "adapter": os.path.abspath(args.out),
        "adapter_sha256": str(adapter_sha256),
        "rank": int(args.rank),
        "alpha": int(args.alpha),
        "dropout": float(args.dropout),
        "seed": int(args.seed),
        "target_modules": list(TARGET_MODULES),
        "vision_trainable": False,
        "zero_initialized": True,
    }


def record_path(out, role):
    return os.path.join(out, "%s_ADAPTER_INIT.json" % role.upper())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None,
                        help="LoRA initialization seed; default depends on --role")
    args = parser.parse_args(argv)
    if args.seed is None:
        args.seed = DEFAULT_SEEDS[args.role]
    if os.path.exists(args.out):
        raise SystemExit("output already exists: %s" % args.out)
    if args.rank <= 0 or args.alpha <= 0 or not 0.0 <= args.dropout < 1.0:
        raise SystemExit("invalid LoRA settings")

    import torch
    from peft import LoraConfig, get_peft_model

    from ..serving import client as LC

    if RS.model_content_sha256(args.model) != args.model_sha256:
        raise SystemExit("base model hash differs from --model-sha256")
    torch.manual_seed(int(args.seed))
    model = LC.model_class(args.model).from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=LC.attention_implementation())
    config = LoraConfig(
        r=int(args.rank), lora_alpha=int(args.alpha),
        lora_dropout=float(args.dropout), target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    if any("lora_B" in name and torch.count_nonzero(parameter).item()
           for name, parameter in model.named_parameters()):
        raise RuntimeError("LoRA adapter is not zero-initialized")
    model.eval()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, selected_adapters=["default"])
    record = adapter_record(
        args, _sha256(os.path.join(args.out, "adapter_model.safetensors")))
    with open(record_path(args.out, args.role), "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
