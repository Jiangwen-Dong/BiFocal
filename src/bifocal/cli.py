"""Command-line entry point for the BiFocal release."""

from __future__ import annotations

import json
import sys

from .serving.lanes import ServingError

COMMANDS = {
    "prepare", "prepare-config", "collect", "evaluate", "train", "campaign",
    "validate-run", "serve", "hash-model", "hash-adapter", "init-adapter",
    "compare",
}

USAGE = """usage: bifocal <command> [options]

commands:
  prepare       build a benchmark manifest
  prepare-config bind a training template to adapters
  collect       collect RL rollout episodes
  evaluate      evaluate a frozen adapter
  train         train one role adapter; requires --role orchestrator|reader
  campaign      run the three-round MLVU campaign
  validate-run  parse and check a run directory
  serve         manage local vLLM role lanes
  hash-model    print the model tree hash used by run records
  hash-adapter  print the adapter weight SHA-256
  init-adapter  create the zero-initialized adapter that training starts from
  compare       paired, video-clustered accuracy difference of two evaluation runs

Run `bifocal <command> --help` for command-specific options.
"""


def _module(module_name, argv):
    from importlib import import_module
    module = import_module(module_name)
    return int(module.main(argv) or 0)


def _take_role(argv):
    output = []
    role = ""
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--role":
            if index + 1 >= len(argv):
                raise ValueError("--role needs orchestrator or reader")
            role = argv[index + 1]
            index += 2
            continue
        if value.startswith("--role="):
            role = value.split("=", 1)[1]
            index += 1
            continue
        output.append(value)
        index += 1
    if role not in {"orchestrator", "reader"}:
        raise ValueError("--role must be orchestrator or reader")
    return role, output


# Test-time decoding is greedy and takes no off-menu exploration step. The campaign
# samples at temperature 0.8 and explores off-menu with probability 0.1 while
# collecting training rollouts; `evaluate` turns both off unless the caller overrides.
EVALUATE_DEFAULTS = (
    ("--temperature", "0"),
    ("--reader-temperature", "0"),
    ("--off-menu-probability", "0"),
)


def evaluate_arguments(rest):
    """Arguments for `evaluate`: fixed mode, greedy decoding, caller flags last."""
    given = {value.split("=", 1)[0] for value in rest if value.startswith("--")}
    defaults = []
    for flag, value in EVALUATE_DEFAULTS:
        if flag not in given:
            defaults += [flag, value]
    return ["--mode", "evaluate", *defaults, *rest]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0 if argv else 2
    command, rest = argv[0], argv[1:]
    if command not in COMMANDS:
        print("bifocal: unknown command %s\n\n%s" % (command, USAGE),
              file=sys.stderr, end="")
        return 2
    try:
        if command == "prepare":
            return _module("bifocal.workflows.manifest", rest)
        if command == "prepare-config":
            return _module("bifocal.workflows.prepare", rest)
        if command == "collect":
            return _module("bifocal.rl.collect_sequence", rest)
        if command == "evaluate":
            mode_values = []
            for index, value in enumerate(rest):
                if value == "--mode" and index + 1 < len(rest):
                    mode_values.append(rest[index + 1])
                elif value.startswith("--mode="):
                    mode_values.append(value.split("=", 1)[1])
            if any(value != "evaluate" for value in mode_values):
                raise ValueError("evaluate fixes --mode evaluate")
            rest = [value for index, value in enumerate(rest)
                    if not (value == "--mode" or value.startswith("--mode=")
                            or (index and rest[index - 1] == "--mode"))]
            return _module("bifocal.rl.collect_sequence",
                           evaluate_arguments(rest))
        if command == "train":
            if any(value in {"-h", "--help"} for value in rest) and not any(
                    value == "--role" or value.startswith("--role=")
                    for value in rest):
                print("usage: bifocal train --role orchestrator|reader [options]")
                return 0
            role, training_args = _take_role(rest)
            module = ("bifocal.rl.train_reader" if role == "reader"
                      else "bifocal.rl.train_o")
            return _module(module, training_args)
        if command == "init-adapter":
            return _module("bifocal.rl.init_adapter", rest)
        if command == "compare":
            return _module("bifocal.workflows.compare", rest)
        if command == "campaign":
            return _module("bifocal.rl.campaign", rest)
        if command == "validate-run":
            if rest in (["-h"], ["--help"]):
                print("usage: bifocal validate-run RUN_DIR")
                return 0
            if len(rest) != 1:
                print("usage: bifocal validate-run RUN_DIR", file=sys.stderr)
                return 2
            from .workflows.validation import validate_run
            print(json.dumps(validate_run(rest[0]), indent=2, sort_keys=True))
            return 0
        if command == "serve":
            return _module("bifocal.serving.vllm_process", rest)
        if command in {"hash-model", "hash-adapter"}:
            if rest in (["-h"], ["--help"]):
                print("usage: bifocal %s PATH" % command)
                return 0
            if len(rest) != 1:
                print("usage: bifocal %s PATH" % command, file=sys.stderr)
                return 2
            if command == "hash-model":
                from .serving.lanes import model_content_sha256
                print(model_content_sha256(rest[0]))
                return 0
            from .serving.lanes import adapter_content_sha256
            print(adapter_content_sha256(rest[0]))
            return 0
    except (OSError, ValueError, KeyError, RuntimeError, ServingError) as error:
        print("bifocal: %s" % error, file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
