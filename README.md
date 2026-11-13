<div align="center">

# 🔭 BiFocal

**Evolving Global and Local Memory for Multi-Agent Long-Video Reasoning**

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/model-Qwen3--VL--8B%20base-8A2BE2)](https://github.com/QwenLM/Qwen3-VL)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/badge/release-code--only-lightgrey)](#-what-is-not-included)

</div>

BiFocal is a team of agents that reads a long video under a fixed budget and answers a
multiple-choice question. A team has one **orchestrator (O)** and several **Readers**. Each
Reader inspects one 64-second chunk of the video. The team keeps two kinds of memory:
**local memory** holds the detailed records one Reader wrote, and **global memory** is a
shared, time-ordered ledger that O and the Readers can check.

This archive holds the harness, the memory protocol, the serving boundary, and the
reinforcement learning (RL) code that trains O and the Reader as separate policies. It is
code only. Benchmark data and model weights are not included.

<div align="center">
  <img src="docs/assets/framework.svg" alt="BiFocal framework" width="88%">
</div>

---

## 🌟 Key Features

- 🔭 **Two memory scopes** – Local memory stores append-only records owned by one Reader. Global memory is a video-scoped, chronological ledger shared by the team.
- ✅ **A memory protocol the harness checks** – The harness validates every typed action, compares evidence with stored frames, charges calls and frames, and records each memory item from *exposed* to *cited*.
- 🤖 **Role-specific learning** – O and the Reader each train their own LoRA adapter with group-relative PPO. Both learn from task outcome minus a capped, role-specific call and frame cost.
- 💰 **Fixed budgets** – Every question has caps on calls, frames, time, and tokens. Three levels (B1, B2, B3) are identical for every arm.
- 📏 **Paired comparison** – `bifocal compare` pairs the same question in two runs and bootstraps over whole videos.
- 🧪 **Offline tests** – The test suite needs no GPU, model, or video. It includes an end-to-end run on synthetic data.

---

## 📋 Table of Contents

- [✨ What Is Included](#-what-is-included)
- [🚫 What Is Not Included](#-what-is-not-included)
- [🚀 Quick Start](#-quick-start)
- [💰 Budgets](#-budgets)
- [🧪 Arms and Controls](#-arms-and-controls)
- [🧾 Run Records](#-run-records)
- [🧭 Scope and Limits](#-scope-and-limits)
- [✅ Tests](#-tests)
- [📄 License](#-license)
- [🙏 Third-Party Assets](#-third-party-assets)

---

## ✨ What Is Included

| Component | Description |
|-----------|-------------|
| `src/bifocal/memory/` | Local records, global ledger, claims, versions, retrieval cards, replay |
| `src/bifocal/harness/` | Typed actions, protocol prompts and parsers, budgets, frame extraction |
| `src/bifocal/rl/` | Episodes, rewards, rollout collection, PPO trainers for O and the Reader, campaign driver |
| `src/bifocal/serving/` | Local Transformers client, vLLM client, vLLM role lanes |
| `src/bifocal/workflows/` | MLVU manifests, config binding, paired comparison, run validation |
| `configs/` | O and Reader training templates, role lanes |
| `tests/` | Offline unit and end-to-end tests |
| `docs/assets/` | Framework figure |

- **O (the orchestrator)** reads the question and chooses one typed action at a time, such as `READ` a video branch, `REVIEW` a stored memory item, or `SUBMIT` a checked answer.
- **Readers** inspect video branches. Each Reader writes local records and returns an evidence-bearing report.
- **The harness** is the environment. It runs each action, checks evidence against stored frames, and logs the memory-use stage of every item: exposed, mentioned, opened, checked, accepted or rejected, and cited.
- **RL training** learns separate adapters for O and the shared Reader policy. A training unit is a *video group*: two questions of one video, asked in order, with four independent rollouts.

### 📂 Project Structure

```text
bifocal-code-anon/
├── src/bifocal/
│   ├── memory/       # local and global memory
│   ├── harness/      # typed actions, protocol, frame calls, media
│   ├── rl/           # episodes, rollouts, rewards, PPO, campaign
│   ├── serving/      # model clients and vLLM role lanes
│   └── workflows/    # manifests, config binding, compare, validation
├── configs/          # orchestrator.json, reader.json, role-lanes.json
├── tests/            # offline tests
├── docs/assets/      # framework.svg
├── LICENSE           # MIT
└── pyproject.toml
```

### 🧰 Command-Line Interface

`pip install` adds one command, `bifocal`. Run `bifocal <command> --help` for every option.

| Command | Purpose |
|---------|---------|
| `bifocal init-adapter` | Create the zero-initialized LoRA adapter that training starts from (needs a GPU) |
| `bifocal prepare` | Build three video-disjoint MLVU training manifests |
| `bifocal prepare-config` | Bind a training template to an adapter and a group count |
| `bifocal collect` | Collect rollouts for O or Reader training |
| `bifocal train --role orchestrator\|reader` | One grouped PPO update for one role |
| `bifocal campaign` | Plan or run three rounds, training O and then the Reader in each round |
| `bifocal evaluate` | Run a frozen adapter on an evaluation manifest, greedy by default |
| `bifocal compare` | Paired, video-clustered accuracy difference of two evaluation runs |
| `bifocal validate-run` | Check that a run directory parses and has a `RUN.json` |
| `bifocal serve up\|status\|down` | Manage local vLLM lanes |
| `bifocal hash-model`, `bifocal hash-adapter` | Print the hashes that run records use |

---

## 🚫 What Is Not Included

- Benchmark videos, annotations, selections, or frame caches
- Model weights, adapters, or tokenizer files
- Run logs, traces, checkpoints, or results
- API keys, credentials, or private paths
- Loaders for Video-MME and LongVideoBench, and the builder for evaluation manifests
- Command-line switches for the unrelated-memory, flat-memory, no-grounding-check and no-revision ablations, the single-agent control, and the shared-adapter arm
- Table-generation scripts

Download data and weights outside the repository. The run record names the exact files and
hashes that an experiment used.

---

## 🚀 Quick Start

### 📦 Installation

Requires **Python 3.11 or newer**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[vision,test]'
```

Install from a checkout with `-e`. The `configs/` templates stay in the repository and are
not packaged into a wheel, and the commands below read them from there.

The base install has no dependencies. It is enough for the offline tests and for
`prepare`, `prepare-config`, `compare`, `validate-run` and the hash commands:

```bash
python -m pip install -e '.[test]'
```

Collection, training, `init-adapter` and local evaluation need the `vision` extra and a CUDA GPU.

The experiments ran on RTX 3090 (24 GB) GPUs with this stack. Other versions may work but are untested.

| Package | Version |
|---------|---------|
| Python | 3.12 |
| PyTorch | 2.8.0 (CUDA 12.8) |
| transformers | 4.57.6 |
| peft | 0.21.2 |
| accelerate | 1.14.0 |
| vLLM | 0.11.0 |
| Attention kernel | SDPA. FlashAttention is used only if installed; set `BIFOCAL_ATTN` to pin one. |

### 🧠 Model

The paper uses the **Qwen3-VL-8B base** checkpoint. Model weights are not distributed with
this archive. The 8B checkpoints in public Hugging Face repositories are Instruct and
Thinking, not the base checkpoint. Do not substitute either one for a reported run.

Use the exact base snapshot that your run record pins. The model family page is
<https://github.com/QwenLM/Qwen3-VL>.

### 🗂️ Data

Download benchmarks from their official pages:

- LVBench: <https://huggingface.co/datasets/THUDM/LVBench>
- Video-MME: <https://video-mme.github.io/>
- MLVU: <https://huggingface.co/datasets/MLVU/MVLU> (code: <https://github.com/JUNJIE99/MLVU>)
- LongVideoBench: <https://longvideobench.github.io/>

`bifocal prepare` reads the MLVU development release. Place it so that this layout exists:

```text
$BIFOCAL_DATA/
└── mlvu/
    └── MLVU/
        ├── json/    1_plotQA.json ... 7_topic_reasoning.json
        └── video/   <task folder>/<video>.mp4
```

`prepare` uses the seven multiple-choice tasks. It skips `8_sub_scene` and `9_summary`
because the letter-based answer loop cannot score them. Pass `--data "$BIFOCAL_DATA"`,
not `"$BIFOCAL_DATA/mlvu"`.

### 🔧 Environment

Set paths outside this repository. The code reads the `BIFOCAL_*` names itself.

```bash
export BIFOCAL_MODEL_DIR=/path/to/Qwen3-VL-8B
export BIFOCAL_DATA=/path/to/benchmarks
export BIFOCAL_RUNS=/path/to/runs
export BIFOCAL_EVAL_MANIFEST=/path/to/evaluation-manifest.json   # see "Evaluate"
export MODEL_SHA256="$(bifocal hash-model "$BIFOCAL_MODEL_DIR")"
```

| Variable | Use |
|----------|-----|
| `BIFOCAL_MODEL_DIR` | Default location of the Qwen3-VL-8B base checkpoint |
| `BIFOCAL_DATA` | Benchmark root |
| `BIFOCAL_RUNS` | Run root |
| `BIFOCAL_ROOT` | Directory that holds `configs/`. It defaults to the current directory, and to the repository for `bifocal campaign`. |
| `BIFOCAL_ATTN` | Force an attention kernel, for example `sdpa` |
| `BIFOCAL_ALLOWED_GPUS` | Comma-separated GPU ids that `bifocal serve` may use. The default is ids 0 to 31. |

### 1️⃣ Create the starting adapters

Training starts from a zero-initialized LoRA adapter for each role. Such an adapter leaves
the base model unchanged, so it is also the untrained *fixed prompt policy*. Loading the
model needs a GPU.

```bash
bifocal init-adapter --role orchestrator --model "$BIFOCAL_MODEL_DIR" \
  --model-sha256 "$MODEL_SHA256" --out "$BIFOCAL_RUNS/adapters/orchestrator-init"
bifocal init-adapter --role reader --model "$BIFOCAL_MODEL_DIR" \
  --model-sha256 "$MODEL_SHA256" --out "$BIFOCAL_RUNS/adapters/reader-init"

export O_ADAPTER="$BIFOCAL_RUNS/adapters/orchestrator-init"
export READER_ADAPTER="$BIFOCAL_RUNS/adapters/reader-init"
export O_SHA256="$(bifocal hash-adapter "$O_ADAPTER")"
export READER_SHA256="$(bifocal hash-adapter "$READER_ADAPTER")"
```

Later, point `O_ADAPTER` and `READER_ADAPTER` at newer adapters and recompute their hashes.

### 2️⃣ Plan the training rounds

```bash
bifocal prepare --data "$BIFOCAL_DATA" --out-dir "$BIFOCAL_RUNS/mlvu-plan"
```

This writes `manifest-round-1.json`, `manifest-round-2.json`, `manifest-round-3.json`, and
`plan.json`. Each round holds 600 questions as 300 video groups of two questions. Each video
appears in one round only. The campaign needs this default plan.

`prepare` draws from the whole development release. It does not hold out a validation block.
Build any validation or test manifest separately, and keep it video-disjoint from training.

### 3️⃣ Collect rollouts

Collection and evaluation share one memory protocol. The campaign passes the flags below, so
pass the same flags whenever you compare arms. Standalone defaults differ (for example,
`--cards-per-scope` defaults to 4).

```bash
PROTOCOL=(--card-list-visible --card-local-visible --card-global-visible
          --cards-per-scope 2 --force-card-check --budget-answer)
```

Collect with the O adapter:

```bash
bifocal collect \
  --manifest "$BIFOCAL_RUNS/mlvu-plan/manifest-round-1.json" \
  --mode collect \
  --model "$BIFOCAL_MODEL_DIR" --model-sha256 "$MODEL_SHA256" \
  --adapter "$O_ADAPTER" --adapter-sha256 "$O_SHA256" \
  --out "$BIFOCAL_RUNS/collect/o" --run-id bifocal-o-round-1 \
  --questions-per-sequence 2 --required-admitted 300 --require-gate \
  "${PROTOCOL[@]}"
```

- `--questions-per-sequence 2` matches the two-question video groups from `prepare`. The default of 6 is refused.
- `--required-admitted` is the number of video groups that must pass admission. It defaults to 300, the size of one round.
- `--require-gate` makes the command exit with code 3 if the run fails its gate.
- The output directory gets `RUN.json`, a question ledger, and `training-sequences.jsonl`. The last file holds the rollouts of admitted groups only, so it is absent when no group is admitted.

Collect again for the Reader with `--reader-policy`, a Reader adapter, and the trained O adapter:

```bash
bifocal collect \
  --manifest "$BIFOCAL_RUNS/mlvu-plan/manifest-round-1.json" \
  --mode collect \
  --model "$BIFOCAL_MODEL_DIR" --model-sha256 "$MODEL_SHA256" \
  --adapter "$O_ADAPTER" --adapter-sha256 "$O_SHA256" \
  --reader-policy \
  --reader-adapter "$READER_ADAPTER" --reader-adapter-sha256 "$READER_SHA256" \
  --out "$BIFOCAL_RUNS/collect/reader" --run-id bifocal-reader-round-1 \
  --questions-per-sequence 2 --required-admitted 300 --require-gate \
  "${PROTOCOL[@]}"
```

### 4️⃣ Train one role at a time

Bind a template to the adapter that you train from. The templates describe the paper's
round: 300 video groups, four per optimizer step, so 75 steps per epoch and two epochs per
phase. `--groups` must match the admitted group count in the collection's `RUN.json`, and
the command refuses fewer than 300. `--batch-size` must divide it:

```bash
bifocal prepare-config \
  --template configs/orchestrator.json \
  --init-adapter "$O_ADAPTER" --reference-adapter "$O_ADAPTER" \
  --groups 300 --batch-size 4 \
  --out "$BIFOCAL_RUNS/train/orchestrator.json"

bifocal train --role orchestrator \
  --config "$BIFOCAL_RUNS/train/orchestrator.json" \
  --episodes "$BIFOCAL_RUNS/collect/o/training-sequences.jsonl" \
  --model "$BIFOCAL_MODEL_DIR" \
  --init-adapter "$O_ADAPTER" \
  --out "$BIFOCAL_RUNS/train/o"
```

Do the same for the Reader with `configs/reader.json` and the Reader collection:

```bash
bifocal prepare-config \
  --template configs/reader.json \
  --init-adapter "$READER_ADAPTER" --reference-adapter "$READER_ADAPTER" \
  --groups 300 --batch-size 4 \
  --out "$BIFOCAL_RUNS/train/reader.json"

bifocal train --role reader \
  --config "$BIFOCAL_RUNS/train/reader.json" \
  --episodes "$BIFOCAL_RUNS/collect/reader/training-sequences.jsonl" \
  --model "$BIFOCAL_MODEL_DIR" \
  --init-adapter "$READER_ADAPTER" \
  --out "$BIFOCAL_RUNS/train/reader"
```

Add `--dry-run` to `bifocal train` to run the corpus and reward audits and stop before any
update. The audits are described under [Scope and Limits](#-scope-and-limits).

### 🔁 Run the three-round campaign

`bifocal campaign` chains the steps above for three rounds: O collection, O training, Reader
collection, Reader training. Without `--execute` it only prints the plan.

```bash
bifocal campaign \
  --plan-dir "$BIFOCAL_RUNS/mlvu-plan" --run-root "$BIFOCAL_RUNS/campaign" \
  --model "$BIFOCAL_MODEL_DIR" --model-sha256 "$MODEL_SHA256" \
  --o-init-adapter "$O_ADAPTER" --o-init-sha256 "$O_SHA256" \
  --reader-init-adapter "$READER_ADAPTER" --reader-init-sha256 "$READER_SHA256"
# add --execute to run it, and --resume to continue an interrupted run
```

The campaign needs the default plan of three rounds with 600 questions each. It requires
exactly 300 admitted groups from each collection. Run long jobs in a detached session
(`screen` or `tmux`) and write logs to disk.

### 5️⃣ Evaluate

`bifocal evaluate` runs a frozen adapter on the `dev` or `test` items of an **evaluation
manifest**. The MLVU round manifests above are for training, and this archive does not build
evaluation manifests. `evaluate` expects a frozen, project-generated manifest with the full
online-sequence envelope: benchmark source, annotation hash, layout version, split-state
fields, split item and sequence arrays, and a canonical `manifest_sha256`.

```bash
bifocal evaluate \
  --manifest "$BIFOCAL_EVAL_MANIFEST" --evaluation-split test \
  --model "$BIFOCAL_MODEL_DIR" --model-sha256 "$MODEL_SHA256" \
  --adapter "$O_ADAPTER" --adapter-sha256 "$O_SHA256" \
  --reader-policy \
  --reader-adapter "$READER_ADAPTER" --reader-adapter-sha256 "$READER_SHA256" \
  --out "$BIFOCAL_RUNS/eval" --run-id bifocal-eval \
  --questions-per-sequence 0 "${PROTOCOL[@]}"
```

- `evaluate` decodes greedily (`--temperature 0`, `--reader-temperature 0`) and takes no off-menu step (`--off-menu-probability 0`). Collection samples at temperature 0.8 and explores off-menu with probability 0.1.
- `--questions-per-sequence 0` accepts videos with different numbers of questions, as in the benchmarks. Questions of one video run in order, and memory persists across them.
- Use the same budget flags (see [Budgets](#-budgets)) and the same `PROTOCOL` flags for every arm that you compare.
- Drop `--reader-policy` and the two Reader flags to evaluate O with the base-model Reader.

Check that a run directory parses, then compare two runs:

```bash
bifocal validate-run "$BIFOCAL_RUNS/eval"
bifocal compare --baseline RUN_A --candidate RUN_B
```

`validate-run` is an integrity check, not a scientific audit. `compare` pairs the same
question and rollout, resamples whole videos for a percentile bootstrap interval (10,000
draws by default), and counts every excluded pair by reason. A pair counts only when both
rows are clean and no earlier question of the same video was unclean in either arm.

### 🖥️ Optional: vLLM lanes

`bifocal serve` starts local vLLM lanes. Each lane uses tensor parallelism over two GPUs.
Choose free GPUs on your host first.

```bash
bifocal serve up \
  --profile single --lane-gpus shared=0,1 \
  --config configs/role-lanes.json \
  --runtime "$BIFOCAL_RUNS/vllm-runtime.json" --log-dir "$BIFOCAL_RUNS/vllm" \
  --vllm vllm --model "$BIFOCAL_MODEL_DIR" --model-revision main
bifocal serve status --runtime "$BIFOCAL_RUNS/vllm-runtime.json"
bifocal serve down   --runtime "$BIFOCAL_RUNS/vllm-runtime.json"
```

In this archive, vLLM serves the **base model only**. Use it for diagnostic evaluation:
`bifocal evaluate ... --vllm-runtime "$BIFOCAL_RUNS/vllm-runtime.json" --diagnostic-only`.
The run record marks such traces as excluded from PPO. Rollout collection and the
evaluation of trained adapters use the local Transformers client with PEFT adapters.

Lanes launch with `--disable-mm-preprocessor-cache`, because the vLLM 0.11 multimodal cache
can assert under concurrent Reader calls.

---

## 💰 Budgets

Every question has caps on calls, frames, time, and tokens. Set them with `--max-calls`,
`--frame-cap`, `--max-minutes` and `--max-tokens` on `collect`, `evaluate` and `campaign`.
The token cap counts prompt plus generated tokens over all calls of the question. The
defaults are budget B2. The three levels are identical for every arm:

| Level | `--max-calls` | `--frame-cap` | `--max-minutes` | `--max-tokens` |
|-------|---------------|---------------|-----------------|----------------|
| B1 | 10 | 128 | 1 | 16384 |
| B2 (default) | 20 | 256 | 3 | 32768 |
| B3 | 40 | 512 | 7 | 65536 |

Each read returns at most 32 frames from a 64-second chunk (0.5 frames per second), with a
long side of 768 pixels. `--max-new-tokens` is a separate per-call limit on generated tokens.

---

## 🧪 Arms and Controls

| Arm in the paper | How to run it |
|------------------|---------------|
| Fixed prompt policy | `bifocal init-adapter` for both roles, then `bifocal evaluate` with those adapters |
| BiFocal | Adapters from `bifocal campaign`, then `bifocal evaluate` |
| No memory | Add `--fresh-memory-each-question`: every question starts from an empty tree |
| Shuffled-reward control | `bifocal prepare-config --shuffle-rewards-seed N`: advantages are permuted within each rollout group, so the update carries no reward signal |
| O-only and Reader-only | Train one role with `bifocal train --role ...` and keep the other adapter fixed |

The paper's single-role arms use LoRA rank 32. For such an arm, pass `--rank 32` to
`bifocal init-adapter` and set `training.rank` in the bound config to the same value. The
trainer checks that the config and the adapter agree on rank and alpha.

A multi-agent claim is about allocation under a matched budget. Compare arms on the same
items with the same budget flags, and report calls and frames per arm.

---

## 🧾 Run Records

`RUN.json` in each run directory is the authority for that run. It records the model, hashes,
adapter, benchmark split, frames, calls, prompts, memory-protocol flags, and gate result.
The commands above are launch templates, not claims about a completed run. This README
makes no benchmark-performance claim. Report results only from reviewed, matched-budget
run records.

---

## 🧭 Scope and Limits

The archive implements the checked memory protocol and harness, local and global memory,
task-plus-cost rewards, group-relative PPO for each role, rollout collection, and a
three-round MLVU campaign that trains O and then the Reader in each round.

Properties of the training pipeline that a run depends on:

- A video group is two ordered questions of one video with four independent rollouts. Collection admits a group only if its four rollouts do not all receive the same reward. The campaign requires exactly 300 admitted groups per round (`--min-groups`).
- The trainer has a second, stricter check. Its reward audit requires every group in the corpus to contain rollouts that differ in task outcome. `bifocal prepare-config` sets that floor to the group count (300) in `collection.min_task_variable_groups`, and `bifocal train` exits with code 3 if the corpus falls short. Run `bifocal train --dry-run` to see the result before any update.
- Only task outcome and the role's capped call and frame cost enter the advantage. The evidence, grounding, memory-reuse, progress and protocol terms are computed and logged but zeroed in the optimization reward.
- O's trained tokens are its typed action tokens. The reply of the separate reviewer call is not trained (`include_reviewer_tokens` is `false` in `configs/orchestrator.json`).
- The campaign carries each trained adapter into the next phase. It does not evaluate, select, or roll back adapters. Evaluate each round's adapter with `bifocal evaluate` and `bifocal compare`. A trainer run that fails its own gates stops the campaign.
- `bifocal prepare` builds training manifests only. It does not define the validation block or any test set.

---

## ✅ Tests

```bash
python -m pytest -q
```

The tests are offline. They do not load a model, open a video, or need a GPU. One test
builds a small synthetic MLVU tree and runs `prepare`, `collect` with a scripted client, and
`validate-run` end to end.

---

## 📄 License

BiFocal is released under the [MIT License](LICENSE).

---

## 🙏 Third-Party Assets

`docs/assets/framework.svg` contains icons adapted from Microsoft Fluent Emoji (MIT
license): <https://github.com/microsoft/fluentui-emoji>.
