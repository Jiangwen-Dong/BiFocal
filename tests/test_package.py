"""Checks for the packaged interface and code-only release boundary."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class PackageTests(unittest.TestCase):
    def test_configs_use_qwen3_vl_8b_base(self):
        for name in ("orchestrator.json", "reader.json"):
            config = json.loads((ROOT / "configs" / name).read_text())
            self.assertEqual(config["training"]["base_model"], "Qwen/Qwen3-VL-8B")
            self.assertNotIn("Instruct", json.dumps(config))
        lanes = json.loads((ROOT / "configs" / "role-lanes.json").read_text())
        self.assertEqual(set(lanes["profiles"]), {"single", "split", "split-readers"})

    def test_cli_help_works_without_optional_gpu_dependencies(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "bifocal.cli", "--help"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("collect", result.stdout)
        self.assertIn("train", result.stdout)

    def test_each_command_exposes_help(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        commands = [
            ["prepare", "--help"], ["prepare-config", "--help"],
            ["collect", "--help"],
            ["evaluate", "--help"], ["campaign", "--help"],
            ["serve", "--help"], ["train", "--help"],
            ["train", "--role", "reader", "--help"],
            ["hash-model", "--help"], ["hash-adapter", "--help"],
            ["init-adapter", "--help"], ["compare", "--help"],
        ]
        for command in commands:
            result = subprocess.run(
                [sys.executable, "-m", "bifocal.cli", *command],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(result.returncode, 0, (command, result.stderr))
            self.assertIn("usage:", result.stdout)

    def test_release_has_no_data_model_or_run_payloads(self):
        forbidden = ("data", "models", "runs", "results", "traces",
                     "checkpoints", "weights")
        for path in ROOT.rglob("*"):
            relative = path.relative_to(ROOT)
            self.assertFalse(set(relative.parts) & set(forbidden), relative)
            self.assertNotIn(path.suffix, {".safetensors", ".pt", ".pth",
                                           ".ckpt", ".mp4", ".mov", ".zip"})

    def test_documented_output_filenames_match_code(self):
        from bifocal.rl.collect_sequence import TRAINING_SEQUENCE_FILE
        from bifocal.workflows.manifest import MANIFEST_NAME_FORMAT
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(MANIFEST_NAME_FORMAT % 1, "manifest-round-1.json")
        self.assertEqual(TRAINING_SEQUENCE_FILE, "training-sequences.jsonl")
        self.assertIn(MANIFEST_NAME_FORMAT % 1, readme)
        self.assertIn(TRAINING_SEQUENCE_FILE, readme)

    def test_evaluate_cannot_switch_to_collection_mode(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "bifocal.cli", "evaluate",
             "--mode", "collect"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("evaluate fixes --mode evaluate", result.stderr)

    def test_evaluate_accepts_its_own_mode_explicitly(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        result = subprocess.run(
            [sys.executable, "-m", "bifocal.cli", "evaluate",
             "--mode", "evaluate", "--help"],
            capture_output=True, text=True, env=env,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--mode {collect,evaluate}", result.stdout)

    def test_prepare_config_binds_an_adapter_and_removes_sentinels(self):
        from bifocal.workflows.prepare import main as prepare_config
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            out = root / "bound.json"
            with contextlib.redirect_stdout(io.StringIO()):
                result = prepare_config([
                    "--template", str(ROOT / "configs" / "orchestrator.json"),
                    "--init-adapter", str(adapter),
                    "--reference-adapter", str(adapter),
                    "--groups", "300", "--batch-size", "4", "--out", str(out),
                ])
            self.assertEqual(result, 0)
            config = json.loads(out.read_text())
        training = config["training"]
        self.assertEqual(config["collection"]["expected_groups"], 300)
        self.assertEqual(config["collection"]["expected_batches"], 75)
        self.assertEqual(training["update_batches"], 75)
        self.assertNotEqual(training["init_adapter_sha256"], "REPLACED_BY_PREPARER")
        self.assertNotEqual(
            training["reference_adapter_sha256"], "REPLACED_BY_PREPARER")

    def test_templates_describe_the_paper_round_of_300_groups(self):
        """Each template is a 300-group round: 75 optimizer steps of 4 groups."""
        from bifocal.rl import train_o, train_reader
        from bifocal.workflows.prepare import main as prepare_config
        loaders = {"orchestrator.json": train_o.load_config,
                   "reader.json": train_reader.load_config}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            for name, load in loaders.items():
                template = json.loads((ROOT / "configs" / name).read_text())
                collection = template["collection"]
                self.assertEqual(collection["expected_groups"], 300, name)
                self.assertEqual(collection["batch_size"], 4, name)
                self.assertEqual(collection["expected_batches"], 75, name)
                self.assertEqual(template["training"]["update_batches"], 75, name)
                self.assertEqual(template["training"]["epochs"], 2, name)
                base = ["--template", str(ROOT / "configs" / name),
                        "--init-adapter", str(adapter),
                        "--reference-adapter", str(adapter)]
                # The template alone binds to the paper schedule, and its
                # trainer accepts the bound config.
                out = root / ("bound-" + name)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        prepare_config(base + ["--out", str(out)]), 0, name)
                bound = json.loads(out.read_text())
                self.assertEqual(bound["collection"]["expected_groups"], 300)
                load(str(out))
                # A small corpus is refused rather than silently accepted.
                with contextlib.redirect_stderr(io.StringIO()), \
                        contextlib.redirect_stdout(io.StringIO()):
                    small = prepare_config(
                        base + ["--groups", "16", "--out", str(root / "x.json")])
                self.assertEqual(small, 2, name)

    def test_hash_commands_match_record_algorithms(self):
        import hashlib
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "model"
            (model / "nested").mkdir(parents=True)
            (model / "config.json").write_bytes(b"config")
            (model / "nested" / "weights").write_bytes(b"weights")
            model_digest = hashlib.sha256()
            for path in (model / "config.json", model / "nested" / "weights"):
                model_digest.update(
                    path.relative_to(model).as_posix().encode() + b"\0")
                model_digest.update(path.read_bytes())
            adapter = root / "adapter"
            adapter.mkdir()
            weight = adapter / "adapter_model.safetensors"
            weight.write_bytes(b"adapter")
            adapter_digest = hashlib.sha256(b"adapter").hexdigest()
            model_run = subprocess.run(
                [sys.executable, "-m", "bifocal.cli", "hash-model", str(model)],
                capture_output=True, text=True, env=env,
            )
            adapter_run = subprocess.run(
                [sys.executable, "-m", "bifocal.cli", "hash-adapter", str(adapter)],
                capture_output=True, text=True, env=env,
            )
        self.assertEqual(model_run.returncode, 0, model_run.stderr)
        self.assertEqual(model_run.stdout.strip(), model_digest.hexdigest())
        self.assertEqual(adapter_run.returncode, 0, adapter_run.stderr)
        self.assertEqual(adapter_run.stdout.strip(), adapter_digest)

    def test_hash_command_errors_are_reported_without_tracebacks(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC)
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing")
            for command in ("hash-model", "hash-adapter"):
                result = subprocess.run(
                    [sys.executable, "-m", "bifocal.cli", command, missing],
                    capture_output=True, text=True, env=env,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn("bifocal:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_model_class_requires_a_checkpoint_config(self):
        from bifocal.serving.client import model_class
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "no config.json"):
                model_class(tmp)

    def test_validate_run_checks_json_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "RUN.json").write_text('{"schema_version": "run-v1"}\n')
            (root / "calls.jsonl").write_text('{"call": 1}\n')
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC)
            result = subprocess.run(
                [sys.executable, "-m", "bifocal.cli", "validate-run", str(root)],
                capture_output=True, text=True, env=env,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(report["ok"])
        self.assertEqual(report["json_files_checked"], 2)

    def test_readme_declares_the_framework_figure(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("docs/assets/framework.svg", text)
        asset = ROOT / "docs" / "assets" / "framework.svg"
        self.assertTrue(asset.is_file())
        self.assertGreater(asset.stat().st_size, 1000)


class EvaluateDefaultsTests(unittest.TestCase):
    def test_evaluate_is_greedy_unless_overridden(self):
        from bifocal.cli import evaluate_arguments
        args = evaluate_arguments(["--manifest", "m.json"])
        self.assertEqual(args[:2], ["--mode", "evaluate"])
        for flag in ("--temperature", "--reader-temperature",
                     "--off-menu-probability"):
            self.assertEqual(args[args.index(flag) + 1], "0")
        override = evaluate_arguments(["--temperature", "0.3"])
        self.assertEqual(override.count("--temperature"), 1)
        self.assertEqual(override[override.index("--temperature") + 1], "0.3")

    def test_init_adapter_record_is_zero_initialized_lora(self):
        from types import SimpleNamespace
        from bifocal.rl import init_adapter
        args = SimpleNamespace(role="reader", model="m", model_sha256="a" * 64,
                               out="o", rank=16, alpha=32, dropout=0.05, seed=0)
        record = init_adapter.adapter_record(args, "b" * 64)
        self.assertTrue(record["zero_initialized"])
        self.assertFalse(record["vision_trainable"])
        self.assertEqual(record["target_modules"],
                         ["q_proj", "k_proj", "v_proj", "o_proj"])
        with self.assertRaises(ValueError):
            init_adapter.adapter_record(SimpleNamespace(**{**vars(args), "role": "x"}), "b")

    def test_compare_pairs_runs_and_clusters_by_video(self):
        from bifocal.workflows import compare

        def write(directory, correct):
            os.makedirs(directory)
            ledger, gold = [], []
            for video in range(6):
                for rollout in range(2):
                    row_id = "v%d-r%d" % (video, rollout)
                    ledger.append({"row_id": row_id, "video_id": "v%d" % video,
                                   "question_id": "q", "rollout_index": rollout,
                                   "question_number": 1, "clean": True,
                                   "cost": {"calls_total": 5, "frames_selected": 100}})
                    gold.append({"row_id": row_id, "correct": correct(video, rollout)})
            for name, rows in (("question-ledger.jsonl", ledger), ("eval-only.jsonl", gold)):
                with open(os.path.join(directory, name), "w") as handle:
                    handle.write("\n".join(json.dumps(row) for row in rows) + "\n")

        with tempfile.TemporaryDirectory() as tmp:
            write(os.path.join(tmp, "a"), lambda video, rollout: video % 2 == 0)
            write(os.path.join(tmp, "b"), lambda video, rollout: video % 2 == 0 or video == 1)
            result = compare.compare(os.path.join(tmp, "a"), os.path.join(tmp, "b"),
                                     iterations=200, seed=1)
        self.assertEqual(result["pairs"], 12)
        self.assertEqual(result["videos"], 6)
        self.assertAlmostEqual(result["delta"], 2 / 12, places=5)
        self.assertEqual(result["excluded"], {})
        self.assertLessEqual(result["interval"][0], result["delta"])
        self.assertGreaterEqual(result["interval"][1], result["delta"])

    def test_training_schemas_have_no_smoke_marker(self):
        for name in ("orchestrator.json", "reader.json"):
            config = json.loads((ROOT / "configs" / name).read_text())
            self.assertNotIn("smoke", config["schema_version"])


if __name__ == "__main__":
    unittest.main()
