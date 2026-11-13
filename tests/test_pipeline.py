"""Offline end-to-end check of the documented data path.

The test builds a tiny synthetic MLVU tree, plans training rounds with
`bifocal prepare`, runs `bifocal collect` with the scripted smoke client, and
validates the run directory. It needs no model, video decoder, or GPU: the model
and adapter are placeholder files, and the scripted client answers every call.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TASKS = ("1_plotQA", "2_needle", "3_ego", "4_count", "5_order",
         "6_anomaly_reco", "7_topic_reasoning")


def bifocal(*args):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    return subprocess.run(
        [sys.executable, "-m", "bifocal.cli", *map(str, args)],
        capture_output=True, text=True, env=env)


def write_mlvu(data_root, videos=12):
    """Write the layout `bifocal prepare --data` expects: two questions per video."""
    base = Path(data_root) / "mlvu" / "MLVU"
    (base / "json").mkdir(parents=True)
    rows = {task: [] for task in TASKS}
    for index in range(videos):
        name = "video%d.mp4" % index
        folder = base / "video" / TASKS[index % len(TASKS)]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"placeholder")
        for offset in range(2):
            rows[TASKS[(index + offset) % len(TASKS)]].append({
                "video": name, "duration": 120.0 + index,
                "question": "question %d.%d" % (index, offset),
                "candidates": ["a", "b", "c", "d"], "answer": "b",
            })
    for task, task_rows in rows.items():
        (base / "json" / (task + ".json")).write_text(json.dumps(task_rows))


class PipelineTests(unittest.TestCase):
    def test_prepare_then_scripted_collect_then_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_mlvu(tmp / "benchmarks")
            plan = tmp / "plan"
            result = bifocal("prepare", "--data", tmp / "benchmarks",
                             "--out-dir", plan, "--questions-per-round", 4)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = plan / "manifest-round-1.json"
            self.assertTrue(manifest.is_file())

            model = tmp / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            adapter = tmp / "adapter"
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
            model_sha = bifocal("hash-model", model).stdout.strip()
            adapter_sha = bifocal("hash-adapter", adapter).stdout.strip()
            self.assertEqual(len(model_sha), 64)
            self.assertEqual(len(adapter_sha), 64)

            common = [
                "--manifest", manifest, "--model", model,
                "--model-sha256", model_sha, "--adapter", adapter,
                "--adapter-sha256", adapter_sha,
                "--card-list-visible", "--card-local-visible",
                "--card-global-visible", "--cards-per-scope", 2,
                "--force-card-check", "--budget-answer",
                "--smoke", "--limit-videos", 2,
                "--scripted-smoke-client", "--diagnostic-only",
            ]
            # The default of six questions per sequence does not fit the
            # two-question groups that `prepare` writes.
            wrong = bifocal("collect", *common, "--out", tmp / "wrong",
                            "--run-id", "wrong")
            self.assertEqual(wrong.returncode, 2)
            self.assertIn("different question count", wrong.stderr)

            out = tmp / "collect"
            result = bifocal("collect", *common, "--out", out,
                             "--run-id", "smoke", "--questions-per-sequence", 2)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads((out / "RUN.json").read_text())
            self.assertEqual(record["questions_per_sequence"], 2)
            self.assertEqual(record["videos"], 2)
            self.assertTrue((out / "question-ledger.jsonl").is_file())

            report = json.loads(bifocal("validate-run", out).stdout)
            self.assertTrue(report["ok"])

    def test_prepare_rejects_the_mlvu_subfolder_as_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_mlvu(tmp / "benchmarks")
            result = bifocal("prepare", "--data", tmp / "benchmarks" / "mlvu",
                             "--out-dir", tmp / "plan",
                             "--questions-per-round", 4)
            self.assertEqual(result.returncode, 2)
            self.assertIn("video directory is missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
