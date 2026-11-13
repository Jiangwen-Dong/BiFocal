"""Wiring check for `bifocal campaign --execute`, with the GPU jobs replaced.

The fake runner writes the files that each phase would produce. The test then checks
the command order, how adapters chain from phase to phase and round to round, the
state file, and `--resume`.
"""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from bifocal.rl import campaign
from bifocal.serving import lanes

TASKS = ("1_plotQA", "2_needle", "3_ego", "4_count", "5_order",
         "6_anomaly_reco", "7_topic_reasoning")
VIDEOS = 900


def write_mlvu(data_root):
    base = Path(data_root) / "mlvu" / "MLVU"
    (base / "json").mkdir(parents=True)
    rows = {task: [] for task in TASKS}
    for index in range(VIDEOS):
        name = "video%d.mp4" % index
        folder = base / "video" / TASKS[index % len(TASKS)]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"placeholder")
        for offset in range(2):
            rows[TASKS[(index + offset) % len(TASKS)]].append({
                "video": name, "duration": 120.0, "question": "q",
                "candidates": ["a", "b"], "answer": "a"})
    for task, task_rows in rows.items():
        (base / "json" / (task + ".json")).write_text(json.dumps(task_rows))


def value(command, flag):
    return command[command.index(flag) + 1]


class FakeRunner:
    """Stands in for `campaign._run` and records every command it receives."""

    def __init__(self):
        self.commands = []
        self.counter = 0

    def adapter(self, directory):
        self.counter += 1
        os.makedirs(directory, exist_ok=True)
        data = ("adapter-%d" % self.counter).encode()
        Path(directory, "adapter_model.safetensors").write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def __call__(self, command, log_path, cwd):
        self.commands.append(list(command))
        module = command[2]
        if module == "bifocal.rl.collect_sequence":
            out = value(command, "--out")
            os.makedirs(out)
            Path(out, "RUN.json").write_text(
                json.dumps({"admitted_sequences": 300}))
        elif module == "bifocal.workflows.prepare":
            Path(value(command, "--out")).write_text(
                json.dumps({"training": {"world_size": 1}}))
        elif module in ("bifocal.rl.train_o", "bifocal.rl.train_reader"):
            if "--dry-run" not in command:
                out = value(command, "--out")
                checkpoint = os.path.join(out, "adapter")
                digest = self.adapter(checkpoint)
                if module.endswith("train_o"):
                    record = {"checkpoints": [{"checkpoint": checkpoint}],
                              "adapter_sha256": digest}
                    name = "ONLINE_O_TRAINING.json"
                else:
                    record = {"checkpoint": checkpoint,
                              "adapter_sha256": digest}
                    name = "ONLINE_READER_TRAINING.json"
                Path(out, name).write_text(json.dumps(record))
        return {"command": command, "returncode": 0, "log": log_path,
                "elapsed_seconds": 0.0}


class CampaignWiringTests(unittest.TestCase):
    def test_execute_chains_phases_rounds_and_resumes(self):
        from bifocal.workflows.manifest import main as prepare
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_mlvu(tmp / "benchmarks")
            plan = tmp / "plan"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(prepare([
                    "--data", str(tmp / "benchmarks"),
                    "--out-dir", str(plan)]), 0)
            model = tmp / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            runner = FakeRunner()
            o_init, reader_init = tmp / "o-init", tmp / "reader-init"
            o_sha = runner.adapter(o_init)
            reader_sha = runner.adapter(reader_init)
            commands = runner.commands
            argv = [
                "--plan-dir", str(plan), "--run-root", str(tmp / "campaign"),
                "--model", str(model),
                "--model-sha256", lanes.model_content_sha256(str(model)),
                "--o-init-adapter", str(o_init), "--o-init-sha256", o_sha,
                "--reader-init-adapter", str(reader_init),
                "--reader-init-sha256", reader_sha,
            ]
            with mock.patch.object(campaign, "_run", runner), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(campaign.main(argv + ["--execute"]), 0)

                modules = [command[2].split(".")[-1] for command in commands]
                one_round = ["collect_sequence", "prepare", "train_o",
                             "train_o", "collect_sequence", "prepare",
                             "train_reader"]
                self.assertEqual(modules, one_round * 3)

                state = json.loads((tmp / "campaign" / "STATE.json").read_text())
                self.assertTrue(state["complete"])
                self.assertEqual([row["round"] for row in state["rounds"]],
                                 [1, 2, 3])

                for index in range(3):
                    block = commands[index * 7:(index + 1) * 7]
                    o_collect, o_dry, o_train = block[0], block[2], block[3]
                    reader_collect, reader_train = block[4], block[6]
                    self.assertIn("--dry-run", o_dry)
                    self.assertNotIn("--dry-run", o_train)
                    self.assertEqual(value(o_collect, "--required-admitted"),
                                     "300")
                    self.assertNotIn("--reader-policy", o_collect)
                    self.assertIn("--reader-policy", reader_collect)
                    self.assertEqual(value(o_collect, "--questions-per-sequence"),
                                     "2")
                    # Reader collection runs with the O adapter just trained.
                    trained_o = state["rounds"][index]["o_adapter"]
                    self.assertEqual(value(reader_collect, "--adapter"), trained_o)
                    self.assertEqual(value(o_dry, "--init-adapter"),
                                     value(o_train, "--init-adapter"))
                    if index == 0:
                        self.assertEqual(value(o_collect, "--adapter"),
                                         str(o_init))
                        self.assertEqual(
                            value(reader_collect, "--reader-adapter"),
                            str(reader_init))
                        self.assertEqual(value(reader_train, "--init-adapter"),
                                         str(reader_init))
                    else:
                        # Each round starts from the previous round's adapters.
                        previous = state["rounds"][index - 1]
                        self.assertEqual(value(o_collect, "--adapter"),
                                         previous["o_adapter"])
                        self.assertEqual(
                            value(reader_collect, "--reader-adapter"),
                            previous["reader_adapter"])

                # A finished run root is protected and can only be resumed.
                before = len(commands)
                self.assertEqual(campaign.main(argv + ["--execute"]), 2)
                self.assertEqual(
                    campaign.main(argv + ["--execute", "--resume"]), 0)
                self.assertEqual(len(commands), before)


if __name__ == "__main__":
    unittest.main()
