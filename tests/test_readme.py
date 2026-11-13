"""The README must describe commands, flags, and defaults that the code accepts."""

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
README = (ROOT / "README.md").read_text(encoding="utf-8")
FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


def help_text(*command):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    result = subprocess.run(
        [sys.executable, "-m", "bifocal.cli", *command, "--help"],
        capture_output=True, text=True, env=env)
    assert result.returncode == 0, (command, result.stderr)
    return result.stdout


def bash_blocks():
    return re.findall(r"```bash\n(.*?)```", README, flags=re.S)


def protocol_flags():
    match = re.search(r"PROTOCOL=\((.*?)\)", README, flags=re.S)
    assert match, "README must define the PROTOCOL array"
    return set(FLAG.findall(match.group(1)))


def documented_commands():
    """Yield (help command, flags) for every `bifocal ...` call in a bash block."""
    for block in bash_blocks():
        text = re.sub(r"\\\n", " ", block)
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            match = re.match(r"bifocal\s+([a-z-]+)(.*)", line)
            if not match:
                continue
            name, rest = match.group(1), match.group(2)
            flags = set(FLAG.findall(rest))
            if '"${PROTOCOL[@]}"' in rest:
                flags |= protocol_flags()
            command = [name]
            if name == "train":
                # `bifocal train` consumes --role before the trainer's parser.
                command += ["--role", rest.split("--role")[1].split()[0]]
                flags.discard("--role")
            elif name == "serve":
                command = ["serve", rest.split()[0]]
            yield command, flags


class ReadmeTests(unittest.TestCase):
    def test_every_documented_flag_exists(self):
        seen = 0
        for command, flags in documented_commands():
            if command[0] == "serve":
                text = subprocess.run(
                    [sys.executable, "-m", "bifocal.serving.vllm_process",
                     command[1], "--help"], capture_output=True, text=True,
                    env={**os.environ, "PYTHONPATH": str(SRC)}).stdout
            else:
                text = help_text(*command)
            known = set(FLAG.findall(text))
            for flag in flags:
                self.assertIn(flag, known, (command, flag))
            seen += 1
        self.assertGreaterEqual(seen, 15)

    def test_documented_commands_cover_the_cli(self):
        from bifocal.cli import COMMANDS
        for name in COMMANDS:
            self.assertIn("bifocal " + name, README, name)

    def test_budget_table_matches_the_defaults(self):
        from bifocal.rl import campaign, collect_sequence
        defaults = {
            "--max-calls": 20, "--frame-cap": 256,
            "--max-minutes": 3.0, "--max-tokens": 32768,
        }
        for parser in (collect_sequence.build_parser(), campaign.build_parser()):
            for flag, expected in defaults.items():
                action = next(a for a in parser._actions
                              if flag in a.option_strings)
                self.assertEqual(action.default, expected, flag)
        row = re.search(r"\| B2 \(default\) \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|",
                        README)
        self.assertEqual(tuple(int(v) for v in row.groups()), (20, 256, 3, 32768))

    def test_license_is_mit_and_declared_everywhere(self):
        text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("MIT License"))
        self.assertIn("Permission is hereby granted, free of charge", text)
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('license = "MIT"', pyproject)
        self.assertIn('license-files = ["LICENSE"]', pyproject)
        self.assertIn("(LICENSE)", README)

    def test_campaign_group_count_matches_the_readme(self):
        from bifocal.rl.campaign import VIDEO_GROUPS_PER_ROUND
        self.assertEqual(VIDEO_GROUPS_PER_ROUND, 300)
        self.assertIn("--groups 300", README)
        self.assertIn("--required-admitted 300", README)

    def test_quick_start_collect_uses_the_prepared_group_size(self):
        from bifocal.workflows.manifest import VIDEO_GROUP_QUESTIONS
        self.assertEqual(VIDEO_GROUP_QUESTIONS, 2)
        self.assertIn("--questions-per-sequence 2", README)
        self.assertIn('--data "$BIFOCAL_DATA" --out-dir', README)

    def test_protocol_flags_match_the_campaign(self):
        from bifocal.rl.campaign import _collect_command

        class Args:
            python = "python"
            model = "m"
            model_sha256 = "s"
            max_calls = max_tokens = frame_cap = max_minutes = 1
            max_new_tokens = turn_new_tokens = temperature = 1
            off_menu_probability = min_groups = 1
            code_commit = ""

        command = _collect_command(Args(), "x", "o", "r", 1, "o", "a", "h")
        campaign_flags = {item for item in command if str(item).startswith("--")}
        self.assertTrue(protocol_flags() <= campaign_flags)
        self.assertEqual(
            command[command.index("--cards-per-scope") + 1], "2")


if __name__ == "__main__":
    unittest.main()
