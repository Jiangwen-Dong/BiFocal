"""Focused checks for the checked action protocol and call accounting."""

import json
import os
import tempfile
import unittest

from bifocal.harness import actions as A
from bifocal.harness import calls as R


class ScriptedClient:
    def __init__(self):
        self.state = {"n": 0, "frames": 0}
        self.last_finish_reason = "stop"

    def chat(self, text, images=(), max_tokens=8, temperature=0.0, **_):
        self.state["n"] += 1
        self.state["frames"] += len(list(images))
        self.last_request_id = "request-1"
        self.last_call = {"request_id": "request-1", "recovered_action": False}
        return "WHY: open\nOPEN B1", {
            "prompt_tokens": 10, "completion_tokens": 3,
            "finish_reason": "stop"}, 0.01


class HarnessTests(unittest.TestCase):
    def test_typed_actions_round_trip(self):
        self.assertEqual(A.parse_action("OPEN B6").branch, 6)
        self.assertEqual(A.parse_action("READ B6 12-20").window, (12.0, 20.0))
        self.assertEqual(A.parse_action("REVIEW R7").revision, 7)
        submit = A.parse_action('SUBMIT B @ R7 | "a red hat"')
        self.assertEqual((submit.option, submit.revision), ("B", 7))
        self.assertEqual(A.parse_action("ABSTAIN no evidence").verb, "abstain")

    def test_multiple_actions_are_rejected(self):
        with self.assertRaises(A.ActionError):
            A.parse_action('READ B1\nSUBMIT B @ R1 | "quote"')

    def test_calls_are_capped_and_logged(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calls.jsonl")
            ask = R.make_ask(
                ScriptedClient(), calls,
                {"video": "v", "uid": "u", "question": "q",
                 "time_reference": "", "window": [0.0, 1.0]},
                {"calls": 1, "frames": 16, "minutes": 1},
                path, turn_parser=A.action_log,
            )
            ask("prompt", role="turn")
            with self.assertRaises(R.StopRun):
                ask("prompt", role="turn")
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["parsed_action"], "open")

    def test_token_budget_stops_the_question(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "calls.jsonl")
            ask = R.make_ask(
                ScriptedClient(), calls,
                {"video": "v", "uid": "u", "question": "q",
                 "time_reference": "", "window": [0.0, 1.0]},
                {"calls": 10, "frames": 16, "minutes": 1, "tokens": 13},
                path, turn_parser=A.action_log,
            )
            ask("prompt", role="turn")  # 10 prompt + 3 generated = 13 tokens
            self.assertEqual(ask.state["tokens"], 13)
            with self.assertRaises(R.StopRun) as stop:
                ask("prompt", role="turn")
        self.assertEqual(stop.exception.kind, "capped")


if __name__ == "__main__":
    unittest.main()
