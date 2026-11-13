"""Deterministic client for offline tests of the harness.

This client is not a model baseline. It exists only to exercise memory rollback,
global promotion and terminal transitions inside a bounded run.
"""

from __future__ import annotations

import re


REVISION = re.compile(r"\bR(\d+)\b")
READ_WINDOW = re.compile(
    r"Read seconds\s+([0-9]+(?:\.[0-9]+)?)\s+to\s+([0-9]+(?:\.[0-9]+)?)")


class ScriptedSmokeClient:
    """Return legal menu replies and a fixed answer-bearing Reader report."""

    backend = "scripted-smoke"
    lane = "deterministic"

    def __init__(self):
        self.last_metadata = {}
        self.last_call = None

    def chat(self, text, images=(), max_tokens=512, temperature=0.0,
             seed=None, role="", request_id="", owner_chunk=None,
             allowed_completions=None, use_cache=None, **_kwargs):
        if str(role).startswith("loopread-"):
            reply = self._reader_reply(text)
        elif str(role).startswith("review-"):
            reply = self._review_reply(role, text)
        elif allowed_completions:
            reply = str(allowed_completions[0])
        else:
            match = re.search(r"Recommended now:\n(ACTION: .+)", str(text))
            reply = match.group(1).strip() if match else "ACTION: ABSTAIN no valid action"
        completion_ids = [1]
        usage = {
            "prompt_tokens": max(100, len(list(images)) * 500 + 100),
            "completion_tokens": 1,
            "finish_reason": "stop",
            "prompt_token_ids": [1],
            "completion_token_ids": completion_ids,
            "completion_logprobs": [-0.1],
            "completion_raw_logprobs": [-0.1],
            "logprob_kind": "scripted_smoke",
            "raw_logprob_kind": "scripted_smoke",
            "use_cache": bool(use_cache),
            "top_p": 1.0,
            "top_k": 0,
        }
        if allowed_completions:
            usage.update({
                "allowed_token_ids": [[1] for _ in completion_ids],
                "allowed_action_completions": list(allowed_completions),
                "generation_constraint": "action_menu_prefix_v1",
            })
        self.last_metadata = {
            "backend": self.backend,
            "lane": self.lane,
            "request_id": request_id,
            "role": role,
            "adapter_active": False,
            "adapter_name": "",
            "adapter_path": "",
            "scripted_smoke": True,
        }
        if owner_chunk is not None:
            self.last_metadata["owner_chunk"] = int(owner_chunk)
        self.last_call = {
            "request_id": request_id,
            "role": role,
            "reply": reply,
            "usage": dict(usage),
        }
        return reply, usage, 0.0

    @staticmethod
    def _reader_reply(text):
        match = READ_WINDOW.search(str(text))
        if match:
            start = int(float(match.group(1)))
            end = int(float(match.group(2)))
        else:
            start, end = 0, 64
        event_end = max(start + 1, min(end, start + 4))
        return (
            "BRANCH_SUMMARY: A dog pulls a sled through snow under a clear "
            "blue sky.\n"
            "CAN_ANSWER: YES\n"
            "ANSWER: B\n"
            "QUOTE: clear blue sky\n"
            "EVENT: %d-%ds | sled moves through snow" % (start, event_end))

    @staticmethod
    def _review_reply(role, text):
        match = REVISION.search(str(role))
        if match is None:
            match = REVISION.search(str(text))
        revision = "R%s" % (match.group(1) if match else "1")
        return 'REVIEW: ACCEPT | %s | B | clear blue sky' % revision
