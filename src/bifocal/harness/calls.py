"""Budgeted, validated model-call adapter used by the BiFocal harness.

This is the small runtime path extracted from the former broad research runner.
It keeps call accounting, frame provenance, turn parsing and role selection.
"""

from __future__ import annotations

import inspect
import json
import os
import time

from . import protocol as C
from . import frames as K
from ..serving import lanes as RS

MAX_NEW_TOKENS = 640
READER_RETRY_MAX_NEW_TOKENS = 192
TOKENS_PER_FRAME = K.TOKENS_PER_FRAME
SLACK = K.SLACK
MAX_SIDE = 768
THINKING_MARKS = ("Thinking Process", "thinking process", "```think", "<think")
READER_ROLES = ("selective", "replan", "all", "verify", "recall", "loopread", "zoom")
READER_ROLE_HEADS = READER_ROLES


def reader_role_head(role):
    """Return the Reader family for a role, or an empty string for O."""
    text = str(role or "")
    for head in sorted(READER_ROLE_HEADS, key=len, reverse=True):
        if text == head or text.startswith(head + "-"):
            return head
    return ""


def adapter_role_enabled(role, adapter_roles):
    role = str(role or "")
    roles = set(adapter_roles or ())
    family = role.split("-", 1)[0]
    return (role in roles or family in roles
            or any(value.startswith(family + "-") for value in roles))


def adapter_family_enabled(family, adapter_roles):
    roles = set(adapter_roles or ())
    return (family in roles
            or any(str(value).startswith(str(family) + "-") for value in roles))



class StopRun(BaseException):
    """A stop that `chunked_team`'s error wrapper must not swallow."""

    def __init__(self, kind, detail=""):
        super().__init__(kind, detail)
        self.kind = kind
        self.detail = detail


class Frames(list):
    """The images handed to a call, plus the manifest of exactly those images."""

    def __init__(self, images, manifest):
        super().__init__(images)
        self.manifest = manifest


def frames_of_factory(cutter, max_side=MAX_SIDE, slack=SLACK):
    """`frames_of(label, first, last, rate, n)` for `chunked_team`.

    `label` is a chunk id or `"coarse"`, and `rate` is the density the frames have
    (the nominal fps for a chunk, the overview's own rate for the coarse read). The
    cutter computes its own rate; what is checked here is what came back.
    """
    def frames_of(label, first, last, rate, n):
        triples = cutter.window(float(first), float(last), K.N_FRAMES)
        if len(triples) < K.N_FRAMES:
            raise StopRun("short_cache",
                          "%s: the window cache holds %d frames, not %d - remove "
                          "the stamp to re-cut" % (label, len(triples), K.N_FRAMES))
        chosen = K.evenly(triples, n)
        if len(chosen) != n:
            raise StopRun("frame_count", "%s: %d frames for a prompt that says %d"
                          % (label, len(chosen), n))
        images, manifest = [], []
        for image, second, md5 in chosen:
            width, height = image.size
            if not (first - slack <= float(second) <= last + slack):
                raise StopRun("frame_window",
                              "%s: a frame's second %.2f is outside %.1f-%.1f"
                              % (label, second, first, last))
            if max(width, height) > max_side:
                scale = float(max_side) / float(max(width, height))
                image = image.resize(
                    (max(1, round(width * scale)), max(1, round(height * scale))))
                width, height = image.size
            images.append(image)
            manifest.append({"second": round(float(second), 3), "md5": md5,
                             "width": width, "height": height})
        return Frames(images, {"label": label, "window": [first, last],
                               "rate": rate, "n": n, "frames": manifest})
    return frames_of


def frames_block(manifest):
    """The `t=Ns <image>` lines `serving client places each image at."""
    lines = ["These are frames from the video, each labelled with the second it "
             "shows."]
    lines += ["t=%ds <image>" % int(round(frame["second"]))
              for frame in manifest["frames"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the call path

def check_prompt(prompt, item):
    """No gold outside the question's own text, **on the items whose gold can leak**."""
    if K.parse_letter(item.get("answer") or ""):
        return
    found = K.gold_in_prompt(prompt, item)
    if found:
        raise StopRun("gold_leak", "%d gold time string(s) in a prompt" % found)


def check_reply(reply, manifest, usage):
    """The frames arrived, and the model did not answer out of a thinking block."""
    attached = len(manifest["frames"]) if manifest else 0
    if attached and usage.get("prompt_tokens", 0) < attached * TOKENS_PER_FRAME:
        raise StopRun("frames_missing",
                      "%d prompt tokens for %d frames - the images did not arrive"
                      % (usage.get("prompt_tokens", 0), attached))
    for mark in THINKING_MARKS:
        if mark in (reply or ""):
            raise StopRun("thinking", "the reply opened a thinking block")


def make_ask(client, calls, item, budget, calls_path, max_new_tokens=MAX_NEW_TOKENS,
             turn_max_new_tokens=None, run_id="", adapter_roles=(), script=None,
             turn_parser=None, policy_top_p=1.0, reader_policy=False,
             role_adapters=None, reader_use_cache=None):
    """The `ask(prompt, images, role)` `chunked_team` is given.

    `budget` is the per-question stop (calls and wall clock); `calls` is the list
    every call is appended to, written to `calls_path` as it goes so a stop still
    leaves the partial log."""
    started = time.time()
    state = {
        "n": 0, "frames": 0, "tokens": 0, "started": started,
        "deadline": started + float(budget.get("minutes") or 0.0) * 60.0,
    }
    chat_parameters = inspect.signature(client.chat).parameters
    chat_takes_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                            for parameter in chat_parameters.values())

    def chat_kwargs(role, request_id, owner_chunk, extra=None):
        names = ("role", "request_id", "owner_chunk")
        values = {name: value for name, value in zip(
            names, (role, request_id, owner_chunk))
                  if chat_takes_kwargs or name in chat_parameters}
        if (reader_role_head(role) == "loopread"
                and (chat_takes_kwargs
                     or "stop_at_first_event" in chat_parameters)):
            values["stop_at_first_event"] = True
        if reader_policy and reader_role_head(role) == "loopread":
            if (chat_takes_kwargs or "no_repeat_ngram_size" in chat_parameters):
                values["no_repeat_ngram_size"] = 6
            if chat_takes_kwargs or "top_p" in chat_parameters:
                values["top_p"] = 1.0
        if (reader_role_head(role) == "loopread"
                and reader_use_cache is not None
                and (chat_takes_kwargs or "use_cache" in chat_parameters)):
            values["use_cache"] = bool(reader_use_cache)
        if (str(role).startswith("review-")
                and (chat_takes_kwargs
                     or "stop_after_review_line" in chat_parameters)):
            values["stop_after_review_line"] = True
        if (extra and "allowed_completions" in extra
                and (chat_takes_kwargs
                     or "allowed_completions" in chat_parameters)):
            values["allowed_completions"] = list(
                extra["allowed_completions"])
        return values

    def ask(prompt, images=(), role="", temperature=0.0, seed=None):
        """One call. `temperature` and `seed` come from the caller; a call that passes none is greedy."""
        # **`Frames` carries the manifest, so it must survive this line.** Coercing
        # to a plain list first silently dropped it, and with it every check that
        # the frames the prompt promises are the frames the model got.
        raw = images if images is not None else ()
        manifest = getattr(raw, "manifest", None) if len(raw) else None
        images = list(raw)
        ask.last_finish_reason = None
        ask.last_role = role
        supplied_extra = dict(getattr(ask, "extra", None) or {})
        owner_chunk = supplied_extra.get("owner_chunk")
        if state["n"] >= budget["calls"]:
            raise StopRun("capped", "the call budget (%d) is spent" % budget["calls"])
        frame_cap = budget.get("frames")
        if frame_cap is not None and state["frames"] + len(images) > int(frame_cap):
            raise StopRun(
                "capped", "the frame budget (%d) would be exceeded" % int(frame_cap))
        token_cap = int(budget.get("tokens") or 0)
        if token_cap and state["tokens"] >= token_cap:
            raise StopRun("capped", "the token budget (%d) is spent" % token_cap)
        elapsed_minutes = (
            time.time() - state["started"]) / 60.0
        if elapsed_minutes > budget["minutes"]:
            raise StopRun("capped", "the time budget (%g min) is spent"
                          % budget["minutes"])
        remaining_seconds = max(
            1.0, float(budget["minutes"]) * 60.0
            - elapsed_minutes * 60.0)
        check_prompt(prompt, item)
        text = prompt if not manifest else frames_block(manifest) + "\n\n" + prompt
        adapter_active = False
        if role_adapters and hasattr(client, "use_adapter_role"):
            adapter_active = client.use_adapter_role(role, role_adapters)
        elif hasattr(client, "use_adapter") and adapter_roles:
            adapter_active = client.use_adapter(
                adapter_role_enabled(role, adapter_roles))
        request_id = "%s:%s:%s:%d" % (run_id, item["uid"], role, state["n"])
        reader_retry = (role.endswith("-retry")
                        and reader_role_head(role) in READER_ROLE_HEADS)
        cap = (turn_max_new_tokens if role == "turn" and turn_max_new_tokens is not None
               else min(max_new_tokens, READER_RETRY_MAX_NEW_TOKENS)
               if reader_retry else max_new_tokens)
        scripted_turn = bool(script is not None and role == "turn" and script)
        if scripted_turn:
            answer, usage, seconds = script.pop(0), {}, 0.0
        else:
            extra_chat = chat_kwargs(
                role, request_id, owner_chunk, supplied_extra)
            policy_call = (role == "turn" or (
                str(role).startswith("review-")
                and adapter_family_enabled("review", adapter_roles)))
            if policy_call and (chat_takes_kwargs or "top_p" in chat_parameters):
                extra_chat["top_p"] = float(policy_top_p)
            if chat_takes_kwargs or "max_seconds" in chat_parameters:
                extra_chat["max_seconds"] = remaining_seconds
            try:
                answer, usage, seconds = client.chat(
                    text, images, max_tokens=cap, temperature=temperature, seed=seed,
                    **extra_chat)
            except RS.ServingError as error:
                raise StopRun("lane", "the serving lane failed: %s" % error)
        state["n"] += 1
        state["frames"] += len(images)
        state["tokens"] += int((usage or {}).get("prompt_tokens") or 0) + int(
            (usage or {}).get("completion_tokens") or 0)
        reply_stop = None
        try:
            check_reply(answer, manifest, usage or {})
        except StopRun as stop:
            reply_stop = stop
        ask.last_finish_reason = (usage or {}).get("finish_reason")
        ask.last_usage = dict(usage or {})
        ask.last_request_id = request_id
        call = {"video": item["video"], "uid": str(item["uid"]), "role": role,
                "request_id": request_id,
                # **Who wrote this call** (see the docstring): the audit flags a file whose
                # lines disagree about their owner, which is the mixture itself.
                "run_id": run_id,
                "video_path": item.get("video_path", ""),
                "frame_dir": item.get("frame_dir", ""),
                "frames": len(images), "manifest": manifest, "prompt": text,
                "scripted_turn": bool(scripted_turn),
                "max_new_tokens": cap,
                "max_seconds": (None if scripted_turn else remaining_seconds),
                "reply": answer, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "finish_reason": ask.last_finish_reason,
                "temperature": temperature, "seed": seed,
                "top_p": (usage or {}).get("top_p"),
                "top_k": (usage or {}).get("top_k"),
                "no_repeat_ngram_size": (usage or {}).get(
                    "no_repeat_ngram_size"),
                "use_cache": (usage or {}).get("use_cache"),
                "adapter_active": bool(adapter_active),
                "prompt_token_ids": (usage or {}).get("prompt_token_ids"),
                "completion_token_ids": (usage or {}).get("completion_token_ids"),
                "completion_logprobs": (usage or {}).get("completion_logprobs"),
                "completion_raw_logprobs": (usage or {}).get(
                    "completion_raw_logprobs"),
                "generation_completion_raw_logprobs": (usage or {}).get(
                    "generation_completion_raw_logprobs"),
                "tensor_dump": (usage or {}).get("tensor_dump", ""),
                "logprob_kind": (usage or {}).get("logprob_kind"),
                "raw_logprob_kind": (usage or {}).get("raw_logprob_kind"),
                "allowed_token_ids": (usage or {}).get("allowed_token_ids"),
                "allowed_action_completions": (usage or {}).get(
                    "allowed_action_completions"),
                "memory_decision_alternative_text": (usage or {}).get(
                    "memory_decision_alternative_text"),
                "memory_decision_revision_id": (usage or {}).get(
                    "memory_decision_revision_id"),
                "memory_decision_token_mask": (usage or {}).get(
                    "memory_decision_token_mask"),
                "memory_decision_chosen_token_ids": (usage or {}).get(
                    "memory_decision_chosen_token_ids"),
                "memory_decision_alternative_token_ids": (usage or {}).get(
                    "memory_decision_alternative_token_ids"),
                "memory_decision_eos_token_ids": (usage or {}).get(
                    "memory_decision_eos_token_ids"),
                "generation_constraint": (usage or {}).get(
                    "generation_constraint"),
                "seconds": round(seconds, 3)}
        if role == "turn":
            if turn_parser is None:
                parsed_turn = C.parse_turn(answer)
                call["action_line_count"] = C.action_line_count(answer)
                call["parsed_action"] = parsed_turn.get("action")
            else:
                parsed_action, action_count = turn_parser(answer)
                call["action_line_count"] = action_count
                call["parsed_action"] = parsed_action
            call["reply_nonempty_lines"] = sum(
                1 for line in str(answer or "").splitlines() if line.strip())
            call["turn_cut"] = (call.get("finish_reason") == "length")
        else:
            call["action_line_count"] = None
            call["parsed_action"] = None
            call["reply_nonempty_lines"] = sum(
                1 for line in str(answer or "").splitlines() if line.strip())
            call["turn_cut"] = False
        if "perspective" not in call:
            call["perspective"] = (
                "reader" if reader_role_head(role) in READER_ROLE_HEADS
                else "orchestrator")
        metadata = getattr(client, "last_metadata", None)
        if scripted_turn:
            metadata = None
        if metadata:
            call.update(metadata)
        call.update(supplied_extra)
        ask.extra = {}
        ask.last_call = call
        cut = ask.last_finish_reason == "length"
        reader_retry_cut = (cut and role.endswith("-retry")
                            and reader_role_head(role) in READER_ROLE_HEADS)
        if reader_retry_cut:
            # A cut repair is one failed repair, not a failed question. The
            # framework records it and keeps the first reader reply.
            call["reader_retry_cut"] = True
        calls.append(call)
        with open(calls_path, "a") as handle:
            handle.write(json.dumps(call) + "\n")
        if (not scripted_turn and getattr(client, "backend", "") == "vllm-role"
                and cut and not reader_retry_cut):
            raise StopRun(
                "backend_cut",
                "%s reply was cut by the vLLM token cap; the call is in the record"
                % role)
        if (not scripted_turn and (usage or {}).get("timed_out")):
            raise StopRun(
                "capped",
                "%s call reached the %g-second remaining time budget; "
                "the call is in the record" % (role, remaining_seconds))
        if reply_stop is not None:
            raise reply_stop
        return answer

    ask.state = state
    ask.run_id = run_id
    def mark_recovered(request_id, recovered_by_request_id):
        records = list(calls)
        if os.path.exists(calls_path):
            with open(calls_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        for call in calls:
            if str(call.get("request_id") or "") == str(request_id):
                call["recovered_action"] = True
                if call.get("finish_reason") == "length":
                    call["recovered_turn_cut"] = True
                call["recovered_by_request_id"] = recovered_by_request_id
        for call in records:
            if str(call.get("request_id") or "") == str(request_id):
                call["recovered_action"] = True
                if call.get("finish_reason") == "length":
                    call["recovered_turn_cut"] = True
                call["recovered_by_request_id"] = recovered_by_request_id
        with open(calls_path, "w", encoding="utf-8") as handle:
            for call in records:
                handle.write(json.dumps(call) + "\n")

    ask.mark_recovered = mark_recovered

    def mark_normalized(request_id, normalized_reply, parsed_action):
        records = list(calls)
        if os.path.exists(calls_path):
            with open(calls_path, encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]
        for call in list(calls) + records:
            if str(call.get("request_id") or "") == str(request_id):
                call["normalized_action"] = True
                call["normalized_reply"] = normalized_reply
                call["parsed_action"] = parsed_action
                call["action_line_count"] = 1
        with open(calls_path, "w", encoding="utf-8") as handle:
            for call in records:
                handle.write(json.dumps(call) + "\n")

    ask.mark_normalized = mark_normalized
    return ask
