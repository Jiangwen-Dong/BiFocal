#!/usr/bin/env python3
"""Local Qwen3-VL backend used by the BiFocal harness.

The runner does not know which backend it is talking to: both expose
`chat(text, images, max_tokens, temperature) -> (answer, usage, seconds)`, where
`usage["prompt_tokens"]` is the model's own input length. That is what lets the
the same prompts and records run with the local model or through a serving client.

Two things this does that are easy to forget:

* **the last-position head.** The language-model head projects every position to
  a large vocabulary; at long contexts that matmul can exhaust a card before the
  attention does. Wrapping it to emit only the final position is the difference
  between fitting and not fitting.
* **a measured capacity.** The GPU, not the model's context window, is the limit
  here. `probe_capacity` finds the number of images one call survives by trying,
  rather than by guessing.
"""

import os
import re
import time

from ..rl import credit as MD
from ..paths import model_path

DEFAULT_MODEL = model_path("Qwen3-VL-8B")
CAP = 12845056          # max pixels per image
READER_REPORT_REQUIRED_FIELDS = (
    "BRANCH_SUMMARY", "CAN_ANSWER", "ANSWER", "QUOTE")
READER_POLICY_REQUIRED_FIELDS = (
    "MEMORY_OP",) + READER_REPORT_REQUIRED_FIELDS


def _event_line_complete(text):
    """Whether the generated report has finished its one required EVENT line."""
    return bool(re.search(r"(?m)^EVENT:\s*\S[^\n]*\n", str(text or "")))


def _review_line_complete(text):
    """Whether the generated reply has finished its one required REVIEW line."""
    return bool(re.search(r"(?m)^REVIEW:\s*\S[^\n]*\n", str(text or "")))


def reader_report_fields_complete(
        text, required_fields=READER_REPORT_REQUIRED_FIELDS):
    """Whether every requested Reader field line has a non-empty value."""
    return all(re.search(
        r"(?m)^\s*%s\s*:\s*\S" % re.escape(name), str(text or ""))
        for name in required_fields)


def eos_token_ids(tokenizer, model=None):
    """Return every EOS id generation treats as terminal."""
    values = {getattr(tokenizer, "eos_token_id", None)}
    config = getattr(model, "config", None)
    configured = getattr(config, "eos_token_id", None)
    if isinstance(configured, (list, tuple)):
        values |= set(configured)
    else:
        values.add(configured)
    values.discard(None)
    return {int(value) for value in values}


def _base_model_for_lm_head(model):
    """Return the unwrapped model that owns the language-model head."""
    get_base_model = getattr(model, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else model


def attention_implementation():
    """Which attention kernel to ask for, chosen by what is actually installed.

    `flash_attention_2` is used when `flash-attn` is installed, and `sdpa` otherwise.
    `sdpa` is the PyTorch-native path: slower, no extra wheel, and correct. The
    choice is an environment property, not a result, so it is detected rather
    than assumed, and `BIFOCAL_ATTN` overrides it when a run needs to pin the kernel.
    """
    override = os.environ.get("BIFOCAL_ATTN")
    if override:
        return override
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def configure_sdpa(torch):
    """Choose SDPA kernels by GPU generation.

    Blackwell has no usable Flash SDPA kernel in this PyTorch build, so the
    memory-efficient kernel stays on and math stays off. Ampere on the group
    server has no usable memory-efficient kernel with this wheel, so leave the
    native set enabled instead of forcing a kernel that aborts.
    """
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception:
        major = 0
    if major >= 12:
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(False)
    else:
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)


def model_class(model_dir):
    """The model class the checkpoint's own `config.json` names.

    `architectures[0]` is what the directory declares, and transformers exports
    compatible Qwen vision-language classes exposed by the installed Transformers
    version. A checkpoint whose class is
    not in the installed transformers fails here, by name, instead of as an
    `AttributeError` deep inside a run.
    """
    import json

    path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(path):
        raise RuntimeError("model directory has no config.json: %s" % model_dir)
    with open(path, encoding="utf-8") as handle:
        architectures = json.load(handle).get("architectures") or []
    if not architectures or not isinstance(architectures[0], str):
        raise RuntimeError("model config does not name an architecture: %s" % path)
    name = architectures[0]
    import transformers
    if not hasattr(transformers, name):
        raise RuntimeError(
            "transformers %s has no %s, which %s declares - install a transformers "
            "that exports it (Qwen3-VL arrives in 4.57) in an environment where "
            "that version wins" % (transformers.__version__, name, model_dir))
    return getattr(transformers, name)


def accepts_max_pixels(processor):
    """Whether this processor's image path takes `max_pixels`.

    Qwen3-VL does (it reuses `Qwen2VLImageProcessorFast`),
    but the argument is a *cap*, not the resolution setting: the frames are already
    cut at 768 px, so the cutter is what makes the pin true. Passing an argument a
    processor does not accept would raise on every call, so it is checked once.
    """
    import inspect

    try:
        signature = inspect.signature(processor.image_processor.__call__)
    except (AttributeError, TypeError, ValueError):
        return False
    if "max_pixels" in signature.parameters:
        return True
    return any(parameter.kind == parameter.VAR_KEYWORD
               for parameter in signature.parameters.values())


def generation_kwargs(max_tokens, temperature, seed=None, top_p=None):
    """The `generate` kwargs for one call."""
    kwargs = {
        "max_new_tokens": max_tokens,
        "do_sample": temperature > 0,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_logits": True,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
        kwargs["top_k"] = 0
    return kwargs


def completion_prefix_processor(tokenizer, allowed_completions, prompt_len,
                                history):
    """Build the trie callback for an exact completion menu."""
    trie = {"children": {}, "terminal": False}
    for text in allowed_completions:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if not token_ids:
            raise ValueError("allowed completion tokenizes to nothing")
        node = trie
        for token_id in token_ids:
            node = node["children"].setdefault(
                int(token_id), {"children": {}, "terminal": False})
        node["terminal"] = True
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def allowed_tokens(batch_id, input_ids):
        if int(batch_id) != 0:
            raise ValueError("completion constraints support one sequence")
        generated = input_ids[int(prompt_len):].tolist()
        node = trie
        for token_id in generated:
            child = node["children"].get(int(token_id))
            if child is None:
                raise ValueError("generated token leaves the action menu")
            node = child
        tokens = sorted(node["children"])
        if node["terminal"] and eos_token_id is not None:
            tokens.append(int(eos_token_id))
        if not tokens:
            raise ValueError("action menu has no continuation")
        history.append(tokens)
        return tokens

    return allowed_tokens


def sampling_setup(torch, temperature, seed):
    """Seed the process RNG for a sampled call; a greedy call seeds nothing."""
    if temperature > 0 and seed is not None:
        torch.manual_seed(int(seed))


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *_info):
        return False


class LocalClient:
    """One model, an optional LoRA adapter, and **per-call control over which is answering**.

    O is trained while the Readers stay frozen, and the cheapest honest way to run that is one
    resident model with the O adapter switched on for the calls O makes and off for every other
    call: `use_adapter(True)` before a turn call, `use_adapter(False)` after. `adapter` is the
    LoRA directory of the O adapter; when it is absent the base model answers every call.
    """

    def __init__(self, model_dir=DEFAULT_MODEL, max_pixels=CAP, device="cuda",
                 adapter=None, reader_adapter=None, use_fast=False,
                 load_in_4bit=False, teacher_forced_logprobs=False):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.attn = attention_implementation()
        # **The class comes from the checkpoint, not from this file.** Hardcoding the class
        # meant a second client per model, and the point of the shared interface is
        # that one runner works on both.
        self.model_class = model_class(model_dir)
        if self.attn == "sdpa":
            configure_sdpa(torch)
        self.processor_fast = bool(use_fast)
        self.processor = AutoProcessor.from_pretrained(
            model_dir, use_fast=self.processor_fast)
        self.max_pixels_kwargs = ({"max_pixels": max_pixels}
                                  if accepts_max_pixels(self.processor) else {})
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "attn_implementation": self.attn,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.model = self.model_class.from_pretrained(
            model_dir, **model_kwargs)
        self.model_quantization = "4bit_nf4" if load_in_4bit else "bf16"
        self.teacher_forced_logprobs = bool(teacher_forced_logprobs)
        self.model.eval()
        self.max_pixels = max_pixels

        class LastPosHead(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, hidden):
                return self.inner(hidden[:, -1:, :])

        self.model.lm_head = LastPosHead(self.model.lm_head)

        self.adapter = adapter
        self.reader_adapter = reader_adapter
        self.adapters = {}
        if adapter:
            self.adapters["o"] = adapter
        if reader_adapter:
            self.adapters["reader"] = reader_adapter
        self.adapter_active = False
        self.active_adapter_name = None
        self.last_metadata = None
        if adapter and reader_adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter, adapter_name="o")
            self.model.load_adapter(
                reader_adapter, adapter_name="reader", is_trainable=False)
            self.model.set_adapter("o")
            self.model.eval()
        elif adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(
                self.model, adapter, adapter_name="o")
            self.model.eval()

    def use_adapter(self, on):
        """Backward-compatible boolean toggle for the O adapter."""
        return self.use_adapter_name("o" if on else None)

    def use_adapter_name(self, name):
        """Select one named resident adapter, or the base model for None."""
        name = str(name or "")
        if not name:
            self.adapter_active = False
            self.active_adapter_name = None
            return False
        if name not in self.adapters:
            self.adapter_active = False
            self.active_adapter_name = None
            return False
        self.adapter_active = True
        self.active_adapter_name = name
        self.model.set_adapter(name)
        return True

    def use_adapter_role(self, role, role_adapters):
        """Select the adapter assigned to a role family."""
        role = str(role or "")
        for pattern, name in (role_adapters or {}).items():
            pattern = str(pattern or "")
            if role == pattern or role.startswith(pattern + "-"):
                return self.use_adapter_name(name)
        return self.use_adapter_name(None)

    def _adapter_context(self):
        if not getattr(self, "adapters", None):
            class _Noop:
                def __enter__(self):
                    return None

                def __exit__(self, *_info):
                    return False
            return _Noop()
        if self.adapter_active and self.active_adapter_name:
            self.model.set_adapter(self.active_adapter_name)
            return _NullContext()
        return self.model.disable_adapter()

    def chat(self, text, images=(), max_tokens=512, temperature=0.0, seed=None,
             role="", request_id="", owner_chunk=None,
             stop_at_first_event=False, stop_after_review_line=False,
             top_p=None, no_repeat_ngram_size=None,
             allowed_completions=None, use_cache=None,
             max_seconds=None, **_):
        """One user message, everything inside it.

        Kept because every run so far used it. For a tool loop it is the wrong
        shape - the model's own earlier replies end up inside a *user* message,
        with no assistant turn around them - so `chat_turns` exists."""
        if "<image>" in text and len(images):
            parts = text.split("<image>")
            content = []
            for index, piece in enumerate(parts):
                if piece:
                    content.append({"type": "text", "text": piece})
                if index < len(parts) - 1 and index < len(images):
                    content.append({"type": "image"})
            content += [{"type": "image"} for _ in images[len(parts) - 1:]]
        else:
            content = [{"type": "text", "text": text}]
            content += [{"type": "image"} for _ in images]
        messages = [{"role": "user", "content": content}]
        bounds = {}
        trained_role = role == "turn" or str(role).startswith("loopread")
        if getattr(self, "teacher_forced_logprobs", False) and trained_role:
            bounds["teacher_forced_logprobs"] = True
        if stop_at_first_event:
            bounds["stop_at_first_event"] = True
        if stop_after_review_line:
            bounds["stop_after_review_line"] = True
        if no_repeat_ngram_size is not None:
            bounds["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
        if role.startswith("loopread") and os.environ.get(
                "BIFOCAL_READER_TENSOR_DIR"):
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)
            bounds["tensor_dump"] = os.path.join(
                os.environ["BIFOCAL_READER_TENSOR_DIR"], safe + ".pt")
        if role.startswith("loopread"):
            # PPO replays one full teacher-forced pass. Collection stores the
            # matching teacher-forced raw values when this flag is enabled.
            bounds["teacher_forced_logprobs"] = bool(getattr(
                self, "teacher_forced_logprobs", False))
            # A Reader report is a harness contract. Do not let the model end
            # the turn before every required field line exists.
            reader_adapter_active = bool(
                self.adapter_active
                and getattr(self, "active_adapter_name", None) == "reader")
            bounds["required_fields"] = (
                READER_POLICY_REQUIRED_FIELDS if reader_adapter_active
                else READER_REPORT_REQUIRED_FIELDS)
        generate_kwargs = dict(bounds)
        # Reader generation stays cache-free. PPO replays one teacher-forced
        # full-sequence pass, and collection stores matching raw values when
        # teacher_forced_logprobs is enabled.
        cache_enabled = (False if role.startswith("loopread")
                         else role != "turn"
                         and not (str(role).startswith("review-")
                                  and self.adapter_active))
        generate_kwargs.update({
            "top_p": top_p,
            "use_cache": cache_enabled if use_cache is None else bool(use_cache),
        })
        if allowed_completions is not None:
            generate_kwargs["allowed_completions"] = allowed_completions
        result = self._generate(
            messages, images, max_tokens, temperature, seed,
            max_seconds=max_seconds, **generate_kwargs)
        if allowed_completions is not None:
            result[1]["allowed_action_completions"] = list(allowed_completions)
        self.last_metadata = {
            "backend": "huggingface",
            "lane": "local",
            "request_id": request_id,
            "role": role,
            "adapter_active": self.adapter_active,
            "adapter_name": getattr(self, "active_adapter_name", None) or "",
            "adapter_path": getattr(self, "adapters", {}).get(
                getattr(self, "active_adapter_name", None), ""),
        }
        if owner_chunk is not None:
            self.last_metadata["owner_chunk"] = int(owner_chunk)
        if (result[1] or {}).get("format_stop_reason"):
            self.last_metadata["format_stop_reason"] = result[1]["format_stop_reason"]
        return result

    def next_token_logprobs(self, text, images, words):
        """Log-probabilities of each of `words` as the first token of the reply,
        from one forward pass: a verifier's score without generating. Each word
        must be one token; the last-position head is all this needs.
        Returns (logprobs, prompt_tokens, seconds)."""
        torch = self.torch
        ids = [self.processor.tokenizer.encode(w, add_special_tokens=False) for w in words]
        for w, i in zip(words, ids):
            if len(i) != 1:
                raise ValueError("%r is %d tokens, not one" % (w, len(i)))
        content = []
        parts = text.split("<image>")
        for index, piece in enumerate(parts):
            if piece:
                content.append({"type": "text", "text": piece})
            if index < len(parts) - 1:
                content.append({"type": "image"})
        if len(parts) - 1 != len(images):
            raise ValueError("%d <image> markers for %d images" % (len(parts) - 1, len(images)))
        prompt = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False,
            add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=list(images),
                                return_tensors="pt",
                                **self.max_pixels_kwargs).to(self.model.device)
        started = time.time()
        try:
            with torch.no_grad():
                logits = self.model(**inputs).logits[0, -1].float()
        except Exception:
            torch.cuda.empty_cache()
            raise
        logp = torch.log_softmax(logits, dim=-1)
        return ([float(logp[i[0]]) for i in ids], int(inputs["input_ids"].shape[1]),
                time.time() - started)

    def chat_turns(self, turns, max_tokens=512, temperature=0.0, seed=None, **_):
        """A real conversation: user and assistant turns kept apart.

        `turns` is a list of {"role": ..., "text": ..., "images": [...]}. The
        images of every turn are gathered in order, which is the order the
        processor expects them in when the "<image>" markers are read left to
        right across the whole conversation.
        """
        messages, flat_images = [], []
        for turn in turns:
            text = turn.get("text") or ""
            turn_images = list(turn.get("images") or ())
            flat_images.extend(turn_images)
            if "<image>" in text and turn_images:
                parts = text.split("<image>")
                content = []
                for index, piece in enumerate(parts):
                    if piece:
                        content.append({"type": "text", "text": piece})
                    if index < len(parts) - 1 and index < len(turn_images):
                        content.append({"type": "image"})
                content += [{"type": "image"} for _ in turn_images[len(parts) - 1:]]
            else:
                content = [{"type": "text", "text": text}]
                content += [{"type": "image"} for _ in turn_images]
            messages.append({"role": turn["role"], "content": content})
        return self._generate(messages, flat_images, max_tokens, temperature, seed)

    def _generate(self, messages, images, max_tokens, temperature, seed=None,
                  stop_at_first_event=False, stop_after_review_line=False,
                  top_p=None, use_cache=True, no_repeat_ngram_size=None,
                  tensor_dump=None, teacher_forced_logprobs=False,
                  allowed_completions=None, required_fields=(),
                  max_seconds=None):
        torch = self.torch
        started = time.time()
        prompt = self.processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        if len(images):
            inputs = self.processor(text=[prompt], images=list(images),
                                    return_tensors="pt",
                                    **self.max_pixels_kwargs).to(self.model.device)
        else:
            inputs = self.processor(text=[prompt], return_tensors="pt").to(self.model.device)
        input_len = int(inputs["input_ids"].shape[1])
        if tensor_dump:
            os.makedirs(os.path.dirname(tensor_dump), exist_ok=True)
            torch.save({
                name: value.detach().cpu()
                for name, value in inputs.items()
                if hasattr(value, "detach")
            }, tensor_dump)
        if max_seconds is not None and float(max_seconds) <= 0:
            raise ValueError("max_seconds must be positive")
        timed_out = [False]
        try:
            # `temperature` is passed only when sampling. Some transformers
            # versions reject `temperature=None` alongside `do_sample=False`, and
            # every arm here runs greedy - so the parameter is omitted rather than
            # nulled, which is the portable form.
            kwargs = generation_kwargs(
                max_tokens, temperature, seed, top_p=top_p)
            kwargs["use_cache"] = bool(use_cache)
            allowed_history = []
            logits_processors = []
            if allowed_completions:
                from transformers import (
                    PrefixConstrainedLogitsProcessor)
                logits_processors.append(PrefixConstrainedLogitsProcessor(
                    completion_prefix_processor(
                        self.processor.tokenizer, allowed_completions,
                        input_len, allowed_history),
                    num_beams=1))
            if required_fields:
                from transformers import LogitsProcessor
                if allowed_completions:
                    raise ValueError(
                        "required fields cannot be combined with an action menu")
                tokenizer = self.processor.tokenizer
                stop_ids = eos_token_ids(tokenizer, self.model)

                class CompleteRequiredFields(LogitsProcessor):
                    def __call__(self, input_ids, scores):
                        generated_ids = input_ids[0, input_len:]
                        text = tokenizer.decode(
                            generated_ids, skip_special_tokens=True)
                        if not reader_report_fields_complete(
                                text, required_fields):
                            scores[:, sorted(stop_ids)] = -float("inf")
                        return scores

                logits_processors.append(CompleteRequiredFields())
            if logits_processors:
                from transformers import LogitsProcessorList
                kwargs["logits_processor"] = LogitsProcessorList(
                    logits_processors)
            if no_repeat_ngram_size is not None:
                kwargs["no_repeat_ngram_size"] = int(no_repeat_ngram_size)
            stopping_criteria = []
            if stop_at_first_event or stop_after_review_line or max_seconds:
                from transformers import StoppingCriteria, StoppingCriteriaList
            if stop_at_first_event or stop_after_review_line:
                tokenizer = self.processor.tokenizer
                complete = (_event_line_complete if stop_at_first_event
                            else _review_line_complete)

                class CompleteFormatLine(StoppingCriteria):
                    def __call__(self, input_ids, scores, **_kwargs):
                        generated_ids = input_ids[0, input_len:]
                        text = tokenizer.decode(
                            generated_ids, skip_special_tokens=True)
                        return (complete(text)
                                and (not required_fields
                                     or reader_report_fields_complete(
                                         text, required_fields)))

                stopping_criteria.append(CompleteFormatLine())
            if max_seconds is not None:
                limit = float(max_seconds)

                class Deadline(StoppingCriteria):
                    def __call__(self, input_ids, scores, **_kwargs):
                        expired = time.time() - started >= limit
                        timed_out[0] = timed_out[0] or expired
                        return expired

                stopping_criteria.append(Deadline())
            if stopping_criteria:
                kwargs["stopping_criteria"] = StoppingCriteriaList(
                    stopping_criteria)
            sampling_setup(torch, temperature, seed)
            with torch.no_grad(), self._adapter_context():
                out = self.model.generate(**inputs, **kwargs)
        except Exception:
            # An out-of-memory failure leaves the caching allocator holding
            # fragmented blocks, and the next item -- smaller, and therefore
            # supposed to fit -- can then fail too. Releasing the cache before
            # re-raising keeps one bad item from poisoning the rest of a run.
            torch.cuda.empty_cache()
            raise
        seconds = time.time() - started
        generated = out.sequences[:, input_len:]
        completion_ids = generated[0].tolist()
        if len(out.scores) < len(completion_ids):
            raise RuntimeError("generation returned fewer score steps than tokens")
        if getattr(out, "logits", None) is None:
            raise RuntimeError("generation returned no raw pre-warper logits")
        if len(out.logits) < len(completion_ids):
            raise RuntimeError("generation returned fewer raw logits than tokens")
        completion_logprobs = [
            float(torch.log_softmax(out.scores[index], dim=-1)[
                0, int(token_id)].item())
            for index, token_id in enumerate(completion_ids)
        ]
        completion_raw_logprobs = [
            float(torch.log_softmax(out.logits[index], dim=-1)[
                0, int(token_id)].item())
            for index, token_id in enumerate(completion_ids)
        ]
        generation_raw_logprobs = list(completion_raw_logprobs)
        if teacher_forced_logprobs and completion_ids:
            with torch.no_grad(), self._adapter_context():
                completion_raw_logprobs = self._teacher_forced_raw_logprobs(
                    inputs, completion_ids)
        decoded_answer = self.processor.batch_decode(
            generated, skip_special_tokens=True)[0]
        format_stop_reason = (
            "one_event" if stop_at_first_event
            and _event_line_complete(decoded_answer) else
            "review_line" if stop_after_review_line
            and _review_line_complete(decoded_answer) else "")
        answer = decoded_answer.strip()
        # `finish_reason` is "stop" only when generation ended on an eos token;
        # anything else hit the token cap, so guards that key on truncation can fire.
        stop_ids = eos_token_ids(self.processor.tokenizer, self.model)
        finished = (bool(format_stop_reason) or bool(generated.shape[1])
                    and int(generated[0, -1]) in stop_ids)
        finish_reason = ("timeout" if timed_out[0] else
                         "stop" if finished else "length")
        image_processor = getattr(self.processor, "image_processor", None)
        usage = {"prompt_tokens": input_len,
                 "completion_tokens": int(generated.shape[1]),
                 "finish_reason": finish_reason,
                 "timed_out": bool(timed_out[0]),
                 "max_seconds": (float(max_seconds)
                                 if max_seconds is not None else None),
                 "processor_fast": bool(getattr(self, "processor_fast", False)),
                 "image_processor_class": type(
                     image_processor or self.processor).__name__,
                 "top_p": top_p,
                 "top_k": 0,
                 "no_repeat_ngram_size": no_repeat_ngram_size,
                 "use_cache": bool(use_cache),
                 "logprob_kind": "sampling_post_warper",
                 "raw_logprob_kind": (
                     "teacher_forced_raw_pre_warper"
                     if teacher_forced_logprobs else "raw_pre_warper"),
                 "prompt_token_ids": inputs["input_ids"][0].tolist(),
                 "completion_token_ids": completion_ids,
                 "completion_logprobs": completion_logprobs,
                 "completion_raw_logprobs": completion_raw_logprobs}
        if allowed_completions:
            usage["allowed_token_ids"] = allowed_history
            usage["generation_constraint"] = "action_menu_prefix_v1"
            try:
                MD.parse_action_text(answer)
            except ValueError:
                pass
            else:
                usage.update(MD.annotate_decision_tokens(
                    answer, completion_ids, self.processor.tokenizer,
                    eos_ids=stop_ids))
        if format_stop_reason:
            usage["format_stop_reason"] = format_stop_reason
        if tensor_dump:
            usage["tensor_dump"] = tensor_dump
        if teacher_forced_logprobs:
            usage["generation_completion_raw_logprobs"] = generation_raw_logprobs
        return answer, usage, seconds

    def _teacher_forced_raw_logprobs(self, inputs, completion_ids):
        """Replay completion logits under the call's active adapter."""
        torch = self.torch
        head_model = _base_model_for_lm_head(self.model)
        original_head = head_model.lm_head
        inner_head = getattr(original_head, "inner", None)
        if inner_head is None:
            raise RuntimeError("cannot unwrap serving lm_head for replay")
        completion = torch.tensor(
            [completion_ids], dtype=torch.long,
            device=inputs["input_ids"].device)
        full_ids = torch.cat([
            inputs["input_ids"], completion], dim=1)
        prompt_attention = inputs.get("attention_mask")
        if prompt_attention is None:
            prompt_attention = torch.ones_like(inputs["input_ids"])
        full_attention = torch.cat([
            prompt_attention, torch.ones_like(completion)], dim=1)
        vision = {name: value for name, value in inputs.items()
                  if name not in ("input_ids", "attention_mask")}
        try:
            head_model.lm_head = inner_head
            forward = self.model(
                input_ids=full_ids, attention_mask=full_attention,
                logits_to_keep=len(completion_ids) + 1,
                use_cache=False, **vision)
            teacher_logits = forward.logits[:, :-1, :].float()[0]
            if teacher_logits.shape[0] != len(completion_ids):
                raise RuntimeError(
                    "teacher-forced forward returned %d logits for %d tokens"
                    % (teacher_logits.shape[0], len(completion_ids)))
            return [
                float(torch.log_softmax(teacher_logits[index], dim=-1)[
                    int(token_id)].item())
                for index, token_id in enumerate(completion_ids)]
        finally:
            head_model.lm_head = original_head


def probe_capacity(client, pages, sizes=(8, 12, 16, 20, 24, 32, 40)):
    """Largest number of images one call survives on this card."""
    capacity, observed = 1, []
    for size in sizes:
        if size > len(pages):
            break
        try:
            _, usage, seconds = client.chat("Reply with the single word: ok.",
                                            pages[:size], max_tokens=4)
        except Exception as error:
            observed.append((size, "failed", str(error)[:60]))
            break
        observed.append((size, usage.get("prompt_tokens"), round(seconds, 1)))
        capacity = size
    return capacity, observed
