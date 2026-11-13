"""A small local vLLM client with one fixed lane per call role.

The client speaks the local OpenAI-compatible chat endpoint. It keeps the same
three-value return as ``bifocal.serving.client.LocalClient.chat``: reply, usage and
seconds. A failed lane is retried only on that same lane and never falls back to
another card pair.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request

from . import lanes as R


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
ERROR_DETAIL_CHARS = 1200


def _http_error_detail(error):
    """A bounded server message, kept when a vLLM request fails.

    ``HTTPError`` alone reports only ``HTTP Error 500``, which hides the server's
    JSON detail and makes a failed Reader call hard to diagnose after the run.
    Read at most a small prefix so an error body cannot flood the run log, and
    pull the OpenAI-style error message out when the body has one.
    """
    try:
        raw = error.read(ERROR_DETAIL_CHARS + 1)
    except Exception:
        return str(error)
    text = raw.decode("utf-8", "replace").strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None
    if isinstance(payload, dict):
        detail = payload.get("error", payload)
        if isinstance(detail, dict):
            detail = (detail.get("message") or detail.get("detail")
                      or detail.get("type") or detail)
        if not isinstance(detail, str):
            detail = json.dumps(detail, ensure_ascii=False)
        text = detail
    text = " ".join(str(text or "").split())
    if len(text) > ERROR_DETAIL_CHARS:
        text = text[:ERROR_DETAIL_CHARS] + "...[truncated]"
    return text or "the server returned no error detail"


def _image_data_url(image):
    buffer = io.BytesIO()
    # PNG is lossless, so the vLLM path receives the same pixels as the HF path.
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def content_blocks(text, images):
    """Turn the runner's text and images into ordered chat content blocks.

    The runner writes ``<image>`` markers where the frames belong. Those markers
    are removed and replaced by image blocks, exactly as the Hugging Face client
    does. An image is encoded once for this request.
    """
    text = str(text or "")
    images = list(images or ())
    blocks = []
    if "<image>" in text:
        parts = text.split("<image>")
        for index, piece in enumerate(parts):
            if piece:
                blocks.append({"type": "text", "text": piece})
            if index < len(parts) - 1:
                if index >= len(images):
                    raise R.ServingError("%d <image> markers for %d images"
                                         % (len(parts) - 1, len(images)))
                blocks.append({"type": "image_url",
                               "image_url": {"url": _image_data_url(images[index])}})
        for image in images[max(0, len(parts) - 1):]:
            blocks.append({"type": "image_url",
                           "image_url": {"url": _image_data_url(image)}})
    else:
        if text:
            blocks.append({"type": "text", "text": text})
        for image in images:
            blocks.append({"type": "image_url",
                           "image_url": {"url": _image_data_url(image)}})
    return blocks


def post_json(url, payload, headers=None, timeout=1800):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _sampling_fields(response, choice):
    content = (choice.get("logprobs") or {}).get("content") or []
    return {
        "prompt_token_ids": response.get("prompt_token_ids") or [],
        "completion_token_ids": choice.get("token_ids") or [],
        "completion_logprobs": [item.get("logprob") for item in content],
    }


class RoleClient:
    """Route each role to its declared local lane and record the call trace."""

    backend = "vllm-role"

    def __init__(self, runtime, transport=post_json, max_retries=2, timeout=1800):
        R.validate_config(runtime.get("role_config"))
        self.runtime = runtime
        self.config = runtime["role_config"]
        self.profile = runtime["profile"]
        self.lanes = runtime["lanes"]
        self.transport = transport
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.scheduler = R.LaneScheduler({
            lane: int(entry["max_concurrency"])
            for lane, entry in self.config["lanes"].items()
        })
        self.last_metadata = None

    @classmethod
    def from_runtime(cls, path, **kwargs):
        return cls(R.load_runtime(path), **kwargs)

    def route_for(self, role, owner_chunk=None):
        return R.route_metadata(self.config, self.profile, role, owner_chunk=owner_chunk)

    def _request(self, lane_entry, payload, request_id):
        endpoint = str(lane_entry["endpoint"]).rstrip("/") + "/chat/completions"
        last_error = None
        for retry in range(self.max_retries + 1):
            try:
                response = self.transport(endpoint, payload,
                                          {"X-Request-ID": request_id}, self.timeout)
                return response, retry
            except urllib.error.HTTPError as error:
                detail = _http_error_detail(error)
                last_error = "HTTP %s: %s" % (error.code, detail)
                if error.code not in TRANSIENT_STATUS:
                    raise R.ServingError("vLLM refused %s with HTTP %s: %s"
                                         % (request_id, error.code, detail))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                last_error = error
            if retry < self.max_retries:
                time.sleep(min(2.0, 0.25 * (2 ** retry)))
        raise R.ServingError("vLLM lane failed after %d attempts: %s"
                             % (self.max_retries + 1, last_error))

    def chat(self, text, images=(), max_tokens=512, temperature=0.0, seed=None,
             role="", request_id="", owner_chunk=None, **_):
        route = self.route_for(role, owner_chunk=owner_chunk)
        lane = route["lane"]
        if lane not in self.lanes:
            raise R.ServingError("lane %s has no runtime endpoint" % lane)
        lane_entry = self.lanes[lane]
        payload = {
            "model": self.runtime.get("served_model_name", "qwen3-vl-8b"),
            "messages": [{"role": "user", "content": content_blocks(text, images)}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "logprobs": True,
            "top_logprobs": 1,
            "return_token_ids": True,
            "top_p": 1.0,
            "top_k": -1,
        }
        if seed is not None:
            payload["seed"] = int(seed)

        def request():
            return self._request(lane_entry, payload, request_id)

        trace = self.scheduler.call(lane, request_id, request)
        response, retry = trace["result"]
        try:
            choice = response["choices"][0]
            reply = choice.get("message", {}).get("content") or ""
            finish = choice.get("finish_reason")
            usage = response.get("usage") or {}
        except (KeyError, IndexError, TypeError) as error:
            raise R.ServingError("vLLM returned no choice for %s: %s"
                                 % (request_id, error))
        usage = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": finish,
            "top_p": 1.0,
            "top_k": -1,
            **_sampling_fields(response, choice),
        }
        metadata = {
            "backend": self.backend,
            "lane": lane,
            "card_pair": list(lane_entry.get("card_pair") or []),
            "server_label": lane_entry.get("server_label"),
            "perspective": lane_entry.get("perspective"),
            "request_id": request_id,
            "retry_number": retry,
            "queue_seconds": round(trace["queue_seconds"], 6),
            "request_seconds": round(trace["request_seconds"], 6),
            "role_map_version": route["role_map_version"],
        }
        if owner_chunk is not None:
            metadata["owner_chunk"] = int(owner_chunk)
        self.last_metadata = metadata
        return reply, usage, trace["request_seconds"]

