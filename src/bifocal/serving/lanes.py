"""Role-to-lane routing for the BiFocal serving profiles.

The module is deliberately offline. It validates the reviewed role map, checks
explicit two-card lanes, and schedules calls without learning anything about the
video, the question or gold. The active runner uses the same functions that the
tests exercise.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time


class ServingError(BaseException):
    """A serving configuration or request that must stop the run."""


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def model_content_sha256(model_dir):
    """sha256 over every regular file in the local model directory."""
    path = os.path.abspath(os.path.expanduser(str(model_dir)))
    if not os.path.isdir(path):
        raise ServingError("model directory does not exist: %s" % path)
    digest = hashlib.sha256()
    files = []
    for root, dirs, names in os.walk(path):
        dirs.sort()
        for name in sorted(names):
            full = os.path.join(root, name)
            if os.path.isfile(full):
                files.append((os.path.relpath(full, path), full))
    if not files:
        raise ServingError("model directory is empty: %s" % path)
    for relative, full in files:
        digest.update(relative.encode("utf-8") + b"\0")
        with open(full, "rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def adapter_content_sha256(path):
    """SHA-256 of an adapter directory's weight file or one weight file."""
    candidate = str(path)
    if os.path.isdir(candidate):
        candidate = os.path.join(candidate, "adapter_model.safetensors")
    if not os.path.isfile(candidate):
        raise ServingError("adapter weights do not exist: %s" % candidate)
    digest = hashlib.sha256()
    with open(candidate, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ServingError("the role config must be an object")
    if not str(config.get("version") or "").strip():
        raise ServingError("the role config has no version")
    profiles = config.get("profiles")
    lanes = config.get("lanes")
    roles = config.get("roles")
    if not isinstance(profiles, dict) or not profiles:
        raise ServingError("the role config has no profiles")
    if not isinstance(lanes, dict) or not lanes:
        raise ServingError("the role config has no lanes")
    expected = {
        "single": ["shared"],
        "split": ["policy", "reader-a"],
        "split-readers": ["policy", "reader-a", "reader-b"],
    }
    for name, names in expected.items():
        got = (profiles.get(name) or {}).get("lanes")
        if got != names:
            raise ServingError("profile %s must name lanes %s" % (name, names))
    for name, lane in lanes.items():
        if not isinstance(lane, dict):
            raise ServingError("lane %s is not an object" % name)
        endpoint = str(lane.get("endpoint") or "").rstrip("/")
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ServingError("lane %s must use a local endpoint" % name)
        if int(lane.get("tensor_parallel_size") or 0) != 2:
            raise ServingError("lane %s must have tensor parallel size 2" % name)
        if int(lane.get("max_concurrency") or 0) < 1:
            raise ServingError("lane %s has no concurrency cap" % name)
        if lane.get("perspective") not in (None, "shared", "orchestrator", "reader"):
            raise ServingError("lane %s has an unknown perspective" % name)
    for profile, entry in profiles.items():
        endpoints = set()
        for name in entry.get("lanes") or ():
            endpoint = str(lanes[name].get("endpoint") or "").rstrip("/")
            if endpoint in endpoints:
                raise ServingError("profile %s gives two lanes the same endpoint: %s"
                                   % (profile, endpoint))
            endpoints.add(endpoint)
    if not isinstance(roles, dict):
        raise ServingError("the role config has no role grammar")
    for key in ("policy_exact", "reader_prefixes", "policy_patterns"):
        if not isinstance(roles.get(key), list):
            raise ServingError("role grammar %s must be a list" % key)
    for pattern in roles["policy_patterns"]:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ServingError("bad policy role pattern %r: %s" % (pattern, error))


def parse_gpu_pair(text, allowed=range(32)):
    try:
        values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    except ValueError:
        raise ServingError("a card pair must be two comma-separated integers")
    if len(values) != 2:
        raise ServingError("a lane needs exactly two cards")
    if len(set(values)) != len(values):
        raise ServingError("a card pair contains a duplicate")
    if any(value not in allowed for value in values):
        raise ServingError("cards must be in the allowed range")
    return values


def parse_lane_pairs(items, profile, config):
    pairs = {}
    for item in items or ():
        if "=" not in item:
            raise ServingError("--lane-gpus must look like lane=card,card")
        lane, text = item.split("=", 1)
        lane = lane.strip()
        if lane in pairs:
            raise ServingError("lane %s was named more than once" % lane)
        pairs[lane] = parse_gpu_pair(text)
    active = list(config["profiles"][profile]["lanes"])
    if sorted(pairs) != sorted(active):
        raise ServingError("profile %s needs explicit pairs for exactly %s; got %s"
                           % (profile, active, sorted(pairs)))
    used = []
    for lane in active:
        used.extend(pairs[lane])
    if len(set(used)) != len(used):
        raise ServingError("two lanes name the same card: %s" % sorted(used))
    return pairs


def _base_role(role):
    role = str(role or "").strip()
    if not role:
        raise ServingError("a call has no role")
    if role.endswith("-retry"):
        role = role[:-6]
        if role.endswith("-retry"):
            raise ServingError("a role may carry only one retry suffix")
    return role


def _numeric_owner(role, owner_chunk=None):
    if owner_chunk is not None:
        try:
            value = int(owner_chunk)
        except (TypeError, ValueError):
            raise ServingError("the owner chunk must be an integer")
        if value < 1:
            raise ServingError("the owner chunk must be positive")
        return value
    tail = str(role).rpartition("-")[2]
    if tail.isdigit():
        return int(tail)
    return None


def lane_for_role(config, profile, role, owner_chunk=None):
    validate_config(config)
    if profile not in config["profiles"]:
        raise ServingError("unknown serving profile %r" % profile)
    base = _base_role(role)
    roles = config["roles"]
    is_policy = (base in roles["policy_exact"] or any(
        re.fullmatch(pattern, base) for pattern in roles["policy_patterns"]))
    prefix, _, tail = base.rpartition("-")
    is_reader = prefix in roles["reader_prefixes"] and tail.isdigit()
    is_handle_verify = prefix == "verify" and tail[:1] == "M" and tail[1:].isdigit()
    if not (is_policy or is_reader or is_handle_verify):
        raise ServingError("the role %r is not declared" % role)
    number = None
    if is_reader or is_handle_verify:
        if is_handle_verify:
            number = _numeric_owner(base, owner_chunk)
            if number is None:
                raise ServingError("handle verify %s has no owning chunk" % role)
        else:
            number = _numeric_owner(base, owner_chunk)
            if number is None:
                raise ServingError("reader role %s has no chunk number" % role)
    if profile == "single":
        return "shared"
    if is_policy:
        return "policy"
    if profile == "split":
        return "reader-a"
    return "reader-a" if number % 2 else "reader-b"


def route_metadata(config, profile, role, owner_chunk=None):
    lane = lane_for_role(config, profile, role, owner_chunk=owner_chunk)
    return {"lane": lane, "role_map_version": str(config["version"])}


class LaneScheduler:
    """One bounded queue per lane, with request-order results."""

    def __init__(self, max_concurrency):
        self.max_concurrency = dict(max_concurrency)
        self._sems = {lane: threading.BoundedSemaphore(int(cap))
                      for lane, cap in self.max_concurrency.items()}
        self._active = set()
        self._lock = threading.Lock()

    def call(self, lane, request_id, function):
        if lane not in self._sems:
            raise ServingError("unknown lane %r" % lane)
        if not request_id:
            raise ServingError("a request id is required")
        with self._lock:
            if request_id in self._active:
                raise ServingError("duplicate active request id %s" % request_id)
            self._active.add(request_id)
        enqueue = time.time()
        try:
            with self._sems[lane]:
                started = time.time()
                result = function()
                ended = time.time()
            return {"result": result, "enqueue_time": enqueue, "start_time": started,
                    "end_time": ended, "queue_seconds": started - enqueue,
                    "request_seconds": ended - started}
        finally:
            with self._lock:
                self._active.discard(request_id)

    def map_ordered(self, jobs):
        """Run jobs and return results in input order, never completion order."""
        if not jobs:
            return []
        workers = sum(max(1, int(value)) for value in self.max_concurrency.values())
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        futures = []
        try:
            for job in jobs:
                futures.append(executor.submit(
                    self.call, job["lane"], job["request_id"], job["function"]))
            return [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)


def load_runtime(path):
    with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
        runtime = json.load(handle)
    validate_config(runtime.get("role_config"))
    profile = runtime.get("profile")
    active = list(runtime["role_config"]["profiles"][profile]["lanes"])
    lanes = runtime.get("lanes")
    if not isinstance(lanes, dict) or sorted(lanes) != sorted(active):
        raise ServingError("the runtime does not hold exactly the profile's lanes")
    used = []
    for lane in active:
        entry = lanes[lane]
        pair = parse_gpu_pair(",".join(str(value) for value in entry.get("card_pair") or []))
        if entry.get("health") != "ok":
            raise ServingError("lane %s is not healthy" % lane)
        used.extend(pair)
    if len(set(used)) != len(used):
        raise ServingError("the runtime has overlapping card pairs")
    if runtime.get("config_sha256") != canonical_sha256(runtime["role_config"]):
        raise ServingError("the runtime role config hash does not match")
    return runtime


def public_runtime(runtime):
    """The run record's serving view, with no credentials or public host names."""
    return {
        "backend": "vllm-role",
        "profile": runtime["profile"],
        "role_map_version": runtime["role_config"]["version"],
        "config_sha256": runtime["config_sha256"],
        "model_revision": runtime.get("model_revision"),
        "tensor_parallel_size": runtime.get("tensor_parallel_size", 2),
        "mm_preprocessor_cache": runtime.get("mm_preprocessor_cache"),
        "lanes": {
            name: {
                "endpoint": entry.get("endpoint"),
                "card_pair": list(entry.get("card_pair") or []),
                "server_label": entry.get("server_label"),
                "health": entry.get("health"),
                "perspective": entry.get("perspective"),
            }
            for name, entry in runtime["lanes"].items()
        },
        "role_config": runtime["role_config"],
    }
