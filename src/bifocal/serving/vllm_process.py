#!/usr/bin/env python3
"""Start, inspect or stop the local vLLM lanes for a BiFocal serving profile."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import gpu as G
from . import lanes as R


def _utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _port_is_free(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _health_url(endpoint):
    parsed = urlparse(endpoint)
    return "%s://%s/health" % (parsed.scheme, parsed.netloc)


def health_ok(endpoint):
    try:
        with urllib.request.urlopen(_health_url(endpoint), timeout=5) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_health(endpoint, timeout):
    deadline = time.time() + float(timeout)
    while time.time() < deadline:
        if health_ok(endpoint):
            return True
        time.sleep(2.0)
    return False


def _pid_start_ticks(pid):
    try:
        with open("/proc/%d/stat" % int(pid), "r", encoding="utf-8") as handle:
            fields = handle.read().split()
        return int(fields[21])
    except (OSError, IndexError, ValueError):
        return None


def _terminate_group(pid):
    try:
        os.killpg(int(pid), signal.SIGTERM)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _kill_group(pid):
    try:
        os.killpg(int(pid), signal.SIGKILL)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _vllm_command(args, lane, port):
    command = [
        args.vllm, "serve", args.model,
        "--served-model-name", args.served_model_name,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--dtype", "bfloat16",
        "--tensor-parallel-size", "2",
        "--data-parallel-size", "1",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--limit-mm-per-prompt", json.dumps({"image": int(args.image_limit)}),
        "--mm-processor-kwargs", json.dumps({"max_pixels": 12845056}),
        # vLLM 0.11 can assert in its P1 multimodal processor cache when
        # concurrent reader requests share image hashes. The cache is an
        # optimisation, not part of the model contract; disable it so a valid
        # request cannot kill the engine thread.
        "--disable-mm-preprocessor-cache",
    ]
    if args.prefix_caching:
        command.append("--enable-prefix-caching")
    else:
        command.append("--no-enable-prefix-caching")
    return command


def _record_path(path):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_runtime(path, runtime):
    path = _record_path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def start(args, query_fn=G.query_gpus, spawn_fn=subprocess.Popen,
          health_fn=wait_health, port_fn=_port_is_free,
          pid_ticks_fn=_pid_start_ticks,
          digest_fn=R.model_content_sha256):
    config = R.load_config(args.config)
    pairs = R.parse_lane_pairs(args.lane_gpus, args.profile, config)
    model_content_sha256 = digest_fn(args.model)
    runtime_path = Path(args.runtime).expanduser().resolve()
    if runtime_path.exists():
        raise R.ServingError("runtime record already exists: %s" % runtime_path)
    rows, busy = query_fn(args.nvidia_smi)
    selected = {}
    for lane, pair in pairs.items():
        selected[lane] = G.select(rows, busy, pair)
    for lane, pair in pairs.items():
        if not port_fn("127.0.0.1", urlparse(config["lanes"][lane]["endpoint"]).port):
            raise R.ServingError("lane %s port is already in use" % lane)
    log_dir = Path(args.log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    started = []
    lanes = {}
    try:
        for lane in config["profiles"][args.profile]["lanes"]:
            pair = pairs[lane]
            endpoint = config["lanes"][lane]["endpoint"].rstrip("/")
            port = urlparse(endpoint).port
            log_path = log_dir / ("%s-%s-%s.log" % (stamp, args.profile, lane))
            log_handle = open(log_path, "ab", buffering=0)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in pair)
            try:
                process = spawn_fn(
                    _vllm_command(args, lane, port), env=environment,
                    stdout=log_handle, stderr=subprocess.STDOUT,
                    start_new_session=True)
            finally:
                log_handle.close()
            started.append((lane, process))
            lanes[lane] = {
                "endpoint": endpoint,
                "card_pair": pair,
                "perspective": config["lanes"][lane].get("perspective"),
                "server_label": "%s-%s-%s" % (args.profile, lane, stamp),
                "pid": int(process.pid),
                "pid_start_ticks": pid_ticks_fn(process.pid),
                "port": int(port),
                "log": str(log_path),
                "health": "starting",
                "started_utc": _utc(),
                "processes": selected[lane],
            }
            healthy = health_fn(endpoint, args.health_timeout)
            if not healthy:
                raise R.ServingError("lane %s did not become healthy" % lane)
            lanes[lane]["health"] = "ok"
        runtime = {
            "version": "runtime-v1",
            "created_utc": _utc(),
            "profile": args.profile,
            "config_path": str(Path(args.config).expanduser().resolve()),
            "config_sha256": R.canonical_sha256(config),
            "role_config": config,
            "model": str(Path(args.model).expanduser().resolve()),
            "model_revision": args.model_revision,
            "model_content_sha256": model_content_sha256,
            "served_model_name": args.served_model_name,
            "tensor_parallel_size": 2,
            "prefix_caching": bool(args.prefix_caching),
            "mm_preprocessor_cache": False,
            "lanes": lanes,
        }
        _write_runtime(runtime_path, runtime)
        return runtime
    except BaseException:
        for lane, process in reversed(started):
            entry = lanes.get(lane) or {}
            current = pid_ticks_fn(process.pid)
            if current is not None and current == entry.get("pid_start_ticks"):
                _terminate_group(process.pid)
        deadline = time.time() + 30.0
        while time.time() < deadline and any(process.poll() is None
                                             for _, process in started):
            time.sleep(0.5)
        for lane, process in reversed(started):
            entry = lanes.get(lane) or {}
            current = pid_ticks_fn(process.pid)
            if (process.poll() is None and current is not None
                    and current == entry.get("pid_start_ticks")):
                _kill_group(process.pid)
        raise


def stop(args, query_fn=G.query_gpus, terminate_fn=_terminate_group,
         kill_fn=_kill_group):
    runtime = R.load_runtime(args.runtime)
    stopped = []
    safe_to_signal = {}
    for lane, entry in runtime["lanes"].items():
        pid = int(entry["pid"])
        current = _pid_start_ticks(pid)
        safe = current is not None and current == entry.get("pid_start_ticks")
        safe_to_signal[lane] = safe
        if safe:
            terminate_fn(pid)
            stopped.append({"lane": lane, "pid": pid, "signal": "TERM"})
    deadline = time.time() + float(args.stop_timeout)
    while time.time() < deadline and any(_pid_start_ticks(int(entry["pid"])) is not None
                                        for entry in runtime["lanes"].values()):
        time.sleep(0.5)
    for lane, entry in runtime["lanes"].items():
        pid = int(entry["pid"])
        current = _pid_start_ticks(pid)
        if safe_to_signal.get(lane) and current is not None \
                and current == entry.get("pid_start_ticks"):
            kill_fn(pid)
            stopped.append({"lane": lane, "pid": pid, "signal": "KILL"})
    rows, busy = query_fn(args.nvidia_smi)
    selected = []
    for entry in runtime["lanes"].values():
        selected.extend(entry["card_pair"])
    G.select(rows, busy, selected)
    runtime["stopped_utc"] = _utc()
    for entry in runtime["lanes"].values():
        entry["health"] = "stopped"
    active_path = Path(args.runtime).expanduser().resolve()
    stopped_path = active_path.with_suffix(active_path.suffix + ".stopped")
    _write_runtime(stopped_path, runtime)
    active_path.unlink()
    return {"runtime": str(Path(args.runtime).expanduser().resolve()),
            "stopped_record": str(stopped_path), "stopped": stopped,
            "selected_cards_free": selected}


def status(args):
    runtime = R.load_runtime(args.runtime)
    out = {"runtime": str(Path(args.runtime).expanduser().resolve()),
           "profile": runtime["profile"], "lanes": {}}
    for lane, entry in runtime["lanes"].items():
        pid = int(entry["pid"])
        out["lanes"][lane] = {
            "pid": pid,
            "alive": _pid_start_ticks(pid) == entry.get("pid_start_ticks"),
            "endpoint": entry["endpoint"],
            "card_pair": entry["card_pair"],
            "health": health_ok(entry["endpoint"]),
        }
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up")
    up.add_argument("--profile", required=True,
                    choices=("single", "split", "split-readers"))
    up.add_argument("--lane-gpus", action="append", required=True,
                    help="lane=card,card; repeat once for every active lane")
    up.add_argument("--config", required=True)
    up.add_argument("--runtime", required=True)
    up.add_argument("--log-dir", required=True)
    up.add_argument("--vllm", required=True)
    up.add_argument("--model", required=True)
    up.add_argument("--model-revision", required=True)
    up.add_argument("--served-model-name", default="qwen3-vl-8b")
    up.add_argument("--max-model-len", type=int, default=32768)
    up.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    up.add_argument("--image-limit", type=int, default=64,
                    help="maximum images in one prompt; the active harness can send 64")
    up.add_argument("--health-timeout", type=float, default=900.0)
    up.add_argument("--nvidia-smi", default="nvidia-smi")
    up.add_argument("--prefix-caching", action="store_true")
    down = sub.add_parser("down")
    down.add_argument("--runtime", required=True)
    down.add_argument("--stop-timeout", type=float, default=60.0)
    down.add_argument("--nvidia-smi", default="nvidia-smi")
    stat = sub.add_parser("status")
    stat.add_argument("--runtime", required=True)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "up":
            result = start(args)
        elif args.command == "down":
            result = stop(args)
        else:
            result = status(args)
    except (R.ServingError, ValueError, OSError, subprocess.SubprocessError) as error:
        print("SERVING_REFUSED: %s" % error, file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
