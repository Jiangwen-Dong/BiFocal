#!/usr/bin/env python3
"""Check that a selected GPU set is idle and launch one command on it."""

import argparse
import csv
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

def _allowed_gpus():
    declared = os.environ.get("BIFOCAL_ALLOWED_GPUS", "")
    if declared.strip():
        return {int(value.strip()) for value in declared.split(",")
                if value.strip()}
    return set(range(32))


ALLOWED = _allowed_gpus()
MAX_IDLE_MIB = 512
MAX_IDLE_UTIL = 5


def parse_gpus(text):
    try:
        values = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError:
        raise ValueError("--gpus must be comma-separated integers")
    if not 1 <= len(values) <= 3:
        raise ValueError("select one, two or three GPUs")
    if len(set(values)) != len(values):
        raise ValueError("GPU list contains a duplicate")
    bad = [value for value in values if value not in ALLOWED]
    if bad:
        raise ValueError("GPUs outside the allowed range: %s" % bad)
    return values


def query_gpus(nvidia_smi):
    fields = "index,uuid,name,memory.used,utilization.gpu,driver_version"
    run = subprocess.run(
        [nvidia_smi, "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True)
    rows = {}
    for row in csv.reader(run.stdout.splitlines(), skipinitialspace=True):
        if len(row) != 6:
            raise ValueError("unexpected nvidia-smi row: %r" % (row,))
        index = int(row[0])
        rows[index] = {"index": index, "uuid": row[1].strip(),
                       "name": row[2].strip(), "memory_used_mib": int(row[3]),
                       "utilization": int(row[4]), "driver": row[5].strip()}
    apps = subprocess.run(
        [nvidia_smi, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
         "--format=csv,noheader,nounits"],
        check=True, text=True, capture_output=True).stdout
    busy = set()
    for row in csv.reader(apps.splitlines(), skipinitialspace=True):
        if row:
            busy.add(row[0].strip())
    return rows, busy


def select(rows, busy, wanted):
    selected = []
    for index in wanted:
        if index not in rows:
            raise ValueError("GPU %d is not present" % index)
        row = rows[index]
        if row["uuid"] in busy or row["memory_used_mib"] > MAX_IDLE_MIB \
                or row["utilization"] > MAX_IDLE_UTIL:
            raise ValueError("GPU %d is busy: %s MiB, %s%%, process=%s" %
                             (index, row["memory_used_mib"], row["utilization"],
                              row["uuid"] in busy))
        selected.append(row)
    return selected


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True,
                    help="comma-separated one to three allowed GPU indices")
    ap.add_argument("--nvidia-smi", default="nvidia-smi")
    ap.add_argument("--record", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command and not args.dry_run:
        ap.error("a command is required unless --dry-run is used")
    try:
        wanted = parse_gpus(args.gpus)
        rows, busy = query_gpus(args.nvidia_smi)
        selected = select(rows, busy, wanted)
    except (ValueError, subprocess.CalledProcessError) as error:
        print("GPU_CHECK_FAILED: %s" % error, file=sys.stderr)
        return 2
    record = {
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpus": selected,
        "command": list(args.command),
    }
    if args.record:
        path = Path(args.record)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), file=sys.stderr, flush=True)
    if args.dry_run:
        return 0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(row["index"]) for row in selected)
    return subprocess.run(args.command, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
