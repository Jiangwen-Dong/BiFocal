"""Frame cutting, sampling, and source-time checks for BiFocal reads."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess

from .media import ffmpeg


N_FRAMES = 64
OVERSHOOT = 1.1
SLACK = 0.5
TOKENS_PER_FRAME = 16
LETTER_FIRST = re.compile(r"^\s*\(?([A-H])(?:[).:\]\s]|$)")
LETTER_ANY = re.compile(r"\b([A-H])\b")


def evenly(items, n):
    """Take `n` items evenly across `items`."""
    if len(items) <= n:
        return list(items)
    step = len(items) / float(n)
    return [items[int(i * step)] for i in range(n)]


def parse_letter(answer):
    """Return the multiple-choice letter in `answer`, or an empty string."""
    text = str(answer or "")
    found = LETTER_FIRST.match(text) or LETTER_ANY.search(text)
    return found.group(1) if found else ""


def clock(seconds):
    """Return the time strings a benchmark may use for one second."""
    seconds = int(seconds)
    forms = {
        "%02d:%02d" % (seconds // 60, seconds % 60),
        "%d:%02d" % (seconds // 60, seconds % 60),
    }
    if seconds >= 3600:
        forms.add("%d:%02d:%02d" % (
            seconds // 3600, seconds // 60 % 60, seconds % 60))
    return forms


def gold_strings(item):
    """All source-time spellings that must stay out of policy prompts."""
    reference = str(item.get("time_reference") or "").strip()
    values = {reference} | {part.strip() for part in reference.split("-")}
    window = item.get("window") or []
    if len(window) == 2:
        values |= clock(window[0])
        values |= clock(window[1] - 1)
    return sorted(value for value in values if value)


def gold_in_prompt(prompt, item):
    """Count gold time strings outside the benchmark question itself."""
    rest = str(prompt).replace(str(item.get("question") or "").strip(), "")
    return sum(1 for value in gold_strings(item) if value in rest)


class Frames:
    """Cut and cache the frames of one video window."""

    def __init__(self, path, cache_dir):
        self.path = path
        self.cache_dir = cache_dir

    def window(self, start, end, n=N_FRAMES):
        """Return `(image, second, jpeg_md5)` tuples spread over the window."""
        from PIL import Image

        out_dir = os.path.join(
            self.cache_dir, "w-%.1f-%.1f" % (start, end))
        stamp = out_dir + ".done"
        if not os.path.exists(stamp):
            os.makedirs(out_dir, exist_ok=True)
            for old in glob.glob(os.path.join(out_dir, "*.jpg")):
                os.remove(old)
            rate = OVERSHOOT * n / max(end - start, 1e-3)
            subprocess.run(
                [ffmpeg(), "-v", "error", "-ss", "%.3f" % start,
                 "-i", self.path, "-t", "%.3f" % (end - start),
                 "-vf", "fps=%.6f" % rate, "-q:v", "2",
                 os.path.join(out_dir, "f-%05d.jpg")],
                check=True, stdin=subprocess.DEVNULL,
            )
            files = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
            times = [start + index / rate for index in range(len(files))]
            if len(files) < n:
                raise RuntimeError(
                    "%s: %d frames from %.1f-%.1f s, need %d; not stamped"
                    % (self.path, len(files), start, end, n))
            with open(out_dir + ".times.json", "w", encoding="utf-8") as handle:
                json.dump({"times": times, "rate": rate}, handle)
            with open(stamp, "w", encoding="utf-8") as handle:
                handle.write("ok\n")
        files = sorted(glob.glob(os.path.join(out_dir, "*.jpg")))
        with open(out_dir + ".times.json", encoding="utf-8") as handle:
            times = json.load(handle)["times"]
        if len(files) != len(times) or len(files) < n:
            raise RuntimeError(
                "%s is stamped but holds %d frames for %d times - remove the "
                "stamp to re-cut" % (out_dir, len(files), len(times)))
        output = []
        for path, second in evenly(list(zip(files, times)), n):
            with open(path, "rb") as handle:
                raw = handle.read()
            output.append((
                Image.open(path).convert("RGB"), second,
                hashlib.md5(raw).hexdigest(),
            ))
        return output
