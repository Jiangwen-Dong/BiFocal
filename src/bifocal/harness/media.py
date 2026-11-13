#!/usr/bin/env python3
"""Video media for BiFocal reads: time-reference parsing, window arithmetic and frame sampling.

The loop asks a media object for a place in the input: where a part sits (a time window
and frame indices), what it covers (seconds a..b), how to open it more densely
(re-sample the window at a higher fps), and how to look inside it (crop a box in a frame).

The checks at the bottom of this file need no video and no GPU: they cover the
time-reference parsing, the window arithmetic and the fidelity ladder (a re-open at higher
fps must be a different, denser sample)."""

import os
import re
import shutil
import subprocess

from . import time as LT  # noqa: E402


def ffmpeg():
    """The ffmpeg to call: the system one if there is one, else the bundled one.

    `imageio-ffmpeg` ships a static binary, so the decoder works without asking the
    host to install anything.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

# The two fidelities a video place can be read at. "read" is what a first pass
# uses; "verify" is the denser sample a re-check pays for.
READ_FPS = 0.5
VERIFY_FPS = 2.0

CLOCK = LT.CLOCK
CLOCK_RANGE = LT.CLOCK_RANGE
MINUTE = re.compile(r"\b(?:minute|min)\s+(\d{1,3})\b")
FRACTION = re.compile(r"\b(first|second|third|last|final)\s+(half|quarter|part)\b")


def _seconds(hours, minutes, seconds=None):
    if seconds is None:
        # "3:20" is minutes:seconds in a video, not hours:minutes - a 5-minute
        # clip that is read as 3 hours would send every read to the end.
        return int(hours) * 60 + int(minutes)
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def _tag(value):
    """A number as a filename: `32.0` is `32p000`, so a cache key is legal."""
    return ("%.3f" % float(value)).replace("-", "m").replace(".", "p")


def time_references(question, duration):
    """Every place the question itself points at, as [t0, t1] in seconds.

    Whatever the question names is
    read first, and - if that read settles nothing - the same window is opened
    again at a higher fps rather than the team wandering to another part.
    """
    found = []
    for match in CLOCK_RANGE.finditer(question or ""):
        a, b = _seconds(match.group(1), match.group(2), match.group(3)), \
            _seconds(match.group(4), match.group(5), match.group(6))
        if b > a:
            found.append((float(a), float(b)))
    for match in CLOCK.finditer(question or ""):
        at = _seconds(match.group(1), match.group(2), match.group(3))
        found.append((max(0.0, at - 30.0), min(float(duration), at + 30.0)))
    for match in MINUTE.finditer(question or ""):
        at = float(match.group(1)) * 60
        found.append((max(0.0, at - 30.0), min(float(duration), at + 30.0)))
    for match in FRACTION.finditer(question or ""):
        which, span = match.group(1), match.group(2)
        size = {"half": 0.5, "quarter": 0.25, "part": 0.5}[span]
        start = {"first": 0.0, "second": size, "third": 2 * size,
                 "last": 1 - size, "final": 1 - size}[which]
        found.append((start * duration, (start + size) * duration))
    out, seen = [], set()
    for t0, t1 in found:
        if t1 - t0 < 1.0 or t0 >= duration:
            continue
        key = (round(t0, 1), round(t1, 1))
        if key not in seen:
            seen.add(key)
            out.append([round(t0, 1), round(t1, 1)])
    return out



class VideoMedia:
    """Video: a place is a time window, at a readable frame rate."""

    modality = "frame"

    def __init__(self, path, duration=None, fps=READ_FPS, work_dir=None,
                 max_side=768, max_frames=24):
        self.path = path
        self.fps = fps
        self.max_side = max_side
        self.max_frames = max_frames
        self.work_dir = work_dir or os.path.join(
            os.path.dirname(path), ".frames")
        self.duration = duration or self._probe()
        self.unit_count = 1        # one unit: the video

    def _probe(self):
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", self.path],
            capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 0.0

    def frames(self, t0, t1, rate):
        """The frames of a window, cut from a decode of the whole video.

        **This is the baselines' pattern and we were not following it.** VideoTree
        extracts CLIP features for the whole corpus *once* and its query-time code
        only selects from them; MovieChat and MA-LMM pre-build a memory bank of
        frame features; VideoHV-Agent extracts captions offline. All of them pay
        the decode cost once, off the query path. The first version here ran
        ffmpeg *per window*, which is why ten videos cost minutes of CPU and
        1.9 GB - it was decoding the same video eight times.

        Now the video is decoded once at each rate we use, the frames stay on
        disk, and a window is a slice of that decode. A re-check at a higher rate
        is therefore a second decode - once - and every later look at that rate
        is free.
        """
        from PIL import Image
        files = self._decoded(rate)
        return [Image.open(files[i]).convert("RGB")
                for i in self._chosen(len(files), t0, t1, rate)]

    def _chosen(self, n_files, t0, t1, rate):
        """The indices into the decode that `frames()` hands back."""
        first = max(0, int(t0 * rate))
        last = min(n_files, max(first + 1, int(t1 * rate)))
        chosen = list(range(first, last))
        if len(chosen) > self.max_frames:
            step = len(chosen) / float(self.max_frames)
            chosen = [chosen[int(index * step)] for index in range(self.max_frames)]
        return chosen

    def frame_times(self, t0, t1, rate):
        """**The second each of `frames(t0, t1, rate)` shows**, same selection.
        Frame `i` of a decode at `rate` is second `i / rate`."""
        files = self._decoded(rate)
        return [i / float(rate) for i in self._chosen(len(files), t0, t1, rate)]

    def _too_few(self, files, rate, seconds=None):
        """Why `files` cannot be a decode of `seconds` of this video at `rate`, or `""`.

        **A stamp says the decode finished; it does not say the frames are still
        there.** With the frames on another disk behind a symlink, a re-image
        leaves the stamp and an empty folder, and a loader that trusts the stamp
        reads zero frames and raises nothing. So the count is checked against the
        duration every time, not only when the stamp is written.

        `seconds` is the window's length; `None` is the whole video. **An unknown
        duration fails closed**: the box has no ffprobe, so `_probe` returns 0,
        and a check against 0 frames would pass any truncated decode.
        """
        if not files:
            return "no frames"
        duration = float(self.duration or 0)
        if duration <= 0:
            return ("%d frames and no known duration to check them against - "
                    "pass duration=" % len(files))
        if seconds is None:
            seconds = duration
        expected = int(min(float(seconds), duration) * float(rate) * 0.9)
        if len(files) < expected:
            return "%d frames, expected at least %d" % (len(files), expected)
        return ""

    def _decoded(self, rate):
        """Decode the whole video once at `rate`, keeping the frames on disk."""
        import glob
        out_dir = os.path.join(self.work_dir, os.path.basename(self.path),
                               "all_%s" % rate)
        stamp = out_dir + ".done"
        if os.path.exists(stamp):
            files = sorted(glob.glob(os.path.join(out_dir, "*.png")))
            why = self._too_few(files, rate)
            if why:
                raise RuntimeError("%s is stamped done but has %s - the frames "
                                   "were lost; remove the stamp to re-decode"
                                   % (out_dir, why))
            return files
        os.makedirs(out_dir, exist_ok=True)
        subprocess.run(
            [ffmpeg(), "-v", "error", "-i", self.path, "-vf",
             "fps=%s,scale='min(%d,iw)':-2" % (rate, self.max_side),
             os.path.join(out_dir, "f-%05d.png")], check=True)
        files = sorted(glob.glob(os.path.join(out_dir, "*.png")))
        why = self._too_few(files, rate)
        if why:
            raise RuntimeError("decode of %s gave %s; not stamped" % (self.path, why))
        open(stamp, "w").write("ok\n")
        return files

    def window_dir(self, t0, t1, rate):
        """**The cache of one window at one rate**, keyed by both.

        `frames()` keys its cache by the rate alone, which is right when every
        window is a slice of one whole-video decode. A dense read is not: at 2 fps
        over 32 seconds it needs 64 frames of a video that may be an hour long,
        and decoding the whole thing to take 64 frames of it is 7,200 PNGs and
        several GB on a disk with 4.5 GB free.
        """
        return os.path.join(self.work_dir, os.path.basename(self.path),
                            "w-%s-%s-%s" % (_tag(rate), _tag(t0), _tag(t1)))

    def window_frames(self, t0, t1, rate):
        """The frames of **this window only**, decoded once and cached."""
        import glob
        from PIL import Image
        out_dir = self.window_dir(t0, t1, rate)
        stamp = out_dir + ".done"
        # the window may run past the end; only the seconds that exist count
        seconds = max(0.0, min(float(t1), float(self.duration or t1)) - float(t0))
        if os.path.exists(stamp):
            # **the same trap as the whole decode**: a stamp over a re-imaged
            # folder must be loud, not zero frames
            why = self._too_few(sorted(glob.glob(os.path.join(out_dir, "*.png"))),
                                rate, seconds)
            if why:
                raise RuntimeError("%s is stamped done but has %s - the frames "
                                   "were lost; remove the stamp to re-decode"
                                   % (out_dir, why))
        else:
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run(
                # **`_tag` is for filenames, not for ffmpeg.** It writes 768.0 as
                # `768p000` so a path stays legal, and passing that to `-ss`
                # produced `-ss 768p000`: the decode died with "Invalid argument"
                # on the first real run, after 138 seconds, and the unit tests
                # had not caught it because a cache hit never builds a command.
                [ffmpeg(), "-v", "error", "-ss", "%.3f" % float(t0),
                 "-i", self.path, "-t",
                 "%.3f" % max(0.0, float(t1) - float(t0)), "-vf",
                 "fps=%s,scale='min(%d,iw)':-2" % (rate, self.max_side),
                 os.path.join(out_dir, "f-%05d.png")], check=True)
            why = self._too_few(sorted(glob.glob(os.path.join(out_dir, "*.png"))),
                                rate, seconds)
            if why:
                raise RuntimeError("window decode of %s gave %s; not stamped"
                                   % (out_dir, why))
            with open(stamp, "w") as handle:
                handle.write("ok\n")
        files = sorted(glob.glob(os.path.join(out_dir, "*.png")))
        if len(files) > self.max_frames:
            step = len(files) / float(self.max_frames)
            files = [files[int(index * step)] for index in range(self.max_frames)]
        return [Image.open(f).convert("RGB") for f in files]

    def open(self, pointer, fidelity="read"):
        t0, t1 = pointer.get("time") or (0.0, self.duration)
        rate = VERIFY_FPS if fidelity == "verify" else self.fps
        return self.frames(t0, t1, rate)

    def crop(self, image, box):
        return _crop(image, box)

    def references(self, question):
        return time_references(question, self.duration)


def _crop(image, box):
    """The same padding rule the document path uses, on any image."""
    width, height = image.size
    left, top, right, bottom = [min(1.0, max(0.0, float(v))) for v in box]
    x0, x1 = int(left * width), int(right * width)
    y0, y1 = int(top * height), int(bottom * height)
    minimum_x, minimum_y = max(32, width // 20), max(32, height // 20)
    x0, x1 = max(0, min(x0, x1 - minimum_x)), max(x1, x0 + minimum_x)
    y0, y1 = max(0, min(y0, y1 - minimum_y)), max(y1, y0 + minimum_y)
    return image.crop((x0, y0, min(width, x1), min(height, y1)))


def for_item(item, plan=None):
    """The media object an item needs, chosen from what the item declares.

    One place decides, so the loop never asks what kind of media it has. An item
    carrying `video_path` is video; a document item is refused; anything else is
    an error rather than a silent default, because a silent default is how an
    arm ends up reading the wrong thing and nobody notices until the numbers are
    strange.
    """
    plan = plan or {}
    if item.get("video_path"):
        return VideoMedia(item["video_path"], duration=item.get("duration"),
                          fps=plan.get("read_fps", READ_FPS),
                          work_dir=plan.get("frame_dir") or None)
    if item.get("doc_id"):
        # Only video items are supported.
        raise ValueError("document items are unsupported in the video-only study")
    raise ValueError("item names neither a video_path nor a doc_id: %r"
                     % (sorted(item.keys())[:8],))


def gate():
    checks = []
    refs = time_references("What happens at 3:20 in the video?", 600)
    checks.append(("a clock reference is minutes:seconds, not hours",
                   refs and abs(refs[0][0] - 170) < 1 and abs(refs[0][1] - 230) < 1))
    refs = time_references("what is said between 10:00 and 11:30?", 3600)
    checks.append(("a range is read as a range",
                   refs and refs[0] == [600.0, 690.0]))
    refs = time_references("What happens in the second half?", 600)
    checks.append(("a fractional reference lands in that half",
                   refs and refs[0] == [300.0, 600.0]))
    refs = time_references("What happens in minute 3?", 600)
    checks.append(("a minute reference is found",
                   refs and refs[0][0] <= 180 <= refs[0][1]))
    checks.append(("nothing is invented when the question names no time",
                   time_references("How many people are in the video?", 600) == []))
    checks.append(("a reference past the end is dropped",
                   time_references("at 99:00", 600) == []))
    class Fake:
        size = (200, 100)
        def crop(self, box):
            self.box = box
            return self
    thin = Fake()
    _crop(thin, [0.5, 0.2, 0.502, 0.4])
    checks.append(("a thin box is padded on any modality",
                   thin.box[2] - thin.box[0] >= max(32, 200 // 20)))
    video = for_item({"video_path": "/tmp/none.mp4", "duration": 600.0},
                     {"read_fps": 0.5})
    checks.append(("a video item gets the video medium, at the declared rate",
                   isinstance(video, VideoMedia) and video.fps == 0.5
                   and video.duration == 600.0))
    try:
        for_item({"question": "q"})
        checks.append(("an item that names no medium is refused", False))
    except ValueError:
        checks.append(("an item that names no medium is refused", True))
    # **One check that runs the real binary.** Every unit test of `window_frames`
    # stubs `subprocess.run`, so the entire command line was untested -- and the
    # first real dense read on the box died on `-ss 768p000` after 138 seconds,
    # because the filename tag was being passed to ffmpeg as a number. A cache
    # hit builds no command, so nothing between the tests and the batch could see
    # it. This builds a two-second clip out of `lavfi` and decodes a window of it.
    try:
        import shutil
        import tempfile
        work = tempfile.mkdtemp(prefix="bifocal-gate-media-")
        clip = os.path.join(work, "clip.mp4")
        subprocess.run([ffmpeg(), "-v", "error", "-f", "lavfi", "-i",
                        "testsrc=size=64x64:rate=8:duration=2",
                        "-pix_fmt", "yuv420p", clip],
                       check=True, capture_output=True)
        medium = VideoMedia(clip, duration=2.0,
                            work_dir=os.path.join(work, "frames"), max_frames=16)
        decoded = medium.window_frames(0.5, 1.5, 8.0)
        checks.append(("a dense window decodes for real, at the rate asked",
                       len(decoded) == 8))
        shutil.rmtree(work, ignore_errors=True)
    except FileNotFoundError:
        print("skip a dense window decodes for real (no ffmpeg on this machine)")
    except Exception as error:                       # noqa: BLE001 - report it
        checks.append(("a dense window decodes for real: %r" % error, False))
    bad = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print("%s %s" % ("ok  " if ok else "FAIL", name))
    if bad:
        raise SystemExit("the media layer is wrong; no run")
    print("gate: %d/%d" % (len(checks), len(checks)))


if __name__ == "__main__":
    gate()
