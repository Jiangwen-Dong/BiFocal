"""Shared parsers for LVBench clock strings.

LVBench uses ``mm:ss`` as its normal form and some videos run past 100
minutes, so minute fields may have one to three digits.  Keep every consumer
on this module: a stamp detector that sees ``100:05`` while question-window
extraction does not creates a silent split between the recorded item and the
state the policy sees.
"""

from __future__ import annotations

import re


STAMP = re.compile(r"(?<!\d)(\d{1,3}:\d{2}(?::\d{2})?)(?!\d)")
TIME_REFERENCE = re.compile(r"(?<!\d)(\d{1,3}):(\d{2})(?::(\d{2}))?(?!\d)")
RANGE = re.compile(
    r"(?<!\d)(\d{1,3}:\d{2}(?::\d{2})?)\s*[-\u2013]\s*"
    r"(\d{1,3}:\d{2}(?::\d{2})?)")
CLOCK = re.compile(
    r"\b(?:at|from|around)\s+(\d{1,3}):(\d{2})(?::(\d{2}))?")
CLOCK_RANGE = re.compile(
    r"(?<!\d)(\d{1,3}):(\d{2})(?::(\d{2}))?\s*"
    r"(?:-|to|until|and)\s*"
    r"(\d{1,3}):(\d{2})(?::(\d{2}))?(?!\d)")
MALFORMED_RANGE = re.compile(
    r"(?:\d{1,3}:\d{2}(?::\d{2})?\s*[-\u2013]\s*(?:None\b|$))|"
    r"(?:(?:^|\b)None\b\s*[-\u2013]\s*\d{1,3}:\d{2}(?::\d{2})?)",
    re.I)


def seconds(text):
    """``mm:ss`` or ``h:mm:ss`` -> seconds."""
    parts = [int(part) for part in str(text).strip().split(":")]
    if not 2 <= len(parts) <= 3:
        raise ValueError("not a clock: %r" % (text,))
    return sum(value * 60 ** index
               for index, value in enumerate(reversed(parts)))


def window(time_reference):
    """Return the inclusive LVBench window as ``(start, end_exclusive)``."""
    found = RANGE.fullmatch(str(time_reference or "").strip())
    if not found:
        raise ValueError("not a window: %r" % (time_reference,))
    start, last = seconds(found.group(1)), seconds(found.group(2))
    if last < start:
        raise ValueError("window ends before it starts: %r" % (time_reference,))
    return float(start), float(last + 1)


def time_references(question, duration):
    """Every place the question names, as ``[start, end]`` seconds."""
    found = []
    question = str(question or "")
    for match in CLOCK_RANGE.finditer(question):
        first = _groups_seconds(match.group(1), match.group(2), match.group(3))
        last = _groups_seconds(match.group(4), match.group(5), match.group(6))
        if last > first:
            found.append((float(first), float(last)))
    for match in CLOCK.finditer(question):
        at = _groups_seconds(match.group(1), match.group(2), match.group(3))
        found.append((max(0.0, at - 30.0), min(float(duration), at + 30.0)))
    out, seen = [], set()
    for first, last in found:
        if last - first < 1.0 or first >= float(duration):
            continue
        key = (round(first, 1), round(last, 1))
        if key not in seen:
            seen.add(key)
            out.append([key[0], key[1]])
    return out


def _groups_seconds(first, second, third=None):
    if third is None:
        return int(first) * 60 + int(second)
    return int(first) * 3600 + int(second) * 60 + int(third)
