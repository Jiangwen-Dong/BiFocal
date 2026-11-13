"""Shared checks for concrete memory claims."""

from __future__ import annotations

import re

GENERIC_CLAIMS = {
    "visible action changes",
    "visible actions change",
    "action changes",
    "scene changes",
    "visible scene changes",
    "visible people and actions",
    "one concrete fact from these frames",
    "no usable structured event",
}
STOPWORDS = set(
    "a an the and or of to in on for with is are was were be been this that "
    "these those what which who whom whose why how when where from as at by it "
    "its into about over under after before during while do does did can could "
    "should would may might will".split())
GENERIC_TERMS = set(
    "he she they her his their it this that these those video videos frame frames "
    "scene scenes clip clips shot shots image images shown show shows visible "
    "appear appears appeared person people someone vlogger vloggers host hosts "
    "hostess speaker man men woman women do does doing happen happens happening "
    "action actions change changes one two first second".split())
WEAK_COLOR_TERMS = set(
    "black white green red blue grey gray brown yellow pink".split())
PLACEHOLDERS = {
    "none", "n/a", "na", "unknown", "placeholder", "event text",
    "one short event", "short summary", "s", "fact",
}


def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_generic_claim(value):
    text = norm(value).strip(" .").lower()
    if not text or text in PLACEHOLDERS:
        return True
    if text in GENERIC_CLAIMS:
        return True
    if text.startswith("visible ") and len(text.split()) <= 4:
        return True
    if "placeholder" in text or "<" in text or ">" in text:
        return True
    return False


def terms(value):
    words = set()
    for word in re.findall(r"[a-z0-9]+", norm(value).lower()):
        if word.isdigit() or (len(word) > 2 and word not in STOPWORDS):
            words.add(word)
    return words


def concrete_terms(value):
    return terms(value) - GENERIC_TERMS


def card_agreement(question, card, options, option, quote):
    """Compare blind-check evidence with the remembered card conservatively."""
    quote_terms = concrete_terms(quote)
    quote_all_terms = terms(quote)
    timeline_text = " ".join(
        str(entry.get("summary") or entry.get("claim") or "")
        for entry in card.get("timeline_entries") or ())
    evidence_terms = concrete_terms(" ".join((
        str(card.get("claim") or ""),
        str(card.get("quote") or ""),
        str(card.get("anchor_source_claim") or ""),
        timeline_text)))
    strong_evidence = evidence_terms - WEAK_COLOR_TERMS
    evidence_match = sorted(quote_terms.intersection(strong_evidence))
    options = list(options or ())
    letter = str(option or "").strip().upper()
    selected = set()
    if len(letter) == 1 and "A" <= letter <= "H":
        index = ord(letter) - ord("A")
        if 0 <= index < len(options):
            selected = terms(options[index])
    option_match = sorted(quote_all_terms.intersection(selected))
    prompt_terms = concrete_terms(question) | {
        term for value in options for term in concrete_terms(value)}
    novel_evidence = sorted(quote_terms - prompt_terms)
    # A visible description may use synonyms not present in the card. Accept
    # either card-grounded evidence or genuinely new visual terms, while still
    # requiring the quote to match the selected option.
    ok = (bool(evidence_match) or bool(novel_evidence)) and (
        not selected or bool(option_match))
    return {
        "ok": ok,
        "evidence_terms": evidence_match,
        "novel_evidence_terms": novel_evidence,
        "option_terms": option_match,
        "quote_terms": sorted(quote_terms),
        "selected_option_terms": sorted(selected),
    }


def covers_question_attribute(question, quote, options=None, option=""):
    quote_terms = terms(quote)
    if not quote_terms:
        return False
    question_terms = terms(question)
    if question_terms and not quote_terms.intersection(question_terms):
        return False
    options = list(options or ())
    letter = str(option or "").strip().upper()
    selected_terms = set()
    if len(letter) == 1 and "A" <= letter <= "H":
        index = ord(letter) - ord("A")
        if 0 <= index < len(options):
            selected_terms = terms(options[index])
    if selected_terms and not quote_terms.intersection(selected_terms):
        return False
    if not selected_terms:
        option_terms = set()
        for value in options:
            option_terms.update(terms(value))
        if option_terms and not quote_terms.intersection(option_terms):
            return False
    return True


_OPTION_MARKERS = (
    re.compile(r"\(\s*[A-Ha-h]\s*\)\s*"),
    re.compile(r"(?<![\w])[A-H]\)\s*"),
    re.compile(r"(?i:\b(?:the\s+)?(?:correct\s+|right\s+|final\s+)?option)"
               r"\s+(?:is\s+)?[A-H]\b[:.]?\s*"),
    re.compile(r"(?i:\b(?:the\s+)?(?:correct\s+|right\s+|final\s+)?answer)"
               r"\s*(?:is|=|:)\s*[A-H]\b[:.]?\s*"),
)


def strip_option_markers(text):
    """Remove answer-letter forms a Reader may write. Bare letters stay."""
    value = str(text or "")
    for pattern in _OPTION_MARKERS:
        value = pattern.sub("", value)
    return re.sub(r"\s+", " ", value).strip()
