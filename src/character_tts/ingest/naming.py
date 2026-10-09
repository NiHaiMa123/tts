"""Inbox filename convention: 【<情绪标签>】<台词原文>.wav under inbox/<标签>/."""
from __future__ import annotations

import re
from pathlib import Path

_NAME_RE = re.compile(r"^【(?P<label>[^】]+)】(?P<text>.+)$")
_PUNCT_RE = re.compile(r"[\s，。！？、；：「」『』（）《》…—\-,.!?;:\"'~·]")


def parse_inbox_name(path: Path) -> tuple[str | None, str]:
    """Return (label, transcript) from an inbox wav path.

    Label comes from 【…】 prefix when present, else the parent dir
    (which carries `<标签>_<eng>` names like 中立_neutral).
    """
    stem = path.stem
    label = path.parent.name if path.parent.name != "inbox" else None
    m = _NAME_RE.match(stem)
    if m:
        label = m.group("label")
        text = m.group("text").strip()
        # "【label】_10" style numbered exports carry no transcript
        if re.fullmatch(r"[_\-\d]+", text):
            text = ""
        return label, text
    bare = re.match(r"^【(?P<l>[^】]+)】\s*$", stem)
    if bare:
        return label or bare.group("l"), ""
    text = stem.strip()
    # bare numbered stems ("1001", "_10") carry no transcript either
    if re.fullmatch(r"[_\-\d]+", text):
        text = ""
    return label, text


def norm_text(text: str) -> str:
    """Normalized comparison form: strip all punctuation/space."""
    return _PUNCT_RE.sub("", text or "")


def text_fid(text: str, audio_sha256: str) -> str:
    """Stable utterance id: sha256(norm_text | audio_sha256)."""
    import hashlib
    return hashlib.sha256(
        f"{norm_text(text)}|{audio_sha256}".encode("utf-8")).hexdigest()
