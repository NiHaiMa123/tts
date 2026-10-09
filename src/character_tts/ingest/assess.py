"""Assess raw inbox audio: per-file sanity report before standardizing."""
from __future__ import annotations

import json
from pathlib import Path

from .audio import TARGET_SR, flag_issues, load_mono, measure
from .naming import parse_inbox_name
from .standardize import iter_inbox


def assess_inbox(inbox: Path, report_path: Path,
                 target_sr: int = TARGET_SR) -> dict:
    """Scan inbox wavs; write a flag report. Speaker-embedding check is a
    reserved extension — not wired (needs an encoder backend)."""
    inbox = Path(inbox)
    rows = []
    for src in iter_inbox(inbox):
        label, text = parse_inbox_name(src)
        try:
            x, sr = load_mono(src, target_sr)
            m = measure(x, sr)
        except Exception as exc:
            rows.append({"source": str(src.relative_to(inbox)),
                         "label": label, "text": text,
                         "flags": [f"unreadable:{exc}"]})
            continue
        rows.append({"source": str(src.relative_to(inbox)),
                     "label": label, "text": text, "metrics": m,
                     "flags": flag_issues(m)})

    flagged = [r for r in rows if r["flags"]]
    by_flag: dict[str, int] = {}
    for r in flagged:
        for f in r["flags"]:
            by_flag[f.split(":")[0]] = by_flag.get(f.split(":")[0], 0) + 1
    report = {
        "inbox": str(inbox), "files": len(rows), "flagged": len(flagged),
        "flags_by_kind": by_flag,
        "speaker_check": "see scripts/ingest/speaker_check.py",
        "rows": rows,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    return report
