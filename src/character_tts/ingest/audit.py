"""Audit a frozen dataset: integrity, dedup, split leakage, stats."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .freeze import repo_root
from .naming import norm_text, text_fid
from .audio import sha256_file

SPLITS = ("train", "validation", "test")


def audit_dataset(dataset_dir: Path, verify_hash: bool = True) -> dict:
    """Check a frozen dataset dir. Returns report; violations non-empty on
    any failure (caller decides exit code)."""
    dataset_dir = Path(dataset_dir).resolve()
    root = repo_root()
    violations: list[str] = []
    by_split: dict[str, list[dict]] = {}

    for name in SPLITS:
        path = dataset_dir / f"{name}.jsonl"
        if not path.is_file():
            violations.append(f"missing split file: {path}")
            by_split[name] = []
            continue
        rows = [json.loads(l) for l in
                path.read_text(encoding="utf-8").splitlines() if l.strip()]
        by_split[name] = rows
        for r in rows:
            # audio is repo-relative ("data/.../audio/x.wav") or
            # dataset-relative ("audio/x.wav"); try both.
            cands = [dataset_dir / r["audio"], root / r["audio"]]
            wav = next((c for c in cands if c.is_file()), None)
            if wav is None:
                violations.append(f"{name}: missing audio {r['audio']}")
                continue
            expected_fid = text_fid(r.get("text") or "",
                                    r.get("audio_sha256") or "")
            if r.get("audio_sha256") and r["fid"] != expected_fid:
                violations.append(f"{name}: fid mismatch {r['fid'][:12]}")
            if verify_hash and r.get("audio_sha256"):
                actual = sha256_file(wav)
                if actual != r["audio_sha256"]:
                    violations.append(
                        f"{name}: audio sha mismatch {r['audio']}")

    # leakage: same audio hash or normalized text across splits
    seen_audio: dict[str, str] = {}
    seen_text: dict[str, str] = {}
    for name, rows in by_split.items():
        for r in rows:
            for table, key, kind in (
                    (seen_audio, r.get("audio_sha256"), "audio_leak"),
                    (seen_text, norm_text(r.get("text") or ""),
                     "text_leak")):
                if not key:
                    continue
                if key in table and table[key] != name:
                    violations.append(
                        f"{kind}: {key[:16]} in {table[key]} and {name}")
                table.setdefault(key, name)

    return {
        "dataset": str(dataset_dir),
        "ok": not violations,
        "violations": violations,
        "counts": {n: len(r) for n, r in by_split.items()},
        "labels": dict(Counter(r.get("label") for rows in by_split.values()
                               for r in rows)),
    }
