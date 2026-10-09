"""Freeze a dataset: pool index -> train/validation/test jsonl + manifest.

Deterministic split: each row's bucket comes from hash("split"|<fid>), so
re-freezes are stable and new utterances never reshuffle existing ones.
fid = sha256(norm_text | audio_sha256) — unique per (text, audio) pair.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .naming import norm_text, text_fid


def _split_of(fid: str, val_frac: float, test_frac: float) -> str:
    h = int(hashlib.sha256(f"split|{fid}".encode()).hexdigest()[:8], 16) \
        / 0xFFFFFFFF
    if h < test_frac:
        return "test"
    if h < test_frac + val_frac:
        return "validation"
    return "train"


def load_index(index_path: Path) -> list[dict]:
    return [json.loads(l) for l in
            Path(index_path).read_text(encoding="utf-8").splitlines()
            if l.strip()]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def freeze_dataset(index_path: Path, dataset_dir: Path,
                   pool_dir: Path | None = None,
                   val_frac: float = 0.1, test_frac: float = 0.1,
                   exclude_fids: set[str] | None = None) -> dict:
    """Build train/validation/test.jsonl + manifest.json under dataset_dir.

    Audio is copied from ``pool_dir`` (defaults to ``dataset_dir/audio``)
    into ``dataset_dir/audio/<sha>.wav``; every jsonl ``audio`` field is
    verified to resolve to an existing file.
    """
    rows = [r for r in load_index(index_path) if r.get("status") == "ok"]
    exclude_fids = exclude_fids or set()
    dataset_dir = Path(dataset_dir).resolve()
    pool_dir = Path(pool_dir).resolve() if pool_dir \
        else dataset_dir / "audio"
    root = repo_root()

    seen_audio, seen_text, entries, dropped = set(), set(), [], []
    for r in rows:
        text = r.get("text") or ""
        text_source = "filename"
        if not norm_text(text) and r.get("text_asr"):
            text, text_source = r["text_asr"], "asr"
        elif r.get("provenance", {}).get("text") and \
                not norm_text(text):
            text, text_source = r["provenance"]["text"], "provenance"
        fid = text_fid(text, r["audio_sha256"])
        nt = norm_text(text)
        if fid in exclude_fids or r["audio_sha256"] in exclude_fids:
            dropped.append({"fid": fid, "why": "reviewed_out"})
            continue
        if not nt:
            dropped.append({"fid": fid, "why": "no_text"})
            continue
        if r["audio_sha256"] in seen_audio:
            dropped.append({"fid": fid, "why": "dup_audio"})
            continue
        if nt in seen_text:
            dropped.append({"fid": fid, "why": "dup_text"})
            continue
        src_wav = pool_dir / r["audio"]
        if not src_wav.exists():
            dropped.append({"fid": fid, "why": "missing_audio",
                            "src": str(src_wav)})
            continue
        seen_audio.add(r["audio_sha256"])
        seen_text.add(nt)
        entries.append({"fid": fid, "audio": r["audio"],
                        "audio_sha256": r["audio_sha256"],
                        "source_sha256": r.get("source_sha256"),
                        "text": text, "text_source": text_source,
                        "label": r.get("label"),
                        "provenance": r.get("provenance")})

    audio_dir = dataset_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict]] = {"train": [], "validation": [],
                                     "test": []}
    for e in sorted(entries, key=lambda e: e["fid"]):
        wav = audio_dir / e["audio"]
        src = pool_dir / e["audio"]
        if src.resolve() != wav.resolve():
            import shutil
            shutil.copy2(src, wav)
        try:
            audio_field = wav.relative_to(root).as_posix()
        except ValueError:
            audio_field = f"audio/{e['audio']}"
        rec = {"audio": audio_field, "fid": e["fid"], "text": e["text"],
               "audio_sha256": e["audio_sha256"], "label": e["label"],
               "text_source": e["text_source"]}
        if e.get("source_sha256"):
            rec["source_sha256"] = e["source_sha256"]
        if e.get("provenance"):
            rec["provenance"] = e["provenance"]
        splits[_split_of(e["fid"], val_frac, test_frac)].append(rec)

    stats = {}
    for name, recs in splits.items():
        path = dataset_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[name] = len(recs)

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "index": str(Path(index_path).resolve()),
        "counts": {**stats, "dropped": len(dropped)},
        "labels": dict(Counter(e["label"] for e in entries)),
        "split": {"method": "sha256(split|fid) fraction",
                  "validation_frac": val_frac, "test_frac": test_frac},
        "fid_format": "sha256(norm_text|audio_sha256)",
        "row_fields": ["audio", "fid", "text", "text_source", "label",
                        "audio_sha256", "source_sha256?", "provenance?"],
        "text_sources": dict(Counter(e["text_source"] for e in entries)),
        "dropped": dropped,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"dataset_dir": str(dataset_dir), "counts": stats,
            "dropped": dropped}
