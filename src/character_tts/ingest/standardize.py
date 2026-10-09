"""Standardize inbox wavs into a content-addressed audio pool.

Each accepted file is resampled to TARGET_SR mono PCM16, hashed by content,
and stored as ``pool_dir/<sha256>.wav``. Duplicates collapse naturally —
same content, same name. Produces an index.jsonl recording src -> pool
mapping with label/text/metrics/flags.
"""
from __future__ import annotations

import json
from pathlib import Path

from .audio import TARGET_SR, flag_issues, load_mono, measure, sha256_file, \
    write_wav_pcm16
from .naming import parse_inbox_name


def iter_inbox(inbox: Path) -> list[Path]:
    return sorted(p for p in inbox.rglob("*")
                  if p.is_file() and p.suffix.lower()
                  in (".wav", ".ogg", ".flac", ".mp3"))


def load_provenance(inbox: Path) -> dict[str, dict]:
    """Optional sidecar ``<inbox>/provenance.jsonl`` written by importers.

    Rows: {"file": "<rel path>", "text"?, "label"?, "event"?, "wem_hash"?,
    "source_wem"?, "tool"?, ...} — merged into index rows by file name.
    """
    path = Path(inbox) / "provenance.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        f = row.pop("file", None)
        if f:
            out[f.replace("\\", "/")] = row
    return out


def standardize_inbox(inbox: Path, pool_dir: Path, index_path: Path,
                      target_sr: int = TARGET_SR,
                      reject_flagged: bool = True) -> dict:
    """Convert every inbox wav into the pool. Returns stats summary."""
    inbox, pool_dir = Path(inbox), Path(pool_dir)
    pool_dir.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = load_provenance(inbox)

    rows, accepted, rejected = [], 0, 0
    for src in iter_inbox(inbox):
        rel = src.relative_to(inbox).as_posix()
        label, text = parse_inbox_name(src)
        prov = provenance.get(rel) or {}
        # provenance text/label (from importer sidecar) wins over filename
        if prov.get("text"):
            text = prov["text"]
        if prov.get("label"):
            label = prov["label"]
        try:
            x, sr = load_mono(src, target_sr)
        except Exception as exc:
            rows.append({"source": rel, "status": "reject",
                         "reason": f"unreadable: {exc}"})
            rejected += 1
            continue
        m = measure(x, sr)
        flags = flag_issues(m)
        if not text:
            flags = flags + ["no_text"]
        row = {"source": rel,
               "label": label, "text": text, "metrics": m, "flags": flags,
               "source_sha256": sha256_file(src)}
        if prov:
            row["provenance"] = prov
        hard = [f for f in flags
                if f.split(":")[0] in ("too_short", "non_finite", "empty")]
        if reject_flagged and hard:
            row.update(status="reject", reason=";".join(hard))
            rejected += 1
        else:
            tmp = pool_dir / f".{src.stem[:24]}.tmp.wav"
            write_wav_pcm16(tmp, x, sr)
            sha = sha256_file(tmp)
            dst = pool_dir / f"{sha}.wav"
            tmp.replace(dst)
            row.update(status="ok", audio_sha256=sha, audio=dst.name)
            accepted += 1
        rows.append(row)

    with index_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"inbox": str(inbox), "pool": str(pool_dir),
            "total": len(rows), "accepted": accepted, "rejected": rejected,
            "index": str(index_path)}
