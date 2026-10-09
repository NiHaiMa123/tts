"""Import decoded game audio into a character inbox with provenance.

Boundary contract: Ludiglot (FModel + vgmstream + event mapping) hands us
a directory of decoded audio plus a jsonl manifest joining each file to
its text and extraction provenance::

    {"audio": "<stem or rel path inside --audio-dir>",
     "text": "<台词原文>",
     "label": "<情绪标签, optional>",
     "event": "play_vo_...",        # wwise event name, optional
     "wem_hash": 1001142754,         # numeric wem id, optional
     "source_wem": "Media/zh/1001142754.wem",  # optional
     "tool": "vgmstream ...",        # optional free text
     ...any extra keys are preserved}

import writes ``<inbox>/<label>/<stem>.<ext>`` (flat when no label) and an
``<inbox>/provenance.jsonl`` sidecar that standardize_inbox merges.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

AUDIO_EXTS = (".wav", ".ogg", ".flac", ".mp3")


def load_manifest(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "audio" not in row:
            raise ValueError(f"manifest line {i}: missing 'audio' field")
        rows.append(row)
    return rows


def _find_audio(audio_dir: Path, key: str) -> Path | None:
    p = audio_dir / key
    if p.is_file():
        return p
    if Path(key).suffix:
        return None
    for ext in AUDIO_EXTS:
        cand = audio_dir / f"{key}{ext}"
        if cand.is_file():
            return cand
    matches = [p for p in audio_dir.rglob(f"{key}.*")
               if p.suffix.lower() in AUDIO_EXTS]
    return matches[0] if matches else None


def import_ludiglot(audio_dir: Path, manifest_path: Path, out_inbox: Path,
                    default_label: str | None = None,
                    copy: bool = True) -> dict:
    """Materialize inbox files + provenance.jsonl. Returns stats."""
    audio_dir, out_inbox = Path(audio_dir), Path(out_inbox)
    rows = load_manifest(manifest_path)
    out_inbox.mkdir(parents=True, exist_ok=True)

    prov_rows, imported, missing, seen = [], 0, [], set()
    for row in rows:
        src = _find_audio(audio_dir, str(row["audio"]))
        if src is None:
            missing.append(row["audio"])
            continue
        label = row.get("label") or default_label or ""
        stem = src.stem
        rel = f"{label}/{stem}{src.suffix.lower()}" if label \
            else f"{stem}{src.suffix.lower()}"
        if rel in seen:                     # same file listed twice
            continue
        seen.add(rel)
        dst = out_inbox / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                dst.hardlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        prov = {"file": rel, "text": row.get("text") or "",
                "label": label or None}
        for k in ("event", "wem_hash", "source_wem", "tool"):
            if row.get(k) is not None:
                prov[k] = row[k]
        extra = {k: v for k, v in row.items()
                 if k not in prov and k not in ("audio", "text", "label")}
        prov.update(extra)
        prov_rows.append(prov)
        imported += 1

    prov_path = out_inbox / "provenance.jsonl"
    with prov_path.open("w", encoding="utf-8", newline="\n") as f:
        for pr in prov_rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    return {"inbox": str(out_inbox), "imported": imported,
            "missing": len(missing), "missing_keys": missing,
            "provenance": str(prov_path)}
