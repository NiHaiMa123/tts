"""Audio file helpers: hashing, wav loading, output path safety."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a wav as float32 mono array + sample rate."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return mono, int(sr)


def wav_info(path: str | Path) -> dict:
    info = sf.info(str(path))
    return {
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "frames": info.frames,
        "duration": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


def safe_output_path(root: str | Path, rel: str | Path) -> Path:
    """Resolve ``rel`` under ``root``; reject escapes via .. or absolutes."""
    root = Path(root).resolve()
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise ValueError(f"output path must be relative: {rel}")
    candidate = (root / rel_path).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"output path escapes root: {rel}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate
