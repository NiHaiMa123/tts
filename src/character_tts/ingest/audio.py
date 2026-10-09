"""Audio primitives for ingest: load/resample/measure. stdlib+numpy+soundfile."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 48000          # repo convention: dataset pool is 48kHz mono
MIN_DURATION_S = 0.4
MAX_DURATION_S = 120.0


def load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Read wav, downmix to mono, resample to target_sr."""
    data, sr = sf.read(str(path), always_2d=True)
    x = data.mean(axis=1).astype(np.float64)
    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        x = resample_poly(x, target_sr // g, sr // g)
    return x, target_sr


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def measure(x: np.ndarray, sr: int) -> dict:
    """Sanity metrics for one utterance."""
    n = len(x)
    if n == 0:
        return {"ok": False, "reason": "empty"}
    peak = float(np.abs(x).max())
    rms = float(np.sqrt((x ** 2).mean()))
    # fraction of 20ms frames below -50 dBFS
    frame = int(sr * 0.02) or 1
    frames = np.abs(x[: n // frame * frame]).reshape(-1, frame)
    silent = float((frames.max(axis=1) < 10 ** (-50 / 20)).mean()) if len(frames) else 1.0
    clipped = int((np.abs(x) > 0.999).sum())
    return {
        "ok": True,
        "sample_rate": sr,
        "duration_s": round(n / sr, 3),
        "peak": round(peak, 4),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2),
        "clipped_samples": clipped,
        "silence_ratio": round(silent, 3),
        "finite": bool(np.isfinite(x).all()),
    }


def flag_issues(m: dict, min_s: float = MIN_DURATION_S,
                max_s: float = MAX_DURATION_S) -> list[str]:
    """Return human-readable rejection/flag reasons for measured audio."""
    if not m.get("ok"):
        return [m.get("reason", "unreadable")]
    flags = []
    if m["duration_s"] < min_s:
        flags.append(f"too_short:{m['duration_s']}s")
    if m["duration_s"] > max_s:
        flags.append(f"too_long:{m['duration_s']}s")
    if m["clipped_samples"] > 0:
        flags.append(f"clipped:{m['clipped_samples']}")
    if m["silence_ratio"] > 0.6:
        flags.append(f"mostly_silent:{m['silence_ratio']}")
    if m["rms_dbfs"] < -45:
        flags.append(f"too_quiet:{m['rms_dbfs']}dBFS")
    if m["rms_dbfs"] > -6:
        flags.append(f"too_hot:{m['rms_dbfs']}dBFS")
    if not m["finite"]:
        flags.append("non_finite")
    return flags


def write_wav_pcm16(path: Path, x: np.ndarray, sr: int) -> None:
    """Write mono PCM_16 wav (dataset/ref convention)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(x, -1.0, 1.0)
    sf.write(str(path), (y * 32767).astype(np.int16), sr, subtype="PCM_16")
