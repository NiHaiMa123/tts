"""音频预处理段：入库前的清洗（trim / 高通 / 响度归一）。

游戏解包音（Wwise wem→wav）常带底噪、编码伪影、首尾静音和响度
不一致——LoRA 会把这些声学签名一并学走（用户听感"磨砂"）。本段在
standardize 之前对 inbox 做无损保留原件的镜像清洗：

    inbox/ ──preprocess──> inbox_clean/   （子目录结构保留）

操作顺序：trim → highpass → loudness。denoise（ZipEnhancer ANS）
是可选重型操作，由 CLI 通过 voxcpm env 子进程执行后再归一化；
其输出为 16kHz，重采样回 48k 后有效带宽 ≤8kHz——对 VoxCPM2
训练无损（AudioVAE 编码输入本来就是 16k），写 provenance 时如实记录。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_SR = 48000


def trim_silence(wav: np.ndarray, sr: int, gate_db: float = -45.0,
                 pad_ms: int = 80, win_ms: int = 10) -> np.ndarray:
    """Energy-gate leading/trailing silence; keeps ``pad_ms`` padding."""
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = max(1, int(sr * win_ms / 1000))
    frames = len(x) // n
    if frames < 3:
        return x
    rms = np.sqrt((x[:frames * n].reshape(frames, n) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    active = np.flatnonzero(db > gate_db)
    if not len(active):
        return x                       # 全静音：不动，交给下游判 bad
    pad = int(sr * pad_ms / 1000)
    lo = max(0, active[0] * n - pad)
    hi = min(len(x), (active[-1] + 1) * n + pad)
    return x[lo:hi]


def highpass(wav: np.ndarray, sr: int, cutoff_hz: float = 70.0,
             order: int = 4) -> np.ndarray:
    """Butterworth high-pass — removes rumble/DC drift."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    x = np.asarray(wav, dtype=np.float32)
    return sosfiltfilt(sos, x).astype(np.float32)


def normalize_loudness(wav: np.ndarray, target_rms_dbfs: float = -25.0,
                       peak_ceiling: float = 0.98) -> np.ndarray:
    """RMS normalize; gain is clipped so peaks never exceed ceiling."""
    x = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    if rms < 1e-9:
        return x
    cur_db = 20 * math.log10(rms)
    gain = 10 ** ((target_rms_dbfs - cur_db) / 20)
    peak = float(np.abs(x).max())
    if peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
    return (x * gain).astype(np.float32)


def preprocess_wav(wav: np.ndarray, sr: int, *,
                   trim: bool = True, hp: bool = True,
                   loudness: bool = True,
                   trim_db: float = -45.0, hp_hz: float = 70.0,
                   target_dbfs: float = -25.0) -> np.ndarray:
    """In-memory pipeline: trim → highpass → loudness."""
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if trim:
        x = trim_silence(x, sr, gate_db=trim_db)
    if hp:
        x = highpass(x, sr, cutoff_hz=hp_hz)
    if loudness:
        x = normalize_loudness(x, target_rms_dbfs=target_dbfs)
    return x


def preprocess_file(in_path: Path, out_path: Path, *,
                    ops: dict | None = None,
                    sr: int = DEFAULT_SR) -> dict:
    """Load → preprocess → write PCM16. Returns a per-file report row."""
    ops = ops or {}
    wav, in_sr = sf.read(str(in_path), always_2d=False)
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if in_sr != sr:                     # 统一到管线采样率
        from scipy.signal import resample_poly
        import math as _m
        g = _m.gcd(in_sr, sr)
        x = resample_poly(x, sr // g, in_sr // g).astype(np.float32)
    in_dur = len(x) / sr
    out = preprocess_wav(x, sr, **{k: v for k, v in ops.items()
                                 if k in ("trim", "hp", "loudness",
                                          "trim_db", "hp_hz",
                                          "target_dbfs")})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr, subtype="PCM_16")
    return {
        "source": str(in_path), "output": str(out_path),
        "duration_in": round(in_dur, 3),
        "duration_out": round(len(out) / sr, 3),
        "ops": {k: v for k, v in ops.items() if k != "denoise"},
        "denoise": bool(ops.get("denoise")),
        "bandwidth_note": ("denoise: 模型输出16k,有效带宽≤8kHz"
                           if ops.get("denoise") else None),
    }
