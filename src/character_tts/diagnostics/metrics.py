"""Objective audio diagnostics for backend gates.

Metrics are descriptive only — per project rules no single metric may be
used to declare a quality winner; subjective listening stays authoritative.

Implemented metrics (plan section 9):
  duration, sample_rate, rms_dbfs, lufs, true_peak_dbtp,
  band energy ratios (4-8k, 8-12k, 12-18k),
  voiced spectral flatness, spectral crest, spectral entropy,
  temporal spectral delta (2-9kHz), optional speaker cosine (hook only).
"""

from __future__ import annotations

import numpy as np
from scipy import signal

_EPS = 1e-12


def _db(x: float) -> float:
    return float(20.0 * np.log10(max(x, _EPS)))


def rms_dbfs(x: np.ndarray) -> float:
    return _db(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))))


def true_peak_dbtp(x: np.ndarray, sr: int, oversample: int = 4) -> float:
    """Approximate BS.1770 true peak via polyphase oversampling."""
    if len(x) < oversample * 8:
        return _db(float(np.max(np.abs(x))) if len(x) else 0.0)
    up = signal.resample_poly(x, oversample, 1)
    return _db(float(np.max(np.abs(up))))


def integrated_lufs(x: np.ndarray, sr: int) -> float | None:
    """BS.1770 integrated loudness via pyloudnorm; None when unavailable."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    try:
        meter = pyln.Meter(sr)
        value = meter.integrated_loudness(x.astype(np.float64))
    except Exception:
        return None
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _stft_mag(x: np.ndarray, sr: int, nperseg: int = 2048,
              hop: int = 512) -> tuple[np.ndarray, np.ndarray]:
    # No boundary/padding so frame count == 1 + (len - nperseg) // hop,
    # matching _voiced_frame_mask. Short inputs shrink nperseg to len.
    freqs, _, z = signal.stft(
        x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
        window="hann", padded=False, boundary=None,
    )
    return freqs, np.abs(z)


def band_energy_ratio(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    """Fraction of total spectral energy inside [lo, hi] Hz."""
    if len(x) < 64:
        return 0.0
    freqs, mag = _stft_mag(x, sr)
    power = np.square(mag)
    nyq = sr / 2.0
    hi = min(hi, nyq)
    if lo >= hi:
        return 0.0
    band = power[(freqs >= lo) & (freqs < hi)].sum()
    total = power.sum()
    return float(band / max(total, _EPS))


def _voiced_frame_mask(x: np.ndarray, hop: int = 512,
                       nperseg: int = 2048, gate_db: float = 40.0) -> np.ndarray:
    padded = np.pad(x, (0, nperseg - len(x))) if len(x) < nperseg else x
    n_frames = 1 + (len(padded) - nperseg) // hop
    frames = np.lib.stride_tricks.sliding_window_view(
        padded, nperseg)[::hop][:n_frames]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    peak = rms.max()
    if peak <= 0:
        return np.zeros(len(rms), dtype=bool)
    return rms >= peak * 10.0 ** (-gate_db / 20.0)


def voiced_spectral_stats(x: np.ndarray, sr: int) -> dict[str, float | None]:
    """Flatness / crest / entropy averaged over voiced frames only."""
    freqs, mag = _stft_mag(x, sr)
    if mag.shape[1] == 0:
        return {"spectral_flatness": None, "spectral_crest": None,
                "spectral_entropy": None}
    mask = _voiced_frame_mask(x)
    mask = mask[: mag.shape[1]]
    if not mask.any():
        return {"spectral_flatness": None, "spectral_crest": None,
                "spectral_entropy": None}
    m = mag[:, mask] + _EPS
    flat = np.exp(np.mean(np.log(m), axis=0)) / np.mean(m, axis=0)
    crest = np.max(m, axis=0) / np.mean(m, axis=0)
    psd = m / m.sum(axis=0, keepdims=True)
    entropy = -(psd * np.log2(psd)).sum(axis=0) / np.log2(m.shape[0])
    return {
        "spectral_flatness": float(np.mean(flat)),
        "spectral_crest": float(np.mean(crest)),
        "spectral_entropy": float(np.mean(entropy)),
    }


def temporal_spectral_delta(x: np.ndarray, sr: int,
                            lo: float = 2000.0, hi: float = 9000.0) -> float | None:
    """Mean |Δ log10 band-energy| between consecutive frames in [lo, hi]."""
    freqs, mag = _stft_mag(x, sr)
    hi = min(hi, sr / 2.0)
    band = (freqs >= lo) & (freqs < hi)
    if not band.any() or mag.shape[1] < 2:
        return None
    energy = np.square(mag[band]).sum(axis=0)
    log_e = np.log10(energy + _EPS)
    return float(np.mean(np.abs(np.diff(log_e))))


def analyze_audio(x: np.ndarray, sr: int) -> dict[str, float | int | None]:
    """Full metric bundle for one mono signal."""
    x = np.asarray(x, dtype=np.float64)
    out: dict[str, float | int | None] = {
        "sample_rate": int(sr),
        "duration": float(len(x) / sr) if sr else 0.0,
        "rms_dbfs": rms_dbfs(x) if len(x) else None,
        "lufs": integrated_lufs(x, sr) if len(x) else None,
        "true_peak_dbtp": true_peak_dbtp(x, sr) if len(x) else None,
        "band_energy_4k_8k": band_energy_ratio(x, sr, 4000, 8000),
        "band_energy_8k_12k": band_energy_ratio(x, sr, 8000, 12000),
        "band_energy_12k_18k": band_energy_ratio(x, sr, 12000, 18000),
        "temporal_delta_2k_9k": temporal_spectral_delta(x, sr),
    }
    out.update(voiced_spectral_stats(x, sr))
    return out


def analyze_file(path: str) -> dict[str, float | int | None]:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return analyze_audio(data.mean(axis=1), int(sr))
