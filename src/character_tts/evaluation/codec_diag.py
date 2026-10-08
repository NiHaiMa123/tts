"""Host-side codec roundtrip diagnostics (Phase 5A).

Pure numpy/scipy analysis used to attribute a bad-sounding roundtrip:
time alignment via cross-correlation, least-squares gain matching,
band-limited spectral comparison, waveform residual, and a stdlib-only
grayscale PNG spectrogram writer (no matplotlib/PIL dependency).

All numbers are descriptive evidence for a human reviewer — they never
decide quality on their own.
"""
from __future__ import annotations

import struct
import zlib
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

_EPS = 1e-12


def resample_to(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Polyphase resample (matches diagnostics.metrics._resample)."""
    if sr == target_sr or len(x) == 0:
        return np.asarray(x, dtype=np.float64)
    g = gcd(int(sr), int(target_sr))
    return signal.resample_poly(
        np.asarray(x, dtype=np.float64), target_sr // g, sr // g)


def audio_sanity(path: str | Path) -> dict[str, Any]:
    """Mechanical checks: readable, finite, non-silent, clipping stats."""
    from ..audio.io import read_wav, wav_info

    rec: dict[str, Any] = {"path": str(path), "ok": True}
    try:
        x, sr = read_wav(path)
        info = wav_info(path)
    except Exception as exc:  # unreadable file is a hard failure
        return {"path": str(path), "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}
    rec.update({
        "sample_rate": sr,
        "duration_s": round(float(len(x) / sr), 3) if sr else 0.0,
        "format": info.get("format"),
        "subtype": info.get("subtype"),
    })
    if len(x) == 0:
        rec.update(ok=False, error="empty wav")
        return rec
    finite = bool(np.isfinite(x).all())
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    rec.update({
        "finite": finite,
        "peak": round(peak, 4),
        "rms_dbfs": round(20.0 * np.log10(max(rms, _EPS)), 2),
        "clip_ratio": round(float(np.mean(np.abs(x) >= 0.999)), 6),
        "silence_ratio": round(float(np.mean(np.abs(x) < 1e-3)), 4),
    })
    if not finite:
        rec.update(ok=False, error="NaN/Inf samples")
    elif rms < 1e-5:
        rec.update(ok=False, error="all-silence output")
    return rec


def estimate_delay_samples(ref: np.ndarray, test: np.ndarray,
                           max_lag: int = 9600) -> tuple[int, float]:
    """Delay of ``test`` relative to ``ref`` in samples + normalized peak.

    Positive delay means ``test`` lags ``ref`` (starts later). Uses FFT
    cross-correlation over a +/-max_lag window.
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if len(ref) == 0 or len(test) == 0:
        return 0, 0.0
    corr = signal.fftconvolve(test, ref[::-1], mode="full")
    mid = len(ref) - 1  # lag 0 index
    lo = max(0, mid - max_lag)
    hi = min(len(corr), mid + max_lag + 1)
    window = corr[lo:hi]
    idx = int(np.argmax(np.abs(window)))
    lag = (idx + lo) - mid
    denom = np.sqrt(np.sum(ref ** 2) * np.sum(test ** 2))
    npeak = float(window[idx] / max(denom, _EPS))
    return lag, npeak


def best_gain(ref: np.ndarray, test: np.ndarray) -> float:
    """Least-squares gain g minimizing ||ref - g*test||."""
    denom = float(np.dot(test, test))
    if denom < _EPS:
        return 0.0
    return float(np.dot(ref, test) / denom)


def aligned_pair(ref: np.ndarray, test: np.ndarray, lag: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Crop ref/test to their overlapping region given a delay."""
    if lag > 0:        # test starts later
        ref_seg = ref[: len(test) - lag] if len(test) > lag else ref[:0]
        test_seg = test[lag: lag + len(ref_seg)]
    elif lag < 0:      # ref starts later
        test_seg = test[: len(ref) + lag] if len(ref) > -lag else test[:0]
        ref_seg = ref[-lag: -lag + len(test_seg)]
    else:
        n = min(len(ref), len(test))
        ref_seg, test_seg = ref[:n], test[:n]
    return ref_seg, test_seg


def _stft_mag(x: np.ndarray, sr: int, nperseg: int = 1024,
              hop: int = 256) -> tuple[np.ndarray, np.ndarray]:
    nperseg = min(nperseg, len(x))
    if nperseg < 16:
        return np.zeros(0), np.zeros((0, 0))
    hop = min(hop, nperseg)  # very short inputs -> non-overlapping frames
    freqs, _, z = signal.stft(
        x, fs=sr, nperseg=nperseg, noverlap=nperseg - hop,
        window="hann", padded=False, boundary=None)
    return freqs, np.abs(z)


def band_comparison(ref: np.ndarray, test: np.ndarray, sr: int,
                    bands: list[tuple[float, float]]
                    ) -> dict[str, Any]:
    """Per-band log-magnitude comparison of equal-length signals.

    Both signals must already be aligned/resampled to the same rate.
    Frame count is matched by truncating to the shorter STFT.
    """
    freqs, mr = _stft_mag(ref, sr)
    _, mt = _stft_mag(test, sr)
    if mr.size == 0 or mt.size == 0:
        return {"error": "signal too short for STFT"}
    n = min(mr.shape[1], mt.shape[1])
    lr = np.log10(mr[:, :n] + _EPS)
    lt = np.log10(mt[:, :n] + _EPS)
    out: dict[str, Any] = {"analysis_sr": sr, "frames": n, "bands": {}}
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            out["bands"][f"{int(lo)}-{int(hi)}"] = None
            continue
        diff_db = (lt[sel] - lr[sel]) * 10.0
        # Pearson correlation of the *shapes* (flattened log spectra)
        a = lr[sel].ravel()
        b = lt[sel].ravel()
        corr = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 else None
        out["bands"][f"{int(lo)}-{int(hi)}"] = {
            "mean_diff_db": round(float(diff_db.mean()), 2),
            "rmse_db": round(float(np.sqrt(np.mean(diff_db ** 2))), 2),
            "logmag_corr": None if corr is None or not np.isfinite(corr)
            else round(corr, 4),
        }
    return out


def compare_pair(ref: np.ndarray, ref_sr: int, test: np.ndarray,
                 test_sr: int, *, analysis_sr: int = 16000,
                 bands: list[tuple[float, float]] | None = None
                 ) -> dict[str, Any]:
    """Full comparison: resample to analysis_sr, time+gain align, then
    waveform residual + per-band spectral diff."""
    bands = bands or [(0, 4000), (4000, 7500), (7500, 12000)]
    r = resample_to(ref, ref_sr, analysis_sr)
    t = resample_to(test, test_sr, analysis_sr)
    lag, npeak = estimate_delay_samples(r, t)
    r_seg, t_seg = aligned_pair(r, t, lag)
    if len(r_seg) < 256:
        return {"error": "overlap too short after alignment", "lag": lag}
    g = best_gain(r_seg, t_seg)
    resid = r_seg - g * t_seg
    nmse = float(np.sum(resid ** 2) / max(np.sum(r_seg ** 2), _EPS))
    wcorr = float(np.corrcoef(r_seg, t_seg)[0, 1])
    out: dict[str, Any] = {
        "analysis_sr": analysis_sr,
        "delay_samples_at_analysis_sr": lag,
        "delay_ms": round(1000.0 * lag / analysis_sr, 2),
        "xcorr_peak": round(npeak, 4),
        "gain_test_to_ref": round(g, 4),
        "waveform_corr": round(wcorr, 4),
        "waveform_nmse": round(nmse, 4),
        "overlap_seconds": round(len(r_seg) / analysis_sr, 2),
    }
    out["spectral"] = band_comparison(
        r_seg, t_seg, analysis_sr,
        [b for b in bands if b[0] < analysis_sr / 2.0])
    return out


# ---------------------------------------------------------------------------
# stdlib-only PNG writer (grayscale) for spectrograms
# ---------------------------------------------------------------------------

def write_png_gray(path: str | Path, img: np.ndarray) -> Path:
    """Write a HxW uint8 array as an 8-bit grayscale PNG."""
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data +
                struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(png)
    return p


def spectrogram_png(x: np.ndarray, sr: int, path: str | Path, *,
                    nperseg: int = 1024, hop: int = 256,
                    fmax: float | None = None,
                    db_floor: float = -80.0) -> Path:
    """Render a linear-frequency log-magnitude spectrogram to grayscale PNG.

    Time runs left->right; low frequencies at the bottom.
    """
    freqs, mag = _stft_mag(x, sr, nperseg=nperseg, hop=hop)
    if mag.size == 0:
        img = np.zeros((64, 64), dtype=np.uint8)
        return write_png_gray(path, img)
    sel = freqs <= (fmax if fmax else sr / 2.0)
    db = 20.0 * np.log10(mag[sel] + _EPS)
    lo, hi = db_floor, max(0.0, float(db.max()))
    img = np.clip((db - lo) / max(hi - lo, _EPS) * 255.0, 0, 255)
    img = img.astype(np.uint8)[::-1]  # flip: low freq at bottom
    return write_png_gray(path, img)
