from __future__ import annotations

import numpy as np
import pytest

from character_tts.diagnostics.metrics import (
    analyze_audio,
    band_energy_ratio,
    rms_dbfs,
    temporal_spectral_delta,
    true_peak_dbtp,
    voiced_spectral_stats,
)

SR = 48000


def _sine(freq, seconds=1.0, amp=0.5, sr=SR):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_rms_dbfs():
    # sine amplitude 0.5 -> RMS = 0.5/sqrt(2) ≈ -9.03 dBFS
    assert rms_dbfs(_sine(440)) == pytest.approx(-9.03, abs=0.05)


def test_true_peak_near_amplitude():
    tp = true_peak_dbtp(_sine(1000, amp=0.25), SR)
    assert tp == pytest.approx(-12.04, abs=0.3)


def test_band_energy_concentration():
    x = _sine(6000, amp=0.5)
    assert band_energy_ratio(x, SR, 4000, 8000) > 0.9
    assert band_energy_ratio(x, SR, 8000, 12000) < 0.05


def test_band_energy_respects_nyquist():
    # 12-18k band on a 16kHz file: hi clamps to nyquist, no crash
    x = _sine(6000, sr=16000)
    assert band_energy_ratio(x, 16000, 12000, 18000) == 0.0


def test_flatness_orders_noise_above_sine():
    rng = np.random.default_rng(0)
    sine_stats = voiced_spectral_stats(_sine(2000, seconds=1.0), SR)
    noise_stats = voiced_spectral_stats(rng.normal(0, 0.1, SR), SR)
    assert sine_stats["spectral_flatness"] < noise_stats["spectral_flatness"]
    assert 0.0 < sine_stats["spectral_entropy"] <= 1.0


def test_temporal_delta_positive_for_mixed_signal():
    rng = np.random.default_rng(1)
    x = _sine(5000, seconds=1.0) + rng.normal(0, 0.01, SR)
    d = temporal_spectral_delta(x, SR)
    assert d is not None and d >= 0


def test_analyze_audio_bundle():
    out = analyze_audio(_sine(3000), SR)
    for key in ("duration", "sample_rate", "analysis_sr", "rms_dbfs",
                "true_peak_dbtp", "band_energy_4k_8k",
                "spectral_flatness", "temporal_delta_2k_9k"):
        assert key in out
    assert out["duration"] == pytest.approx(1.0, abs=0.01)


def test_shared_band_metrics_comparable_across_rates():
    """Same 8 kHz-dominant signal at 48 kHz vs its 24 kHz version must
    produce nearly identical shared-band numbers — the whole point of
    analyzing at a common rate."""
    from scipy import signal as _sig

    rng = np.random.default_rng(0)
    base = (_sine(6000, seconds=1.0, amp=0.4)
            + _sine(2000, seconds=1.0, amp=0.3)
            + rng.normal(0, 0.02, SR))
    at_48 = analyze_audio(base, SR)
    down = _sig.resample_poly(base, 1, 2)
    at_24 = analyze_audio(down, 24000)
    assert at_48["analysis_sr"] == 24000 == at_24["analysis_sr"]
    for key in ("band_energy_4k_8k", "band_energy_8k_12k",
                "temporal_delta_2k_9k"):
        assert at_48[key] == pytest.approx(at_24[key], abs=0.02)


def test_native_hf_context_only_when_supported():
    x = _sine(15000, amp=0.3) + _sine(3000, amp=0.5)
    out48 = analyze_audio(x, SR)
    assert out48["native_energy_above_12k"] is not None
    assert out48["native_energy_above_12k"] > 0.1
    down = _sine(3000, amp=0.5, sr=24000)
    out24 = analyze_audio(down, 24000)
    assert out24["native_energy_above_12k"] is None
