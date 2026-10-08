"""Phase 5A tests: codec diagnostics math, manifest/runner wiring,
prompt-A/B pairing and reuse verification, blind-listen page."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml

from character_tts.backends.manager import BackendManager
from character_tts.evaluation import codec_diag
from character_tts.evaluation import phase5a
from character_tts.registry import loader
from character_tts.registry.models import BackendProfile, CharacterProfile


def _wav(path: Path, seconds: float = 0.1, sr: int = 8000,
         freq: float = 0.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        if freq:
            frames = b"".join(
                struct.pack("<h", int(12000 * np.sin(2 * np.pi * freq * i / sr)))
                for i in range(n))
        else:
            frames = b"".join(
                struct.pack("<h", int(8000 * ((i * 7) % 100 - 50) / 50))
                for i in range(n))
        w.writeframes(frames)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- codec_diag --------------------------------------------------------

def test_audio_sanity_ok_and_silent(tmp_path):
    ok_wav = _wav(tmp_path / "ok.wav", freq=440)
    rec = codec_diag.audio_sanity(ok_wav)
    assert rec["ok"] and rec["finite"] and rec["sample_rate"] == 8000
    assert rec["rms_dbfs"] > -60

    silent = tmp_path / "sil.wav"
    silent.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    rec = codec_diag.audio_sanity(silent)
    assert not rec["ok"] and "silence" in rec["error"]


def test_estimate_delay_and_gain():
    sr = 16000
    t = np.arange(sr // 2) / sr
    ref = np.sin(2 * np.pi * 300 * t) * 0.5
    lag_true = 400
    test = np.concatenate([np.zeros(lag_true), ref]) * 0.7
    lag, peak = codec_diag.estimate_delay_samples(ref, test)
    assert lag == lag_true and peak > 0.5
    r, te = codec_diag.aligned_pair(ref, test, lag)
    g = codec_diag.best_gain(r, te)
    assert abs(g - 1.0 / 0.7) < 0.02


def test_compare_pair_identical_vs_noise():
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr) / sr
    ref = np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    same = codec_diag.compare_pair(ref, sr, ref.copy(), sr)
    assert same["waveform_corr"] > 0.999
    assert same["waveform_nmse"] < 0.01
    noise = codec_diag.compare_pair(ref, sr, rng.standard_normal(sr), sr)
    assert noise["waveform_corr"] < 0.3
    assert noise["waveform_nmse"] > same["waveform_nmse"]


def test_write_png_gray_valid(tmp_path):
    img = np.tile(np.linspace(0, 255, 200, dtype=np.uint8), (100, 1))
    p = codec_diag.write_png_gray(tmp_path / "s.png", img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (200, 100)


# -- runner ------------------------------------------------------------

@pytest.fixture
def phase5a_env(tmp_path, monkeypatch):
    """Fake backend + tiny config driving run_phase5a end-to-end."""
    from conftest import FAKE_WORKER, REPO_ROOT  # noqa: N813

    ref = _wav(tmp_path / "ref.wav", freq=330)
    gt = _wav(tmp_path / "gt.wav", seconds=0.2, freq=440)
    extra = _wav(tmp_path / "extra.wav", seconds=0.15, freq=550)

    character = CharacterProfile(
        character_id="suoming", display_name="锁暝", dataset={},
        reference={"audio": str(ref), "text": "prompt text",
                   "sha256": _sha(ref)},
        evaluation={"anchor_texts": [
            {"id": "rain", "text": "anchor text", "audio": str(gt),
             "sha256": _sha(gt), "source": "validation.jsonl"}]},
    )

    profile = BackendProfile(
        backend_id="fake", family="fake", enabled=True,
        worker={"python": sys.executable, "script": str(FAKE_WORKER),
                "cwd": str(REPO_ROOT), "startup_timeout_seconds": 30,
                "env": {"FAKE_ROUNDTRIP_OK": "1"}},
        model={"path_or_id": "fake/model", "revision": "r1"},
        generation={"cfg_value": 2.0, "inference_timesteps": 10},
    )
    cfgs = tmp_path / "backend_cfgs"
    cfgs.mkdir()
    (cfgs / "fake.yaml").write_text(
        yaml.safe_dump({"backend_id": "fake", "family": "fake",
                        "enabled": True, "worker": profile.worker,
                        "model": profile.model,
                        "generation": profile.generation},
                       allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(
        phase5a, "load_backend",
        lambda name: loader.load_backend(str(cfgs / f"{name}.yaml")))
    monkeypatch.setattr(phase5a, "repo_root", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)

    p1 = _wav(tmp_path / "p1.wav", freq=500)
    cfg = {
        "experiment_id": "test_phase5a",
        "output_dir": "phase5a_out",
        "backend": "fake",
        "prompts": [
            {"id": "P0", "role": "baseline", "audio": str(ref),
             "text": "prompt text", "sha256": _sha(ref)},
            {"id": "P1", "role": "candidate", "audio": str(p1),
             "text": "p1 text", "sha256": _sha(p1)},
        ],
        "texts": [
            {"id": "rain", "text": "anchor text", "split": "validation",
             "audio": str(gt), "sha256": _sha(gt),
             "has_ground_truth": True},
            {"id": "short", "text": "短句", "split": "validation"},
        ],
        "stability": {"texts": ["rain", "short"], "seeds": [42, 43]},
        "prompt_ab": {"baseline_prompt": "P0", "prompts": ["P0", "P1"],
                      "texts": ["rain", "short"], "seed": 42},
        "codec_diagnosis": {
            "variants": ["default", "cond16000"],
            "inputs": [{"id": "gt", "audio": str(gt), "sha256": _sha(gt)},
                       {"id": "extra", "audio": str(extra),
                        "sha256": _sha(extra)}]},
    }
    return character, cfg, tmp_path


def test_run_phase5a_end_to_end(phase5a_env):
    character, cfg, tmp_path = phase5a_env
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        manifest = phase5a.run_phase5a(cfg, character, manager=mgr,
                                       log=lambda *a: None)
    finally:
        mgr.stop()

    stab = [c for c in manifest["cases"] if c["group"] == "stability"]
    ab = [c for c in manifest["cases"] if c["group"] == "prompt_ab"]
    assert len(stab) == 4 and len(ab) == 4          # 2 texts x 2 seeds, 2 prompts x 2 texts
    assert all(c["status"] == "ok" for c in manifest["cases"])

    # every generated case carries the full manifest fields
    for c in stab:
        assert c["output_sha256"] and c["prompt_sha256"]
        assert c["seed"] in (42, 43) and c["prompt_id"] == "P0"
        assert c["text"] and c["sample_rate"] == 8000
        assert c["wall_seconds"] is not None

    # P0 A/B cells must reuse the identical stability wav (hash-verified),
    # P1 cells must be freshly generated.
    for c in ab:
        if c["prompt_id"] == "P0":
            assert c["reused_from"].startswith("stability/")
            src = next(s for s in stab
                       if s["case_id"] == c["reused_from"])
            assert c["output_sha256"] == src["output_sha256"]
        else:
            assert c["reused_from"] is None
            assert Path(c["output"]).name.startswith("P1_")

    assert manifest["budget"]["zero_shot_generated"] == 6  # 4 stab + 2 P1

    diag = manifest["codec_diagnosis"]
    assert set(diag) == {"gt", "extra"}
    for entry in diag.values():
        assert entry["status"] == "ok"
        assert entry["probe_result"]["encode_sample_rate"] == 16000
        assert "original_vs_roundtrip" in entry["analysis"]["pairs"]

    out_dir = tmp_path / "phase5a_out"
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "listen" / "index.html").is_file()
    assert (out_dir / "listen" / "unblind_map.json").is_file()

    html = (out_dir / "listen" / "index.html").read_text(encoding="utf-8")
    assert "test_phase5a" in html and "PENDING_USER_LISTENING" in html
    # labels are anonymized but the reveal map keeps true ids
    unblind = json.loads(
        (out_dir / "listen" / "unblind_map.json").read_text(encoding="utf-8"))
    labels = [v["case_id"] for v in unblind.values()]
    assert "prompt_ab/P1_rain_seed42" in labels


def test_phase5a_prompt_sha_mismatch_blocks(phase5a_env):
    character, cfg, _ = phase5a_env
    cfg["prompts"][1]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        phase5a.run_phase5a(cfg, character,
                            manager=BackendManager(
                                logs_dir=Path(cfg["output_dir"]) / "x"),
                            log=lambda *a: None)


def test_load_phase5a_config_env_expansion(tmp_path, monkeypatch):
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(
        "output_dir: ${TTS_OUT_ROOT}/x\nbackend: fake\n", encoding="utf-8")
    monkeypatch.setenv("TTS_OUT_ROOT", "D:/data")
    cfg = phase5a.load_phase5a_config(str(cfg_path))
    assert cfg["output_dir"] == "D:/data/x"
