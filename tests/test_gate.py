from __future__ import annotations

import hashlib
import json
import struct
import sys
import wave
from pathlib import Path

import pytest
import yaml

from character_tts.backends.manager import BackendManager
from character_tts.evaluation.gate import run_gate, verify_reference
from character_tts.evaluation.report import finalize
from character_tts.registry import loader
from character_tts.registry.models import (
    BackendProfile,
    CharacterProfile,
    EvaluationConfig,
    GateCase,
)


def _wav(path: Path, seconds: float = 0.1, sr: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(8000 * ((i * 7) % 100 - 50) / 50))
            for i in range(n)
        )
        w.writeframes(frames)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def character(tmp_path):
    ref = tmp_path / "ref.wav"
    gt = tmp_path / "gt.wav"
    _wav(ref)
    _wav(gt, seconds=0.15)
    return CharacterProfile(
        character_id="suoming",
        display_name="锁暝",
        dataset={},
        reference={"audio": str(ref), "text": "ref text",
                   "sha256": _sha(ref)},
        evaluation={"anchor_texts": [
            {"id": "rain", "text": "anchor",
             "audio": str(gt), "sha256": _sha(gt),
             "source": "validation.jsonl"},
        ]},
    )


def _fake_backend(tmp_path, monkeypatch, backend_id="fake",
                  env=None):
    from conftest import FAKE_WORKER, REPO_ROOT
    profile = BackendProfile(
        backend_id=backend_id,
        family="fake",
        enabled=True,
        worker={
            "python": sys.executable,
            "script": str(FAKE_WORKER),
            "cwd": str(REPO_ROOT),
            "startup_timeout_seconds": 30,
            "env": env or {},
        },
        model={"path_or_id": "fake/model", "revision": "r1"},
    )
    configs_dir = tmp_path / "backend_cfgs"
    configs_dir.mkdir(exist_ok=True)
    (configs_dir / f"{backend_id}.yaml").write_text(
        yaml.safe_dump({
            "backend_id": backend_id,
            "family": "fake",
            "enabled": True,
            "worker": profile.worker,
            "model": profile.model,
        }, allow_unicode=True),
        encoding="utf-8",
    )
    return configs_dir


def _eval_cfg(tmp_path, cases):
    return EvaluationConfig(
        evaluation_id="test_gate",
        character="suoming",
        output_dir=tmp_path / "gate_out",
        seed=42,
        cases=cases,
    )


def test_verify_reference_hash(tmp_path, character):
    info = verify_reference(character)
    assert info["sha256"] == _sha(Path(character.reference_audio))
    character.reference["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_reference(character)


def test_gate_end_to_end(tmp_path, character, monkeypatch):
    cfgs = _fake_backend(tmp_path, monkeypatch, backend_id="fake")
    monkeypatch.setattr(
        "character_tts.evaluation.gate.load_backend",
        lambda name: loader.load_backend(str(cfgs / f"{name}.yaml")),
    )
    ev = _eval_cfg(tmp_path, [
        GateCase(backend="fake", kind="zero_shot", output="fake/zero.wav"),
        GateCase(backend="fake", kind="codec_roundtrip",
                 output="fake/roundtrip.wav"),
    ])
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        report = run_gate(ev, character, manager=mgr)
    finally:
        mgr.stop()

    by_kind = {c["kind"]: c for c in report["cases"]}
    assert by_kind["zero_shot"]["status"] == "ok"
    # fake worker has no codec -> honest unsupported, not a fake pass
    assert by_kind["codec_roundtrip"]["status"] == "unsupported"

    out_dir = ev.output_dir
    assert (out_dir / "fake" / "zero.wav").is_file()
    assert (out_dir / "fake" / "zero.wav.json").is_file()
    sidecar = json.loads(
        (out_dir / "fake" / "zero.wav.json").read_text(encoding="utf-8"))
    assert sidecar["backend_id"] == "fake"
    assert sidecar["seed"] == 42
    assert sidecar["output_sha256"] == _sha(out_dir / "fake" / "zero.wav")
    assert (out_dir / "reference_original.wav").is_file()
    assert (out_dir / "ground_truth_original.wav").is_file()
    assert report["ground_truth"]["sha256"] == _sha(
        Path(character.ground_truth["audio"]))
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "report.json").is_file()

    paths = finalize(out_dir, "t", "m")
    assert paths["report_md"].is_file()
    assert paths["listen_page"].is_file()
    html = paths["listen_page"].read_text(encoding="utf-8")
    assert "zero.wav" in html


def test_codec_roundtrip_blocked_without_ground_truth(tmp_path, character,
                                                      monkeypatch):
    """No ground truth -> codec case blocked, never silently prompt-based."""
    character.evaluation["anchor_texts"] = [{"id": "rain", "text": "anchor"}]
    cfgs = _fake_backend(tmp_path, monkeypatch, backend_id="fake")
    monkeypatch.setattr(
        "character_tts.evaluation.gate.load_backend",
        lambda name: loader.load_backend(str(cfgs / f"{name}.yaml")),
    )
    ev = _eval_cfg(tmp_path, [
        GateCase(backend="fake", kind="codec_roundtrip",
                 output="fake/roundtrip.wav"),
    ])
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        report = run_gate(ev, character, manager=mgr)
    finally:
        mgr.stop()
    assert report["cases"][0]["status"] == "blocked"
    assert "ground truth" in report["cases"][0]["reason"]


def test_gate_case_error_does_not_abort_others(tmp_path, character,
                                             monkeypatch):
    cfgs = _fake_backend(tmp_path, monkeypatch, backend_id="fake",
                         env={"FAKE_CRASH_ON": "generate"})
    monkeypatch.setattr(
        "character_tts.evaluation.gate.load_backend",
        lambda name: loader.load_backend(str(cfgs / f"{name}.yaml")),
    )
    ev = _eval_cfg(tmp_path, [
        GateCase(backend="fake", kind="zero_shot", output="fake/a.wav"),
        GateCase(backend="fake", kind="generate", output="fake/b.wav"),
    ])
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        report = run_gate(ev, character, manager=mgr)
    finally:
        mgr.stop()
    assert all(c["status"] == "error" for c in report["cases"])
    assert len(report["cases"]) == 2  # second case still attempted
