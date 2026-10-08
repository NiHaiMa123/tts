"""Phase 5A2 tests: reuse sha verification, pair matrix, blind page,
context-variance bookkeeping, ASR merge."""
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
from character_tts.evaluation import phase5a2
from character_tts.registry import loader
from character_tts.registry.models import BackendProfile, CharacterProfile


def _wav(path: Path, seconds: float = 0.1, sr: int = 8000,
         freq: float = 440.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(12000 * (i % 16 < 8)))
            if not freq else
            struct.pack("<h", int(8000 * (1 if (i * freq // sr) % 2 else -1)))
            for i in range(n)))
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def a2_env(tmp_path, monkeypatch):
    """Fake backend + prior-experiment manifest + a2 config."""
    from conftest import FAKE_WORKER, REPO_ROOT  # noqa: N813

    ref = _wav(tmp_path / "ref.wav")
    p2 = _wav(tmp_path / "p2.wav", freq=550)
    gt = _wav(tmp_path / "gt.wav", seconds=0.2)
    extra = _wav(tmp_path / "extra.wav", freq=660)

    character = CharacterProfile(
        character_id="suoming", display_name="锁暝", dataset={},
        reference={"audio": str(ref), "text": "P0 text",
                   "sha256": _sha(ref)},
        evaluation={"anchor_texts": [
            {"id": "rain", "text": "gt text", "audio": str(gt),
             "sha256": _sha(gt), "source": "validation.jsonl"}]},
    )
    profile = BackendProfile(
        backend_id="fake", family="fake", enabled=True,
        worker={"python": sys.executable, "script": str(FAKE_WORKER),
                "cwd": str(REPO_ROOT), "startup_timeout_seconds": 30,
                "env": {}},
        model={"path_or_id": "fake/model", "revision": "rev1"},
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
    monkeypatch.setattr(phase5a2, "load_backend",
                        lambda name: loader.load_backend(
                            str(cfgs / f"{name}.yaml")))

    # Fabricated prior experiment with two reusable P0 stability wavs.
    old_dir = tmp_path / "old_exp"
    old_dir.mkdir()
    old_cases = []
    for seed in (42, 43, 44):
        w = _wav(old_dir / f"stab_seed{seed}.wav", freq=440)
        old_cases.append({
            "case_id": f"stability/short_response_seed{seed}",
            "group": "stability", "status": "ok",
            "text_id": "short_response", "text": "开伞，由我来动手。",
            "prompt_id": "P0", "prompt_sha256": _sha(ref),
            "seed": seed, "output": f"stab_seed{seed}.wav",
            "output_sha256": _sha(w), "sample_rate": 8000,
            "duration_s": 0.1,
            "effective_args": dict(phase5a2.EXPECTED_GEN_ARGS),
            "retry_count": 0, "wall_seconds": 0.1,
        })
    w = _wav(old_dir / "stab_mid_seed44.wav", freq=440)
    old_cases.append({
        "case_id": "stability/mid_exposition_seed44",
        "group": "stability", "status": "ok",
        "text_id": "mid_exposition", "text": "中长句文本",
        "prompt_id": "P0", "prompt_sha256": _sha(ref),
        "seed": 44, "output": "stab_mid_seed44.wav",
        "output_sha256": _sha(w), "sample_rate": 8000,
        "duration_s": 0.1,
        "effective_args": dict(phase5a2.EXPECTED_GEN_ARGS),
        "retry_count": 0, "wall_seconds": 0.1,
    })
    (old_dir / "manifest.json").write_text(json.dumps(
        {"cases": old_cases,
         "model": {"path_or_id": "fake/model", "revision": "rev1"}},
        ensure_ascii=False), encoding="utf-8")

    cfg = {
        "experiment_id": "test_a2",
        "output_dir": "a2_out",
        "backend": "fake",
        "root": str(tmp_path),
        "reuse_from": str(old_dir),
        "prompts": [
            {"id": "P0", "role": "baseline", "audio": str(ref),
             "text": "P0 text", "sha256": _sha(ref)},
            {"id": "P2", "role": "candidate", "audio": str(p2),
             "text": "P2 text", "sha256": _sha(p2)},
        ],
        "texts": [
            {"id": "short_response", "text": "开伞，由我来动手。",
             "split": "validation"},
            {"id": "short_response_2", "text": "有什么在跟着我们",
             "split": "test", "audio": str(extra),
             "sha256": _sha(extra)},
            {"id": "mid_exposition", "text": "中长句文本",
             "split": "validation"},
        ],
        "matrix": [
            {"pair_group": "A", "text": "short_response",
             "seeds": [42, 43, 44], "p0_source": "reuse",
             "reuse_prefix": "stability/short_response"},
            {"pair_group": "B", "text": "short_response_2",
             "seeds": [42, 43, 44], "p0_source": "generate"},
            {"pair_group": "C", "text": "mid_exposition",
             "seeds": [44], "p0_source": "reuse",
             "reuse_prefix": "stability/mid_exposition"},
        ],
    }
    return character, cfg, tmp_path, old_dir


def test_run_phase5a2_pairs(a2_env):
    character, cfg, tmp_path, old_dir = a2_env
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        manifest = phase5a2.run_phase5a2(cfg, character, manager=mgr,
                                         log=lambda *a: None)
    finally:
        mgr.stop()

    assert len(manifest["pairs"]) == 7
    assert len(manifest["cases"]) == 14
    assert all(c["status"] == "ok" for c in manifest["cases"])

    reused = [c for c in manifest["cases"] if c.get("reused_from")]
    assert len(reused) == 4
    for c in reused:
        assert c["output_sha256"]
        assert c["output"].startswith("..")  # lives in the old dir
    new = [c for c in manifest["cases"] if not c.get("reused_from")]
    assert len(new) == 10
    assert manifest["budget"]["zero_shot_generated"] == 10
    assert manifest["reuse_revision_match"] is True

    # each pair = same text+seed, two different prompts
    for pair in manifest["pairs"]:
        a, b = (cases for cases in manifest["cases"]
                if cases["case_id"] in pair["cells"])
        assert {a["prompt_id"], b["prompt_id"]} == {"P0", "P2"}
        assert a["text_id"] == b["text_id"] and a["seed"] == b["seed"]

    out_dir = tmp_path / "a2_out"
    assert (out_dir / "manifest.json").is_file()
    assert (out_dir / "listen" / "index.html").is_file()
    assert (out_dir / "listen" / "unblind_map.json").is_file()
    unblind = json.loads(
        (out_dir / "listen" / "unblind_map.json").read_text(encoding="utf-8"))
    # every pair contributes two anonymized sides
    pair_keys = {v["case_id"].split("/")[0] for v in unblind.values()}
    assert len(pair_keys) == 7
    # new wavs physically inside a2 dir; nothing written to old dir
    assert len(list(out_dir.glob("*.wav"))) >= 3  # originals + some new
    assert not (old_dir / "stab_seed42.wav.json").exists()


def test_blocked_reuse_on_tampered_wav(a2_env):
    character, cfg, tmp_path, old_dir = a2_env
    victim = old_dir / "stab_seed42.wav"
    victim.write_bytes(victim.read_bytes() + b"x")  # corrupt
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        manifest = phase5a2.run_phase5a2(cfg, character, manager=mgr,
                                       log=lambda *a: None)
    finally:
        mgr.stop()
    bad = [c for c in manifest["cases"]
           if c["case_id"] == "pair_A_short_response_seed42/P0"]
    assert bad[0]["status"] == "blocked_reuse"
    assert "sha256" in bad[0]["error"]


def test_context_variance_detection(tmp_path, monkeypatch):
    docs = tmp_path / "docs" / "reports"
    docs.mkdir(parents=True)
    (docs / "suoming-voxcpm-phase5a-ratings.json").write_text(json.dumps({
        "samples": [
            {"case_id": "stability/rain_seed42",
             "output_sha256": "a" * 64,
             "clarity": "4", "naturalness": "5"},
            {"case_id": "prompt_ab/P0_rain_seed42",
             "output_sha256": "a" * 64,
             "clarity": "5", "naturalness": "3"},
            {"case_id": "other", "output_sha256": "b" * 64,
             "clarity": "5"},
        ]}), encoding="utf-8")
    monkeypatch.setattr(
        "character_tts.evaluation.phase5a2.repo_root", lambda: tmp_path)
    cv = phase5a2._context_variance()
    assert len(cv) == 1
    assert cv[0]["divergent_dims"] == ["clarity", "naturalness"]
    assert len(cv[0]["case_ids"]) == 2


def test_attach_asr_results(a2_env):
    character, cfg, tmp_path, _ = a2_env
    mgr = BackendManager(logs_dir=tmp_path / "logs")
    try:
        manifest = phase5a2.run_phase5a2(cfg, character, manager=mgr,
                                         log=lambda *a: None)
    finally:
        mgr.stop()
    asr_path = tmp_path / "asr_check.json"
    cid = manifest["cases"][0]["case_id"]
    asr_path.write_text(json.dumps({
        "backend": "sensevoice", "model": "m",
        "results": [{"case_id": cid, "cer": 0.0, "hyp_norm": "x"}],
    }), encoding="utf-8")
    mpath = tmp_path / "a2_out" / "manifest.json"
    merged = phase5a2.attach_asr_results(mpath, asr_path)
    assert merged["asr"]["status"] == "done"
    hit = [c for c in merged["cases"] if c["case_id"] == cid]
    assert hit[0]["asr_aux"]["cer"] == 0.0
