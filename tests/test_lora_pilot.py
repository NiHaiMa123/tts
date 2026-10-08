"""Phase 5B pilot tests: train-only data gate + leakage blocks, step cap,
eval reuse verification, pair packs, blind page, fatigue export."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import wave
from pathlib import Path

import pytest
import yaml

from character_tts.evaluation import lora_pilot
from character_tts.registry.models import BackendProfile, CharacterProfile

REPO = Path(__file__).resolve().parent.parent


def _wav(path: Path, seconds: float = 0.1, sr: int = 8000,
         amp: int = 8000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", amp if (i // 4) % 2 else -amp)
            for i in range(n)))
    return path


def _silent(path: Path, seconds: float = 0.1, sr: int = 8000) -> Path:
    return _wav(path, seconds, sr, amp=0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- data prep
def test_prepare_blocks_leakage(tmp_path):
    prep = _load_script("prepare_voxcpm_lora_data")
    leak_wav = _wav(tmp_path / "leak.wav")
    block = {"audio_sha256": [_sha(leak_wav)], "norm_texts": ["开伞由我来动手"],
             "fids": ["fid_p0"], "entries": []}
    ok1 = _wav(tmp_path / "ok1.wav", seconds=0.6)
    ok2 = _wav(tmp_path / "ok2.wav", seconds=0.7)
    rows = [
        {"audio": str(ok1), "fid": "f1", "text": "正常训练句"},
        {"audio": str(ok2), "fid": "f2", "text": "另一句训练文本"},
        {"audio": str(leak_wav), "fid": "f3", "text": "完全不同的文本"},
        {"audio": str(ok1), "fid": "f4", "text": "开伞，由我来动手。"},
        {"audio": str(tmp_path / "missing.wav"), "fid": "f5", "text": "缺失"},
        {"audio": str(ok2), "fid": "fid_p0", "text": "fid 撞线"},
        {"audio": str(ok1), "fid": "f6", "text": "正常训练句"},
    ]
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                             for r in rows), encoding="utf-8")
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps({"audio": str(ok1), "fid": "v1",
                               "text": "验证句"}, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    stats = prep.prepare(train, val, block, tmp_path / "out",
                         log=lambda *a: None)
    reasons = stats["reject_reasons"]
    assert reasons["leak_audio_sha"] == 1
    assert reasons["leak_text_exact"] == 1
    assert reasons["missing_file"] == 1
    assert reasons["leak_fid"] == 1
    assert reasons.get("duplicate_audio") == 1 or \
        reasons.get("duplicate_text") == 1
    assert stats["accepted"] == 2
    # val manifest shares train columns exactly (HF datasets requirement)
    train_cols = set(json.loads((tmp_path / "out/train.jsonl")
                     .read_text(encoding="utf-8").splitlines()[0]))
    val_cols = set(json.loads((tmp_path / "out/val_metrics.jsonl")
                   .read_text(encoding="utf-8").splitlines()[0]))
    assert train_cols == val_cols == {"audio", "text"}


def test_prepare_data_blocked(tmp_path):
    prep = _load_script("prepare_voxcpm_lora_data")
    w = _wav(tmp_path / "a.wav", seconds=0.2)
    train = tmp_path / "t.jsonl"
    train.write_text(json.dumps({"audio": str(w), "fid": "x",
                                 "text": "only one"}, ensure_ascii=False),
                     encoding="utf-8")
    stats = prep.prepare(train, train, {"audio_sha256": [],
                                        "norm_texts": [], "fids": [],
                                        "entries": []},
                         tmp_path / "o", log=lambda *a: None)
    assert stats["decision"] == "DATA_BLOCKED"


# ---------------------------------------------------------------- step cap
def test_train_runner_cap(tmp_path, monkeypatch):
    mod = _load_script("train_voxcpm_lora_pilot")
    ck = tmp_path / "ck" / "latest"
    ck.mkdir(parents=True)
    (ck / "training_state.json").write_text('{"step": 150}')
    assert mod._steps_completed(tmp_path / "ck") == 150
    (ck / "training_state.json").write_text('{"step": 42}')
    assert mod._steps_completed(tmp_path / "ck") == 42
    # config above the hard cap must be refused by main()
    cfg = yaml.safe_load((REPO / "configs/training/"
                          "suoming_voxcpm_lora_pilot.yaml")
                         .read_text(encoding="utf-8"))
    assert cfg["num_iters"] <= mod.MAX_OPT_STEPS
    assert cfg["save_interval"] == 50


# ---------------------------------------------------------------- pilot eval
@pytest.fixture
def pilot_env(tmp_path, monkeypatch):
    ref = _wav(tmp_path / "ref.wav")
    p0_text = "prompt text"
    character = CharacterProfile(
        character_id="suoming", display_name="锁暝", dataset={},
        reference={"audio": str(ref), "text": p0_text,
                   "sha256": _sha(ref)},
        evaluation={"anchor_texts": [
            {"id": "rain", "text": "rain text", "audio": str(ref),
             "sha256": _sha(ref), "source": "validation.jsonl"}]},
    )
    profile = BackendProfile(
        backend_id="voxcpm2", family="fake", enabled=True,
        worker={"python": sys.executable, "script": "x"},
        model={"path_or_id": "openbmb/VoxCPM2", "revision": "rev1"},
        generation={},)
    monkeypatch.setattr(lora_pilot, "load_backend", lambda _: profile)

    # reuse dirs: phase5a has 4 base cases, phase5a2 has 2
    texts = {"rain": "rain text", "short_response": "short text",
             "short_response_2": "short2 text",
             "mid_exposition": "mid text"}
    dirs = {}
    for key, case_map in (
            ("phase5a", {"stability/rain_seed42": ("rain", 42),
                         "stability/short_response_seed42":
                             ("short_response", 42),
                         "stability/short_response_seed43":
                             ("short_response", 43),
                         "stability/mid_exposition_seed42":
                             ("mid_exposition", 42)}),
            ("phase5a2",
             {"pair_B_short_response_2_seed42/P0":
                  ("short_response_2", 42),
              "pair_B_short_response_2_seed43/P0":
                  ("short_response_2", 43)})):
        d = tmp_path / key
        d.mkdir()
        old = []
        for cid, (tid, seed) in case_map.items():
            w = _wav(d / f"{cid.replace('/', '_')}.wav")
            old.append({"case_id": cid, "status": "ok",
                        "text": texts[tid], "prompt_sha256": _sha(ref),
                        "seed": seed, "output": w.name,
                        "output_sha256": _sha(w),
                        "effective_args": dict(lora_pilot.EXPECTED_GEN_ARGS),
                        "retry_count": 0})
        (d / "manifest.json").write_text(json.dumps(
            {"cases": old, "model": {"revision": "rev1"}},
            ensure_ascii=False), encoding="utf-8")
        dirs[key] = d

    # checkpoints with real lora weights files
    ck_root = tmp_path / "ckpts"
    for step in (50, 100, 150):
        d = ck_root / f"step_{step:07d}"
        d.mkdir(parents=True)
        (d / "lora_weights.safetensors").write_bytes(b"weights" + str(step).encode())
        (d / "lora_config.json").write_text("{}")

    # fake env subprocess: fabricate wav + _gen_meta.json
    def fake_run(cmd, **kw):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        cases = json.loads(Path(cmd[cmd.index("--cases") + 1])
                           .read_text(encoding="utf-8"))
        out_dir.mkdir(parents=True, exist_ok=True)
        metas = []
        for c in cases:
            w = _wav(out_dir / f"{c['case_id'].replace('/', '_')}.wav")
            metas.append({"case_id": c["case_id"], "status": "ok",
                          "output": w.name, "output_sha256": _sha(w),
                          "sample_rate": 8000, "retry_count": 0,
                          "wall_seconds": 0.1,
                          "effective_args": {}})
        (out_dir / "_gen_meta.json").write_text(json.dumps(metas))
        class R: returncode = 0; stderr = ""; stdout = ""
        return R()
    monkeypatch.setattr(lora_pilot.subprocess, "run", fake_run)
    monkeypatch.setattr(lora_pilot, "_env_python", lambda: Path("py"))
    monkeypatch.setattr(lora_pilot, "_gen_script", lambda: Path("s"))

    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text("{}\n", encoding="utf-8")
    cfg = {
        "experiment_id": "pilot_test", "output_dir": "pilot_out",
        "backend": "voxcpm2", "root": str(tmp_path),
        "snapshot": str(tmp_path / "snap"),
        "train_manifest": str(train_jsonl),
        "prompts": [{"id": "P0", "audio": str(ref), "text": p0_text,
                     "sha256": _sha(ref)}],
        "texts": [{"id": k, "text": v} for k, v in texts.items()],
        "reuse_sources": {k: str(v) for k, v in dirs.items()},
        "checkpoints": {str(s): {"path": str(ck_root / f"step_{s:07d}")}
                        for s in (50, 100, 150)},
        "lora": {"enable_lm": True, "enable_dit": True,
                 "enable_proj": False, "r": 16, "alpha": 16,
                 "dropout": 0.0},
    }
    return character, cfg, tmp_path


def test_run_lora_eval(pilot_env):
    character, cfg, tmp_path = pilot_env
    m = lora_pilot.run_lora_eval(cfg, character, log=lambda *a: None)
    assert len(m["cases"]) == 24          # 6 base + 18 lora
    assert all(c["status"] == "ok" for c in m["cases"])
    assert sum(1 for c in m["cases"] if c.get("reused_from")) == 6
    assert m["budget"]["new_generated"] == 18
    # pairs: 18 base-vs-ckpt + 2 ckpt-vs-ckpt
    assert len(m["pairs"]) == 20
    p1 = next(p for p in m["packs"] if p["pack_id"] == "pack1")
    p2 = next(p for p in m["packs"] if p["pack_id"] == "pack2")
    p3 = next(p for p in m["packs"] if p["pack_id"] == "pack3")
    assert len(p1["pairs"]) == 4 and not p1["collapsed"]
    assert len(p2["pairs"]) == 2 and p2["collapsed"]
    assert len(p3["pairs"]) == 14
    # every pair's cells exist in the manifest
    ids = {c["case_id"] for c in m["cases"]}
    for p in m["pairs"]:
        assert all(cell in ids for cell in p["cells"])
    # checkpoint provenance
    assert all(v["weights_sha256"]
               for v in m["checkpoints"].values())
    # artifacts
    out = tmp_path / "pilot_out"
    assert (out / "manifest.json").is_file()
    assert (out / "screening.json").is_file()
    assert (out / "listen" / "index.html").is_file()
    html = (out / "listen" / "index.html").read_text(encoding="utf-8")
    assert "UNCERTAIN_FATIGUE" in html
    assert "first_impression" in html
    um = json.loads((out / "listen" / "unblind_map.json")
                    .read_text(encoding="utf-8"))
    assert len(um) == 40   # 20 pairs x 2 sides


def test_blocked_reuse_no_silent_regen(pilot_env):
    character, cfg, tmp_path = pilot_env
    victim = tmp_path / "phase5a" / "stability_rain_seed42.wav"
    victim.write_bytes(b"corrupted")
    m = lora_pilot.run_lora_eval(cfg, character, log=lambda *a: None)
    bad = [c for c in m["cases"] if c["case_id"] == "base/rain_seed42"]
    assert bad[0]["status"] == "blocked_reuse"
    assert "sha256" in bad[0]["error"]


def test_missing_checkpoint_blocks_cells(pilot_env):
    character, cfg, tmp_path = pilot_env
    del cfg["checkpoints"]["50"]
    m = lora_pilot.run_lora_eval(cfg, character, log=lambda *a: None)
    blocked = [c for c in m["cases"]
               if c["condition"] == "ckpt50"]
    assert all(c["status"] == "blocked" for c in blocked)
    assert m["checkpoints"]["ckpt50"]["weights_sha256"] is None
