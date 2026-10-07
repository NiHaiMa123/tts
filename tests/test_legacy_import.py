from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import pytest
import yaml

from character_tts.app.legacy_import import import_suoming, write_character


def _wav(path: Path, seconds: float = 0.05, sr: int = 8000):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", i % 300) for i in range(n)))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_fake_legacy(root: Path, bad_hash: bool = False) -> Path:
    """Create a minimal fake legacy tree and return the profile path."""
    (root / "src" / "dots_tts").mkdir(parents=True)
    prompt = root / "data" / "inbox" / "锁暝" / "中立_neutral" / "ref.wav"
    _wav(prompt)
    adapter = root / "ckpt" / "model"
    adapter.mkdir(parents=True)
    (adapter / "trainable_model.safetensors").write_bytes(b"weights")
    (adapter / "trainable_model.json").write_text("{}", encoding="utf-8")
    ds = root / "datasets" / "suoming" / "v1"
    ds.mkdir(parents=True)
    for name in ("train", "validation", "test"):
        (ds / f"{name}.jsonl").write_text(
            '{"audio": "a.wav", "text": "x"}\n', encoding="utf-8")
    (ds / "manifest.json").write_text("{}", encoding="utf-8")

    profile = {
        "model_id": "suoming_step500_v1",
        "base_model": {"path": "pretrained_models/dots.tts-soar",
                        "revision": "abc"},
        "adapter": {"path": "ckpt/model",
                    "format": "dots_tts_trainable_delta",
                    "training_step": 500},
        "prompt": {
            "audio_path": "data/inbox/锁暝/中立_neutral/ref.wav",
            "audio_sha256": ("0" * 64) if bad_hash else _sha(prompt),
            "text": "ref text",
        },
        "generation": {"base_seed": 42},
        "runtime": {"precision": "bfloat16"},
    }
    p = root / "configs" / "voices"
    p.mkdir(parents=True)
    prof = p / "suoming_step500_v1.yaml"
    prof.write_text(yaml.safe_dump(profile, allow_unicode=True),
                    encoding="utf-8")
    return prof


def test_import_ok(tmp_path):
    root = tmp_path / "dotstts"
    _build_fake_legacy(root)
    result = import_suoming(root)
    assert result["status"] == "ok"
    ch = result["character"]
    assert ch["reference"]["sha256"] == _sha(
        root / "data" / "inbox" / "锁暝" / "中立_neutral" / "ref.wav")
    assert ch["reference"]["audio"].startswith("${DOTSTTS_ROOT}/")
    assert ch["legacy_voice_profile"]["adapter"]["training_step"] == 500
    assert ch["dataset"]["train_manifest"].endswith("train.jsonl")


def test_import_hash_mismatch_blocked(tmp_path):
    root = tmp_path / "dotstts"
    _build_fake_legacy(root, bad_hash=True)
    result = import_suoming(root)
    assert result["status"] == "blocked"
    assert any("sha256 mismatch" in p for p in result["problems"])


def test_import_missing_files_blocked(tmp_path):
    root = tmp_path / "dotstts"
    (root / "src" / "dots_tts").mkdir(parents=True)
    result = import_suoming(root)
    assert result["status"] == "blocked"


def test_write_character(tmp_path):
    root = tmp_path / "dotstts"
    _build_fake_legacy(root)
    result = import_suoming(root)
    out = write_character(result, tmp_path / "chars")
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data["character_id"] == "suoming"
    with pytest.raises(ValueError):
        write_character({"status": "blocked", "problems": ["x"]}, tmp_path)
