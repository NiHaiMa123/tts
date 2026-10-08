from __future__ import annotations

import hashlib
import struct
import wave
from pathlib import Path

import pytest
import yaml

from character_tts.app.legacy_import import (
    GATE_ANCHOR_TEXT,
    import_suoming,
    write_character,
)


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
    # Ground truth: an utterance whose text matches the gate anchor text.
    # It is NOT the first validation record — the importer must select by
    # text, not by position.
    gt_wav = ds / "audio" / "anchor.wav"
    _wav(gt_wav)
    import json as _json
    decoy = _json.dumps({"audio": "decoy.wav", "text": "unrelated"},
                        ensure_ascii=False)
    hit = _json.dumps({"audio": str(gt_wav), "text": GATE_ANCHOR_TEXT},
                      ensure_ascii=False)
    for name in ("train", "validation", "test"):
        (ds / f"{name}.jsonl").write_text(
            decoy + "\n" + (hit + "\n" if name == "validation" else ""),
            encoding="utf-8")
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


def test_import_ground_truth_selected_by_text(tmp_path):
    """The ground truth must be the record matching the anchor text —
    not whatever happens to be first in validation.jsonl."""
    root = tmp_path / "dotstts"
    _build_fake_legacy(root)
    result = import_suoming(root)
    assert result["status"] == "ok", result.get("problems")
    anchors = result["character"]["evaluation"]["anchor_texts"]
    gt = next(a for a in anchors if a["id"] == "rain")
    assert gt["text"] == GATE_ANCHOR_TEXT
    expected = root / "datasets" / "suoming" / "v1" / "audio" / "anchor.wav"
    assert gt["sha256"] == _sha(expected)
    assert "anchor.wav" in gt["audio"]
    assert "decoy" not in gt["audio"]
    assert gt["source"].endswith("validation.jsonl")


def test_import_missing_anchor_record_blocked(tmp_path):
    root = tmp_path / "dotstts"
    _build_fake_legacy(root)
    # Remove the matching record -> import must report blocked, not guess.
    ds = root / "datasets" / "suoming" / "v1"
    (ds / "validation.jsonl").write_text(
        '{"audio": "decoy.wav", "text": "unrelated"}\n', encoding="utf-8")
    result = import_suoming(root)
    assert result["status"] == "blocked"
    assert any("anchor" in p for p in result["problems"])


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
