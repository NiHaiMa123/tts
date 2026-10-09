from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

from character_tts.ingest.freeze import freeze_dataset, load_index
from character_tts.ingest.naming import norm_text, parse_inbox_name, text_fid
from character_tts.ingest.review import load_decisions
from character_tts.ingest.standardize import standardize_inbox


def _wav(path: Path, dur_s: float = 1.0, sr: int = 48000,
         freq: float = 220.0, amp: float = 0.2) -> Path:
    n = int(sr * dur_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(amp * 32767 *
                                 math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n))
        w.writeframes(frames)
    return path


def test_parse_inbox_name(tmp_path):
    p = tmp_path / "中立_neutral" / "【中立_neutral】你好世界.wav"
    p.parent.mkdir()
    p.touch()
    label, text = parse_inbox_name(p)
    assert label == "中立_neutral" and text == "你好世界"
    plain = tmp_path / "情绪_emo" / "没有前缀.wav"
    plain.parent.mkdir(exist_ok=True)
    plain.touch()
    assert parse_inbox_name(plain) == ("情绪_emo", "没有前缀")


def test_norm_and_fid():
    assert norm_text("你好，世界！") == "你好世界"
    a = text_fid("你好，世界", "x" * 64)
    assert a == text_fid("你好,世界", "x" * 64)      # punct-insensitive
    assert a != text_fid("你好世界", "y" * 64)        # audio-bound


def test_standardize_and_freeze(tmp_path):
    inbox = tmp_path / "inbox"
    _wav(inbox / "中立_neutral" / "【中立_neutral】第一句话.wav")
    _wav(inbox / "中立_neutral" / "【中立_neutral】第二句话.wav", freq=330)
    _wav(inbox / "中立_neutral" / "【中立_neutral】重复句.wav", freq=440)
    _wav(inbox / "中立_neutral" / "【中立_neutral】重复句.wav.bak.wav",
         freq=440)                                   # same content -> dup audio
    _wav(inbox / "中立_neutral" / "【中立_neutral】太短.wav", dur_s=0.1)

    pool = tmp_path / "pool"
    index = tmp_path / "index.jsonl"
    stats = standardize_inbox(inbox, pool, index)
    assert stats["accepted"] == 4 and stats["rejected"] == 1
    rows = load_index(index)
    ok = [r for r in rows if r["status"] == "ok"]
    # identical content collapses to one pool entry
    assert len({r["audio_sha256"] for r in ok}) == 3

    ds = tmp_path / "dataset"
    res = freeze_dataset(index, ds, pool_dir=pool, val_frac=0.34, test_frac=0.34)
    counts = res["counts"]
    assert sum(counts.values()) == 3
    for name in ("train", "validation", "test"):
        for line in (ds / f"{name}.jsonl").read_text(
                encoding="utf-8").splitlines():
            rec = json.loads(line)
            assert rec["audio"].startswith("audio/") \
                or "/" in rec["audio"]
    man = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    assert man["fid_format"] == "sha256(norm_text|audio_sha256)"


def test_freeze_deterministic_and_excludes(tmp_path):
    inbox = tmp_path / "inbox"
    digits = "一二三四五六七八九十"
    for i in range(10):
        _wav(inbox / f"【中立_neutral】第{digits[i]}句话.wav",
             freq=200 + i * 30)
    pool, index = tmp_path / "pool", tmp_path / "idx.jsonl"
    standardize_inbox(inbox, pool, index)

    ds1, ds2 = tmp_path / "d1", tmp_path / "d2"
    freeze_dataset(index, ds1, pool_dir=pool)
    freeze_dataset(index, ds2, pool_dir=pool)
    for name in ("train", "validation", "test"):
        assert (ds1 / f"{name}.jsonl").read_text(encoding="utf-8") == \
               (ds2 / f"{name}.jsonl").read_text(encoding="utf-8")

    fids = [json.loads(l)["fid"] for l in
            (ds1 / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    ds3 = tmp_path / "d3"
    res = freeze_dataset(index, ds3, pool_dir=pool, exclude_fids={fids[0]})
    assert res["dropped"][0]["why"] == "reviewed_out"
    assert json.loads((ds3 / "train.jsonl").read_text(
        encoding="utf-8").splitlines()[0])["fid"] != fids[0] or \
        fids[0] not in [json.loads(l)["fid"] for l in
                        (ds3 / "train.jsonl").read_text(
                            encoding="utf-8").splitlines()]


def test_load_decisions(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"drop": ["a", "b"]}), encoding="utf-8")
    assert load_decisions(p) == {"a", "b"}
