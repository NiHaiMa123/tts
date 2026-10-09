from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

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


# ---------- provenance / import / audit / enrich ----------

def test_numeric_suffix_is_no_text(tmp_path):
    p = tmp_path / "中立_neutral" / "【中立_neutral】_10.wav"
    p.parent.mkdir()
    p.touch()
    assert parse_inbox_name(p) == ("中立_neutral", "")


def test_provenance_sidecar(tmp_path):
    inbox = tmp_path / "inbox"
    _wav(inbox / "1001.wav")
    (inbox / "provenance.jsonl").write_text(json.dumps({
        "file": "1001.wav", "text": "台词一", "label": "中立_neutral",
        "event": "play_vo_x", "wem_hash": 1001,
        "source_wem": "Media/zh/1001.wem", "tool": "vgmstream"},
        ensure_ascii=False) + "\n", encoding="utf-8")
    pool, index = tmp_path / "pool", tmp_path / "idx.jsonl"
    standardize_inbox(inbox, pool, index)
    row = load_index(index)[0]
    assert row["text"] == "台词一" and row["label"] == "中立_neutral"
    assert row["provenance"]["wem_hash"] == 1001
    assert len(row["source_sha256"]) == 64


def test_import_ludiglot(tmp_path):
    from character_tts.ingest.ludiglot_import import import_ludiglot
    src = tmp_path / "decoded"
    _wav(src / "111.wav")
    _wav(src / "222.wav")
    man = tmp_path / "man.jsonl"
    man.write_text(
        json.dumps({"audio": "111", "text": "你好", "event": "play_vo_a",
                    "wem_hash": 111, "source_wem": "Media/zh/111.wem"},
                   ensure_ascii=False) + "\n" +
        json.dumps({"audio": "222", "text": "再见", "label": "情绪_emo"},
                   ensure_ascii=False) + "\n" +
        json.dumps({"audio": "999", "text": "缺文件"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8")
    out = tmp_path / "inbox"
    res = import_ludiglot(src, man, out, default_label="中立_neutral")
    assert res["imported"] == 2 and res["missing"] == 1
    prov = [json.loads(l) for l in
            (out / "provenance.jsonl").read_text(
                encoding="utf-8").splitlines()]
    assert prov[0]["file"] == "中立_neutral/111.wav"
    assert prov[0]["event"] == "play_vo_a"
    assert (out / "情绪_emo" / "222.wav").exists()


def test_freeze_uses_text_asr_fallback(tmp_path):
    inbox = tmp_path / "inbox"
    _wav(inbox / "中立_neutral" / "【中立_neutral】_1.wav")
    _wav(inbox / "中立_neutral" / "【中立_neutral】有台词.wav", freq=330)
    pool, index = tmp_path / "pool", tmp_path / "idx.jsonl"
    standardize_inbox(inbox, pool, index)
    # simulate enrich: ASR produced text for the numbered row
    rows = load_index(index)
    for r in rows:
        if r.get("status") == "ok" and not r["text"]:
            r["text_asr"] = "识别出来的台词"
    index.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                               for r in rows) + "\n", encoding="utf-8")
    ds = tmp_path / "ds"
    freeze_dataset(index, ds, pool_dir=pool)
    man = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
    assert man["counts"]["dropped"] == 0
    assert man["text_sources"].get("asr") == 1
    texts = [json.loads(l)["text"] for l in
             (ds / "train.jsonl").read_text(encoding="utf-8").splitlines()
             ] + [json.loads(l)["text"] for l in
                  (ds / "test.jsonl").read_text(encoding="utf-8").splitlines()
             ] + [json.loads(l)["text"] for l in
                  (ds / "validation.jsonl").read_text(
                      encoding="utf-8").splitlines()]
    assert "识别出来的台词" in texts


def test_audit_dataset(tmp_path):
    from character_tts.ingest.audit import audit_dataset
    inbox = tmp_path / "inbox"
    for i, f in enumerate((200, 300, 400)):
        _wav(inbox / f"【标签】台词{i}.wav", freq=f)
    pool, index = tmp_path / "pool", tmp_path / "idx.jsonl"
    standardize_inbox(inbox, pool, index)
    ds = tmp_path / "ds"
    freeze_dataset(index, ds, pool_dir=pool)
    rep = audit_dataset(ds)
    assert rep["ok"] and rep["counts"]["train"] + \
        rep["counts"]["validation"] + rep["counts"]["test"] == 3
    # corrupt: delete one wav -> violation
    victim = next((ds / "audio").glob("*.wav"))
    victim.unlink()
    rep2 = audit_dataset(ds, verify_hash=False)
    assert not rep2["ok"] and any("missing audio" in v
                                  for v in rep2["violations"])


def test_enrich_index(tmp_path):
    import importlib.util
    import sys as _sys
    spec = importlib.util.spec_from_file_location(
        "enrich_index", REPO / "scripts" / "ingest" / "enrich_index.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    index = tmp_path / "idx.jsonl"
    index.write_text(json.dumps(
        {"source": "a.wav", "status": "ok", "text": "",
         "audio": "x.wav", "flags": []}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    asr = tmp_path / "asr.json"
    asr.write_text(json.dumps({"results": [
        {"source": "a.wav", "cer": 0.1, "hypothesis": "台词"}]}),
        encoding="utf-8")
    spk = tmp_path / "spk.json"
    spk.write_text(json.dumps({"results": [
        {"source": "a.wav", "speaker_cos": 0.3}]}), encoding="utf-8")

    _sys.argv = ["enrich_index.py", "--index", str(index),
                 "--speaker", str(spk), "--asr", str(asr)]
    assert mod.main() == 0
    row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    assert row["text_asr"] == "台词"
    assert "speaker_mismatch:0.3" in row["flags"]
    assert "text_from_asr" in row["flags"]
