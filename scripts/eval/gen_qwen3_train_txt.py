"""一次性：Qwen3-TTS 零样本克隆生成《训练台词》长文，供 A/B 对比。

用法（qwen3_tts env）：
    backend_envs/qwen3_tts/Scripts/python.exe \
        scripts/eval/gen_qwen3_train_txt.py [--text ...] [--name ...] \
        [--out outputs/_tmp/train_txt/qwen3]

与 voxcpm2 路径同策略：100 字句切分 + 每块独立 ref 锚定 + 200ms 缝。
"""
import inspect
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SNAP_ROOT = Path("hf_cache/qwen3_tts")
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REV = "fd4b254389122332181a7c3db7f27e918eec64e3"
REF_WAV = ("data/characters/aimisi/datasets/v1/audio/"
           "d8b99e88731cd1e6e9dc5bec4667ba75727adc843089d67b04e4bc503f00677a.wav")
REF_TEXT = ("啊……是陆医生。和他说话的时候，就当我不在吧？"
            "不然会被当成对着空气谈天的怪人哦。")
GEN = {"language": "Chinese", "max_new_tokens": 2048, "do_sample": True,
       "top_k": 50, "top_p": 1.0, "temperature": 0.9,
       "repetition_penalty": 1.05, "subtalker_dosample": True,
       "subtalker_top_k": 50, "subtalker_top_p": 1.0,
       "subtalker_temperature": 0.9, "x_vector_only_mode": False,
       "non_streaming_mode": True}


def split_text(text: str, max_chars: int) -> list[str]:
    pieces = re.split(r"(?<=[。！？!?；;…])", text)
    chunks, cur = [], ""
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        subs = (re.split(r"(?<=[，,、：:])", piece)
                if len(piece) > max_chars else [piece])
        for s in subs:
            s = s.strip()
            if not s:
                continue
            if len(cur) + len(s) <= max_chars:
                cur += s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
    if cur:
        chunks.append(cur)
    return chunks


def main() -> int:
    text_path = Path("outputs/_tmp/aimisi_train_lines.txt")
    name = "训练台词"
    out_dir = Path("outputs/_tmp/train_txt/qwen3")
    args = sys.argv[1:]
    for i, a in enumerate(list(args)):
        if a == "--text":
            text_path = Path(args[i + 1])
        if a == "--name":
            name = args[i + 1]
        if a == "--out":
            out_dir = Path(args[i + 1])
    max_chunks = 0
    for i, a in enumerate(list(args)):
        if a == "--max-chunks":
            max_chunks = int(args[i + 1])

    import numpy as np
    import soundfile as sf
    import torch
    from huggingface_hub import snapshot_download
    from qwen_tts import Qwen3TTSModel

    snap = snapshot_download(MODEL_ID, revision=REV,
                             cache_dir=str(SNAP_ROOT),
                             local_files_only=True)
    model = Qwen3TTSModel.from_pretrained(
        snap, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa")
    fn = model.generate_voice_clone
    sig = inspect.signature(fn)
    call_kwargs = {k: v for k, v in GEN.items()
                   if k in sig.parameters or any(
                       p.kind == inspect.Parameter.VAR_KEYWORD
                       for p in sig.parameters.values())}

    chunks = split_text(text_path.read_text(encoding="utf-8"), 100)
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"{len(chunks)} chunks", flush=True)
    waves, t0 = [], time.monotonic()
    for i, chunk in enumerate(chunks):
        kw = dict(call_kwargs, text=chunk, ref_audio=REF_WAV,
                  ref_text=REF_TEXT)
        if "seed" in sig.parameters:
            kw["seed"] = 42 + i
        else:
            torch.manual_seed(42 + i)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42 + i)
        wavs, sr = fn(**kw)
        w = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        waves.append(w)
        print(f"  chunk {i + 1}/{len(chunks)} {len(w) / sr:.1f}s",
              flush=True)
    gap = np.zeros(int(sr * 0.2), dtype=np.float32)
    joined = []
    for i, w in enumerate(waves):
        if i:
            joined.append(gap)
        joined.append(w)
    out = np.concatenate(joined)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / f"{name}.wav"), out, int(sr))
    meta = {"tag": "qwen3_tts", "model": MODEL_ID, "rev": REV,
            "ref": "d8b9", "seed_base": 42, "chunks": len(chunks),
            "sample_rate": int(sr), "duration": round(len(out) / sr, 1),
            "wall_seconds": round(time.monotonic() - t0, 1),
            "text": str(text_path), "gen": call_kwargs}
    (out_dir / f"{name}.wav.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done {meta['duration']}s wall={meta['wall_seconds']}s "
          f"sr={sr}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
