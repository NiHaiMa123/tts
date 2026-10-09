#!/usr/bin/env python3
"""Speed A/B for voxcpm2: prompt-cache and inference_timesteps.

Runs INSIDE backend_envs/voxcpm2 (needs voxcpm+torch). Generates the
same texts under each condition with a fixed seed, records wall time,
writes wavs + report.json under --out.

    backend_envs\\voxcpm2\\Scripts\\python.exe scripts\\eval\\speed_ab.py ^
        --char aimisi --out outputs\\_tmp\\speedtest ^
        [--steps 10,8,6] [--texts file.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _texts(args) -> list[str]:
    if args.texts:
        return [json.loads(l)["text"] for l in
                Path(args.texts).read_text(encoding="utf-8").splitlines()
                if l.strip()][: args.n]
    # default: a few lines from the character's test split
    p = (ROOT / "data" / "characters" / args.char / "datasets" / "v1"
         / "test.jsonl")
    rows = [json.loads(l)["text"] for l in
            p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=len)
    picks = [rows[len(rows) // 4], rows[len(rows) // 2],
             rows[3 * len(rows) // 4]]
    return picks[: args.n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="10,8,6")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--texts", default=None,
                    help="jsonl with .text fields; default: char test split")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "characters" /
                          f"{args.char}.yaml").read_text(encoding="utf-8"))
    ref = cfg["reference"]
    ref_wav = str(ROOT / Path(ref["audio"].replace("${TTS_ROOT}/", "")))
    ref_text = ref.get("text") or ""
    texts = _texts(args)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from voxcpm import VoxCPM
    from voxcpm.model.utils import next_and_close
    import soundfile as sf

    model = VoxCPM.from_pretrained(
        "openbmb/VoxCPM2",
        cache_dir=str(ROOT / "hf_cache" / "voxcpm2"),
        load_denoiser=False, local_files_only=True)
    tts = model.tts_model
    sr = int(getattr(tts, "sample_rate", 48000))

    t0 = time.monotonic()
    pcache = tts.build_prompt_cache(
        prompt_text=ref_text, prompt_wav_path=ref_wav,
        reference_wav_path=None)
    cache_build_s = round(time.monotonic() - t0, 2)
    print(f"prompt cache built in {cache_build_s}s", flush=True)

    steps_list = [int(s) for s in args.steps.split(",") if s.strip()]
    conds = [("nocache", steps_list[0])] + \
            [("cache", s) for s in steps_list]
    report = {"char": args.char, "ref": ref_wav, "seed": args.seed,
              "cache_build_s": cache_build_s, "conditions": []}
    for ci, text in enumerate(texts):
        clean = re.sub(r"\s+", " ", text.replace("\n", " "))
        for tag, steps in conds:
            name = f"t{ci}_{tag}_t{steps}"
            torch.manual_seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            t0 = time.monotonic()
            if tag == "nocache":
                wav = model.generate(
                    text=clean, prompt_wav_path=ref_wav,
                    prompt_text=ref_text, cfg_value=2.0,
                    inference_timesteps=steps, normalize=False,
                    denoise=False, retry_badcase=True)
            else:
                wav, _, _ = next_and_close(tts._generate_with_prompt_cache(
                    target_text=clean, prompt_cache=pcache,
                    inference_timesteps=steps, cfg_value=2.0,
                    retry_badcase=True, streaming=False))
                wav = wav.detach().float().cpu().numpy()
            wall = time.monotonic() - t0
            import numpy as np
            wav_np = np.asarray(wav, dtype=np.float32).squeeze()
            wp = out_dir / f"{name}.wav"
            sf.write(str(wp), wav_np, sr)
            report["conditions"].append({
                "name": name, "text_i": ci, "text": clean,
                "mode": tag, "timesteps": steps,
                "wall_s": round(wall, 2),
                "dur_s": round(float(wav_np.size) / sr, 2)})
            print(f"{name}: {wall:.1f}s wall "
                  f"({wav_np.size/sr:.1f}s audio)", flush=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"report: {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
