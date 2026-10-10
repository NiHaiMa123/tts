"""一次性：对每个 LoRA ckpt 生成《睡眠》分块版，供 A/B 对比。

用法（voxcpm2 env）：
    backend_envs/voxcpm2/Scripts/python.exe scripts/lora/gen_aimisi_lora_ab.py

产物：outputs/_tmp/lora_ab/{base,step50,step100,step150}/睡眠.wav + .json
"""
import sys, os, json, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "workers")

SNAP = ("hf_cache/voxcpm2/models--openbmb--VoxCPM2/snapshots/"
        "32279effe8c19989596f05d353d1447f51d9e915")
CKPT_ROOT = Path("outputs/training/aimisi_voxcpm_lora_pilot/checkpoints")
OUT = Path("outputs/_tmp/lora_ab")
LORA_CFG = {"enable_lm": True, "enable_dit": True, "enable_proj": False,
            "r": 16, "alpha": 16, "dropout": 0.0}
# d8b9 = 实测最优参考音（mean cos_centroid 0.899，当前 ref 0.842）
REF_WAV = ("data/characters/aimisi/datasets/v1/audio/"
           "d8b99e88731cd1e6e9dc5bec4667ba75727adc843089d67b04e4bc503f00677a.wav")
REF_TEXT = "啊……是陆医生。和他说话的时候，就当我不在吧？不然会被当成对着空气谈天的怪人哦。"
TEXT = Path("inputs/睡眠.txt").read_text(encoding="utf-8")
GEN = {"cfg_value": 2.0, "inference_timesteps": 10, "normalize": True,
       "denoise": False, "retry_badcase": True}


def run_one(tag: str, lora_weights: str | None):
    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM
    from voxcpm_worker import VoxCPMWorker

    kwargs = {"load_denoiser": False, "optimize": True,
              "local_files_only": True}
    if lora_weights:
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
        kwargs["lora_config"] = LoRAConfigV2(**LORA_CFG)
        kwargs["lora_weights_path"] = lora_weights
    model = VoxCPM.from_pretrained(SNAP, **kwargs)

    chunks = VoxCPMWorker._split_text(TEXT, 100)   # staticmethod 复用
    print(f"[{tag}] {len(chunks)} chunks, lora={lora_weights or 'none'}",
          flush=True)
    tts = model.tts_model
    pcache = tts.build_prompt_cache(prompt_text=REF_TEXT,
                                    prompt_wav_path=REF_WAV,
                                    reference_wav_path=None)
    from voxcpm.model.utils import next_and_close
    waves = []
    t0 = time.monotonic()
    for i, chunk in enumerate(chunks):
        torch.manual_seed(42 + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42 + i)
        wav, _a, _b = next_and_close(
            tts._generate_with_prompt_cache(
                target_text=chunk, prompt_cache=pcache,
                min_len=2, max_len=4096,
                inference_timesteps=GEN["inference_timesteps"],
                cfg_value=GEN["cfg_value"],
                retry_badcase=GEN["retry_badcase"],
                retry_badcase_max_times=3,
                retry_badcase_ratio_threshold=6.0,
                streaming=False))
        waves.append(wav.reshape(-1))
    gap = torch.zeros(int(getattr(tts, "sample_rate", 48000) * 0.2))
    joined = []
    for i, wv in enumerate(waves):
        if i:
            joined.append(gap)
        joined.append(wv)
    wav = torch.cat(joined).float().cpu().numpy()
    sr = int(getattr(tts, "sample_rate", 48000))
    d = OUT / tag
    d.mkdir(parents=True, exist_ok=True)
    sf.write(str(d / "睡眠.wav"), np.asarray(wav, dtype=np.float32), sr)
    meta = {"tag": tag, "lora_weights": lora_weights, "lora": LORA_CFG,
            "ref": "d8b9", "seed_base": 42, "chunks": len(chunks),
            "duration": round(len(wav) / sr, 2),
            "wall_seconds": round(time.monotonic() - t0, 1),
            "gen": GEN}
    (d / "睡眠.wav.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{tag}] done {meta['duration']}s wall={meta['wall_seconds']}s",
          flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    only = sys.argv[1:] or None
    for tag, ck in [("step50", "step_0000050"), ("step100", "step_0000100"),
                    ("step150", "step_0000150")]:
        if only and tag not in only:
            continue
        run_one(tag, str(CKPT_ROOT / ck))
    if not only or "base" in only:
        run_one("base", None)               # 对照：无 LoRA、同 ref/分块
