"""Env-side LoRA/ baseline generator for the Phase 5B pilot.

Runs INSIDE backend_envs/voxcpm2 (the same inference env as production
zero-shot) so generation conditions are identical except the adapter.
Invoked by the main-env orchestrator (src/character_tts/evaluation/
lora_pilot.py); not meant to be run by hand, but can be:

    backend_envs/voxcpm2/Scripts/python.exe scripts/eval_voxcpm_lora_env.py \
        --snapshot <model dir> --cases cases.json --out-dir out \
        [--lora-weights <ckpt dir>] [--lora-config lora_cfg.json]

Each case: {case_id, text, seed, prompt_wav, prompt_text}
Writes <out>/<case_id>.wav plus <out>/<case_id>.json metadata
(retry count, wall time, args, model/adapter provenance).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lora-weights", default=None)
    ap.add_argument("--lora-config", default=None,
                    help="JSON file with enable_lm/enable_dit/enable_proj/"
                         "r/alpha/dropout")
    ap.add_argument("--gen-args", default=None,
                    help="JSON file overriding generation kwargs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    gen_args = {"cfg_value": 2.0, "inference_timesteps": 10,
                "normalize": True, "denoise": False, "retry_badcase": True}
    if args.gen_args:
        gen_args.update(json.loads(
            Path(args.gen_args).read_text(encoding="utf-8")))

    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    lora_config = None
    if args.lora_config:
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
        lora_config = LoRAConfigV2(**json.loads(
            Path(args.lora_config).read_text(encoding="utf-8")))

    model = VoxCPM.from_pretrained(
        args.snapshot,
        load_denoiser=False,
        optimize=True,
        local_files_only=True,
        lora_config=lora_config,
        lora_weights_path=args.lora_weights,
    )
    sr = int(getattr(model.tts_model, "sample_rate", 48000))

    manifest_rows = []
    for case in cases:
        cid = case["case_id"]
        out_path = out_dir / f"{cid.replace('/', '_')}.wav"
        meta_path = out_path.with_suffix(".json")
        seed = int(case["seed"])
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        errbuf = io.StringIO()
        t0 = time.monotonic()
        status, error = "ok", None
        try:
            with contextlib.redirect_stderr(errbuf):
                wav = model.generate(
                    text=case["text"],
                    prompt_wav_path=case["prompt_wav"],
                    prompt_text=case["prompt_text"],
                    **gen_args,
                )
            wav_np = np.asarray(wav, dtype=np.float32).squeeze()
            sf.write(str(out_path), wav_np, sr)
        except Exception as exc:  # keep evidence; never fabricate
            status, error = "error", f"{type(exc).__name__}: {exc}"
        wall = time.monotonic() - t0

        meta = {
            "case_id": cid,
            "status": status,
            "error": error,
            "seed": seed,
            "effective_args": {"text": case["text"],
                               "prompt_wav_path": case["prompt_wav"],
                               "prompt_text": case["prompt_text"],
                               **gen_args},
            "retry_badcase_enabled": gen_args["retry_badcase"],
            "retry_count": errbuf.getvalue().count("Badcase detected"),
            "wall_seconds": round(wall, 2),
            "sample_rate": sr,
            "output": out_path.name,
            "output_sha256": _sha256(out_path) if out_path.is_file() else None,
            "lora_weights": args.lora_weights,
            "lora_config": (json.loads(Path(args.lora_config).read_text())
                            if args.lora_config else None),
            "snapshot": args.snapshot,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        manifest_rows.append(meta)
        print(f"[eval] {cid} {status} "
              f"retry={meta['retry_count']} wall={wall:.1f}s",
              file=sys.stderr)

    (out_dir / "_gen_meta.json").write_text(
        json.dumps(manifest_rows, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0 if all(r["status"] == "ok" for r in manifest_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
