"""Phase 5C — VoxCPM2 LoRA effect audit runner (inference env side).

Runs INSIDE backend_envs/voxcpm2 (torch + voxcpm + safetensors present).
Pure analysis lives in src/character_tts/evaluation/lora_audit.py; this
script only collects raw evidence:

    backend_envs/voxcpm2/Scripts/python.exe scripts/audit_voxcpm_lora_effect.py \
        --snapshot <model snapshot dir> \
        --ckpt-root outputs/training/suoming_voxcpm_lora_pilot/checkpoints \
        --out-dir outputs/diagnostics/suoming_voxcpm_lora_effect_v1 \
        --steps static,load,onoff \
        --text "开伞，由我来动手。" --prompt-wav <P0 wav> --prompt-text <P0 text> --seed 42

Steps: static (no GPU), load (GPU, integrity), onoff (GPU, 4 WAVs max).
Read-only wrt checkpoints/base model. Never trains, never writes outside
--out-dir.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import contextlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.evaluation.lora_audit import (  # noqa: E402
    classify_lora_key, compare_generation_args, evaluate_load_integrity,
    evaluate_onoff, evaluate_weight_diffs, safetensors_header,
    sha256_file, weights_census,
)

CKPT_STEPS = {"ckpt50": "step_0000050", "ckpt100": "step_0000100",
              "ckpt150": "step_0000150"}
GEN_ARGS = {"cfg_value": 2.0, "inference_timesteps": 10,
            "normalize": True, "denoise": False, "retry_badcase": True}


def _jdump(obj, path: Path):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(f"[audit] wrote {path}", file=sys.stderr)


# --------------------------------------------------------------------- static
def step_static(ckpt_root: Path, out_dir: Path) -> dict:
    import torch  # noqa: F401
    from safetensors.torch import load_file

    report = {"checkpoints": {}, "diffs": {}, "configs": {}}
    tensors_by_ckpt = {}
    for tag, dname in CKPT_STEPS.items():
        d = ckpt_root / dname
        st_path = d / "lora_weights.safetensors"
        cfg_path = d / "lora_config.json"
        entry = {"dir": str(d), "exists": st_path.is_file()}
        if not st_path.is_file():
            report["checkpoints"][tag] = entry
            continue
        header = safetensors_header(st_path)
        sd = load_file(str(st_path), device="cpu")
        tensors_by_ckpt[tag] = sd
        stats = {}
        for k, t in sd.items():
            tf = t.float()
            stats[k] = {
                "l2": float(tf.norm()),
                "nonzero": float((tf != 0).float().mean()),
                "nan": int(tf.isnan().sum()), "inf": int(tf.isinf().sum()),
            }
        entry.update({
            "sha256": sha256_file(st_path),
            "key_count": len(header),
            "keys_by_group": {},
            "census": weights_census(header, stats),
            "tensor_stats_sample": {
                k: stats[k] for k in list(stats)[:4]},
        })
        for k in header:
            g = classify_lora_key(k)["group"]
            entry["keys_by_group"].setdefault(g, []).append(k)
        if cfg_path.is_file():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            report["configs"][tag] = cfg
            entry["lora_config"] = cfg
        report["checkpoints"][tag] = entry

    # step-to-step diffs
    order = ["ckpt50", "ckpt100", "ckpt150"]
    diffs = {}
    for a, b in zip(order, order[1:]):
        if a not in tensors_by_ckpt or b not in tensors_by_ckpt:
            continue
        sa, sb = tensors_by_ckpt[a], tensors_by_ckpt[b]
        per_key = {}
        for k in sa:
            if k not in sb:
                per_key[k] = {"missing_in_b": True}
                continue
            ta, tb = sa[k].float(), sb[k].float()
            if ta.shape != tb.shape:
                per_key[k] = {"shape_mismatch": [list(ta.shape), list(tb.shape)]}
                continue
            delta = (tb - ta)
            na = float(ta.norm())
            per_key[k] = {
                "max_abs": float(delta.abs().max()),
                "rel_l2": float(delta.norm()) / na if na else None,
                "identical": bool((ta == tb).all()),
            }
        for k in set(sb) - set(sa):
            per_key[k] = {"missing_in_a": True}
        diffs[f"{a}->{b}"] = per_key
    report["diffs_eval"] = evaluate_weight_diffs(diffs, None)
    report["config_identical"] = (
        len({json.dumps(c, sort_keys=True)
             for c in report["configs"].values()}) <= 1)
    _jdump(report, out_dir / "weights_metrics.json")
    return report


# ------------------------------------------------------------------ gpu load
def _pin_seed(seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lora_module_census(model) -> dict:
    import torch.nn as nn
    from voxcpm.modules.layers.lora import LoRALinear
    census = {"total": 0, "enabled": 0, "by_group": {}, "param_keys": 0}
    lora_param_keys = 0
    for name, mod in model.tts_model.named_modules():
        if isinstance(mod, LoRALinear):
            grp = ("lm" if "base_lm." in name or "residual_lm." in name
                   else "dit" if "feat_decoder." in name
                   else "proj" if "_proj" in name else "other")
            b = census["by_group"].setdefault(grp, {"modules": 0, "enabled": 0})
            b["modules"] += 1
            census["total"] += 1
            if mod.enabled:
                b["enabled"] += 1
                census["enabled"] += 1
    for n, _ in model.tts_model.named_parameters():
        if "lora_" in n:
            lora_param_keys += 1
    census["param_keys"] = lora_param_keys
    return census


def _build_model(snapshot: str, lora_cfg: dict | None, lora_weights=None):
    from voxcpm import VoxCPM
    cfg = None
    if lora_cfg is not None:
        from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
        cfg = LoRAConfigV2(**lora_cfg)
    return VoxCPM.from_pretrained(
        snapshot, load_denoiser=False, optimize=True,
        local_files_only=True, lora_config=cfg,
        lora_weights_path=lora_weights)


def _inner_lora_cfg(cfg_json: dict | None) -> dict | None:
    """ckpt lora_config.json wraps config as {base_model, lora_config};
    eval-side flat configs are accepted as-is."""
    if not cfg_json:
        return None
    inner = cfg_json.get("lora_config")
    return inner if isinstance(inner, dict) else cfg_json


def step_load(snapshot: str, ckpt_root: Path, out_dir: Path,
              static_report: dict) -> dict:
    import torch  # noqa: F401
    tag = "ckpt100"
    ckpt_dir = ckpt_root / CKPT_STEPS[tag]
    lora_cfg = _inner_lora_cfg(static_report["configs"].get(tag))
    ckpt_keys = list(
        safetensors_header(ckpt_dir / "lora_weights.safetensors"))

    model = _build_model(snapshot, lora_cfg, lora_weights=None)
    census_pre = _lora_module_census(model)
    loaded, skipped = model.tts_model.load_lora_weights(str(ckpt_dir))
    integrity = evaluate_load_integrity(ckpt_keys, loaded, skipped)
    census_post = _lora_module_census(model)

    # enabled toggling check
    model.tts_model.set_lora_enabled(False)
    off = _lora_module_census(model)
    model.tts_model.set_lora_enabled(True)
    on = _lora_module_census(model)

    report = {
        "checkpoint": tag,
        "ckpt_sha256": sha256_file(ckpt_dir / "lora_weights.safetensors"),
        "lora_config_used": lora_cfg,
        "model_lora_modules_pre": census_pre,
        "load": integrity,
        "toggle": {"after_disable": off, "after_enable": on},
        "model_lora_modules_post": census_post,
        "sample_loaded_keys": loaded[:6],
    }
    _jdump(report, out_dir / "load_report.json")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


# -------------------------------------------------------------------- onoff
def step_onoff(snapshot: str, ckpt_root: Path, out_dir: Path,
               static_report: dict, text: str, prompt_wav: str,
               prompt_text: str, seed: int) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import correlate  # noqa: F401  (only if needed later)

    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    tag = "ckpt100"
    ckpt_dir = ckpt_root / CKPT_STEPS[tag]
    lora_cfg = _inner_lora_cfg(static_report["configs"].get(tag))
    sr = 48000
    results = {}

    # If a previous mis-configured run left base-only WAVs, reuse the
    # determinism evidence: condition A wav (native base, same params).
    a_meta_path = wav_dir / "A.json"

    def gen(model, label):
        _pin_seed(seed)
        errbuf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stderr(errbuf):
            wav = model.generate(
                text=text, prompt_wav_path=prompt_wav,
                prompt_text=prompt_text, **GEN_ARGS)
        wall = time.monotonic() - t0
        arr = np.asarray(wav, dtype=np.float32).squeeze()
        p = wav_dir / f"{label}.wav"
        sf.write(str(p), arr, sr)
        meta = {
            "label": label, "sha256": sha256_file(p), "seed": seed,
            "text": text, "prompt_wav": prompt_wav,
            "prompt_sha256": sha256_file(prompt_wav),
            "gen_args": dict(GEN_ARGS), "wall_seconds": round(wall, 2),
            "retry_count": errbuf.getvalue().count("Badcase detected"),
            "duration_s": round(len(arr) / sr, 3),
            "lora_weights": str(ckpt_dir) if label != "A" else None,
            "ckpt_sha256": sha256_file(
                ckpt_dir / "lora_weights.safetensors") if label != "A" else None,
        }
        (wav_dir / f"{label}.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[audit] gen {label}: sha={meta['sha256'][:12]} "
              f"dur={meta['duration_s']}s wall={wall:.1f}s "
              f"retry={meta['retry_count']}", file=sys.stderr)
        return meta, arr

    def wave_diff(x, y):
        n = min(len(x), len(y))
        if n == 0:
            return {"corr": None}
        a, b = x[:n], y[:n]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return {
            "corr": float(np.dot(a, b) / denom) if denom else 1.0,
            "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
            "len_a": len(x), "len_b": len(y),
        }

    # Condition A: native base (no LoRA modules). If a previous run left a
    # valid A (same text/seed/prompt/gen args), reuse it — generation under
    # manual_seed is deterministic, so re-generating adds no evidence.
    wa = None
    if a_meta_path.is_file():
        prev = json.loads(a_meta_path.read_text(encoding="utf-8"))
        same = (prev.get("seed") == seed and prev.get("text") == text
                and prev.get("prompt_sha256") == sha256_file(prompt_wav)
                and prev.get("gen_args") == GEN_ARGS)
        a_wav = wav_dir / "A.wav"
        if same and a_wav.is_file() and sha256_file(a_wav) == prev["sha256"]:
            results["A"] = prev
            results["A"]["reused"] = True
            import soundfile as sf
            wa, _ = sf.read(str(a_wav), dtype="float32")
            print(f"[audit] gen A: reused sha={prev['sha256'][:12]}",
                  file=sys.stderr)
    if wa is None:
        model_a = _build_model(snapshot, None, None)
        results["A"], wa = gen(model_a, "A")
        del model_a
        torch.cuda.empty_cache()

    # Conditions B/C/C-repeat: one instance, weights loaded, toggled
    model_l = _build_model(snapshot, lora_cfg, str(ckpt_dir))
    model_l.tts_model.set_lora_enabled(False)
    results["B"], wb = gen(model_l, "B")
    model_l.tts_model.set_lora_enabled(True)
    results["C"], wc = gen(model_l, "C")
    results["Crep"], wcrep = gen(model_l, "Crep")
    del model_l
    torch.cuda.empty_cache()

    diffs = {
        "A_vs_B": wave_diff(wa, wb),
        "B_vs_C": wave_diff(wb, wc),
        "C_vs_Crep": wave_diff(wc, wcrep),
    }
    verdict = evaluate_onoff(
        results["A"]["sha256"], results["B"]["sha256"],
        results["C"]["sha256"], results["Crep"]["sha256"], diffs)
    report = {"conditions": results, "diffs": diffs, "onoff": verdict}
    _jdump(report, out_dir / "onoff_report.json")
    return report


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", default="static,load,onoff")
    ap.add_argument("--text", default="开伞，由我来动手。")
    ap.add_argument("--prompt-wav", default=None)
    ap.add_argument("--prompt-text", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = Path(args.ckpt_root)
    steps = {s.strip() for s in args.steps.split(",") if s.strip()}

    env_info = {}
    try:
        import torch, voxcpm  # noqa: F401
        import importlib.metadata as im
        env_info = {
            "torch": torch.__version__,
            "voxcpm_dist": im.version("voxcpm"),
            "cuda": torch.cuda.is_available(),
            "device": (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else None),
            "voxcpm_file": voxcpm.__file__,
        }
    except Exception as e:  # static step still works
        env_info["error"] = str(e)
    manifest = {"env": env_info, "steps": {}}
    _jdump(env_info, out_dir / "env_info.json")

    static_report = None
    if "static" in steps:
        static_report = step_static(ckpt_root, out_dir)
        manifest["steps"]["static"] = "ok"
    elif (out_dir / "weights_metrics.json").is_file():
        static_report = json.loads(
            (out_dir / "weights_metrics.json").read_text(encoding="utf-8"))

    if "load" in steps:
        manifest["steps"]["load"] = step_load(
            args.snapshot, ckpt_root, out_dir, static_report)["load"]["verdict"]

    if "onoff" in steps:
        manifest["steps"]["onoff"] = step_onoff(
            args.snapshot, ckpt_root, out_dir, static_report,
            args.text, args.prompt_wav, args.prompt_text, args.seed
        )["onoff"]["verdict"]

    _jdump(manifest, out_dir / "audit_run_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
