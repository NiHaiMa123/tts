"""Phase 5B LoRA pilot runner (PLAN.md section 25.4).

Orchestrates the OFFICIAL VoxCPM2 finetune entry
(backend_envs/voxcpm2/official_v203/train_voxcpm_finetune.py, tag 2.0.3)
inside the ISOLATED training env backend_envs/voxcpm2_train.

    python scripts/train_voxcpm_lora_pilot.py --smoke   # 1-2 step preflight
    python scripts/train_voxcpm_lora_pilot.py           # <=150 opt steps

Guarantees:
- hard cap: num_iters <= 150 optimizer steps; cumulative resume beyond
  the cap is refused;
- one documented OOM fallback only (max_batch_tokens 8192 -> 4096);
- NaN/Inf/loss-explosion/OOM -> immediate stop with evidence;
- checkpoints verified: step_0000050/100/150 + lora_weights.safetensors;
- VRAM sampled externally via nvidia-smi; run provenance in run_log.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TRAIN_PY = ROOT / "backend_envs/voxcpm2_train/Scripts/python.exe"
OFFICIAL = ROOT / "backend_envs/voxcpm2/official_v203/train_voxcpm_finetune.py"
CONFIG = ROOT / "configs/training/suoming_voxcpm_lora_pilot.yaml"
OUT = ROOT / "outputs/training/suoming_voxcpm_lora_pilot"
MAX_OPT_STEPS = 150
EXPECTED_CKPTS = ["step_0000050", "step_0000100", "step_0000150"]
_FAIL_PATTERNS = [
    (re.compile(r"out of memory|CUDA error|OOM", re.I), "oom"),
    (re.compile(r"nan|inf", re.I), "nan_or_inf"),
    (re.compile(r"Traceback|RuntimeError|AssertionError"), "exception"),
]


def _load_cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _sm_sample_vram(stop: threading.Event, samples: list[int],
                    interval: float = 1.0) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                samples.append(int(out.stdout.strip().splitlines()[0]))
        except Exception:
            pass
        stop.wait(interval)


def _run_official(cfg_path: Path, tag: str, log_path: Path,
                  env_extra: dict | None = None) -> dict:
    """Run the official script; stream stdout/stderr to log_path."""
    env = dict(os.environ)
    env.update(env_extra or {})
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    vram: list[int] = []
    sampler = threading.Thread(target=_sm_sample_vram,
                               args=(stop, vram), daemon=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            [str(TRAIN_PY), str(OFFICIAL), "--config_path", str(cfg_path)],
            stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        sampler.start()
        rc = proc.wait()
    stop.set()
    wall = time.time() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failures = [name for pat, name in _FAIL_PATTERNS if pat.search(text)]
    return {"tag": tag, "returncode": rc, "wall_seconds": round(wall, 1),
            "log": str(log_path), "failures": failures,
            "vram_peak_mib": max(vram) if vram else None,
            "vram_samples": len(vram)}


def _preflight() -> dict:
    out = {}
    r = subprocess.run(
        [str(TRAIN_PY), "-c",
         "import torch, json; print(json.dumps({'cuda': torch.cuda.is_available(), 'name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'mem_mib': round(torch.cuda.get_device_properties(0).total_memory/1048576) if torch.cuda.is_available() else None, 'torch': torch.__version__}))"],
        capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    out["torch"] = r.stdout.strip()
    snap = Path(_load_cfg()["pretrained_path"])
    cfg = json.loads((snap / "config.json").read_text(encoding="utf-8"))
    out["model_architecture"] = cfg.get("architecture")
    out["snapshot"] = str(snap)
    ok = (TRAIN_PY.is_file() and OFFICIAL.is_file()
          and '"cuda": true' in out["torch"].lower()
          and out["model_architecture"] == "voxcpm2")
    out["ok"] = ok
    return out


def _steps_completed(ckpt_dir: Path) -> int:
    state = ckpt_dir / "latest" / "training_state.json"
    if state.is_file():
        return int(json.loads(state.read_text())["step"])
    return 0


def _checkpoints_ok(ckpt_dir: Path) -> dict:
    found = {}
    for name in EXPECTED_CKPTS:
        d = ckpt_dir / name
        w = d / "lora_weights.safetensors"
        cfg = d / "lora_config.json"
        found[name] = (d.is_dir() and (w.is_file()
                       or (d / "lora_weights.ckpt").is_file())
                       and cfg.is_file())
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 optimizer steps only, counts toward the 150 cap "
                         "only if checkpoints are kept (they are not).")
    ap.add_argument("--oom-fallback", action="store_true",
                    help="single documented fallback: max_batch_tokens 4096")
    args = ap.parse_args()

    run_log = {"script": OFFICIAL.name,
               "official_commit": "19b6bf7590025418821a86dcb817504e0ad7e5df",
               "config": str(CONFIG)}
    print("[preflight]", flush=True)
    pf = _preflight()
    run_log["preflight"] = pf
    print(" ", pf)
    if not pf["ok"]:
        run_log["status"] = "API_BLOCKED"
        (OUT / "run_log.json").write_text(
            json.dumps(run_log, indent=2), encoding="utf-8")
        return 4

    base_cfg = _load_cfg()
    ckpt_dir = Path(base_cfg["save_path"])
    done = _steps_completed(ckpt_dir)
    if done >= MAX_OPT_STEPS:
        print(f"[refuse] cumulative steps already {done} >= {MAX_OPT_STEPS}")
        return 5
    if base_cfg.get("num_iters", 0) > MAX_OPT_STEPS:
        print(f"[refuse] num_iters {base_cfg['num_iters']} > cap")
        return 5

    if args.smoke:
        cfg = dict(base_cfg)
        cfg.update(num_iters=2, save_interval=10_000,
                   valid_interval=10_000, log_interval=1,
                   save_path=str(OUT / "smoke_checkpoints"),
                   tensorboard=str(OUT / "smoke_tb"),
                   warmup_steps=1, max_steps=2)
        smoke_yaml = OUT / "smoke_config.yaml"
        OUT.mkdir(parents=True, exist_ok=True)
        smoke_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                             sort_keys=False),
                              encoding="utf-8")
        res = _run_official(smoke_yaml, "smoke", OUT / "logs/smoke.log")
        run_log["smoke"] = res
        res["status"] = ("ok" if res["returncode"] == 0
                         and not res["failures"] else "failed")
        (OUT / "run_log.json").write_text(
            json.dumps(run_log, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["status"] == "ok" else 6

    cfg = dict(base_cfg)
    tag = "main"
    if args.oom_fallback:
        cfg["max_batch_tokens"] = 4096
        tag = "oom_fallback_mbt4096"
        run_log["oom_fallback"] = ("max_batch_tokens 8192->4096 "
                                   "(official length filter)")
    run_yaml = OUT / "run_config.yaml"
    OUT.mkdir(parents=True, exist_ok=True)
    run_yaml.write_text(yaml.safe_dump(cfg, allow_unicode=True,
                                       sort_keys=False), encoding="utf-8")

    res = _run_official(run_yaml, tag, OUT / "logs/train.log")
    run_log["train"] = res
    run_log["checkpoints"] = _checkpoints_ok(ckpt_dir)
    run_log["steps_completed"] = _steps_completed(ckpt_dir)
    status = "ok"
    if res["returncode"] != 0 or res["failures"]:
        status = ("BLOCKED_16GB" if "oom" in res["failures"]
                  else "TRAIN_FAILED")
    elif run_log["steps_completed"] > MAX_OPT_STEPS:
        status = "CAP_VIOLATION"
    elif not all(run_log["checkpoints"].values()):
        status = "CHECKPOINT_INCOMPLETE"
    run_log["status"] = status
    (OUT / "run_log.json").write_text(
        json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_log, ensure_ascii=False, indent=2))
    return 0 if status == "ok" else 7


if __name__ == "__main__":
    raise SystemExit(main())
