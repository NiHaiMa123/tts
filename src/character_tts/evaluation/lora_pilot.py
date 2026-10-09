"""Phase 5B pilot evaluation orchestrator (PLAN.md section 25.5-25.7).

Compares base zero-shot vs LoRA checkpoints under identical conditions.
Base cells reuse Phase 5A/5A2 wavs after full identity verification
(model revision, reference sha, text, seed, frozen args, on-disk sha);
LoRA cells are generated through the official inference path
(VoxCPM.from_pretrained + load_lora_weights) inside the SAME inference
env — never through teacher-forced training reconstructions.

New-WAV budget: <= 6 base + <= 18 LoRA = 24 hard cap.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..audio.io import safe_output_path, sha256_file, wav_info
from ..registry.loader import load_backend, repo_root
from ..registry.models import CharacterProfile
from . import codec_diag
from .gate import collect_metrics, verify_ground_truth, verify_reference
from .phase5a import _git_commit, _utc_now, _verify_sha
from .phase5a2 import (EXPECTED_GEN_ARGS, _load_reuse_manifest,
                       _verify_reuse, repo_root_of)

#: Fixed eval set (PLAN 25.5). prompt is always P0.
EVAL_CASES = [
    {"case_id": "rain_seed42", "text_id": "rain", "seed": 42,
     "reuse": ("phase5a", "stability/rain_seed42")},
    {"case_id": "short_response_seed42", "text_id": "short_response",
     "seed": 42, "reuse": ("phase5a", "stability/short_response_seed42")},
    {"case_id": "short_response_seed43", "text_id": "short_response",
     "seed": 43, "reuse": ("phase5a", "stability/short_response_seed43")},
    {"case_id": "short_response_2_seed42", "text_id": "short_response_2",
     "seed": 42,
     "reuse": ("phase5a2", "pair_B_short_response_2_seed42/P0")},
    {"case_id": "short_response_2_seed43", "text_id": "short_response_2",
     "seed": 43,
     "reuse": ("phase5a2", "pair_B_short_response_2_seed43/P0")},
    {"case_id": "mid_exposition_seed42", "text_id": "mid_exposition",
     "seed": 42, "reuse": ("phase5a", "stability/mid_exposition_seed42")},
]

CHECKPOINT_STEPS = (50, 100, 150)
MAX_NEW_WAVS = 24

#: Low-fatigue packs (PLAN 25.7): pack1 default, rest collapsed.
def build_packs(case_ids: list[str]) -> list[dict[str, Any]]:
    pack1_cases = ["rain_seed42", "short_response_seed42",
                   "short_response_2_seed42", "mid_exposition_seed42"]
    pack2_cases = ["short_response_seed42", "short_response_2_seed42"]
    p1 = [f"base-vs-100:{c}" for c in pack1_cases if c in case_ids]
    p2 = [f"50-vs-150:{c}" for c in pack2_cases if c in case_ids]
    seen = set(p1) | set(p2)
    rest = []
    for c in case_ids:
        for step in CHECKPOINT_STEPS:
            pid = f"base-vs-{step}:{c}"
            if pid not in seen:
                rest.append(pid)
    packs = [
        {"pack_id": "pack1", "collapsed": False,
         "title": "第 1 组：base vs step100（两短一中一雨，4 对）",
         "pairs": p1},
        {"pack_id": "pack2", "collapsed": True,
         "title": "第 2 组：step50 vs step150（最有区分力的短句，"
                  "检查是否早期已够）",
         "pairs": p2},
        {"pack_id": "pack3", "collapsed": True,
         "title": "第 3 组：其余配对（可选，不必一次听完）",
         "pairs": rest},
    ]
    return packs


def _case_pair_ids(case_ids: list[str]) -> list[str]:
    ids = [f"base-vs-{s}:{c}" for c in case_ids for s in CHECKPOINT_STEPS]
    ids += [f"50-vs-150:{c}" for c in
            ("short_response_seed42", "short_response_2_seed42")
            if c in case_ids]
    return ids


def _env_python() -> Path:
    return repo_root() / "backend_envs/voxcpm2/Scripts/python.exe"


def _gen_script() -> Path:
    return repo_root() / "scripts/lora/eval_voxcpm_lora_env.py"


def run_lora_eval(cfg: dict[str, Any], character: CharacterProfile,
                  log=print) -> dict[str, Any]:
    """Generate/verify the fixed comparison set and all artifacts."""
    out_dir = safe_output_path(repo_root_of(cfg), cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = load_backend(cfg["backend"])

    ref = verify_reference(character)
    gt = verify_ground_truth(character)
    prompts = {p["id"]: p for p in cfg.get("prompts") or []}
    p0 = prompts["P0"]
    p0["_sha"] = _verify_sha(p0["audio"], p0.get("sha256"), "prompt P0")
    texts = {t["id"]: t for t in cfg.get("texts") or []}
    for tid, t in texts.items():
        if t.get("audio"):
            t["_sha"] = _verify_sha(t["audio"], t.get("sha256"),
                                    f"text {tid} source audio")

    import shutil
    if gt is not None:
        shutil.copy2(gt["audio"], out_dir / "reference_clean.wav")
    shutil.copy2(p0["audio"], out_dir / "prompt_P0.wav")

    model_rev = profile.model.get("revision")
    train_data_sha = sha256_file(Path(cfg["train_manifest"])) \
        if cfg.get("train_manifest") else None

    reuse_dirs = {}
    for key, rel in (cfg.get("reuse_sources") or {}).items():
        d = Path(rel)
        reuse_dirs[key] = d if d.is_absolute() else repo_root_of(cfg) / d

    ckpts = cfg.get("checkpoints") or {}
    lora_cfg = cfg.get("lora") or {}

    manifest: dict[str, Any] = {
        "experiment_id": cfg.get("experiment_id", "lora_pilot_v1"),
        "character": character.character_id,
        "backend": cfg["backend"],
        "created_at": _utc_now(),
        "git_commit": _git_commit(repo_root_of(cfg)),
        "model": {"path_or_id": profile.model.get("path_or_id"),
                  "revision": model_rev},
        "frozen_generation": dict(profile.generation),
        "prompt": {"id": "P0", "sha256": p0["_sha"],
                   "note": "P0 is the fixed reference; P2 is NOT used"},
        "train_manifest_sha256": train_data_sha,
        "checkpoints": {},
        "cases": [], "pairs": [], "packs": [],
        "subjective": "PENDING_USER_LISTENING",
        "asr": "pending",
        "screening": "pending",
    }

    # ---- base cells: reuse with full identity verification ----------
    case_records: dict[str, dict[str, Any]] = {}
    text_by_case = {}
    for spec in EVAL_CASES:
        cid = spec["case_id"]
        t = texts[spec["text_id"]]
        text_by_case[cid] = t.get("text")
        src_key, src_cid = spec["reuse"]
        old_cases = _load_reuse_manifest(reuse_dirs[src_key])
        old = old_cases.get(src_cid)
        wav, reason = _verify_reuse(
            old, reuse_dirs[src_key], prompt_sha=p0["_sha"],
            text=t.get("text") or "", seed=spec["seed"],
            model_revision=model_rev)
        rec = {"case_id": f"base/{cid}", "condition": "base",
               "text_id": spec["text_id"], "text": t.get("text"),
               "prompt_id": "P0", "prompt_sha256": p0["_sha"],
               "seed": spec["seed"], "status": "pending",
               "reused_from": None}
        if wav is not None:
            rel = Path(os.path.relpath(wav, out_dir)).as_posix()
            info = wav_info(wav)
            rec.update(status="ok", reused_from=f"{src_key}:{src_cid}",
                       output=rel, output_sha256=old["output_sha256"],
                       sample_rate=info["sample_rate"],
                       duration_s=round(info["duration"], 3),
                       sanity=codec_diag.audio_sanity(wav))
            log(f"[pilot] base/{cid} reused+verified {src_cid}")
        else:
            rec.update(status="blocked_reuse", error=reason)
            log(f"[pilot] base/{cid} BLOCKED_REUSE: {reason}")
        manifest["cases"].append(rec)
        case_records[f"base/{cid}"] = rec

    # ---- LoRA cells: official inference path per checkpoint ---------
    cases_json = out_dir / "_eval_cases.json"
    gen_cases = [{"case_id": spec["case_id"],
                  "text": text_by_case[spec["case_id"]],
                  "seed": spec["seed"],
                  "prompt_wav": p0["audio"],
                  "prompt_text": p0.get("text")}
                 for spec in EVAL_CASES]
    cases_json.write_text(json.dumps(gen_cases, ensure_ascii=False),
                          encoding="utf-8")
    lora_cfg_json = out_dir / "_lora_cfg.json"
    lora_cfg_json.write_text(json.dumps(lora_cfg), encoding="utf-8")

    new_generated = 0
    for step in CHECKPOINT_STEPS:
        ck = ckpts.get(str(step)) or {}
        ck_dir = Path(ck.get("path") or
                      f"missing_step_{step}")
        weights = ck_dir / "lora_weights.safetensors"
        cond = f"ckpt{step}"
        manifest["checkpoints"][cond] = {
            "step": step, "path": str(ck_dir),
            "weights_sha256": sha256_file(weights)
            if weights.is_file() else None,
            "lora": lora_cfg,
        }
        for spec in EVAL_CASES:
            cid = spec["case_id"]
            rec = {"case_id": f"{cond}/{cid}", "condition": cond,
                   "text_id": spec["text_id"], "text": text_by_case[cid],
                   "prompt_id": "P0", "prompt_sha256": p0["_sha"],
                   "seed": spec["seed"], "status": "pending",
                   "checkpoint": cond,
                   "reused_from": None}
            manifest["cases"].append(rec)
            case_records[rec["case_id"]] = rec

        if not ck_dir.is_dir() or not weights.is_file():
            for spec in EVAL_CASES:
                case_records[f"{cond}/{spec['case_id']}"].update(
                    status="blocked", error=f"checkpoint {cond} missing")
            log(f"[pilot] {cond} checkpoint missing — cells blocked")
            continue
        if new_generated + len(EVAL_CASES) > MAX_NEW_WAVS:
            for spec in EVAL_CASES:
                case_records[f"{cond}/{spec['case_id']}"].update(
                    status="blocked_budget", error="24-wav cap")
            continue

        sub_dir = out_dir / cond
        res = subprocess.run(
            [str(_env_python()), str(_gen_script()),
             "--snapshot", str(cfg["snapshot"]),
             "--cases", str(cases_json),
             "--out-dir", str(sub_dir),
             "--lora-weights", str(ck_dir),
             "--lora-config", str(lora_cfg_json)],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=str(repo_root_of(cfg)),
            env={**os.environ, "HF_HUB_OFFLINE": "1"}, timeout=3600)
        log(f"[pilot] {cond} gen rc={res.returncode}\n"
            f"{(res.stderr or '')[-800:]}")
        metas = {}
        meta_file = sub_dir / "_gen_meta.json"
        if meta_file.is_file():
            metas = {m["case_id"]: m
                     for m in json.loads(meta_file.read_text(
                         encoding="utf-8"))}
        for spec in EVAL_CASES:
            cid = spec["case_id"]
            rec = case_records[f"{cond}/{cid}"]
            m = metas.get(cid)
            if not m:
                rec.update(status="error", error="no metadata returned")
                continue
            wav = sub_dir / m["output"] if m.get("output") else None
            rec.update({
                "status": m["status"],
                "error": m.get("error"),
                "retry_count": m.get("retry_count"),
                "wall_seconds": m.get("wall_seconds"),
                "effective_args": m.get("effective_args"),
                "sample_rate": m.get("sample_rate"),
                "adapter_sha256": manifest["checkpoints"][cond][
                    "weights_sha256"],
            })
            if m["status"] == "ok" and wav and wav.is_file():
                info = wav_info(wav)
                rec.update(
                    output=f"{cond}/{m['output']}",
                    output_sha256=sha256_file(wav),
                    duration_s=round(info["duration"], 3),
                    sanity=codec_diag.audio_sanity(wav))
                new_generated += 1

    # ---- pairs + packs ----------------------------------------------
    case_ids = [s["case_id"] for s in EVAL_CASES]
    for pid in _case_pair_ids(case_ids):
        left, cid = pid.split(":", 1)
        a, b = left.split("-vs-")
        if a == "base":
            cells = [f"base/{cid}", f"ckpt{b}/{cid}"]
            ptype = "base_vs_ckpt"
        else:
            cells = [f"ckpt{a}/{cid}", f"ckpt{b}/{cid}"]
            ptype = "ckpt_vs_ckpt"
        manifest["pairs"].append({"pair_id": pid, "type": ptype,
                                  "text_id": next(
                                      s["text_id"] for s in EVAL_CASES
                                      if s["case_id"] == cid),
                                  "seed": next(
                                      s["seed"] for s in EVAL_CASES
                                      if s["case_id"] == cid),
                                  "cells": cells})
    manifest["packs"] = build_packs(case_ids)
    manifest["budget"] = {"new_generated": new_generated,
                          "limit": MAX_NEW_WAVS}
    manifest["finished_at"] = _utc_now()

    # ---- screening (hard errors only; subjective = UNDETERMINED) ----
    screening = {"cases": {}, "flags": []}
    for c in manifest["cases"]:
        s = c.get("sanity") or {}
        flags = []
        if c["status"] != "ok":
            flags.append(c["status"])
        else:
            if not s.get("finite", True):
                flags.append("non_finite")
            if s.get("clip_ratio", 0) > 0:
                flags.append("clipping")
            if s.get("silence_ratio", 0) > 0.85:
                flags.append("mostly_silent")
            if (c.get("retry_count") or 0) > 0:
                flags.append(f"retry_{c['retry_count']}")
        screening["cases"][c["case_id"]] = {
            "flags": flags, "subjective": "UNDETERMINED",
            "duration_s": c.get("duration_s"),
            "sha256": c.get("output_sha256")}
        if flags:
            screening["flags"].append({"case_id": c["case_id"],
                                       "flags": flags})
    manifest["screening"] = "done"
    (out_dir / "screening.json").write_text(
        json.dumps(screening, ensure_ascii=False, indent=2),
        encoding="utf-8")

    metrics = collect_metrics(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from ..web.listen_page import write_lora_pilot_listen_page
    write_lora_pilot_listen_page(out_dir, manifest)
    return manifest


def attach_asr_results(manifest_path: str | Path,
                       asr_json_path: str | Path) -> dict[str, Any]:
    """Merge auxiliary ASR transcripts/CER into the pilot manifest.

    ASR is auxiliary evidence only — it never decides pair winners."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr = json.loads(Path(asr_json_path).read_text(encoding="utf-8"))
    by_case = {r["case_id"]: r for r in asr.get("results") or []}
    for case in manifest.get("cases") or []:
        rec = by_case.get(case["case_id"])
        if rec:
            case["asr_aux"] = rec
    manifest["asr"] = {
        "status": "done",
        "backend": asr.get("backend"),
        "model": asr.get("model"),
        "results_file": Path(asr_json_path).name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
