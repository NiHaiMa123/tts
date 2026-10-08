"""Phase 5A2 runner — paired P0/P2 stability check (PLAN.md section 24).

Matrix: 7 pairs (same text + seed, prompt is the only variable).
Four P0 cells reuse Phase 5A wavs after per-file sha + parameter
verification; anything that fails verification is marked
``BLOCKED_REUSE`` and never silently regenerated.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from ..audio.io import read_wav, safe_output_path, sha256_file, wav_info
from ..backends.manager import BackendManager
from ..registry.loader import load_backend, repo_root
from ..registry.models import CharacterProfile
from . import codec_diag
from .gate import collect_metrics, verify_ground_truth, verify_reference
from .phase5a import _git_commit, _utc_now, _verify_sha, GENERATE_TIMEOUT_S

logger = logging.getLogger(__name__)

NEW_WAV_BUDGET = 10

#: Frozen generation args every case must report — reuse verification
#: compares the old manifest's effective_args against this set.
EXPECTED_GEN_ARGS = {
    "cfg_value": 2.0,
    "inference_timesteps": 10,
    "normalize": True,
    "denoise": False,
    "retry_badcase": True,
}


def _load_reuse_manifest(reuse_dir: Path) -> dict[str, dict[str, Any]]:
    path = reuse_dir / "manifest.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {c["case_id"]: c for c in data.get("cases") or []}
    out["__model__"] = data.get("model") or {}
    return out


def _verify_reuse(old_case: dict[str, Any], reuse_dir: Path,
                  *, prompt_sha: str, text: str, seed: int,
                  model_revision: str | None) -> tuple[Path | None, str | None]:
    """Return (wav_path, None) when every identity check passes,
    else (None, reason). Never substitutes a fresh wav."""
    if not old_case or old_case.get("status") != "ok":
        return None, "prior case missing or not ok"
    wav = reuse_dir / str(old_case.get("output", ""))
    if not wav.is_file():
        return None, f"reused wav missing on disk: {wav}"
    if sha256_file(wav) != old_case.get("output_sha256"):
        return None, "on-disk wav sha256 != prior manifest"
    if old_case.get("prompt_sha256") != prompt_sha:
        return None, "prompt sha256 mismatch"
    if int(old_case.get("seed", -1)) != seed:
        return None, "seed mismatch"
    if (old_case.get("text") or "") != text:
        return None, "text mismatch"
    args = old_case.get("effective_args") or {}
    for k, v in EXPECTED_GEN_ARGS.items():
        if args.get(k) != v:
            return None, f"effective arg {k}={args.get(k)} != frozen {v}"
    if model_revision:
        prior = old_case.get("model_revision") or old_case.get("_model_rev")
        # phase5a manifest stores revision at top level; fall through ok
        # when the per-case record lacks it — the experiment pins one rev.
        if prior and prior != model_revision:
            return None, "model revision mismatch"
    return wav, None


def run_phase5a2(cfg: dict[str, Any], character: CharacterProfile,
                 manager: BackendManager | None = None,
                 log=print) -> dict[str, Any]:
    out_dir = safe_output_path(repo_root_of(cfg), cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    backend_id = cfg["backend"]
    profile = load_backend(backend_id)
    if not profile.enabled:
        raise RuntimeError(f"backend {backend_id} disabled")

    ref = verify_reference(character)
    gt = verify_ground_truth(character)

    prompts = {p["id"]: p for p in cfg.get("prompts") or []}
    texts = {t["id"]: t for t in cfg.get("texts") or []}
    for pid, p in prompts.items():
        p["_sha"] = _verify_sha(p["audio"], p.get("sha256"), f"prompt {pid}")
    for tid, t in texts.items():
        if t.get("audio"):
            t["_sha"] = _verify_sha(t["audio"], t.get("sha256"),
                                    f"text {tid} source audio")

    # Calibration originals for the listening page: ground truth +
    # both prompt audios (P0 baseline, P2 candidate) side by side.
    import shutil
    if gt is not None:
        shutil.copy2(gt["audio"], out_dir / "ground_truth_original.wav")
    shutil.copy2(ref["audio"], out_dir / "prompt_P0_original.wav")
    shutil.copy2(prompts["P2"]["audio"],
                 out_dir / "prompt_P2_original.wav")

    reuse_dir = Path(cfg["reuse_from"])
    if not reuse_dir.is_absolute():
        reuse_dir = repo_root_of(cfg) / reuse_dir
    old_cases = _load_reuse_manifest(reuse_dir)
    model_rev = profile.model.get("revision")

    manifest: dict[str, Any] = {
        "experiment_id": cfg.get("experiment_id", "phase5a2"),
        "character": character.character_id,
        "backend": backend_id,
        "created_at": _utc_now(),
        "git_commit": _git_commit(repo_root_of(cfg)),
        "model": {"path_or_id": profile.model.get("path_or_id"),
                  "revision": model_rev},
        "frozen_generation": dict(profile.generation),
        "reuse_from": cfg["reuse_from"],
        "prompts": {pid: {"role": p.get("role"), "sha256": p["_sha"],
                         "audio": p["audio"], "text": p.get("text"),
                         "split": p.get("split"), "source": p.get("source")}
                    for pid, p in prompts.items()},
        "texts": {tid: {"text": t.get("text"), "split": t.get("split"),
                        "source": t.get("source")}
                  for tid, t in texts.items()},
        "reference_sha256": ref.get("sha256"),
        "ground_truth_sha256": (gt or {}).get("sha256"),
        "cases": [],
        "pairs": [],
        "context_variance": _context_variance(),
        "subjective": "PENDING_USER_LISTENING",
        "asr": "pending",
        "reuse_revision_match": (
            (old_cases.get("__model__") or {}).get("revision")
            == model_rev),
    }

    own_manager = manager is None
    manager = manager or BackendManager(logs_dir=out_dir / "logs")
    new_generated = 0

    def _gen(text: str, prompt: dict, seed: int, out_path: Path,
             case: dict) -> None:
        nonlocal new_generated
        if new_generated >= NEW_WAV_BUDGET:
            case.update(status="blocked_budget",
                        error=f"new-wav budget {NEW_WAV_BUDGET} exhausted")
            return
        if not manager.acquire_generate():
            case.update(status="error",
                        error="concurrent generation in progress")
            return
        t0 = time.monotonic()
        try:
            client = manager.ensure(profile)
            result = client.generate(
                text=text, output_path=str(out_path),
                reference_audio=str(prompt["audio"]),
                reference_text=prompt.get("text"), seed=seed,
                options={}, timeout=GENERATE_TIMEOUT_S)
        finally:
            manager.release_generate()
        wall = time.monotonic() - t0
        meta = result.get("metadata") or {}
        case.update({
            "effective_args": meta.get("generation"),
            "retry_badcase_enabled": meta.get("retry_badcase_enabled"),
            "retry_count": meta.get("retry_count"),
            "wall_seconds": round(result.get("wall_seconds", wall), 2),
            "sample_rate": result.get("sample_rate"),
        })
        if out_path.is_file():
            case.update(status="ok",
                        output_sha256=sha256_file(out_path),
                        sample_rate=wav_info(out_path)["sample_rate"],
                        duration_s=round(wav_info(out_path)["duration"], 3),
                        sanity=codec_diag.audio_sanity(out_path))
            new_generated += 1
        else:
            case.update(status="error",
                        error="worker returned ok but output file missing")

    try:
        for group in cfg.get("matrix") or []:
            gid = group["pair_group"]
            tid = group["text"]
            spec = texts[tid]
            for seed in [int(s) for s in group["seeds"]]:
                pair_id = f"pair_{gid}_{tid}_seed{seed}"
                pair = {"pair_id": pair_id, "group": gid,
                        "text_id": tid, "seed": seed, "cells": []}
                for pid in ("P0", "P2"):
                    prompt = prompts[pid]
                    cid = f"{pair_id}/{pid}"
                    case: dict[str, Any] = {
                        "case_id": cid, "pair_id": pair_id,
                        "pair_group": gid, "text_id": tid,
                        "text": spec.get("text"), "prompt_id": pid,
                        "prompt_sha256": prompt["_sha"], "seed": seed,
                        "status": "pending", "reused_from": None,
                    }
                    want_reuse = (pid == "P0"
                                  and group.get("p0_source") == "reuse")
                    if want_reuse:
                        src_cid = f"{group['reuse_prefix']}_seed{seed}"
                        old = old_cases.get(src_cid)
                        wav, reason = _verify_reuse(
                            old, reuse_dir, prompt_sha=prompt["_sha"],
                            text=spec.get("text") or "", seed=seed,
                            model_revision=model_rev)
                        if wav is not None:
                            rel_old = Path(os.path.relpath(
                                wav, out_dir)).as_posix()
                            info = wav_info(wav)
                            case.update({
                                "status": "ok", "reused_from": src_cid,
                                "reuse_experiment": cfg["reuse_from"],
                                "output": rel_old,
                                "output_sha256": old["output_sha256"],
                                "sample_rate": info["sample_rate"],
                                "duration_s": round(info["duration"], 3),
                                "effective_args": old.get("effective_args"),
                                "retry_count": old.get("retry_count"),
                                "wall_seconds": old.get("wall_seconds"),
                                "sanity": codec_diag.audio_sanity(wav),
                            })
                            log(f"[phase5a2] {cid} reused+verified "
                                f"{src_cid}")
                        else:
                            case.update(status="blocked_reuse",
                                        error=reason)
                            log(f"[phase5a2] {cid} BLOCKED_REUSE: {reason}")
                    else:
                        out_path = out_dir / f"{pair_id}_{pid}.wav"
                        case["output"] = \
                            out_path.relative_to(out_dir).as_posix()
                        log(f"[phase5a2] {cid}")
                        try:
                            _gen(spec["text"], prompt, seed,
                                 out_path, case)
                        except Exception as exc:
                            case.update(
                                status="error",
                                error=f"{type(exc).__name__}: {exc}")
                            logger.exception("phase5a2 failed: %s", cid)
                    manifest["cases"].append(case)
                    pair["cells"].append(cid)
                manifest["pairs"].append(pair)
        try:
            client = manager._client
            if client is not None:
                manifest["backend_health"] = client.health(timeout=15)
        except Exception:
            pass
    finally:
        if own_manager:
            manager.stop()

    manifest["budget"] = {"zero_shot_generated": new_generated,
                          "zero_shot_limit": NEW_WAV_BUDGET}
    manifest["finished_at"] = _utc_now()

    metrics = collect_metrics(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from ..web.listen_page import write_phase5a2_listen_page
    write_phase5a2_listen_page(out_dir, manifest)
    _write_report_md(out_dir, manifest)
    return manifest


def repo_root_of(cfg: dict[str, Any]) -> Path:
    """Repo root, overridable in tests via cfg['root']."""
    from ..registry.loader import repo_root
    return Path(cfg["root"]) if cfg.get("root") else repo_root()


def _context_variance() -> list[dict[str, Any]]:
    """Same-sha Phase 5A samples rated differently in two contexts.

    Recorded verbatim — never averaged into a single score.
    """
    ratings_path = (repo_root() / "docs" / "reports" /
                    "suoming-voxcpm-phase5a-ratings.json")
    if not ratings_path.is_file():
        return []
    data = json.loads(ratings_path.read_text(encoding="utf-8"))
    by_sha: dict[str, list[dict[str, Any]]] = {}
    for s in data.get("samples") or []:
        sha = s.get("output_sha256")
        if sha:
            by_sha.setdefault(sha, []).append(s)
    out = []
    for sha, group in by_sha.items():
        if len(group) < 2:
            continue
        dims = ["clarity", "grit", "likeness", "naturalness"]
        divergent = {
            d for d in dims
            if len({g.get(d) for g in group}) > 1
        }
        out.append({
            "output_sha256": sha,
            "case_ids": [g["case_id"] for g in group],
            "ratings": [{d: g.get(d) for d in dims} for g in group],
            "divergent_dims": sorted(divergent),
            "note": "same wav rated in different listening contexts; "
                    "counted once as acoustic evidence",
        })
    return out


def attach_asr_results(manifest_path: str | Path,
                       asr_json_path: str | Path) -> dict[str, Any]:
    """Merge auxiliary ASR transcripts/CER into a phase5a2 manifest and
    regenerate report.md. ASR is auxiliary evidence only — the
    ``text_complete`` field stays user-owned."""
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
        "results_file": str(Path(asr_json_path).name),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report_md(manifest_path.parent, manifest)
    return manifest


def _write_report_md(out_dir: Path, manifest: dict[str, Any]) -> Path:
    lines = [
        f"# Phase 5A2 paired stability — {manifest['experiment_id']}",
        "",
        f"- generated: {manifest['created_at']}",
        f"- git: `{manifest['git_commit']}`",
        f"- backend `{manifest['backend']}` rev "
        f"`{manifest['model'].get('revision')}`",
        f"- frozen args: `{manifest['frozen_generation']}`",
        f"- subjective: **{manifest['subjective']}**",
        "",
        "## Pairs (P0 vs P2, same text+seed)",
        "",
        "| pair | text | seed | P0 cell | P2 cell |",
        "|---|---|---|---|---|",
    ]
    cases = {c["case_id"]: c for c in manifest["cases"]}
    for pair in manifest["pairs"]:
        cells = pair["cells"]
        p0 = cases.get(cells[0], {})
        p2 = cases.get(cells[1], {})
        def _cell(c: dict) -> str:
            if c.get("status") != "ok":
                return f"{c.get('status')}"
            tag = "reuse" if c.get("reused_from") else "new"
            return (f"{tag} `{str(c.get('output_sha256'))[:10]}` "
                    f"{c.get('duration_s')}s")
        lines.append(f"| {pair['pair_id']} | {pair['text_id']} "
                     f"| {pair['seed']} | {_cell(p0)} | {_cell(p2)} |")
    lines += ["", "## Cases", "",
              "| case | prompt | seed | status | retry | dur | sha |",
              "|---|---|---|---|---|---|---|"]
    for c in manifest["cases"]:
        lines.append(
            f"| {c['case_id']} | {c['prompt_id']} | {c['seed']} "
            f"| {c['status']} | "
            f"{c.get('retry_count') if c.get('retry_count') is not None else '-'} "
            f"| {c.get('duration_s') or '-'} "
            f"| `{str(c.get('output_sha256') or '')[:10]}` |")
    if manifest.get("context_variance"):
        lines += ["", "## Context variance (same wav, prior ratings)", ""]
        for cv in manifest["context_variance"]:
            lines.append(f"- `{cv['output_sha256'][:12]}`: "
                         f"{cv['case_ids']} ratings={cv['ratings']} "
                         f"diff={cv['divergent_dims']}")
    lines += ["", "## Pending", "",
              "- Subjective fields: PENDING_USER_LISTENING",
              "- Training decision: PENDING_USER_DECISION", ""]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
