"""Phase 5A experiment runner — VoxCPM2 stability + codec attribution +
prompt A/B (PLAN.md section 23).

Reuses the existing BackendManager / worker client / audio io. Each case
lands in a machine-readable manifest with prompt hash, seed, effective
generation args, output hash, wall time and retry telemetry — nothing is
inferred about subjective quality; human listening stays authoritative.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ..audio.io import read_wav, safe_output_path, sha256_file, wav_info
from ..backends.manager import BackendManager
from ..registry.loader import load_backend, load_env_file, repo_root
from ..registry.loader import expand_env
from ..registry.models import CharacterProfile
from . import codec_diag
from .gate import collect_metrics, verify_ground_truth, verify_reference

logger = logging.getLogger(__name__)

GENERATE_TIMEOUT_S = 1800
PROBE_TIMEOUT_S = 900


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_phase5a_config(path_or_id: str,
                        env: dict[str, str] | None = None
                        ) -> dict[str, Any]:
    env = env or {**load_env_file(repo_root() / ".env.local"),
                  **os.environ}
    path = Path(path_or_id)
    if not path.suffix:
        path = repo_root() / "configs" / "evaluations" / f"{path_or_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"phase5a config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    return expand_env(data, env)


def _git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), check=True,
            capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _verify_sha(path: str | Path, expected: str | None, label: str) -> str:
    actual = sha256_file(path)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def _wav_stats(path: Path) -> dict[str, Any]:
    info = wav_info(path)
    return {"sample_rate": info["sample_rate"],
            "duration_s": round(info["duration"], 3)}


def _new_case(case_id: str, group: str, **fields: Any) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "group": group,
        "status": "pending",
        "retry_count": None,
        "reused_from": None,
        **fields,
    }


def run_phase5a(cfg: dict[str, Any], character: CharacterProfile,
                manager: BackendManager | None = None,
                log=print) -> dict[str, Any]:
    """Execute the bounded Phase 5A experiment; returns the manifest."""
    out_dir = safe_output_path(repo_root(), cfg["output_dir"])
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
    codec_inputs = cfg.get("codec_diagnosis", {}).get("inputs") or []
    for ci in codec_inputs:
        ci["_sha"] = _verify_sha(ci["audio"], ci.get("sha256"),
                                 f"codec input {ci['id']}")

    # Originals for the listening package
    ref_dest = out_dir / "reference_original.wav"
    if not ref_dest.exists() or sha256_file(ref_dest) != ref.get("sha256"):
        shutil.copy2(ref["audio"], ref_dest)
    gt_dest: Path | None = None
    if gt is not None:
        gt_dest = out_dir / "ground_truth_original.wav"
        if not gt_dest.exists() or sha256_file(gt_dest) != gt.get("sha256"):
            shutil.copy2(gt["audio"], gt_dest)

    own_manager = manager is None
    manager = manager or BackendManager(logs_dir=out_dir / "logs")
    manifest: dict[str, Any] = {
        "experiment_id": cfg.get("experiment_id", "phase5a"),
        "character": character.character_id,
        "backend": backend_id,
        "created_at": _utc_now(),
        "git_commit": _git_commit(repo_root()),
        "model": {
            "path_or_id": profile.model.get("path_or_id"),
            "revision": profile.model.get("revision"),
        },
        "frozen_generation": dict(profile.generation),
        "retry_badcase_enabled": bool(
            (profile.generation or {}).get("retry_badcase", True)),
        "prompts": {
            pid: {"role": p.get("role"), "sha256": p["_sha"],
                  "audio": p["audio"], "text": p.get("text"),
                  "split": p.get("split"), "source": p.get("source")}
            for pid, p in prompts.items()
        },
        "texts": {
            tid: {"text": t.get("text"), "split": t.get("split"),
                  "source": t.get("source"),
                  "has_ground_truth": bool(t.get("has_ground_truth"))}
            for tid, t in texts.items()
        },
        "cases": [],
        "codec_diagnosis": {},
        "sanity": {},
        "subjective": "PENDING_USER_LISTENING",
    }

    cases: list[dict[str, Any]] = manifest["cases"]
    zs_generated = 0

    def _gen(text: str, prompt: dict, seed: int, out_path: Path,
             case: dict) -> None:
        """One zero-shot generation; fills the case record."""
        if not manager.acquire_generate():
            case["status"] = "error"
            case["error"] = "concurrent generation in progress"
            return
        t0 = time.monotonic()
        try:
            client = manager.ensure(profile)
            result = client.generate(
                text=text,
                output_path=str(out_path),
                reference_audio=str(prompt["audio"]),
                reference_text=prompt.get("text"),
                seed=seed,
                options={},
                timeout=GENERATE_TIMEOUT_S,
            )
        finally:
            manager.release_generate()
        wall = time.monotonic() - t0
        meta = (result.get("metadata") or {})
        case.update({
            "effective_args": meta.get("generation"),
            "retry_badcase_enabled": meta.get("retry_badcase_enabled"),
            "retry_count": meta.get("retry_count"),
            "wall_seconds": round(result.get("wall_seconds", wall), 2),
            "sample_rate": result.get("sample_rate"),
        })
        if out_path.is_file():
            case["status"] = "ok"
            case["output_sha256"] = sha256_file(out_path)
            case.update(_wav_stats(out_path))
        else:
            case["status"] = "error"
            case["error"] = "worker returned ok but output file missing"

    try:
        client = None
        # -- Task B: stability set -------------------------------------
        stab = cfg.get("stability") or {}
        seeds = [int(s) for s in stab.get("seeds") or [42]]
        p0 = prompts[cfg.get("prompt_ab", {}).get("baseline_prompt", "P0")]
        for tid in stab.get("texts") or []:
            spec = texts[tid]
            for seed in seeds:
                cid = f"stability/{tid}_seed{seed}"
                out_path = out_dir / f"{cid}.wav"
                case = _new_case(
                    cid, "stability", text_id=tid,
                    text=spec.get("text"), prompt_id=p0["id"],
                    prompt_sha256=p0["_sha"], seed=seed,
                    output=out_path.relative_to(out_dir).as_posix())
                log(f"[phase5a] {cid}")
                try:
                    _gen(spec["text"], p0, seed, out_path, case)
                    if case["status"] == "ok":
                        zs_generated += 1
                except Exception as exc:
                    case["status"] = "error"
                    case["error"] = f"{type(exc).__name__}: {exc}"
                    logger.exception("phase5a case failed: %s", cid)
                cases.append(case)
                case["sanity"] = codec_diag.audio_sanity(out_path) \
                    if out_path.is_file() else None

        # -- Task D: prompt A/B ----------------------------------------
        ab = cfg.get("prompt_ab") or {}
        ab_seed = int(ab.get("seed", 42))
        ab_texts = ab.get("texts") or []
        for p in ab.get("prompts") or []:
            prompt = prompts[p["id"]] if isinstance(p, dict) else prompts[p]
            pid = prompt["id"]
            for tid in ab_texts:
                spec = texts[tid]
                cid = f"prompt_ab/{pid}_{tid}_seed{ab_seed}"
                stab_cid = f"stability/{tid}_seed{ab_seed}"
                stab_case = next(
                    (c for c in cases if c["case_id"] == stab_cid), None)
                case = _new_case(
                    cid, "prompt_ab", text_id=tid,
                    text=spec.get("text"), prompt_id=pid,
                    prompt_sha256=prompt["_sha"], seed=ab_seed,
                    output=None)
                if (pid == p0["id"] and stab_case
                        and stab_case["status"] == "ok"):
                    # Identical (prompt, text, seed, args) -> reuse the
                    # stability wav; verified by its real output hash.
                    case.update({
                        "status": "ok",
                        "reused_from": stab_cid,
                        "output": stab_case["output"],
                        "output_sha256": stab_case["output_sha256"],
                        "sample_rate": stab_case["sample_rate"],
                        "duration_s": stab_case["duration_s"],
                        "effective_args": stab_case.get("effective_args"),
                        "wall_seconds": stab_case.get("wall_seconds"),
                        "retry_count": stab_case.get("retry_count"),
                        "retry_badcase_enabled":
                            stab_case.get("retry_badcase_enabled"),
                        "sanity": stab_case.get("sanity"),
                    })
                    log(f"[phase5a] {cid} (reused {stab_cid})")
                else:
                    out_path = out_dir / f"{cid}.wav"
                    case["output"] = \
                        out_path.relative_to(out_dir).as_posix()
                    log(f"[phase5a] {cid}")
                    try:
                        _gen(spec["text"], prompt, ab_seed, out_path, case)
                        if case["status"] == "ok":
                            zs_generated += 1
                    except Exception as exc:
                        case["status"] = "error"
                        case["error"] = f"{type(exc).__name__}: {exc}"
                        logger.exception("phase5a case failed: %s", cid)
                    if out_path.is_file():
                        case["sanity"] = codec_diag.audio_sanity(out_path)
                cases.append(case)

        # -- Task C: codec probe + host analysis ------------------------
        diag_cfg = cfg.get("codec_diagnosis") or {}
        variants = diag_cfg.get("variants") or ["default", "cond16000"]
        if not manager.acquire_generate():
            raise RuntimeError("concurrent generation in progress")
        try:
            client = manager.ensure(profile)
        finally:
            manager.release_generate()

        for ci in codec_inputs:
            iid = ci["id"]
            log(f"[phase5a] codec_probe {iid}")
            entry: dict[str, Any] = {
                "input_id": iid, "audio": ci["audio"],
                "input_sha256": ci["_sha"], "split": ci.get("split"),
                "variants": variants,
            }
            if not manager.acquire_generate():
                entry["status"] = "error"
                entry["error"] = "concurrent generation in progress"
                manifest["codec_diagnosis"][iid] = entry
                continue
            try:
                result = client.codec_probe(
                    audio_path=str(ci["audio"]),
                    output_dir=str(out_dir / "codec"),
                    stem=iid, variants=list(variants),
                    timeout=PROBE_TIMEOUT_S)
            except Exception as exc:
                entry["status"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("codec probe failed: %s", iid)
                manifest["codec_diagnosis"][iid] = entry
                continue
            finally:
                manager.release_generate()
            entry["probe_result"] = {
                k: v for k, v in result.items() if k != "artifacts"}
            entry["artifacts"] = result.get("artifacts")
            entry["status"] = result.get("status")
            try:
                entry["analysis"] = _analyze_codec_chain(
                    ci, result, out_dir)
            except Exception as exc:
                entry["analysis_error"] = f"{type(exc).__name__}: {exc}"
                logger.exception("codec analysis failed: %s", iid)
            manifest["codec_diagnosis"][iid] = entry

        try:
            manifest["backend_health"] = client.health(timeout=15)
        except Exception:
            pass
    finally:
        if own_manager:
            manager.stop()

    manifest["budget"] = {"zero_shot_generated": zs_generated,
                          "zero_shot_limit": 18}
    manifest["finished_at"] = _utc_now()

    metrics = collect_metrics(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    from ..web.listen_page import write_phase5a_listen_page
    write_phase5a_listen_page(out_dir, manifest)
    _write_report_md(out_dir, manifest)
    return manifest


def _analyze_codec_chain(ci: dict[str, Any], result: dict[str, Any],
                         out_dir: Path) -> dict[str, Any]:
    """Three-chain comparison: original vs encoder input vs roundtrip.

    Everything is compared on the shared 0-8 kHz band (analysis at
    16 kHz) after time+gain alignment; the >8 kHz content of the
    original is reported separately since the 16 kHz encode physically
    cannot carry it.
    """
    src_x, src_sr = read_wav(ci["audio"])
    analysis: dict[str, Any] = {"source_sr": src_sr, "pairs": {}}

    enc = result.get("encode_input") or {}
    enc_path = enc.get("path")
    input16 = None
    if enc_path and Path(enc_path).is_file():
        input16, in_sr = read_wav(enc_path)
        # Bandwidth-loss-only reference: original lowpassed to the
        # encoder's real input (torchaudio resample in-worker) vs our
        # polyphase resample — checks the capture path is sane.
        src16 = codec_diag.resample_to(src_x, src_sr, in_sr)
        analysis["pairs"]["original_vs_encode_input"] = \
            codec_diag.compare_pair(src16, in_sr, input16, in_sr,
                                    analysis_sr=in_sr,
                                    bands=[(0, 4000), (4000, 7500)])

    artifacts = result.get("artifacts") or {}
    rt = artifacts.get("default") or {}
    rt_path = rt.get("path")
    if rt_path and Path(rt_path).is_file():
        rt_x, rt_sr = read_wav(rt_path)
        analysis["pairs"]["original_vs_roundtrip"] = codec_diag.compare_pair(
            src_x, src_sr, rt_x, rt_sr,
            bands=[(0, 4000), (4000, 7500)])
        if input16 is not None:
            # Codec-added distortion inside the representable band —
            # pure downsample loss is removed by comparing to input16k.
            analysis["pairs"]["encode_input_vs_roundtrip"] = \
                codec_diag.compare_pair(input16, in_sr, rt_x, rt_sr,
                                        analysis_sr=in_sr,
                                        bands=[(0, 4000), (4000, 7500)])
        # Descriptive native-band view (>=8 kHz cannot come from the
        # encode; whatever is there is decoder resynthesis).
        analysis["native_hf"] = {
            "original_energy_8k_16k": round(
                _band_energy(src_x, src_sr, 8000, min(16000, src_sr // 2)), 5),
            "roundtrip_energy_8k_16k": round(
                _band_energy(rt_x, rt_sr, 8000, min(16000, rt_sr // 2)), 5),
            "roundtrip_energy_16k_plus": round(
                _band_energy(rt_x, rt_sr, 16000, rt_sr // 2), 5)
            if rt_sr // 2 > 16000 else None,
        }
        codec_diag.spectrogram_png(
            src_x, src_sr, out_dir / "codec" /
            f"{ci['id']}_spec_original.png", fmax=16000)
        codec_diag.spectrogram_png(
            rt_x, rt_sr, out_dir / "codec" /
            f"{ci['id']}_spec_roundtrip.png", fmax=16000)

    cond = artifacts.get("cond16000") or {}
    cond_path = cond.get("path")
    if cond_path and Path(cond_path).is_file() and rt_path:
        c_x, c_sr = read_wav(cond_path)
        rt_x2, rt_sr2 = read_wav(rt_path)
        analysis["pairs"]["roundtrip_cond16k_vs_cond48k"] = \
            codec_diag.compare_pair(c_x, c_sr, rt_x2, rt_sr2,
                                    analysis_sr=16000,
                                    bands=[(0, 4000), (4000, 7500)])
        codec_diag.spectrogram_png(
            c_x, c_sr, out_dir / "codec" /
            f"{ci['id']}_spec_roundtrip_cond16k.png", fmax=8000)
    if input16 is not None:
        codec_diag.spectrogram_png(
            input16, in_sr, out_dir / "codec" /
            f"{ci['id']}_spec_input16k.png", fmax=8000)
    return analysis


def _band_energy(x, sr: int, lo: float, hi: float) -> float:
    from ..diagnostics.metrics import band_energy_ratio
    return band_energy_ratio(x, sr, lo, hi)


def _write_report_md(out_dir: Path, manifest: dict[str, Any]) -> Path:
    lines = [
        f"# Phase 5A experiment — {manifest['experiment_id']}",
        "",
        f"- generated: {manifest['created_at']}",
        f"- git: `{manifest['git_commit']}`",
        f"- backend: `{manifest['backend']}` "
        f"model `{manifest['model'].get('path_or_id')}` "
        f"rev `{manifest['model'].get('revision')}`",
        f"- frozen generation args: `{manifest['frozen_generation']}`",
        f"- retry_badcase enabled: {manifest['retry_badcase_enabled']}",
        f"- subjective ratings: **{manifest['subjective']}**",
        "",
        "## Cases",
        "",
        "| case | text | prompt | seed | sr | dur(s) | wall(s) | "
        "retries | status | sha256 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in manifest["cases"]:
        sha = (c.get("output_sha256") or "")[:12]
        lines.append(
            f"| {c['case_id']} | {c.get('text_id')} | {c.get('prompt_id')} "
            f"| {c.get('seed')} | {c.get('sample_rate') or '-'} "
            f"| {c.get('duration_s') or '-'} | {c.get('wall_seconds') or '-'} "
            f"| {c.get('retry_count') if c.get('retry_count') is not None else '-'} "
            f"| {c['status']} | `{sha}` |")
    lines += ["", "## Codec diagnosis", ""]
    for iid, e in (manifest.get("codec_diagnosis") or {}).items():
        lines.append(f"### {iid}")
        lines.append(f"- status: {e.get('status')}")
        pr = e.get("probe_result") or {}
        lines.append(f"- encode_sr={pr.get('encode_sample_rate')} "
                     f"decode_sr={pr.get('decode_sample_rate')} "
                     f"vae_dtype={pr.get('vae_dtype')} "
                     f"latent_shape={pr.get('latent_shape')} "
                     f"latent_rate={pr.get('latent_rate_hz')}Hz")
        for pair, res in (e.get("analysis") or {}).get("pairs", {}).items():
            if "error" in res:
                lines.append(f"- {pair}: ERROR {res['error']}")
                continue
            bands = (res.get("spectral") or {}).get("bands") or {}
            btxt = ", ".join(
                f"{b}: Δ{bv['mean_diff_db']}dB rmse={bv['rmse_db']}dB "
                f"corr={bv['logmag_corr']}"
                for b, bv in bands.items() if bv)
            lines.append(
                f"- {pair}: delay={res.get('delay_ms')}ms "
                f"gain={res.get('gain_test_to_ref')} "
                f"wavecorr={res.get('waveform_corr')} "
                f"nmse={res.get('waveform_nmse')} | {btxt}")
        hf = (e.get("analysis") or {}).get("native_hf") or {}
        if hf:
            lines.append(f"- native HF: {hf}")
        lines.append("")
    lines += [
        "## Pending",
        "",
        "- All subjective fields: PENDING_USER_LISTENING",
        "- Training decision: PENDING_USER_DECISION",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
