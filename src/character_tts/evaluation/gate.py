"""Gate evaluation runner.

Executes an EvaluationConfig against backend workers: copies the verified
reference audio, runs codec_roundtrip / zero_shot / generate cases, writes
per-case sidecar provenance JSON, then aggregates objective metrics.

Failures are recorded per-case (status=error/blocked) and never abort the
remaining backends. No subjective judgement is made here — listening is
deferred to the HTML page and the human report fields.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..audio.io import read_wav, safe_output_path, sha256_file, wav_info
from ..backends.manager import BackendManager
from ..diagnostics.metrics import analyze_audio
from ..registry.loader import load_backend
from ..registry.models import (
    CharacterProfile,
    EvaluationConfig,
    GateCase,
)

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def verify_reference(character: CharacterProfile) -> dict[str, Any]:
    """Check reference audio exists and its sha256 matches the profile."""
    audio = character.reference_audio
    if not audio or not Path(audio).is_file():
        raise FileNotFoundError(
            f"character '{character.character_id}' reference audio missing: {audio}"
        )
    result: dict[str, Any] = {"audio": audio}
    expected = character.reference_sha256
    if expected:
        actual = sha256_file(audio)
        result["sha256"] = actual
        if actual.lower() != expected.lower():
            raise ValueError(
                f"reference audio sha256 mismatch: expected {expected}, got {actual}"
            )
    return result


def verify_ground_truth(character: CharacterProfile) -> dict[str, Any] | None:
    """Check the anchor's real utterance exists and matches its sha256.

    Returns None when the character has no ground truth — codec_roundtrip
    cases then report ``blocked`` instead of silently reusing the clone
    prompt as reconstruction input.
    """
    gt = character.ground_truth
    if gt is None:
        return None
    audio = gt.get("audio")
    if not audio or not Path(audio).is_file():
        raise FileNotFoundError(
            f"character '{character.character_id}' ground truth audio "
            f"missing: {audio}"
        )
    result: dict[str, Any] = {"audio": audio, "text": gt.get("text"),
                              "anchor_id": gt.get("anchor_id"),
                              "source": gt.get("source")}
    expected = gt.get("sha256")
    if expected:
        actual = sha256_file(audio)
        result["sha256"] = actual
        if actual.lower() != expected.lower():
            raise ValueError(
                f"ground truth sha256 mismatch: expected {expected}, "
                f"got {actual}"
            )
    return result


def _anchor_text(character: CharacterProfile) -> str:
    anchors = character.anchor_texts
    if not anchors:
        raise ValueError(
            f"character '{character.character_id}' has no evaluation.anchor_texts"
        )
    return anchors[0].text


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".json")


def _write_sidecar(output_path: Path, payload: dict[str, Any]) -> None:
    _sidecar_path(output_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_case(manager: BackendManager, case: GateCase, *,
             character: CharacterProfile, evaluation: EvaluationConfig,
             reference_wav: Path,
             ground_truth_wav: Path | None = None) -> dict[str, Any]:
    """Run one gate case. Always returns a record dict; never raises."""
    record: dict[str, Any] = {
        "backend": case.backend,
        "kind": case.kind,
        "output": case.output,
        "status": "pending",
        "started_at": _utc_now(),
    }
    if case.kind == "codec_roundtrip" and ground_truth_wav is None:
        record["status"] = "blocked"
        record["reason"] = ("character has no ground truth utterance — "
                            "codec_roundtrip refuses to use the clone prompt")
        record["finished_at"] = _utc_now()
        return record
    out_path = safe_output_path(evaluation.output_dir, case.output)
    try:
        profile = load_backend(case.backend)
        if not profile.enabled:
            record["status"] = "blocked"
            record["reason"] = "backend disabled in config"
            return record
        if not manager.acquire_generate():
            record["status"] = "error"
            record["error"] = "concurrent generation in progress"
            return record
        try:
            # ensure() may switch workers — that must happen *inside* the
            # generate gate so external switch/stop calls stay blocked.
            client = manager.ensure(profile)
            seed = int(case.options.get("seed", evaluation.seed))
            if case.kind == "codec_roundtrip":
                t0 = time.monotonic()
                result = client.codec_roundtrip(
                    audio_path=str(ground_truth_wav),
                    output_path=str(out_path),
                    timeout=600,
                )
                wall = time.monotonic() - t0
            else:
                gen_options = dict(case.options)
                if case.kind == "zero_shot":
                    gen_options.setdefault("variant", "base")
                t0 = time.monotonic()
                result = client.generate(
                    text=_anchor_text(character),
                    output_path=str(out_path),
                    reference_audio=str(reference_wav),
                    reference_text=character.reference_text,
                    seed=seed,
                    options=gen_options,
                    timeout=1800,
                )
                wall = time.monotonic() - t0
        finally:
            manager.release_generate()

        record["result"] = result
        record["wall_seconds"] = result.get("wall_seconds", wall)
        if case.kind == "codec_roundtrip" and result.get("status") == "unsupported":
            record["status"] = "unsupported"
            record["reason"] = result.get("reason", "")
        else:
            record["status"] = "ok" if out_path.is_file() else "error"
            if record["status"] == "error":
                record["error"] = "worker returned ok but output file missing"

        if out_path.is_file():
            sidecar = {
                "character_id": character.character_id,
                "backend_id": case.backend,
                "kind": case.kind,
                "model_id": (manager.profile.model.get("path_or_id")
                             if manager.profile else None),
                "model_revision": (manager.profile.model.get("revision")
                                   if manager.profile else None),
                "adapter": (manager.profile.model.get("adapter")
                            if manager.profile else None),
                "reference_sha256": sha256_file(reference_wav),
                "ground_truth_sha256": (sha256_file(ground_truth_wav)
                                      if ground_truth_wav is not None
                                      else None),
                "text": _anchor_text(character) if case.kind != "codec_roundtrip"
                        else None,
                "seed": case.options.get("seed", evaluation.seed),
                "options": case.options,
                "worker_result": result,
                "output_sha256": sha256_file(out_path),
                "generated_at": _utc_now(),
            }
            _write_sidecar(out_path, sidecar)
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        logger.exception("gate case failed: %s %s", case.backend, case.kind)
    finally:
        record["finished_at"] = _utc_now()
    return record


def collect_metrics(output_dir: Path) -> dict[str, dict[str, Any]]:
    """Analyze every wav under output_dir; returns rel-path -> metrics."""
    metrics: dict[str, dict[str, Any]] = {}
    for wav in sorted(Path(output_dir).rglob("*.wav")):
        rel = wav.relative_to(output_dir).as_posix()
        try:
            x, sr = read_wav(wav)
            entry = analyze_audio(x, sr)
            entry["sha256"] = sha256_file(wav)
            entry.update({f"file_{k}": v for k, v in wav_info(wav).items()})
            metrics[rel] = entry
        except Exception as exc:
            metrics[rel] = {"error": f"{type(exc).__name__}: {exc}"}
    return metrics


def run_gate(evaluation: EvaluationConfig, character: CharacterProfile,
             manager: BackendManager | None = None,
             backends: list[str] | None = None) -> dict[str, Any]:
    """Execute the full gate; returns the report dict (also persisted)."""
    out_dir = Path(evaluation.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = verify_reference(character)
    ref_dest = out_dir / "reference_original.wav"
    if not ref_dest.exists() or \
            sha256_file(ref_dest) != ref.get("sha256"):
        shutil.copy2(ref["audio"], ref_dest)

    # Ground truth (the anchor's real utterance) is the codec-roundtrip
    # input and the A/B original — never the clone prompt.
    gt = verify_ground_truth(character)
    gt_dest: Path | None = None
    if gt is not None:
        gt_dest = out_dir / "ground_truth_original.wav"
        if not gt_dest.exists() or \
                sha256_file(gt_dest) != gt.get("sha256"):
            shutil.copy2(gt["audio"], gt_dest)

    own_manager = manager is None
    manager = manager or BackendManager(logs_dir=out_dir / "logs")
    cases = evaluation.cases
    if backends:
        cases = [c for c in cases if c.backend in backends]

    records = []
    backend_health: dict[str, Any] = {}
    try:
        for case in cases:
            logger.info("gate case: %s %s -> %s",
                        case.backend, case.kind, case.output)
            record = run_case(
                manager, case,
                character=character,
                evaluation=evaluation,
                reference_wav=ref_dest,
                ground_truth_wav=gt_dest,
            )
            records.append(record)
            logger.info("  -> %s (%s)", record["status"],
                        record.get("error") or record.get("reason") or "")
            # Re-probe after every case so the recorded health reflects the
            # lazily-loaded model's real VRAM, not the pre-load snapshot.
            if manager.is_alive() \
                    and manager.active_backend_id == case.backend:
                try:
                    backend_health[case.backend] = manager._client.health(timeout=15)
                except Exception:
                    pass
    finally:
        if own_manager:
            manager.stop()

    metrics = collect_metrics(out_dir)

    # Merge with any existing report so incremental --backend runs do not
    # clobber results from earlier runs. Same (backend,kind,output) keys
    # are replaced by the fresh record.
    prev_cases: list[dict[str, Any]] = []
    prev_health: dict[str, Any] = {}
    report_path = out_dir / "report.json"
    if report_path.is_file():
        try:
            prev = json.loads(report_path.read_text(encoding="utf-8"))
            prev_cases = prev.get("cases") or []
            prev_health = prev.get("backend_health") or {}
        except (json.JSONDecodeError, OSError):
            pass
    new_keys = {(r["backend"], r["kind"], r["output"]) for r in records}
    merged_cases = [c for c in prev_cases
                    if (c.get("backend"), c.get("kind"), c.get("output"))
                    not in new_keys]
    merged_cases.extend(records)
    merged_health = {**prev_health, **backend_health}

    report = {
        "evaluation_id": evaluation.evaluation_id,
        "character": character.character_id,
        "generated_at": _utc_now(),
        "seed": evaluation.seed,
        "anchor_text": _anchor_text(character),
        "reference": {
            "audio": ref["audio"],
            "sha256": ref.get("sha256"),
            "text": character.reference_text,
        },
        "ground_truth": ({
            "audio": gt["audio"],
            "sha256": gt.get("sha256"),
            "text": gt.get("text"),
            "anchor_id": gt.get("anchor_id"),
        } if gt else None),
        "backend_health": merged_health,
        "cases": [r if not isinstance(r, GateCase) else asdict(r)
                  for r in merged_cases],
        "metrics_file": "metrics.json",
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
