"""Markdown gate report writer (plan section 12/21 tables)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PENDING = "PENDING_USER_LISTENING"

_BACKENDS = ("dots_legacy", "voxcpm2", "qwen3_tts")


def _yes_no(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "?"


def write_report_md(output_dir: str | Path, report: dict[str, Any],
                    metrics: dict[str, Any]) -> Path:
    output_dir = Path(output_dir)
    lines: list[str] = []
    lines.append(f"# Gate report — {report.get('evaluation_id')}")
    lines.append("")
    lines.append(f"- character: `{report.get('character')}`")
    lines.append(f"- generated_at: {report.get('generated_at')}")
    lines.append(f"- seed: {report.get('seed')}")
    ref = report.get("reference") or {}
    lines.append(f"- reference sha256: `{ref.get('sha256')}`")
    lines.append(f"- anchor text: {report.get('anchor_text')}")
    lines.append("")

    # -- per-case results ---------------------------------------------------
    lines.append("## Case results")
    lines.append("")
    lines.append("| Backend | Case | Output | Status | Wall (s) | Notes |")
    lines.append("|---|---|---|---|---|---|")
    case_status: dict[tuple[str, str], str] = {}
    for case in report.get("cases", []):
        status = case.get("status", "?")
        case_status[(case.get("backend"), case.get("kind"))] = status
        note = case.get("error") or case.get("reason") or ""
        wall = case.get("wall_seconds")
        wall_s = f"{wall:.1f}" if isinstance(wall, (int, float)) else ""
        lines.append(
            f"| {case.get('backend')} | {case.get('kind')} | "
            f"`{case.get('output')}` | {status} | {wall_s} | {note} |"
        )
    lines.append("")

    # -- objective metrics ----------------------------------------------------
    lines.append("## Objective metrics")
    lines.append("")
    lines.append(
        "| Sample | dur (s) | RMS dBFS | LUFS | TP dBTP | E4-8k% | E8-12k% | "
        "E12-18k% | flatness | crest | entropy | Δ2-9k |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for rel, m in metrics.items():
        if m.get("error"):
            lines.append(f"| `{rel}` | error: {m['error']} | | | | | | | | | |")
            continue
        def f(key, nd=2):
            v = m.get(key)
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ""
        lines.append(
            f"| `{rel}` | {f('duration')} | {f('rms_dbfs')} | {f('lufs', 1)} | "
            f"{f('true_peak_dbtp', 1)} | {f('band_energy_4k_8k', 3)} | "
            f"{f('band_energy_8k_12k', 3)} | {f('band_energy_12k_18k', 3)} | "
            f"{f('spectral_flatness', 3)} | {f('spectral_crest', 1)} | "
            f"{f('spectral_entropy', 3)} | {f('temporal_delta_2k_9k', 3)} |"
        )
    lines.append("")
    lines.append(
        "> Objective metrics are descriptive only. No single metric decides "
        "quality — see plan section 0.5. Subjective scores remain "
        f"`{PENDING}` until the user listens."
    )
    lines.append("")

    # -- env status table -------------------------------------------------------
    lines.append("### A. 环境状态")
    lines.append("")
    lines.append("| Backend | Env | Model download | Smoke | Peak VRAM |")
    lines.append("|---|---|---|---|---|")
    health = report.get("backend_health") or {}
    seen = set()
    ordered = [b for b in _BACKENDS if b in
               {c.get("backend") for c in report.get("cases", [])}]
    ordered += [b for b in _BACKENDS if b not in ordered]
    for backend in ordered:
        h = health.get(backend) or {}
        vram = h.get("vram") or {}
        peak = vram.get("peak_allocated")
        peak_s = f"{peak / 2**30:.1f} GiB" if isinstance(peak, (int, float)) else "?"
        env_ok = backend in health
        model_ok = bool(h.get("model_id"))
        smoke = any(
            s == "ok" for (b, k), s in case_status.items() if b == backend
        )
        lines.append(
            f"| {backend} | {_yes_no(env_ok)} | {_yes_no(model_ok or None)} | "
            f"{_yes_no(smoke if env_ok else None)} | {peak_s} |"
        )
        seen.add(backend)
    lines.append("")

    # -- gate table ---------------------------------------------------------------
    lines.append("### B. Gate 状态")
    lines.append("")
    lines.append(
        "| Backend | Codec gate | Zero-shot gate | 主观评分 | 是否建议训练 |")
    lines.append("|---|---|---|---|---|")
    for backend in ordered:
        codec = case_status.get((backend, "codec_roundtrip"), "not_run")
        zero = case_status.get((backend, "zero_shot"), "not_run")
        lines.append(
            f"| {backend} | {codec} | {zero} | {PENDING} | {PENDING} |"
        )
    lines.append("")

    out = output_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def finalize(output_dir: str | Path, title: str, meta: str) -> dict[str, Path]:
    """Write report.md + listen page from report.json/metrics.json."""
    from ..web.listen_page import write_listen_page

    output_dir = Path(output_dir)
    report = json.loads(
        (output_dir / "report.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "report_md": write_report_md(output_dir, report, metrics),
        "listen_page": write_listen_page(output_dir, title, meta),
    }
