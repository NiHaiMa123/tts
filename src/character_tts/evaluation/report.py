"""Markdown gate report writer (plan section 12/21 tables)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PENDING = "PENDING_USER_LISTENING"
PENDING_DECISION = "PENDING_USER_DECISION"

_BACKENDS = ("dots_legacy", "voxcpm2", "qwen3_tts")
_RATING_DIMS = ("clarity", "grit", "likeness", "naturalness")


def _yes_no(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "?"


def _load_ratings(output_dir: Path) -> dict[str, dict[str, str]]:
    """User-exported blind-listening ratings, if present."""
    for cand in (output_dir / "listen" / "listen-ratings.json",
                 output_dir / "listen-ratings.json"):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            ratings = data.get("ratings")
            return ratings if isinstance(ratings, dict) else {}
    return {}


def _score_cell(ratings: dict[str, dict[str, str]], rel: str) -> str:
    """Compact c/g/l/n summary for one sample; '-' when unrated."""
    r = ratings.get(rel) or {}
    vals = [r.get(d) for d in _RATING_DIMS]
    if not any(vals):
        return "-"
    return "c/g/l/n " + "/".join(v or "-" for v in vals)


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
    gt = report.get("ground_truth") or {}
    lines.append(f"- reference sha256: `{ref.get('sha256')}`")
    if gt:
        lines.append(f"- ground truth sha256: `{gt.get('sha256')}`")
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
        "| Sample | SR | dur (s) | RMS dBFS | LUFS | TP dBTP | E4-8k% | "
        "E8-12k% | flatness | crest | entropy | Δ2-9k |"
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
            f"| `{rel}` | {f('sample_rate', 0)} | {f('duration')} | "
            f"{f('rms_dbfs')} | {f('lufs', 1)} | "
            f"{f('true_peak_dbtp', 1)} | {f('band_energy_4k_8k', 3)} | "
            f"{f('band_energy_8k_12k', 3)} | "
            f"{f('spectral_flatness', 3)} | {f('spectral_crest', 1)} | "
            f"{f('spectral_entropy', 3)} | {f('temporal_delta_2k_9k', 3)} |"
        )
    lines.append("")
    lines.append(
        "> Spectral metrics are computed at a shared analysis rate "
        "(analysis_sr ≤ 24 kHz) — bands above the common Nyquist are "
        "excluded because 24 kHz and 48 kHz outputs are not comparable "
        "there. Per-file native-band energy >12 kHz is in metrics.json "
        "as descriptive context only."
    )
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

    # -- subjective listening -----------------------------------------------------
    ratings = _load_ratings(output_dir)
    if ratings:
        lines.append("### 主观试听（用户评分，1–5，越低越差）")
        lines.append("")
        lines.append(
            "| Sample | 清澈度 | 磨砂/颗粒 | 像角色 | 自然度 | "
            "稳定性 | 备注 |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for rel in sorted(ratings, key=lambda k: ("/" in k, k)):
            r = ratings[rel] or {}

            def cell(dim: str) -> str:
                v = r.get(dim)
                return str(v) if v not in (None, "") else "-"
            lines.append(
                f"| `{rel}` | {cell('clarity')} | {cell('grit')} | "
                f"{cell('likeness')} | {cell('naturalness')} | "
                f"{cell('stability')} | {cell('notes')} |"
            )
        lines.append("")
        lines.append(
            "> 评分来自 `listen/listen-ratings.json`（用户盲听导出）。"
            "平台如实展示，不自动宣布胜负。"
        )
        lines.append("")

    # -- gate table ---------------------------------------------------------------
    lines.append("### B. Gate 状态")
    lines.append("")
    lines.append(
        "| Backend | Codec gate | Zero-shot gate | 主观评分(用户) | 是否建议训练 |")
    lines.append("|---|---|---|---|---|")
    case_outputs = {(c.get("backend"), c.get("kind")): c.get("output")
                    for c in report.get("cases", [])}
    for backend in ordered:
        codec = case_status.get((backend, "codec_roundtrip"), "not_run")
        zero = case_status.get((backend, "zero_shot"), "not_run")
        # Show the user's scores for the two gate samples when rated.
        codec_r = _score_cell(
            ratings, case_outputs.get((backend, "codec_roundtrip")) or "")
        zero_r = _score_cell(
            ratings, case_outputs.get((backend, "zero_shot")) or "")
        if ratings:
            subj = f"codec {codec_r}; zs {zero_r}"
            train = PENDING_DECISION
        else:
            subj = train = PENDING
        lines.append(
            f"| {backend} | {codec} | {zero} | {subj} | {train} |"
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
