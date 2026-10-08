"""Export Phase 5A blind-listening ratings to a sanitized, committed JSON.

Reads outputs/gates/suoming_voxcpm_phase5a/listen/listen-ratings-phase5a.json,
verifies every rating's ``output_sha256`` against the current manifest so
stale ratings can never attach to regenerated WAVs, and writes
docs/reports/suoming-voxcpm-phase5a-ratings.json.

Never fabricates ratings: entries whose sha does not match the manifest
are marked ``stale_sha`` and kept visible as evidence, not silently dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs/gates/suoming_voxcpm_phase5a"
SRC = OUT_DIR / "listen/listen-ratings-phase5a.json"
MANIFEST = OUT_DIR / "manifest.json"
DST = ROOT / "docs/reports/suoming-voxcpm-phase5a-ratings.json"

DIMS = ("clarity", "grit", "likeness", "naturalness")


def _manifest_shas(manifest: dict) -> dict[str, str | None]:
    """case_id -> current output sha256 (None when nothing to verify)."""
    sha: dict[str, str | None] = {}
    for c in manifest.get("cases") or []:
        sha[c["case_id"]] = c.get("output_sha256")
    for iid, entry in (manifest.get("codec_diagnosis") or {}).items():
        enc = entry.get("encode_input") or {}
        sha[f"codec/{iid}:encode_input"] = enc.get("sha256")
        for variant, art in (entry.get("artifacts") or {}).items():
            tag = {"default": "roundtrip48",
                   "cond16000": "roundtrip_cond16k"}.get(
                variant, f"roundtrip_{variant}")
            sha[f"codec/{iid}:{tag}"] = art.get("sha256")
    return sha


def main() -> int:
    if not SRC.is_file():
        print(f"ratings file missing: {SRC}", file=sys.stderr)
        return 1
    data = json.loads(SRC.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_id = manifest.get("experiment_id")
    got_id = data.get("experiment_id")
    if got_id != expected_id:
        print(f"experiment_id mismatch: ratings={got_id} "
              f"manifest={expected_id}", file=sys.stderr)
        return 1

    shas = _manifest_shas(manifest)
    case_group = {c["case_id"]: c.get("group")
                  for c in manifest.get("cases") or []}
    samples = []
    stale = []
    for case_id, entry in (data.get("ratings") or {}).items():
        rated_sha = entry.get("output_sha256")
        current_sha = shas.get(case_id, "ABSENT")
        if current_sha == "ABSENT":
            status = "unknown_case"
        elif rated_sha and current_sha and rated_sha == current_sha:
            status = "verified"
        else:
            status = "stale_sha"
            stale.append(case_id)
        rec = {
            "case_id": case_id,
            "group": case_group.get(case_id,
                                    "codec_diagnostics"
                                    if case_id.startswith("codec/")
                                    else None),
            "output_sha256": rated_sha,
            "sha_status": status,
        }
        for d in DIMS:
            rec[d] = entry.get(d)
        for extra in ("text_complete", "artifact_type", "notes"):
            if entry.get(extra):
                rec[extra] = entry[extra]
        samples.append(rec)

    out = {
        "schema": "suoming_voxcpm_phase5a_ratings/1",
        "experiment_id": expected_id,
        "source": "outputs/gates/suoming_voxcpm_phase5a/listen/"
                  "listen-ratings-phase5a.json",
        "source_type": "raw_export",
        "exported_at": data.get("exported_at"),
        "scale": {
            "range": "1-5",
            "direction": "higher_is_better",
            "grit_semantics": "higher means LESS grit/scratchiness",
        },
        "dimensions": list(DIMS),
        "extra_fields": ["text_complete", "artifact_type", "notes"],
        "samples": samples,
    }
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    n_verified = sum(1 for s in samples if s["sha_status"] == "verified")
    print(f"wrote {DST} ({len(samples)} samples, "
          f"{n_verified} sha-verified)")
    if stale:
        print("WARNING stale/unknown entries (kept as evidence):",
              *stale, sep="\n  ")
    _append_ratings_section(samples)
    return 0


def _append_ratings_section(samples: list[dict]) -> None:
    """Append the user-ratings table to the generated report.md."""
    report = OUT_DIR / "report.md"
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")
    marker = "## Subjective ratings (user-provided)"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        "| case | c | g | l | n | text_complete | artifact_type | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in samples:
        lines.append(
            "| {cid} | {c} | {g} | {l} | {n} | {tc} | {at} | {no} |".format(
                cid=s["case_id"],
                c=s.get("clarity") or "-",
                g=s.get("grit") or "-",
                l=s.get("likeness") or "-",
                n=s.get("naturalness") or "-",
                tc=s.get("text_complete") or "待审",
                at=s.get("artifact_type") or "",
                no=s.get("notes") or "",
            ))
    report.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"updated {report}")


if __name__ == "__main__":
    raise SystemExit(main())
