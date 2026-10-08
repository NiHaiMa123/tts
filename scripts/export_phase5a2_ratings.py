"""Export Phase 5A2 blind-listening ratings to a sanitized, committed JSON.

Reads outputs/gates/suoming_voxcpm_phase5a2/listen/listen-ratings-phase5a2.json,
verifies every rating's ``output_sha256`` against the current manifest so
stale ratings can never attach to regenerated WAVs, writes
docs/reports/suoming-voxcpm-phase5a2-ratings.json, and appends a
per-pair P0-vs-P2 summary to the generated report.md.

Never fabricates ratings: sha mismatches are marked ``stale_sha`` and
kept visible as evidence, not silently dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs/gates/suoming_voxcpm_phase5a2"
SRC = OUT_DIR / "listen/listen-ratings-phase5a2.json"
MANIFEST = OUT_DIR / "manifest.json"
DST = ROOT / "docs/reports/suoming-voxcpm-phase5a2-ratings.json"

DIMS = ("clarity", "grit", "likeness", "naturalness")


def _mean(values: list[str | None]) -> float | None:
    nums = [float(v) for v in values if v not in (None, "")]
    return round(sum(nums) / len(nums), 2) if nums else None


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

    cases = {c["case_id"]: c for c in manifest.get("cases") or []}
    samples, stale = [], []
    for case_id, entry in (data.get("ratings") or {}).items():
        rated_sha = entry.get("output_sha256")
        case = cases.get(case_id)
        current_sha = (case or {}).get("output_sha256")
        if case is None:
            status = "unknown_case"
        elif rated_sha and current_sha and rated_sha == current_sha:
            status = "verified"
        else:
            status = "stale_sha"
            stale.append(case_id)
        rec = {
            "case_id": case_id,
            "pair_id": (case or {}).get("pair_id"),
            "prompt_id": (case or {}).get("prompt_id"),
            "text_id": (case or {}).get("text_id"),
            "seed": (case or {}).get("seed"),
            "reused_from": (case or {}).get("reused_from"),
            "output_sha256": rated_sha,
            "sha_status": status,
        }
        for d in DIMS:
            rec[d] = entry.get(d)
        for extra in ("text_complete", "artifact_type", "notes"):
            if entry.get(extra):
                rec[extra] = entry[extra]
        samples.append(rec)

    # Pair-level verdicts: mean dim score per side, blind the user saw.
    by_pair: dict[str, dict[str, list[str | None]]] = {}
    for s in samples:
        if s["sha_status"] != "verified":
            continue
        slot = by_pair.setdefault(s["pair_id"], {})
        for d in DIMS:
            slot.setdefault(s["prompt_id"], {}).setdefault(d, [])
            slot[s["prompt_id"]][d].append(s.get(d))
    pair_summary = []
    for pair in manifest.get("pairs") or []:
        pid = pair["pair_id"]
        slot = by_pair.get(pid, {})
        row = {"pair_id": pid, "text_id": pair["text_id"],
               "seed": pair["seed"]}
        for side in ("P0", "P2"):
            dims = slot.get(side, {})
            row[side] = {d: _mean(dims.get(d, [])) for d in DIMS}
            vals = [v for v in row[side].values() if v is not None]
            row[f"{side}_mean"] = round(sum(vals) / len(vals), 2) \
                if vals else None
        if row.get("P0_mean") is not None and row.get("P2_mean") is not None:
            row["delta_P2_minus_P0"] = round(
                row["P2_mean"] - row["P0_mean"], 2)
        pair_summary.append(row)

    out = {
        "schema": "suoming_voxcpm_phase5a2_ratings/1",
        "experiment_id": expected_id,
        "source": "outputs/gates/suoming_voxcpm_phase5a2/listen/"
                  "listen-ratings-phase5a2.json",
        "source_type": "raw_export",
        "exported_at": data.get("exported_at"),
        "scale": {
            "range": "1-5",
            "direction": "higher_is_better",
            "grit_semantics": "higher means LESS grit/scratchiness",
        },
        "dimensions": list(DIMS),
        "extra_fields": ["text_complete", "artifact_type", "notes"],
        "pairing": "each pair shares text+seed; only the prompt differs",
        "samples": samples,
        "pair_summary": pair_summary,
    }
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    n_verified = sum(1 for s in samples if s["sha_status"] == "verified")
    print(f"wrote {DST} ({len(samples)} samples, "
          f"{n_verified} sha-verified)")
    if stale:
        print("WARNING stale/unknown entries (kept as evidence):",
              *stale, sep="\n  ")
    _append_ratings_section(samples, pair_summary)
    return 0


def _append_ratings_section(samples: list[dict],
                            pair_summary: list[dict]) -> None:
    report = OUT_DIR / "report.md"
    if not report.is_file():
        return
    text = report.read_text(encoding="utf-8")
    marker = "## Subjective ratings (user-provided)"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n"
    lines = [
        marker, "",
        "### Per-pair means (P2 - P0)", "",
        "| pair | text | seed | P0 mean | P2 mean | delta |",
        "|---|---|---|---|---|---|",
    ]
    for r in pair_summary:
        lines.append(
            f"| {r['pair_id']} | {r['text_id']} | {r['seed']} "
            f"| {r.get('P0_mean')} | {r.get('P2_mean')} "
            f"| {r.get('delta_P2_minus_P0')} |")
    lines += ["", "### All cells", "",
              "| case | c | g | l | n | text_complete | artifact_type "
              "| notes |",
              "|---|---|---|---|---|---|---|---|"]
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
