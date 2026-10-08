"""Export Phase 5B (LoRA pilot) A/B ratings: verify sha256 bindings against the
manifest, persist a sanitized copy under docs/reports/, and append the
subjective section to outputs report.md + committed report.

Usage:
    python scripts/export_lora_pilot_ratings.py \
        --manifest outputs/gates/suoming_voxcpm_lora_pilot_v1/manifest.json \
        --ratings outputs/gates/suoming_voxcpm_lora_pilot_v1/listen/listen-ratings-lora-pilot.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ratings", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man_path = Path(args.manifest)
    exp_dir = man_path.parent
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    ratings = json.loads(Path(args.ratings).read_text(encoding="utf-8"))

    if ratings.get("experiment_id") != manifest.get("experiment_id"):
        print("ERROR: experiment_id mismatch between ratings and manifest")
        return 1

    cases = {c["case_id"]: c for c in manifest["cases"]}
    truth = ratings.get("truth") or {}

    # Verify every truth sha256 against the manifest cell it claims to be.
    mismatches = []
    verified = 0
    for cid, t in truth.items():
        cell = cases.get(cid)
        if cell is None:
            mismatches.append((cid, "unknown case_id"))
            continue
        if t.get("output_sha256") != cell.get("output_sha256"):
            mismatches.append((cid, "sha mismatch"))
            continue
        if t.get("condition") != cell.get("condition"):
            mismatches.append((cid, "condition mismatch"))
            continue
        verified += 1
    if mismatches:
        for cid, why in mismatches:
            print(f"ERROR: truth binding failed for {cid}: {why}")
        return 1

    pairs = ratings.get("pairs") or {}
    rated = {pid: p for pid, p in pairs.items() if p.get("choice")}
    pending = sorted(set(pairs) - set(rated))

    # Sanitize: keep pair_id, batch, real winner, tags/notes; drop absolute paths.
    clean_pairs = []
    tallies = Counter()
    pack1 = []
    for pid, p in sorted(rated.items(), key=lambda kv: kv[1].get("rated_at", "")):
        real = p.get("choice_real") or p.get("choice")
        winner_cond = real.split("/")[0] if isinstance(real, str) and "/" in real else real
        tallies[winner_cond] += 1
        rec = {
            "pair_id": pid,
            "batch": p.get("batch"),
            "winner": real,
            "winner_condition": winner_cond,
            "defect_tags": p.get("defect_tags"),
            "severity": p.get("severity"),
            "first_impression": p.get("first_impression"),
            "notes": p.get("notes"),
            "rated_at": p.get("rated_at"),
        }
        clean_pairs.append(rec)
        if p.get("batch") == "pack1":
            pack1.append(rec)

    out = {
        "experiment_id": manifest["experiment_id"],
        "source": "user A/B listening export (sanitized)",
        "exported_at": ratings.get("exported_at"),
        "session": ratings.get("session"),
        "truth_verified": verified,
        "truth_total": len(truth),
        "pairs_rated": len(clean_pairs),
        "pairs_pending": len(pending),
        "winner_tally": dict(tallies),
        "pack1": pack1,
        "pairs": clean_pairs,
    }
    out_path = Path(args.out) if args.out else Path(
        "docs/reports/suoming-voxcpm-lora-pilot-v1-ratings.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"truth verified: {verified}/{len(truth)}")
    print(f"pairs rated: {len(clean_pairs)} (pending: {len(pending)})")
    print(f"winner tally: {dict(tallies)}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
