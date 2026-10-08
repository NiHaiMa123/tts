"""Run the Phase 5A VoxCPM2 experiment (PLAN.md section 23).

    python scripts/run_voxcpm_phase5a.py
        [--config configs/evaluations/suoming_voxcpm_phase5a.yaml]
        [--character configs/characters/suoming.yaml]

Writes outputs/gates/suoming_voxcpm_phase5a/{manifest.json,metrics.json,
report.md,listen/index.html} plus WAV artifacts (gitignored).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from character_tts.evaluation.phase5a import (  # noqa: E402
    load_phase5a_config, run_phase5a)
from character_tts.registry.loader import load_character  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="suoming_voxcpm_phase5a")
    ap.add_argument("--character", default="suoming")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_phase5a_config(args.config)
    character = load_character(args.character)
    manifest = run_phase5a(cfg, character)

    ok = sum(1 for c in manifest["cases"] if c["status"] == "ok")
    err = [c["case_id"] for c in manifest["cases"] if c["status"] != "ok"]
    print(f"\n== Phase 5A done: {ok}/{len(manifest['cases'])} cases ok")
    if err:
        print("failed:", *err, sep="\n  ")
    print(f"zero-shot generated: "
          f"{manifest['budget']['zero_shot_generated']}"
          f"/{manifest['budget']['zero_shot_limit']}")
    print(f"manifest: {cfg['output_dir']}/manifest.json")
    print(f"listen:   {cfg['output_dir']}/listen/index.html")
    return 0 if not err else 2


if __name__ == "__main__":
    raise SystemExit(main())
