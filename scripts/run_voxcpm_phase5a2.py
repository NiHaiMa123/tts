"""Run Phase 5A2 — paired P0/P2 short-utterance check (PLAN.md sec. 24).

    python scripts/run_voxcpm_phase5a2.py
        [--config configs/evaluations/suoming_voxcpm_phase5a2.yaml]
        [--attach-asr outputs/gates/suoming_voxcpm_phase5a2/asr_check.json]

Writes outputs/gates/suoming_voxcpm_phase5a2/{manifest.json,metrics.json,
report.md,listen/index.html,unblind_map.json} plus WAV artifacts.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from character_tts.evaluation.phase5a import load_phase5a_config  # noqa: E402
from character_tts.evaluation.phase5a2 import (  # noqa: E402
    attach_asr_results, run_phase5a2)
from character_tts.registry.loader import load_character  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="suoming_voxcpm_phase5a2")
    ap.add_argument("--character", default="suoming")
    ap.add_argument("--attach-asr", default=None,
                    help="merge a finished asr_check.json into the manifest")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.attach_asr:
        cfg = load_phase5a_config(args.config)
        mpath = Path(cfg["output_dir"]) / "manifest.json"
        if not mpath.is_absolute():
            from character_tts.registry.loader import repo_root
            mpath = repo_root() / mpath
        manifest = attach_asr_results(mpath, args.attach_asr)
        print(f"ASR results merged into {mpath} "
                  f"({manifest['asr']})")
        return 0

    cfg = load_phase5a_config(args.config)
    character = load_character(args.character)
    manifest = run_phase5a2(cfg, character)

    ok = sum(1 for c in manifest["cases"] if c["status"] == "ok")
    blocked = [c["case_id"] for c in manifest["cases"]
               if c["status"] != "ok"]
    print(f"\n== Phase 5A2 done: {ok}/{len(manifest['cases'])} cells ok, "
          f"{len(manifest['pairs'])} pairs")
    if blocked:
        print("not-ok:", *blocked, sep="\n  ")
    print(f"new zero-shot WAVs: "
          f"{manifest['budget']['zero_shot_generated']}"
          f"/{manifest['budget']['zero_shot_limit']}")
    print(f"manifest: {cfg['output_dir']}/manifest.json")
    print(f"listen:   {cfg['output_dir']}/listen/index.html")
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
