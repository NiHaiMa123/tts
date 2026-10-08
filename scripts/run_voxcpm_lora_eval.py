"""Run the Phase 5B pilot evaluation (PLAN.md section 25.5+).

    python scripts/run_voxcpm_lora_eval.py
        [--config configs/evaluations/suoming_voxcpm_lora_pilot_v1.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from character_tts.evaluation.lora_pilot import run_lora_eval  # noqa: E402
from character_tts.evaluation.phase5a import load_phase5a_config  # noqa: E402
from character_tts.registry.loader import load_character  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="suoming_voxcpm_lora_pilot_v1")
    ap.add_argument("--character", default="suoming")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_phase5a_config(args.config)
    character = load_character(args.character)
    manifest = run_lora_eval(cfg, character)

    ok = sum(1 for c in manifest["cases"] if c["status"] == "ok")
    blocked = [c["case_id"] for c in manifest["cases"]
               if c["status"] != "ok"]
    print(f"\n== pilot eval done: {ok}/{len(manifest['cases'])} cells ok")
    if blocked:
        print("not-ok:", *blocked, sep="\n  ")
    print(f"new WAVs: {manifest['budget']['new_generated']}"
          f"/{manifest['budget']['limit']}")
    print(f"manifest: {cfg['output_dir']}/manifest.json")
    print(f"listen:   {cfg['output_dir']}/listen/index.html")
    return 0 if not blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
