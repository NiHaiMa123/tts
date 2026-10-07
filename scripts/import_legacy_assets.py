#!/usr/bin/env python3
"""Import legacy dotstts character assets into platform config.

Locates the old dotstts repo (env DOTSTTS_ROOT or known candidates),
verifies the suoming voice profile files exist with matching hashes, and
emits configs/characters/suoming.yaml. Large WAV/model files are NOT
copied — paths stay external via ${DOTSTTS_ROOT}.

When the legacy repo or required files are missing the script exits
BLOCKED and never fabricates data.

Usage:
    python scripts/import_legacy_assets.py [--character suoming]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.app.legacy_import import (  # noqa: E402
    find_dotstts,
    import_suoming,
    write_character,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", default="suoming")
    parser.add_argument("--voice-profile",
                        default="configs/voices/suoming_step500_v1.yaml")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "configs" / "characters")
    args = parser.parse_args()

    if args.character != "suoming":
        print(f"BLOCKED: only 'suoming' import is implemented "
              f"(got '{args.character}')")
        return 2

    root = find_dotstts()
    if root is None:
        print("BLOCKED: legacy dotstts repo not found. Set DOTSTTS_ROOT.")
        return 2
    print(f"legacy root: {root}")

    result = import_suoming(root, profile_rel=args.voice_profile)
    if result["status"] != "ok":
        print("BLOCKED:")
        for p in result["problems"]:
            print(f"  - {p}")
        return 2

    out_path = write_character(result, args.out)
    ref = result["character"]["reference"]
    adapter = result["character"]["legacy_voice_profile"]["adapter"]
    print(f"wrote {out_path}")
    print(f"reference sha256: {ref['sha256']}")
    print(f"adapter step: {adapter.get('training_step')} "
          f"weights_sha256: {str(adapter.get('weights_sha256', ''))[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
