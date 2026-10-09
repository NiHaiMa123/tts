#!/usr/bin/env python3
"""Standardize inbox -> content-addressed pool + index.jsonl.

    python scripts/ingest/standardize_inbox.py \
        --inbox data/characters/<id>/inbox \
        --pool data/characters/<id>/datasets/v1/audio \
        --index data/characters/<id>/datasets/v1/index.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.standardize import standardize_inbox  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--keep-flagged", action="store_true",
                    help="also accept hard-flagged files (too_short etc.)")
    args = ap.parse_args()
    stats = standardize_inbox(Path(args.inbox), Path(args.pool),
                              Path(args.index),
                              reject_flagged=not args.keep_flagged)
    print(f"{stats['accepted']}/{stats['total']} accepted, "
          f"{stats['rejected']} rejected -> {stats['index']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
