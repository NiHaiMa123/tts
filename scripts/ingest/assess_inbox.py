#!/usr/bin/env python3
"""Assess an inbox dir: per-file sanity report.

    python scripts/ingest/assess_inbox.py \
        --inbox data/characters/<id>/inbox --out outputs/_tmp/assess_<id>.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.assess import assess_inbox  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rep = assess_inbox(Path(args.inbox), Path(args.out))
    print(f"{rep['files']} files, {rep['flagged']} flagged "
          f"({rep['flags_by_kind']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
