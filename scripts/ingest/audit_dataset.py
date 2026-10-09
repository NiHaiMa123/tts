#!/usr/bin/env python3
"""Audit a frozen dataset dir before registration.

    .venv\\Scripts\\python.exe scripts\\ingest\\audit_dataset.py ^
        --dataset-dir data\\characters\\<id>\\datasets\\v1 [--no-hash]

Checks: split files exist, every ``audio`` resolves, recompute sha256 &
fid per row, no audio/text leakage across splits. Exit 1 on violations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.audit import audit_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--no-hash", action="store_true",
                    help="skip sha256 recompute (fast structural check)")
    args = ap.parse_args()
    report = audit_dataset(Path(args.dataset_dir),
                           verify_hash=not args.no_hash)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
