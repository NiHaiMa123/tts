#!/usr/bin/env python3
"""Freeze dataset split from a standardize index (+ optional review drops).

    python scripts/ingest/freeze_dataset.py \
        --index data/characters/<id>/datasets/v1/index.jsonl \
        --dataset-dir data/characters/<id>/datasets/v1 \
        [--exclude-file decisions.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.freeze import freeze_dataset  # noqa: E402
from character_tts.ingest.review import load_decisions  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--pool",
                    help="audio pool dir from standardize_inbox "
                         "(default: <dataset-dir>/audio)")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--exclude-file",
                    help="decisions.json from the review page (drop list)")
    args = ap.parse_args()
    exclude = load_decisions(Path(args.exclude_file)) \
        if args.exclude_file else set()
    result = freeze_dataset(Path(args.index), Path(args.dataset_dir),
                            pool_dir=Path(args.pool) if args.pool else None,
                            val_frac=args.val_frac,
                            test_frac=args.test_frac,
                            exclude_fids=exclude)
    print(json.dumps(result["counts"], ensure_ascii=False))
    print(f"dropped: {len(result['dropped'])} -> {result['dataset_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
