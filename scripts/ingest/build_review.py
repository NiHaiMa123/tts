#!/usr/bin/env python3
"""Build the keep/drop review page for a standardized pool.

    python scripts/ingest/build_review.py \
        --index data/characters/<id>/datasets/v1/index.jsonl \
        --audio-root <pool_dir_from_standardize> \
        --out-dir outputs/_tmp/review_<id>

Clips are copied into <out-dir>/audio/ so the bundle is portable.

Open review.html, mark drops, export decisions.json, then pass it to
freeze_dataset.py --exclude-file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.review import build_review_page  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--audio-root", required=True,
                    help="pool dir produced by standardize_inbox")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = build_review_page(Path(args.index), Path(args.audio_root),
                            Path(args.out_dir))
    print(f"review page: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
