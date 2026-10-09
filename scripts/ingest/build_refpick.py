#!/usr/bin/env python3
"""Build the reference-audio pick page for a character.

    .venv\\Scripts\\python.exe scripts\\ingest\\build_refpick.py ^
        --index data\\characters\\<id>\\datasets\\v1\\index.jsonl ^
        --audio-root data\\characters\\<id>\\datasets\\v1\\audio ^
        --char <id> --out-dir outputs\\_tmp\\refpick_<id> ^
        [--top-n 40] [--min-cos 0.6]

Then serve it:
    .venv\\Scripts\\python.exe scripts\\ingest\\serve_review.py ^
        --bundle outputs\\_tmp\\refpick_<id>
Open http://127.0.0.1:7865/refpick.html and click 选为参考音 — the file is
written to assets/characters/<id>/reference/ref.wav and
configs/characters/<id>.yaml reference is updated automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.review import build_refpick_page  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--min-cos", type=float, default=None,
                    help="only show rows with speaker_cos >= this")
    args = ap.parse_args()
    out = build_refpick_page(Path(args.index), Path(args.audio_root),
                             Path(args.out_dir), args.char,
                             top_n=args.top_n, min_cos=args.min_cos)
    print(f"refpick page: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
