#!/usr/bin/env python3
"""Import Ludiglot-decoded audio + manifest into a character inbox.

    python scripts/ingest/import_ludiglot.py ^
        --audio-dir <decoded wav/ogg dir> ^
        --manifest manifest.jsonl ^
        --out data/characters/<id>/inbox [--label 中立_neutral] [--link]

manifest.jsonl rows: {"audio": "<stem|relpath>", "text": "...",
                      "label"?, "event"?, "wem_hash"?, "source_wem"?,
                      "tool"?, ...}
Writes <inbox>/<label>/<stem>.<ext> + <inbox>/provenance.jsonl which
standardize_inbox merges automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from character_tts.ingest.ludiglot_import import import_ludiglot  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True,
                    help="dir of decoded audio (wav/ogg/flac/mp3)")
    ap.add_argument("--manifest", required=True,
                    help="jsonl manifest joining audio -> text/provenance")
    ap.add_argument("--out", required=True, help="target inbox dir")
    ap.add_argument("--label", help="default label dir for unlabeled rows")
    ap.add_argument("--link", action="store_true",
                    help="hardlink instead of copy (same volume only)")
    args = ap.parse_args()
    result = import_ludiglot(Path(args.audio_dir), Path(args.manifest),
                             Path(args.out), default_label=args.label,
                             copy=not args.link)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
