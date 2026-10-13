#!/usr/bin/env python3
"""Read the user's saved choices when they tell the Agent to continue."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from character_tts.web.listen_store import agent_review


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True,
                        help="current experiment output dir")
    args = parser.parse_args()
    result = agent_review(Path(args.bundle))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result["status"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
