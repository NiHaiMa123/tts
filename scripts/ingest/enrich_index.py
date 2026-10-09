#!/usr/bin/env python3
"""Merge auxiliary check reports (speaker cos / ASR CER) into an ingest index.

    .venv\\Scripts\\python.exe scripts\\ingest\\enrich_index.py ^
        --index index.jsonl [--speaker speaker.json] [--asr asr.json] ^
        [--speaker-threshold 0.45] [--asr-cer-threshold 0.3]

Adds flag strings to each matching "ok" row (index rewritten in place):
    speaker_cos:<v> / speaker_mismatch:<v>   (cos < threshold)
    asr_cer:<v>     / asr_mismatch:<v>       (cer > threshold)
Flags are advisory evidence for the human reviewer — freeze only drops
reviewed_out rows, never flags alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | None) -> dict:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for r in data.get("results") or []:
        key = r.get("source") or r.get("case_id")
        if key:
            out[key.replace("\\", "/")] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--speaker", help="speaker_check.py report")
    ap.add_argument("--asr", help="asr_check.py --index report")
    ap.add_argument("--speaker-threshold", type=float, default=0.45)
    ap.add_argument("--asr-cer-threshold", type=float, default=0.3)
    args = ap.parse_args()

    speaker = _load(args.speaker)
    asr = _load(args.asr)
    index_path = Path(args.index)
    rows = [json.loads(l) for l in
            index_path.read_text(encoding="utf-8").splitlines()
            if l.strip()]

    touched = 0
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = row["source"].replace("\\", "/")
        flags = row.setdefault("flags", [])
        sp = speaker.get(key)
        if sp and sp.get("speaker_cos") is not None:
            cos = sp["speaker_cos"]
            flags.append(f"speaker_cos:{cos}")
            if cos < args.speaker_threshold:
                flags.append(f"speaker_mismatch:{cos}")
            touched += 1
        ar = asr.get(key)
        if ar:
            hyp = (ar.get("hypothesis") or "").strip()
            if not row.get("text") and hyp:
                row["text_asr"] = hyp
                flags.append("text_from_asr")
            if ar.get("cer") is not None and row.get("text"):
                cer = ar["cer"]
                flags.append(f"asr_cer:{cer}")
                if cer > args.asr_cer_threshold:
                    flags.append(f"asr_mismatch:{cer}")
            touched += 1

    with index_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"enriched {touched} rows in {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
