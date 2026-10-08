"""Export the Phase 0-4 blind-listening ratings into a sanitized,
machine-readable JSON committed under docs/reports/.

Reads outputs/gates/suoming_v1/listen/listen-ratings.json (the file the
listen page writes when the user clicks Export) and produces
docs/reports/suoming-gate-v1-user-ratings.json with relative sample
paths only - no machine-local absolute paths.

If the source file does not exist we do NOT fabricate it; the committed
report table remains the only record (marked report_transcription).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "outputs/gates/suoming_v1/listen/listen-ratings.json"
DST = ROOT / "docs/reports/suoming-gate-v1-user-ratings.json"

# Stable logical ids for each sample file, so the JSON survives even if
# the outputs/ directory is regenerated or moved.
SAMPLE_IDS = {
    "ground_truth_original.wav": "ground_truth/rain_anchor_original",
    "reference_original.wav": "reference/ditianjian_prompt_original",
    "dots/codec_roundtrip.wav": "dots_legacy/codec_roundtrip",
    "dots/trained_or_lora.wav": "dots_legacy/lora_step500_generate",
    "dots/zero_shot.wav": "dots_legacy/zero_shot",
    "qwen3_tts/codec_roundtrip.wav": "qwen3_tts/codec_roundtrip",
    "qwen3_tts/zero_shot.wav": "qwen3_tts/zero_shot",
    "voxcpm2/codec_roundtrip.wav": "voxcpm2/codec_roundtrip",
    "voxcpm2/zero_shot.wav": "voxcpm2/zero_shot",
}


def main() -> int:
    if not SRC.exists():
        print(f"source ratings file missing: {SRC}", file=sys.stderr)
        return 1
    data = json.loads(SRC.read_text(encoding="utf-8"))
    ratings = data.get("ratings") or {}
    samples = []
    unknown = sorted(set(ratings) - set(SAMPLE_IDS))
    if unknown:
        print(f"warning: unrated-id entries not mapped: {unknown}",
              file=sys.stderr)
    for rel_path, entry in ratings.items():
        sid = SAMPLE_IDS.get(rel_path, rel_path)
        rec = {
            "sample_id": sid,
            "sample_rel_path": rel_path,
            "clarity": entry.get("clarity"),
            "grit": entry.get("grit"),
            "likeness": entry.get("likeness"),
            "naturalness": entry.get("naturalness"),
        }
        if entry.get("notes"):
            rec["notes"] = entry["notes"]
        samples.append(rec)
    out = {
        "schema": "suoming_gate_v1_user_ratings/1",
        "source": "outputs/gates/suoming_v1/listen/listen-ratings.json",
        "source_type": "raw_export",
        "exported_at": data.get("exported_at"),
        "scale": {
            "range": "1-5",
            "direction": "higher_is_better",
            "grit_semantics": "higher means LESS grit/scratchiness",
        },
        "dimensions": ["clarity", "grit", "likeness", "naturalness"],
        "samples": samples,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {DST} ({len(samples)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
