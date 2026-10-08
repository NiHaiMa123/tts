"""Auxiliary ASR check for Phase 5A2 (runs in the SenseVoice env).

This script is intentionally standalone: it must run under
``dotstts/data/work/asr_envs/sensevoice`` which has funasr installed and
no ``character_tts`` package.

    <sensevoice env python> scripts/asr_phase5a2.py \
        --manifest outputs/gates/suoming_voxcpm_phase5a2/manifest.json \
        --model-root E:/project/dotstts/data/work/models/asr/sensevoice/3847d57b6bdf2dd8875cb1508d2af43d80a16bf7 \
        --out outputs/gates/suoming_voxcpm_phase5a2/asr_check.json

Transcribes every generated/reused wav in the manifest against its
target text and reports a character-level CER as AUXILIARY evidence —
``text_complete`` remains a human judgement field on the listen page.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TAG_RE = re.compile(r"<\|[^|]+\|>")
_PUNCT_RE = re.compile(r"[，。！？、；：「」『』（）……—…,.!?;:\"'\s]")


def _norm(text: str) -> str:
    """Strip SenseVoice tags and punctuation/space for char CER."""
    text = _TAG_RE.sub("", text or "")
    return _PUNCT_RE.sub("", text)


def _cer(ref: str, hyp: str) -> tuple[float, int, int]:
    """Character-level Levenshtein error rate."""
    r, h = list(ref), list(hyp)
    if not r:
        return (0.0, 0, 0) if not h else (1.0, 0, len(h))
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (rc != hc)))
        prev = cur
    dist = prev[-1]
    return dist / len(r), dist, len(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    from funasr import AutoModel  # noqa: E402  (env-provided)
    model = AutoModel(model=str(Path(args.model_root).resolve()),
                      device=args.device, disable_update=True)

    targets = []  # (case_id, abs wav path, ref text)
    for case in manifest.get("cases") or []:
        if case.get("status") != "ok" or not case.get("output"):
            continue
        wav = (manifest_path.parent / case["output"]).resolve()
        if wav.is_file() and case.get("text"):
            targets.append((case["case_id"], wav, case["text"]))

    results = []
    for cid, wav, ref in targets:
        try:
            out = model.generate(input=str(wav), cache={},
                                 language="zh", use_itn=True)
            hyp_raw = out[0].get("text", "") if out else ""
        except Exception as exc:
            results.append({"case_id": cid, "wav": str(wav),
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        ref_n, hyp_n = _norm(ref), _norm(hyp_raw)
        cer, dist, n = _cer(ref_n, hyp_n)
        results.append({
            "case_id": cid, "wav": str(wav), "ref": ref,
            "hyp_raw": hyp_raw, "ref_norm": ref_n, "hyp_norm": hyp_n,
            "cer": round(cer, 4), "edit_distance": dist,
            "ref_chars": n,
        })
        print(f"{cid}: cer={cer:.3f} hyp={hyp_n[:60]}", file=sys.stderr)

    payload = {
        "backend": "sensevoice",
        "model": str(Path(args.model_root).resolve()),
        "manifest": str(manifest_path),
        "results": results,
        "note": "Auxiliary evidence only; text_complete is user-owned.",
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out} ({len(results)} transcripts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
