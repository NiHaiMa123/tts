#!/usr/bin/env python3
"""Speaker-embedding check for an ingest index. Run under the SenseVoice env:

    backend_envs\\asr_sensevoice\\Scripts\\python.exe ^
        scripts\\ingest\\speaker_check.py ^
        --index <index.jsonl> --audio-root <pool_dir> ^
        --reference assets\\characters\\<id>\\reference\\ref.wav ^
        --out outputs\\_tmp\\speaker_<id>.json

Computes CampPlus speaker embeddings (funasr
``iic/speech_campplus_sv_zh-cn_16k-common``, auto-downloaded into
``models/speaker/``) and writes per-row cosine similarity to the mean
reference embedding. Advisory only — human listening stays authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "iic/speech_campplus_sv_zh-cn_16k-common"
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "speaker"


def _unit(v):
    import numpy as np
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v)) or 1.0
    return v / n


def _embed(model, wav: Path):
    r = model.generate(input=str(wav))
    if not r:
        return None
    emb = r[0].get("spk_embedding")
    return None if emb is None else _unit(emb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="ingest index.jsonl")
    ap.add_argument("--audio-root",
                    help="pool dir (default: <index dir>/audio)")
    ap.add_argument("--reference", action="append", required=True,
                    help="reference wav of the target speaker; repeatable")
    ap.add_argument("--model-root", default=None,
                    help="local model dir; default auto-download into "
                         "models/speaker/ (modelscope cache)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N ok rows (smoke test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index_path = Path(args.index).resolve()
    audio_root = Path(args.audio_root).resolve() if args.audio_root \
        else index_path.parent / "audio"
    model_root = Path(args.model_root).resolve() if args.model_root else None
    if model_root and model_root.is_dir():
        model_ref = str(model_root)
    else:
        os.environ.setdefault("MODELSCOPE_CACHE", str(DEFAULT_MODEL_DIR))
        model_ref = MODEL_ID

    from funasr import AutoModel  # noqa: E402
    model = AutoModel(model=model_ref, device=args.device,
                      disable_update=True)

    refs = []
    for ref_arg in args.reference:
        emb = _embed(model, Path(ref_arg).resolve())
        if emb is None:
            print(f"reference produced no embedding: {ref_arg}",
                  file=sys.stderr)
            return 2
        refs.append(emb)
    import numpy as np
    ref_vec = _unit(np.mean(refs, axis=0))

    rows = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        wav = (audio_root / row["audio"]).resolve()
        if not wav.is_file():
            continue
        if args.limit and len(rows) >= args.limit:
            break
        emb = _embed(model, wav)
        if emb is None:
            rows.append({"source": row["source"], "speaker_cos": None})
            continue
        rows.append({"source": row["source"],
                     "speaker_cos": round(float(np.dot(emb, ref_vec)), 4)})
        print(f"[{len(rows)}] {row['source']} cos={rows[-1]['speaker_cos']}",
              flush=True)

    payload = {"model": model_ref, "references": args.reference,
               "device": args.device, "auxiliary_only": True,
               "results": rows}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {out_path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
