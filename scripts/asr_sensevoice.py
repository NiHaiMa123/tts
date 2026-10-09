"""Auxiliary ASR check (SenseVoice). Runs under the dedicated ASR env:

    backend_envs\\asr_sensevoice\\Scripts\\python.exe scripts\\asr_sensevoice.py ^
        --manifest outputs/gates/<exp>/manifest.json ^
        --out outputs/gates/<exp>/asr_check.json

    backend_envs\\asr_sensevoice\\Scripts\\python.exe scripts\\asr_sensevoice.py ^
        --wav outputs/webui/suoming/睡眠.wav --text-file inputs/睡眠.txt ^
        --out outputs/user_gen/睡眠_asr.json

Standalone by design: the ASR env has funasr but no character_tts package.
Reports a character-level CER as AUXILIARY evidence only — ``text_complete``
remains a human judgement field; never infer subjective quality from CER.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (REPO_ROOT / "models" / "asr" / "sensevoice"
                 / "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7")

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


def _collect_targets(args) -> list[tuple[str, Path, str]]:
    targets: list[tuple[str, Path, str]] = []
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for case in manifest.get("cases") or []:
            if case.get("status") != "ok" or not case.get("output"):
                continue
            wav = (manifest_path.parent / case["output"]).resolve()
            if wav.is_file() and case.get("text"):
                targets.append((case["case_id"], wav, case["text"]))
    for wav_arg, text in zip(args.wav or [], args.text or []):
        targets.append((Path(wav_arg).stem, Path(wav_arg).resolve(), text))
    for wav_arg, txt_path in zip(args.wav or [], args.text_file or []):
        targets.append((Path(wav_arg).stem, Path(wav_arg).resolve(),
                        Path(txt_path).read_text(encoding="utf-8")))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="gate/eval manifest.json (batch mode)")
    ap.add_argument("--wav", action="append",
                    help="wav path; repeatable, pairs with --text/--text-file")
    ap.add_argument("--text", action="append",
                    help="reference text paired positionally with --wav")
    ap.add_argument("--text-file", action="append",
                    help="txt file paired positionally with --wav")
    ap.add_argument("--model-root", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    targets = _collect_targets(args)
    if not targets:
        print("no targets: pass --manifest or --wav with --text/--text-file",
              file=sys.stderr)
        return 2

    from funasr import AutoModel  # noqa: E402  (env-provided)
    model = AutoModel(model=str(Path(args.model_root).resolve()),
                      device=args.device, disable_update=True)

    results = []
    for cid, wav, ref in targets:
        try:
            out = model.generate(input=str(wav), cache={},
                                 language="zh", use_itn=True)
            hyp_raw = out[0].get("text", "") if out else ""
            cer, dist, n = _cer(_norm(ref), _norm(hyp_raw))
            results.append({"case_id": cid, "wav": str(wav),
                            "cer": round(cer, 4), "edit_distance": dist,
                            "ref_chars": n, "hypothesis": hyp_raw})
        except Exception as exc:  # ASR failure is evidence too
            results.append({"case_id": cid, "wav": str(wav),
                            "error": f"{type(exc).__name__}: {exc}"})
        print(f"[{len(results)}/{len(targets)}] {cid} "
              f"cer={results[-1].get('cer')}", flush=True)

    payload = {
        "tool": "sensevoice",
        "model": str(Path(args.model_root).resolve()),
        "device": args.device,
        "auxiliary_only": True,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {out_path} ({len(results)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
