"""Auxiliary ASR check — multi-backend. Run under the matching ASR env:

    backend_envs\\asr_sensevoice\\Scripts\\python.exe    scripts\\asr_check.py --backend sensevoice    ...
    backend_envs\\asr_faster_whisper\\Scripts\\python.exe scripts\\asr_check.py --backend faster_whisper ...
    backend_envs\\asr_qwen3\\Scripts\\python.exe         scripts\\asr_check.py --backend qwen3_asr     ...

Usage (either mode):
    --manifest outputs/gates/<exp>/manifest.json --out <exp>/asr_check.json
    --wav xxx.wav --text-file xxx.txt --out out.json   (repeatable)

Standalone by design: the ASR envs carry the runtime libs and no
character_tts package. Output is char-CER as AUXILIARY evidence only —
``text_complete`` remains a human judgement field; never infer subjective
quality from CER.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

BACKENDS = {
    "sensevoice": REPO_ROOT / "models" / "asr" / "sensevoice"
                  / "3847d57b6bdf2dd8875cb1508d2af43d80a16bf7",
    "faster_whisper": REPO_ROOT / "models" / "asr" / "faster_whisper"
                      / "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf",
    "qwen3_asr": REPO_ROOT / "models" / "asr" / "qwen3_asr"
                 / "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
}

_TAG_RE = re.compile(r"<\|[^|]+\|>")
_PUNCT_RE = re.compile(r"[，。！？、；：「」『』（）……—…,.!?;:\"'\s]")


def _norm(text: str) -> str:
    """Strip tags/emoji/punctuation/space for char CER."""
    text = _TAG_RE.sub("", text or "")
    text = "".join(c for c in text if unicodedata.category(c) != "So")
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


def _transcribe_sensevoice(model_root: Path, wavs: list[Path],
                           device: str) -> list[str]:
    from funasr import AutoModel
    model = AutoModel(model=str(model_root), device=device,
                      disable_update=True)
    out = []
    for wav in wavs:
        r = model.generate(input=str(wav), cache={}, language="zh",
                           use_itn=True)
        out.append(r[0].get("text", "") if r else "")
    return out


def _transcribe_faster_whisper(model_root: Path, wavs: list[Path],
                               device: str) -> list[str]:
    import os
    # ctranslate2 needs CUDA DLLs (nvidia-cublas-cu12 / nvidia-cudnn-cu12);
    # register their bin dirs explicitly so the env is self-contained.
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia = Path(sp) / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]
    from faster_whisper import WhisperModel
    dev = "cuda" if device.startswith("cuda") else "cpu"
    model = WhisperModel(str(model_root), device=dev,
                         compute_type="float16" if dev == "cuda" else "int8")
    out = []
    for wav in wavs:
        segments, _info = model.transcribe(
            str(wav), language="zh", beam_size=5, vad_filter=True,
            condition_on_previous_text=False)
        out.append("".join(s.text for s in segments).strip())
    return out


def _transcribe_qwen3_asr(model_root: Path, wavs: list[Path],
                          device: str) -> list[str]:
    import torch
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        str(model_root), dtype=torch.bfloat16, device_map=device,
        max_inference_batch_size=8, max_new_tokens=1024)
    out = []
    for wav in wavs:
        r = model.transcribe(audio=str(wav), language="Chinese",
                             return_time_stamps=False)[0]
        out.append(r.text)
    return out


_TRANSCRIBERS = {
    "sensevoice": _transcribe_sensevoice,
    "faster_whisper": _transcribe_faster_whisper,
    "qwen3_asr": _transcribe_qwen3_asr,
}


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
    n_wav = len(args.wav or [])
    texts = list(args.text or [])
    files = list(args.text_file or [])
    for i, wav_arg in enumerate(args.wav or []):
        text = None
        if i < len(texts):
            text = texts[i]
        elif i - len(texts) < len(files):
            text = Path(files[i - len(texts)]).read_text(encoding="utf-8")
        if text is None:
            print(f"--wav {wav_arg} has no paired --text/--text-file",
                  file=sys.stderr)
            continue
        targets.append((Path(wav_arg).stem, Path(wav_arg).resolve(), text))
    return targets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--manifest", help="gate/eval manifest.json (batch mode)")
    ap.add_argument("--wav", action="append",
                    help="wav path; repeatable, pairs with --text/--text-file")
    ap.add_argument("--text", action="append",
                    help="reference text paired positionally with --wav")
    ap.add_argument("--text-file", action="append",
                    help="txt file paired positionally with --wav")
    ap.add_argument("--model-root", default=None,
                    help="default: models/asr/<backend>/<rev>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    model_root = Path(args.model_root or BACKENDS[args.backend]).resolve()
    if not model_root.is_dir():
        print(f"model root missing: {model_root}", file=sys.stderr)
        return 2
    targets = _collect_targets(args)
    if not targets:
        print("no targets: pass --manifest or --wav with --text/--text-file",
              file=sys.stderr)
        return 2

    wavs = [w for _, w, _ in targets]
    hyps = _TRANSCRIBERS[args.backend](model_root, wavs, args.device)

    results = []
    for (cid, wav, ref), hyp_raw in zip(targets, hyps):
        cer, dist, n = _cer(_norm(ref), _norm(hyp_raw))
        results.append({"case_id": cid, "wav": str(wav),
                        "cer": round(cer, 4), "edit_distance": dist,
                        "ref_chars": n, "hypothesis": hyp_raw})
        print(f"[{len(results)}/{len(targets)}] {cid} cer={cer:.4f}",
              flush=True)

    payload = {
        "tool": args.backend,
        "model": str(model_root),
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
