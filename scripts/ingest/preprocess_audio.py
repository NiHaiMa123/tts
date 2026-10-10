"""音频预处理段（inbox → inbox_clean 镜像清洗）——管线第 2.5 步。

用法：
    .venv/Scripts/python scripts/ingest/preprocess_audio.py \
        --inbox data/characters/<id>/inbox \
        --out   data/characters/<id>/inbox_clean --cut

默认操作链（可用 --no-* 关闭单项）：
    hp       70Hz 高通（去隆隆声/直流漂移）
    loudness RMS 归一化到 -25dBFS（峰值≤0.98 防爆音）
    cut      VAD 剪切（--cut 启用，推荐）：FSMN-VAD 经 asr env
             子进程检测语音段，删除非语音段/过短爆音段/削波失真段，
             保留段之间截断原静音缝(≤300ms)拼接，边界 10ms 余弦
             淡化防爆音——把数据集里"质量差的片段"整段剪掉
    trim     能量门静音裁剪（首尾；cut 启用时自动跳过）
    denoise  ZipEnhancer ANS（已弃用：16k 输出抹掉音色细节，
             实测磨砂感来自数据声学签名，剪切才是对症方案）

写 <out>/preprocess_report.jsonl（逐文件操作+剪切统计+时长变化）。
原 inbox 不动；之后照常 assess → standardize（指向 clean 目录）。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from character_tts.ingest.preprocess import (  # noqa: E402
    DEFAULT_SR, preprocess_file, preprocess_wav)
from character_tts.registry.loader import repo_root  # noqa: E402

VOXCPM_PY = Path("backend_envs/voxcpm2/Scripts/python.exe")
DENOISE_HELPER = Path("scripts/ingest/_denoise_zipenhancer.py")
ASR_PY = Path("backend_envs/asr_sensevoice/Scripts/python.exe")
VAD_HELPER = Path("scripts/ingest/_vad_fsmn.py")


def _vad_batch(wavs: list[Path]) -> dict[str, list]:
    """Run FSMN-VAD over files inside asr env → {file: segments_ms}."""
    root = repo_root()
    py = root / ASR_PY
    helper = root / VAD_HELPER
    if not py.is_file():
        print(f"[vad] asr env 不存在: {py}", file=sys.stderr)
        return {}
    with tempfile.TemporaryDirectory(prefix="vad_") as td:
        td = Path(td)
        files_in = td / "files.jsonl"
        files_in.write_text("".join(
            json.dumps({"in": str(w)}, ensure_ascii=False) + "\n"
            for w in wavs), encoding="utf-8")
        out_f = td / "segments.jsonl"
        print(f"[vad] {len(wavs)} 条送 FSMN-VAD…")
        proc = subprocess.run([str(py), str(helper), str(files_in),
                               str(out_f)], cwd=str(root),
                              capture_output=True)
        err = (proc.stderr or b"").decode("utf-8", errors="replace")
        out = (proc.stdout or b"").decode("utf-8", errors="replace")
        sys.stderr.write(err[-2000:])
        print(out.strip().splitlines()[-1] if out.strip() else
              "[vad] no output")
        segmap, errs = {}, []
        if out_f.is_file():
            for line in out_f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                segmap[r["in"]] = r["segments"]
                if r.get("error"):
                    errs.append(r["in"])
        if errs:
            print(f"[vad] {len(errs)} 条失败（按未剪切处理）",
                  file=sys.stderr)
        return segmap


def _denoise_batch(pairs: list[dict]) -> dict[str, bool]:
    """Run ZipEnhancer over pairs [{in,out}] inside voxcpm env."""
    root = repo_root()
    py = root / VOXCPM_PY
    helper = root / DENOISE_HELPER
    if not py.is_file():
        print(f"[denoise] voxcpm env 不存在: {py}", file=sys.stderr)
        return {}
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
        pairs_file = f.name
    proc = subprocess.run([str(py), str(helper), pairs_file],
                          cwd=str(root), capture_output=True)
    err = (proc.stderr or b"").decode("utf-8", errors="replace")
    out = (proc.stdout or b"").decode("utf-8", errors="replace")
    sys.stderr.write(err[-2000:])
    print(out.strip().splitlines()[-1] if out.strip() else
          "[denoise] no output")
    return {p["out"]: (Path(p["out"]).is_file()) for p in pairs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inbox", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--no-hp", action="store_true")
    ap.add_argument("--no-loudness", action="store_true")
    ap.add_argument("--cut", action="store_true",
                    help="VAD 剪切（asr env 子进程，推荐）")
    ap.add_argument("--denoise", action="store_true",
                    help="ZipEnhancer ANS（已弃用）")
    ap.add_argument("--trim-db", type=float, default=-45.0)
    ap.add_argument("--hp-hz", type=float, default=70.0)
    ap.add_argument("--target-dbfs", type=float, default=-25.0)
    args = ap.parse_args()

    inbox, out_dir = Path(args.inbox), Path(args.out)
    if inbox.resolve() == out_dir.resolve():
        print("inbox == out：拒绝原地清洗（会毁原件）", file=sys.stderr)
        return 2
    wavs = sorted(inbox.rglob("*.wav"))
    if not wavs:
        print(f"无 wav: {inbox}", file=sys.stderr)
        return 2

    ops = {"trim": not args.no_trim, "hp": not args.no_hp,
           "loudness": not args.no_loudness, "trim_db": args.trim_db,
           "hp_hz": args.hp_hz, "target_dbfs": args.target_dbfs,
           "denoise": args.denoise, "cut": args.cut}

    # 第一遍：外部模型段（denoise 或 VAD）批量跑，再逐文件 DSP。
    segmap = {}
    if args.cut:
        segmap = _vad_batch(wavs)
        if not segmap:
            print("[cut] VAD 无产出，全部按未剪切处理", file=sys.stderr)

    rows, dn_pairs = [], []
    tmp_dir = None
    if args.denoise:
        tmp_dir = Path(tempfile.mkdtemp(prefix="denoise_"))
        for w in wavs:
            rel = w.relative_to(inbox)
            dn_pairs.append({"in": str(w),
                             "out": str(tmp_dir / rel)})
        print(f"[preprocess] {len(dn_pairs)} 条送 ZipEnhancer…")
        dn_ok = _denoise_batch(dn_pairs)
    else:
        dn_ok = {}

    import numpy as np
    from scipy.signal import resample_poly
    import math
    for w in wavs:
        rel = w.relative_to(inbox)
        dst = out_dir / rel
        src_for_dsp = w
        dn_path = (tmp_dir / rel) if tmp_dir else None
        if args.denoise and dn_path and dn_ok.get(str(dn_path)):
            # 16k denoise 输出 → 回读重采样到 48k 再进 DSP 链
            import soundfile as sf
            x16, sr16 = sf.read(str(dn_path), always_2d=False)
            g = math.gcd(sr16, DEFAULT_SR)
            x = resample_poly(np.asarray(x16, dtype=np.float32),
                              DEFAULT_SR // g, sr16 // g)
            out, _st = preprocess_wav(x, DEFAULT_SR, **{
                k: v for k, v in ops.items() if k != "denoise"})
            dst.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst), out, DEFAULT_SR, subtype="PCM_16")
            row = {"source": str(w), "output": str(dst),
                   "duration_in": round(len(x) / DEFAULT_SR, 3),
                   "duration_out": round(len(out) / DEFAULT_SR, 3),
                   "ops": {k: v for k, v in ops.items() if k != "denoise"},
                   "denoise": True,
                   "bandwidth_note": "denoise: 模型输出16k,有效带宽≤8kHz"}
        else:
            row = preprocess_file(w, dst, ops=ops,
                                  segments_ms=segmap.get(str(w)))
            if args.cut and str(w) not in segmap:
                row["cut"] = {"error": "vad_missing"}
            if args.denoise:
                row["denoise"] = False
                row["denoise_error"] = "denoise 未产出，已按未降噪处理"
        rows.append(row)

    if tmp_dir:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "preprocess_report.jsonl"
    report.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                              for r in rows), encoding="utf-8")
    n_dn = sum(1 for r in rows if r.get("denoise"))
    n_cut = sum(1 for r in rows
                if isinstance(r.get("cut"), dict)
                and r["cut"].get("segments_kept"))
    n_flag = sum(1 for r in rows
                 if isinstance(r.get("cut"), dict)
                 and r["cut"].get("all_dropped"))
    print(f"[preprocess] {len(rows)} 条 → {out_dir}  "
          f"cut={n_cut} all_dropped={n_flag} denoise={n_dn}  "
          f"report={report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
