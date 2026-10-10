"""音频预处理段：入库前的清洗（剪切 / 高通 / 响度归一）。

游戏解包音（Wwise wem→wav）常带底噪、编码伪影、句间噪声段和响度
不一致——LoRA 会把这些声学签名一并学走（用户听感"磨砂"）。本段在
standardize 之前对 inbox 做无损保留原件的镜像清洗：

    inbox/ ──preprocess──> inbox_clean/   （子目录结构保留）

操作顺序：highpass → cut（VAD 切段 + 逐段质检 + 拼接）→ loudness。

cut 是主操作：外部 VAD（FSMN-VAD，asr env 子进程）给出语音段，
本模块把「非语音段 + 过短爆音段 + 削波失真段」整段删除，保留下来的
语音段之间截断原静音缝再拼接（保留自然停顿，≤keep_gap_ms），边界
做短余弦淡入淡出防爆音。全段都被丢弃时退回原音频并记
``all_dropped`` flag 交下游人工判。

denoise（ZipEnhancer ANS）已被弃用：输出 16k 有效带宽 ≤8kHz 会抹掉
音色细节，且实测"磨砂"并非底噪而是数据声学签名——剪切是替代方案。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_SR = 48000


def trim_silence(wav: np.ndarray, sr: int, gate_db: float = -45.0,
                 pad_ms: int = 80, win_ms: int = 10) -> np.ndarray:
    """Energy-gate leading/trailing silence; keeps ``pad_ms`` padding."""
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = max(1, int(sr * win_ms / 1000))
    frames = len(x) // n
    if frames < 3:
        return x
    rms = np.sqrt((x[:frames * n].reshape(frames, n) ** 2).mean(axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-9))
    active = np.flatnonzero(db > gate_db)
    if not len(active):
        return x                       # 全静音：不动，交给下游判 bad
    pad = int(sr * pad_ms / 1000)
    lo = max(0, active[0] * n - pad)
    hi = min(len(x), (active[-1] + 1) * n + pad)
    return x[lo:hi]


def highpass(wav: np.ndarray, sr: int, cutoff_hz: float = 70.0,
             order: int = 4) -> np.ndarray:
    """Butterworth high-pass — removes rumble/DC drift."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    x = np.asarray(wav, dtype=np.float32)
    return sosfiltfilt(sos, x).astype(np.float32)


def normalize_loudness(wav: np.ndarray, target_rms_dbfs: float = -25.0,
                       peak_ceiling: float = 0.98) -> np.ndarray:
    """RMS normalize; gain is clipped so peaks never exceed ceiling."""
    x = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    if rms < 1e-9:
        return x
    cur_db = 20 * math.log10(rms)
    gain = 10 ** ((target_rms_dbfs - cur_db) / 20)
    peak = float(np.abs(x).max())
    if peak * gain > peak_ceiling:
        gain = peak_ceiling / peak
    return (x * gain).astype(np.float32)


def _fade_edges(x: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
    """10ms 余弦淡入淡出——拼接边界防爆音（不改动中段）。"""
    n = min(len(x) // 2, int(sr * fade_ms / 1000))
    if n < 2:
        return x
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    x = x.copy()
    x[:n] *= np.sin(t) ** 2
    x[-n:] *= np.cos(t) ** 2
    return x


def cut_segments(wav: np.ndarray, sr: int, segments_ms,
                 *, keep_gap_ms: int = 300, fade_ms: int = 10,
                 min_seg_ms: int = 150, clip_level: float = 0.98,
                 clip_ratio: float = 0.02) -> tuple[np.ndarray, dict]:
    """按 VAD 语音段剪切：丢非语音/过短/削波段，截断长静音缝后拼接。

    ``segments_ms``: ``[[start_ms, end_ms], ...]``（FSMN-VAD 输出）。
    每段质检：时长 < ``min_seg_ms``（孤立爆音）或
    |x|>``clip_level`` 样本占比 > ``clip_ratio``（削波失真）→ 丢弃。
    相邻保留段之间的原静音缝保留 ≤``keep_gap_ms``（自然停顿）。
    返回 ``(audio, stats)``；全部丢弃时返回原音频 + ``all_dropped``。
    """
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    segs = [[max(0, int(s * sr / 1000)), min(len(x), int(e * sr / 1000))]
            for s, e in (segments_ms or [])]
    segs = [s for s in segs if s[1] > s[0]]
    if not segs:
        return x, {"segments_in": 0, "segments_kept": 0,
                   "dropped": [], "removed_s": 0.0,
                   "all_dropped": True}
    kept, dropped = [], []
    for s_i, e_i in segs:
        seg = x[s_i:e_i]
        dur_ms = (e_i - s_i) * 1000 / sr
        cr = float((np.abs(seg) > clip_level).mean()) if len(seg) else 0.0
        if dur_ms < min_seg_ms:
            dropped.append({"ms": [round(s_i * 1000 / sr),
                                   round(e_i * 1000 / sr)],
                            "reason": "too_short"})
        elif cr > clip_ratio:
            dropped.append({"ms": [round(s_i * 1000 / sr),
                                   round(e_i * 1000 / sr)],
                            "reason": "clipped", "clip_ratio": round(cr, 3)})
        else:
            kept.append((s_i, e_i))
    if not kept:
        return x, {"segments_in": len(segs), "segments_kept": 0,
                   "dropped": dropped, "removed_s": 0.0,
                   "all_dropped": True}
    max_gap = int(sr * keep_gap_ms / 1000)
    pieces = [_fade_edges(x[kept[0][0]:kept[0][1]], sr, fade_ms)]
    prev_end = kept[0][1]
    for s_i, e_i in kept[1:]:
        gap = x[prev_end:s_i]
        if len(gap) > max_gap:
            gap = gap[:max_gap]
        if len(gap):
            pieces.append(_fade_edges(gap, sr, fade_ms))
        pieces.append(_fade_edges(x[s_i:e_i], sr, fade_ms))
        prev_end = e_i
    out = np.concatenate(pieces) if pieces else x
    removed_s = round(max(0.0, (len(x) - len(out)) / sr), 3)
    return out, {"segments_in": len(segs), "segments_kept": len(kept),
                 "dropped": dropped, "removed_s": removed_s,
                 "all_dropped": False}


def preprocess_wav(wav: np.ndarray, sr: int, *,
                   trim: bool = True, hp: bool = True,
                   loudness: bool = True,
                   trim_db: float = -45.0, hp_hz: float = 70.0,
                   target_dbfs: float = -25.0,
                   segments_ms=None,
                   cut: bool = False) -> tuple[np.ndarray, dict | None]:
    """In-memory pipeline: hp → cut(VAD 段) → trim → loudness.

    ``cut=True`` 且给了 ``segments_ms`` 时剪切替代首尾 trim。
    返回 ``(audio, cut_stats)``；cut 未启用时 stats 为 None。
    """
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    stats = None
    if hp:
        x = highpass(x, sr, cutoff_hz=hp_hz)
    if cut and segments_ms is not None:
        x, stats = cut_segments(x, sr, segments_ms)
    elif trim:
        x = trim_silence(x, sr, gate_db=trim_db)
    if loudness:
        x = normalize_loudness(x, target_rms_dbfs=target_dbfs)
    return x, stats


def preprocess_file(in_path: Path, out_path: Path, *,
                    ops: dict | None = None,
                    segments_ms=None,
                    sr: int = DEFAULT_SR) -> dict:
    """Load → preprocess → write PCM16. Returns a per-file report row."""
    ops = ops or {}
    wav, in_sr = sf.read(str(in_path), always_2d=False)
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if in_sr != sr:                     # 统一到管线采样率
        from scipy.signal import resample_poly
        import math as _m
        g = _m.gcd(in_sr, sr)
        x = resample_poly(x, sr // g, in_sr // g).astype(np.float32)
    in_dur = len(x) / sr
    out, cut_stats = preprocess_wav(
        x, sr, **{k: v for k, v in ops.items()
                  if k in ("trim", "hp", "loudness", "trim_db", "hp_hz",
                           "target_dbfs", "cut")},
        segments_ms=segments_ms)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), out, sr, subtype="PCM_16")
    return {
        "source": str(in_path), "output": str(out_path),
        "duration_in": round(in_dur, 3),
        "duration_out": round(len(out) / sr, 3),
        "ops": {k: v for k, v in ops.items() if k != "denoise"},
        "cut": cut_stats,
        "denoise": bool(ops.get("denoise")),
        "bandwidth_note": ("denoise: 模型输出16k,有效带宽≤8kHz"
                           if ops.get("denoise") else None),
    }
