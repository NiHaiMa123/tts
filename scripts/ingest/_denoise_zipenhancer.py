"""ZipEnhancer 批处理——在 backend_envs/voxcpm2 内运行（勿手调）.

由 preprocess_audio.py --denoise 经子进程调用：
    voxcpm2/python _denoise_zipenhancer.py <pairs.jsonl>

pairs.jsonl 每行 {"in": "...", "out": "..."}；模型只加载一次。
绕过 modelscope Pipeline.__call__ 的输入归一化（torchcodec 在此环境
DLL 加载失败），直接调 preprocess/forward/postprocess 三段。
输出 16kHz 单声道（模型采样率），上层负责重采样。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    import torch
    pairs = [json.loads(l) for l in
             Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    p = pipeline(Tasks.acoustic_noise_suppression,
                 model="iic/speech_zipenhancer_ans_multiloss_16k_base")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p.model = p.model.to(dev).eval()
    ok = 0
    for i, pair in enumerate(pairs):
        try:
            raw = Path(pair["in"]).read_bytes()
            out = p.forward(p.preprocess(raw))
            Path(pair["out"]).parent.mkdir(parents=True, exist_ok=True)
            p.postprocess(out, output_path=pair["out"])
            ok += 1
        except Exception as exc:
            print(f"[denoise] FAIL {pair['in']}: {exc}", file=sys.stderr)
    print(f"[denoise] {ok}/{len(pairs)} done")
    return 0 if ok == len(pairs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
