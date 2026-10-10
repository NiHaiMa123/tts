"""FSMN-VAD 语音段检测——在 backend_envs/asr_sensevoice 内运行（勿手调）.

由 preprocess_audio.py --cut 经子进程调用：
    asr_sensevoice/python _vad_fsmn.py <files.jsonl> <out.jsonl>

files.jsonl 每行 {"in": "<wav>"}；输出每行 {"in": "...", "segments":
[[start_ms, end_ms], ...], "error": null}。模型只加载一次，CPU 即可。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "MODELSCOPE_CACHE",
    str(Path(__file__).resolve().parents[2] / "models" / "vad"))


def main() -> int:
    files = [json.loads(l) for l in
             Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
             if l.strip()]
    out_path = Path(sys.argv[2])
    from funasr import AutoModel
    model = AutoModel(
        model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        model_revision="v2.0.4", disable_update=True)
    ok = 0
    with out_path.open("w", encoding="utf-8") as f:
        for item in files:
            row = {"in": item["in"], "segments": [], "error": None}
            try:
                res = model.generate(input=item["in"])
                if res and "value" in res[0]:
                    row["segments"] = res[0]["value"]
                ok += 1
            except Exception as exc:
                row["error"] = str(exc)[:300]
                print(f"[vad] FAIL {item['in']}: {exc}", file=sys.stderr)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[vad] {ok}/{len(files)} done")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
