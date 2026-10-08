"""Protocol-speaking fake worker for manager/client tests.

Stdlib only. Behaviour knobs via env:
  FAKE_BACKEND_ID       backend_id reported by health (default "fake")
  FAKE_UNSUPPORTED      comma-separated capability keys forced to false
  FAKE_CRASH_ON         method name that triggers os._exit(2)
  FAKE_NO_HEALTH        when set, health never answers (startup timeout test)
  FAKE_ROUNDTRIP_OK     when set, codec_roundtrip copies input to output
Generate writes a small 16-bit PCM sine wav (wave module, no deps).
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import wave

_OUT = sys.stdout
sys.stdout = sys.stderr


def _write(obj):
    _OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    _OUT.flush()


def _ok(req_id, result):
    _write({"id": req_id, "ok": True, "result": result})


def _err(req_id, message):
    _write({"id": req_id, "ok": False,
            "error": {"type": "test", "message": message}})


def _maybe_crash(method):
    if os.environ.get("FAKE_CRASH_ON") == method:
        os._exit(2)


def _maybe_slow(method):
    """FAKE_SLOW_MS=<ms> sleeps that long before answering a method
    listed in FAKE_SLOW_ON (default: generate)."""
    ms = int(os.environ.get("FAKE_SLOW_MS") or 0)
    if not ms:
        return
    targets = os.environ.get("FAKE_SLOW_ON", "generate").split(",")
    if method in (t.strip() for t in targets):
        import time
        time.sleep(ms / 1000.0)


def _write_sine(path, seconds=0.2, sr=8000, freq=440.0):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    n = int(sr * seconds)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(16000 * math.sin(2 * math.pi * freq * i / sr)))
            for i in range(n)
        )
        w.writeframes(frames)


def _capabilities():
    caps = {
        "zero_shot": True, "codec_roundtrip": True, "fine_tune": False,
        "lora": False, "voice_design": False, "streaming": False,
    }
    for key in (os.environ.get("FAKE_UNSUPPORTED") or "").split(","):
        key = key.strip()
        if key in caps:
            caps[key] = False
    return caps


def handle(method, params):
    _maybe_crash(method)
    _maybe_slow(method)
    if method == "health":
        if os.environ.get("FAKE_NO_HEALTH"):
            import time
            while True:
                time.sleep(60)
        return {
            "backend_id": os.environ.get("FAKE_BACKEND_ID", "fake"),
            "backend_version": "fake 0.0.1",
            "model_id": "fake/model",
            "device": "cpu",
            "dtype": "float32",
            "loaded": True,
            "vram": None,
            "capabilities": _capabilities(),
        }
    if method == "capabilities":
        return _capabilities()
    if method == "generate":
        out = params["output_path"]
        _write_sine(out)
        return {"sample_rate": 8000, "duration": 0.2,
                "output_path": out, "wall_seconds": 0.01,
                "metadata": {"seed": params.get("seed")}}
    if method == "codec_roundtrip":
        if not os.environ.get("FAKE_ROUNDTRIP_OK"):
            return {"status": "unsupported",
                    "reason": "fake worker has no codec"}
        import shutil
        shutil.copy2(params["audio_path"], params["output_path"])
        return {"status": "ok", "sample_rate": 8000,
                "output_path": params["output_path"]}
    if method == "shutdown":
        return {"stopped": True}
    raise ValueError(f"unknown method {method}")


def main():
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            _err(None, "bad json")
            continue
        try:
            result = handle(req.get("method"), req.get("params") or {})
        except Exception as exc:
            _err(req.get("id"), str(exc))
            continue
        _ok(req.get("id"), result)
        if req.get("method") == "shutdown":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
