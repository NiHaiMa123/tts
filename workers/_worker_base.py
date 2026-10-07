"""Stdlib-only JSONL worker loop shared by all backend workers.

Lives in workers/ next to each *_worker.py so a backend env only needs
this directory on sys.path — it never imports character_tts or any
platform package. Keep it dependency-free.

Protocol (one JSON object per line):
  request : {"id": ..., "method": ..., "params": {...}}
  result  : {"id": ..., "ok": true, "result": {...}}
  error   : {"id": ..., "ok": false, "error": {type, message, traceback}}
  event   : {"event": "log"|"progress", ...}

Anything a model library prints to stdout would corrupt the protocol, so
serve() rebinds sys.stdout to stderr and writes responses on the saved
real stdout.
"""

from __future__ import annotations

import json
import sys
import traceback

ALL_METHODS = (
    "health",
    "capabilities",
    "generate",
    "codec_roundtrip",
    "shutdown",
)


def _write_line(stream, obj) -> None:
    stream.write(json.dumps(obj, ensure_ascii=False) + "\n")
    stream.flush()


class WorkerServer:
    """Subclasses implement handler methods via ``handle_<method>``."""

    def __init__(self):
        self._out = sys.stdout
        # Rebind stdout -> stderr so stray library prints cannot corrupt the
        # protocol stream. Responses always go through self._out.
        sys.stdout = sys.stderr
        self._running = True

    # -- helpers for subclasses ------------------------------------------------
    def emit_log(self, message: str, level: str = "info") -> None:
        _write_line(self._out,
                    {"event": "log", "level": level, "message": message})

    def emit_progress(self, stage: str, fraction: float | None = None) -> None:
        _write_line(self._out,
                    {"event": "progress", "stage": stage, "fraction": fraction})

    def request_shutdown(self) -> None:
        self._running = False

    # -- main loop ---------------------------------------------------------------
    def serve(self) -> int:
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                _write_line(self._out, {
                    "id": None, "ok": False,
                    "error": {"type": "protocol",
                              "message": f"malformed line: {raw[:200]}"},
                })
                continue
            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            handler = getattr(self, f"handle_{method}", None)
            if method not in ALL_METHODS or handler is None:
                _write_line(self._out, {
                    "id": req_id, "ok": False,
                    "error": {"type": "unknown_method",
                              "message": f"unsupported method: {method}"},
                })
                continue
            try:
                result = handler(params)
            except Exception as exc:
                _write_line(self._out, {
                    "id": req_id, "ok": False,
                    "error": {
                        "type": "internal",
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                })
                continue
            _write_line(self._out, {"id": req_id, "ok": True,
                                    "result": result or {}})
            if method == "shutdown":
                break
        return 0
