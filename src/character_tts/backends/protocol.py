"""Worker protocol schema shared by the platform client and workers.

Transport: JSONL over stdin/stdout (one JSON object per line).

Request (platform -> worker)::

    {"id": "<req-id>", "method": "<method>", "params": {...}}

Response (worker -> platform)::

    {"id": "<req-id>", "ok": true,  "result": {...}}
    {"id": "<req-id>", "ok": false, "error": {"type": ..., "message": ..., "traceback": ...}}

Unsolicited worker events (no "id")::

    {"event": "log",      "level": "info", "message": "..."}
    {"event": "progress", "stage": "...",  "fraction": 0.5}

Methods: health, capabilities, generate, codec_roundtrip, shutdown.

Binary audio is never embedded in JSON; workers write WAV files to
``output_path`` and return metadata only.

This module is stdlib-only on purpose: it must also be importable from
backend worker environments that do not have ``character_tts`` installed.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

METHOD_HEALTH = "health"
METHOD_CAPABILITIES = "capabilities"
METHOD_GENERATE = "generate"
METHOD_CODEC_ROUNDTRIP = "codec_roundtrip"
METHOD_CODEC_PROBE = "codec_probe"
METHOD_SHUTDOWN = "shutdown"

ALL_METHODS = (
    METHOD_HEALTH,
    METHOD_CAPABILITIES,
    METHOD_GENERATE,
    METHOD_CODEC_ROUNDTRIP,
    METHOD_CODEC_PROBE,
    METHOD_SHUTDOWN,
)

# codec_roundtrip result status values
STATUS_OK = "ok"
STATUS_UNSUPPORTED = "unsupported"

CAPABILITY_KEYS = (
    "zero_shot",
    "codec_roundtrip",
    "fine_tune",
    "lora",
    "voice_design",
    "streaming",
)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def encode_request(method: str, params: dict[str, Any] | None = None,
                   request_id: str | None = None) -> str:
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method: {method}")
    return json.dumps(
        {"id": request_id or new_request_id(), "method": method,
         "params": params or {}},
        ensure_ascii=False,
    )


def encode_result(request_id: str, result: dict[str, Any]) -> str:
    return json.dumps({"id": request_id, "ok": True, "result": result},
                      ensure_ascii=False)


def encode_error(request_id: str, error_type: str, message: str,
                 traceback_text: str | None = None) -> str:
    error: dict[str, Any] = {"type": error_type, "message": message}
    if traceback_text:
        error["traceback"] = traceback_text
    return json.dumps({"id": request_id, "ok": False, "error": error},
                      ensure_ascii=False)


def encode_event(event: str, **fields: Any) -> str:
    payload: dict[str, Any] = {"event": event}
    payload.update(fields)
    return json.dumps(payload, ensure_ascii=False)


def decode_line(line: str) -> dict[str, Any]:
    """Parse one JSONL line into a dict; raises ValueError on bad input."""
    obj = json.loads(line)
    if not isinstance(obj, dict):
        raise ValueError(f"protocol line is not a JSON object: {line[:200]}")
    return obj


def is_response(obj: dict[str, Any]) -> bool:
    return "id" in obj and ("ok" in obj or "result" in obj or "error" in obj)


def is_event(obj: dict[str, Any]) -> bool:
    return "event" in obj and "id" not in obj


class WorkerError(RuntimeError):
    """Raised by the client when the worker returns ok=false."""

    def __init__(self, error_type: str, message: str,
                 traceback_text: str | None = None):
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.traceback_text = traceback_text


class WorkerProtocolError(RuntimeError):
    """Raised when the worker emits malformed protocol data."""
