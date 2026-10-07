"""Client side of the JSONL worker protocol.

Wraps the worker's stdin/stdout pipes. A daemon reader thread drains
stdout into a queue so requests can enforce real timeouts — a blocking
``readline()`` on a pipe can never be interrupted otherwise.

Responses are matched by request id; worker events are forwarded to an
optional callback.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any, Callable, IO

from . import protocol

_EOF = object()


class WorkerClient:
    def __init__(self, stdin: IO[bytes], stdout: IO[bytes],
                 on_event: Callable[[dict[str, Any]], None] | None = None):
        self._stdin = stdin
        self._stdout = stdout
        self._on_event = on_event
        self._closed = False
        self._queue: queue.Queue[Any] = queue.Queue()
        self._call_lock = threading.Lock()
        self._reader = threading.Thread(
            target=self._reader_loop, name="worker-reader", daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        while True:
            try:
                raw = self._stdout.readline()
            except (OSError, ValueError):
                raw = b""
            if not raw:
                self._queue.put(_EOF)
                return
            try:
                obj = protocol.decode_line(raw.decode("utf-8", "replace"))
            except (json.JSONDecodeError, ValueError) as exc:
                self._queue.put(protocol.WorkerProtocolError(
                    f"malformed worker line: {raw[:200]!r} ({exc})"))
                continue
            self._queue.put(obj)

    @property
    def closed(self) -> bool:
        return self._closed

    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout: float | None = None) -> dict[str, Any]:
        # Serialize calls: request/response ids only work when one call is
        # in flight at a time.
        with self._call_lock:
            return self._call_locked(method, params, timeout)

    def _call_locked(self, method: str, params: dict[str, Any] | None,
                     timeout: float | None) -> dict[str, Any]:
        if self._closed:
            raise protocol.WorkerError("closed", "client is closed")
        request_id = protocol.new_request_id()
        line = protocol.encode_request(method, params, request_id)
        try:
            self._stdin.write(line.encode("utf-8") + b"\n")
            self._stdin.flush()
        except OSError as exc:
            self._closed = True
            raise protocol.WorkerError(
                "io", f"worker stdin write failed: {exc}") from exc

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = (None if deadline is None
                         else max(0.0, deadline - time.monotonic()))
            if remaining == 0.0:
                raise TimeoutError(
                    f"worker did not respond to '{method}' within {timeout}s"
                )
            try:
                item = self._queue.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(
                    f"worker did not respond to '{method}' within {timeout}s"
                )
            if item is _EOF:
                self._closed = True
                raise protocol.WorkerError(
                    "eof", f"worker closed stdout while awaiting '{method}'"
                )
            if isinstance(item, protocol.WorkerProtocolError):
                raise item
            obj = item
            if protocol.is_event(obj):
                if self._on_event:
                    self._on_event(obj)
                continue
            if obj.get("id") != request_id:
                continue  # stray/late response for another id
            if obj.get("ok"):
                result = obj.get("result")
                return result if isinstance(result, dict) else {}
            error = obj.get("error") or {}
            raise protocol.WorkerError(
                str(error.get("type", "error")),
                str(error.get("message", "unknown worker error")),
                error.get("traceback"),
            )

    # -- convenience wrappers -------------------------------------------------
    def health(self, timeout: float | None = 30) -> dict[str, Any]:
        return self.call(protocol.METHOD_HEALTH, timeout=timeout)

    def capabilities(self, timeout: float | None = 30) -> dict[str, Any]:
        return self.call(protocol.METHOD_CAPABILITIES, timeout=timeout)

    def generate(self, *, text: str, output_path: str,
                 reference_audio: str | None = None,
                 reference_text: str | None = None,
                 seed: int | None = None,
                 options: dict[str, Any] | None = None,
                 timeout: float | None = None) -> dict[str, Any]:
        return self.call(
            protocol.METHOD_GENERATE,
            {
                "text": text,
                "output_path": output_path,
                "reference_audio": reference_audio,
                "reference_text": reference_text,
                "seed": seed,
                "options": options or {},
            },
            timeout=timeout,
        )

    def codec_roundtrip(self, *, audio_path: str, output_path: str,
                        timeout: float | None = None) -> dict[str, Any]:
        return self.call(
            protocol.METHOD_CODEC_ROUNDTRIP,
            {"audio_path": audio_path, "output_path": output_path},
            timeout=timeout,
        )

    def shutdown(self, timeout: float | None = 10) -> dict[str, Any]:
        try:
            return self.call(protocol.METHOD_SHUTDOWN, timeout=timeout)
        finally:
            self._closed = True

    def close(self) -> None:
        self._closed = True
