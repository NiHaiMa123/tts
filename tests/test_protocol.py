from __future__ import annotations

import io
import json

import pytest

from character_tts.backends import protocol
from character_tts.backends.worker_client import WorkerClient


def test_request_roundtrip():
    line = protocol.encode_request("generate", {"text": "hi"}, "r1")
    obj = protocol.decode_line(line)
    assert obj["id"] == "r1"
    assert obj["method"] == "generate"
    assert obj["params"] == {"text": "hi"}


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        protocol.encode_request("hack", {})


def test_event_and_response_classification():
    event = protocol.decode_line(protocol.encode_event("log", message="x"))
    assert protocol.is_event(event)
    assert not protocol.is_response(event)
    resp = protocol.decode_line(protocol.encode_result("a", {}))
    assert protocol.is_response(resp)
    err = protocol.decode_line(protocol.encode_error("a", "t", "m"))
    assert protocol.is_response(err)


def _pipe_pair():
    """(client stdin, client stdout) wired to a responder callable.

    The responder runs on a thread: it reads one request line from the
    worker-stdin side and writes lines back to the worker-stdout side.
    """
    import os
    import threading

    r_in, w_in = os.pipe()     # client -> worker
    r_out, w_out = os.pipe()   # worker -> client
    return (
        os.fdopen(w_in, "wb"),
        os.fdopen(r_out, "rb"),
        os.fdopen(r_in, "rb"),
        os.fdopen(w_out, "wb"),
        threading,
    )


def _respond_once(req_r, resp_w, replies_fn):
    """Read one request line, run replies_fn(obj)->list[str], write them."""
    line = req_r.readline()
    obj = json.loads(line.decode())
    for reply in replies_fn(obj):
        resp_w.write(reply.encode() + b"\n")
    resp_w.flush()


def test_client_skips_events_and_returns_result():
    import json as _json
    import threading

    stdin, stdout, req_r, resp_w, _ = _pipe_pair()
    events = []
    client = WorkerClient(stdin, stdout, on_event=events.append)

    def replies(req):
        return [
            protocol.encode_event("log", message="loading"),
            protocol.encode_result("WRONG_ID", {}),      # stray id, skipped
            protocol.encode_result(req["id"], {"v": 1}),
        ]

    t = threading.Thread(target=_respond_once,
                         args=(req_r, resp_w, replies), daemon=True)
    t.start()
    result = client.health(timeout=10)
    t.join(timeout=5)
    assert result == {"v": 1}
    assert events and events[0]["event"] == "log"


def test_client_raises_worker_error():
    import threading

    stdin, stdout, req_r, resp_w, _ = _pipe_pair()
    client = WorkerClient(stdin, stdout)

    t = threading.Thread(
        target=_respond_once,
        args=(req_r, resp_w,
              lambda req: [protocol.encode_error(req["id"], "internal", "boom")]),
        daemon=True)
    t.start()
    with pytest.raises(protocol.WorkerError, match="boom"):
        client.health(timeout=10)


def test_client_eof_raises():
    client = WorkerClient(io.BytesIO(), io.BytesIO(b""))
    with pytest.raises(protocol.WorkerError):
        client.health(timeout=5)


def test_call_serializes_concurrent_callers():
    import threading

    stdin, stdout, req_r, resp_w, _ = _pipe_pair()
    client = WorkerClient(stdin, stdout)

    # Responder answers every request after a tiny delay.
    def serve():
        while True:
            line = req_r.readline()
            if not line:
                return
            obj = json.loads(line.decode())
            resp_w.write(
                (protocol.encode_result(obj["id"], {"m": obj["method"]})
                 + "\n").encode())
            resp_w.flush()

    threading.Thread(target=serve, daemon=True).start()
    results = {}
    threads = [
        threading.Thread(
            target=lambda m=m: results.__setitem__(
                m, client.call(m, timeout=10)),
            daemon=True)
        for m in ("health", "capabilities")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert results == {"health": {"m": "health"},
                       "capabilities": {"m": "capabilities"}}

