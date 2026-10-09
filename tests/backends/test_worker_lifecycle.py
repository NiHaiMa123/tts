from __future__ import annotations

import pytest

import threading
import time

from character_tts.backends import protocol
from character_tts.backends.manager import (
    BackendBusyError,
    BackendManager,
    WorkerStartError,
)


def test_launch_health_stop(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        health = mgr.start(profile_factory("fake_a"))
        assert health["backend_id"] == "fake"
        assert mgr.is_alive()
        assert mgr.active_backend_id == "fake_a"
    finally:
        mgr.stop()
    assert not mgr.is_alive()
    assert mgr.active_backend_id is None


def test_graceful_shutdown_reaps_process(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    mgr.start(profile_factory("fake_a"))
    proc = mgr._proc
    mgr.stop()
    assert proc.poll() is not None  # process actually exited


def test_backend_switch(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        mgr.start(profile_factory("fake_a", env={"FAKE_BACKEND_ID": "a"}))
        first = mgr._proc
        mgr.switch(profile_factory("fake_b", env={"FAKE_BACKEND_ID": "b"}))
        assert first.poll() is not None  # old worker fully exited
        assert mgr.active_backend_id == "fake_b"
        assert mgr.is_alive()
    finally:
        mgr.stop()


def test_ensure_restarts_crashed_worker(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        profile = profile_factory("fake_a", env={"FAKE_CRASH_ON": "generate"})
        client = mgr.ensure(profile)
        with pytest.raises(protocol.WorkerError):
            client.generate(text="x", output_path=str(tmp_path / "o.wav"),
                            timeout=10)
        # worker died mid-call; ensure() must recover a fresh one
        client2 = mgr.ensure(profile_factory(
            "fake_a", env={"FAKE_CRASH_ON": "never"}))
        health = client2.health(timeout=10)
        assert health["backend_id"] == "fake"
    finally:
        mgr.stop()


def test_unsupported_capability_reported(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        mgr.start(profile_factory(
            "fake_a", env={"FAKE_UNSUPPORTED": "codec_roundtrip,lora"}))
        caps = mgr._client.capabilities(timeout=10)
        assert caps["codec_roundtrip"] is False
        assert caps["lora"] is False
        assert caps["zero_shot"] is True
    finally:
        mgr.stop()


def test_startup_timeout(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    with pytest.raises(WorkerStartError):
        mgr.start(profile_factory(
            "fake_a", env={"FAKE_NO_HEALTH": "1"}, startup_timeout=3))
    assert not mgr.is_alive()


def test_generate_gate_rejects_concurrent(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        mgr.start(profile_factory("fake_a"))
        assert mgr.acquire_generate() is True
        assert mgr.acquire_generate() is False  # second caller rejected
        mgr.release_generate()
        assert mgr.acquire_generate() is True
        mgr.release_generate()
    finally:
        mgr.stop()


def _generate_on_thread(mgr, profile, out_path, holder):
    """Mirror the real call path: gate held, then ensure+generate."""
    def run():
        try:
            if not mgr.acquire_generate():
                holder["error"] = "gate busy"
                return
            try:
                client = mgr.ensure(profile)
                holder["result"] = client.generate(
                    text="hi", output_path=str(out_path), timeout=30)
            finally:
                mgr.release_generate()
        except Exception as exc:  # noqa: BLE001 - surfaced via holder
            holder["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_switch_and_stop_refused_while_generating(tmp_path,
                                                  profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    holder: dict = {}
    profile = profile_factory(
        "fake_a", env={"FAKE_SLOW_MS": "800", "FAKE_SLOW_ON": "generate"})
    try:
        client = mgr.ensure(profile)
        t = _generate_on_thread(mgr, profile, tmp_path / "g.wav", holder)
        # Wait until the generate call is actually in flight.
        deadline = time.monotonic() + 10
        while not mgr.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        assert mgr.busy
        with pytest.raises(BackendBusyError):
            mgr.switch(profile_factory("fake_b"))
        with pytest.raises(BackendBusyError):
            mgr.stop()
        with pytest.raises(BackendBusyError):
            mgr.ensure(profile_factory("fake_b"))
        t.join(timeout=30)
        assert not t.is_alive()
        assert "result" in holder  # generation completed unharmed
        assert mgr.active_backend_id == "fake_a"
        assert mgr.is_alive()
        # After release, switching works again.
        mgr.switch(profile_factory("fake_b"))
        assert mgr.active_backend_id == "fake_b"
    finally:
        mgr.stop()


def test_stop_wait_then_succeeds(tmp_path, profile_factory):
    mgr = BackendManager(logs_dir=tmp_path)
    holder: dict = {}
    profile = profile_factory(
        "fake_a", env={"FAKE_SLOW_MS": "400", "FAKE_SLOW_ON": "generate"})
    try:
        mgr.ensure(profile)
        t = _generate_on_thread(mgr, profile, tmp_path / "g.wav", holder)
        mgr.stop(wait_seconds=10)  # blocks until generation releases
        t.join(timeout=30)
        assert not mgr.is_alive()
    finally:
        mgr.stop()


def test_stderr_does_not_corrupt_protocol(tmp_path, profile_factory):
    """fake worker prints nothing, but real backends spam stderr — the
    pipe separation is what we verify here end-to-end."""
    mgr = BackendManager(logs_dir=tmp_path)
    try:
        mgr.start(profile_factory("fake_a"))
        out = tmp_path / "g.wav"
        res = mgr._client.generate(text="hi", output_path=str(out), timeout=15)
        assert out.is_file()
        assert res["sample_rate"] == 8000
    finally:
        mgr.stop()
