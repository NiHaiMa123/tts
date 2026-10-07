from __future__ import annotations

import pytest

from character_tts.backends import protocol
from character_tts.backends.manager import (
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
