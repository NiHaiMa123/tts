"""Disk evidence, stale experiments and write failures in local listening."""
from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from character_tts.web import listen_store
from character_tts.web.listen_page import write_listen_page

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def bundle(tmp_path):
    output = tmp_path / "run_a"
    output.mkdir()
    with wave.open(str(output / "sample.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\0\0" * 1600)
    write_listen_page(output, "测试")
    return output


def request_for(bundle, value="4", revision=None):
    c = listen_store.contract_for(bundle)
    state = {"identity": c["identity"], "ratings": {
        "sample.wav": {"likeness": value}}, "session": {}}
    return {"name": c["ratings_name"], "identity": c["identity"],
            "expected_revision": revision, "state": state,
            "data": {"identity": c["identity"],
                     "experiment_id": c["experiment_id"],
                     "ratings": {"sample.wav": {"likeness": value,
                         "output_sha256": c["asset_hashes"]["sample.wav"]}}}}


def test_saved_choices_are_agent_readable_and_clear_returns_pending(bundle):
    assert listen_store.agent_review(bundle)["status"] == "PENDING_USER_LISTENING"
    saved = listen_store.save_snapshot(bundle, request_for(bundle))
    result = listen_store.agent_review(bundle)
    assert result["status"] == "RATED"
    assert result["data"]["ratings"]["sample.wav"]["likeness"] == "4"
    assert result["review"]["rated_count"] == 1
    empty = request_for(bundle, value="", revision=saved["revision"])
    empty["state"]["ratings"]["sample.wav"]["rated_at"] = "now"
    listen_store.save_snapshot(bundle, empty)
    assert listen_store.agent_review(bundle)["status"] == "PENDING_USER_LISTENING"


def test_write_failure_keeps_previous_rating_snapshot(bundle, monkeypatch):
    saved = listen_store.save_snapshot(bundle, request_for(bundle))
    path = Path(saved["path"])
    before = path.read_bytes()
    def fail_replace(source, target):
        raise OSError("disk full")
    monkeypatch.setattr(listen_store.os, "replace", fail_replace)
    with pytest.raises(OSError):
        listen_store.save_snapshot(bundle, request_for(bundle, "2", saved["revision"]))
    assert path.read_bytes() == before
    assert listen_store.read_snapshot(bundle, path.name)["state"]["ratings"]["sample.wav"]["likeness"] == "4"
    assert not list(path.parent.glob("tmp*"))


def test_old_tab_cannot_overwrite_newer_choices(bundle):
    stale = request_for(bundle, "1")
    listen_store.save_snapshot(bundle, request_for(bundle, "5"))
    with pytest.raises(listen_store.ReviewError, match="another page") as err:
        listen_store.save_snapshot(bundle, stale)
    assert err.value.code == 409
    assert listen_store.agent_review(bundle)["data"]["ratings"]["sample.wav"]["likeness"] == "5"


def test_sha_bound_legacy_choices_restore_without_manual_export(bundle):
    contract = listen_store.contract_for(bundle)
    path = bundle / "listen" / contract["ratings_name"]
    old = {"experiment_id": contract["experiment_id"], "ratings": {
        "sample.wav": {"likeness": "5", "output_sha256": contract["asset_hashes"]["sample.wav"]},
        "other.wav": {"likeness": "1", "output_sha256": "unrelated"}}}
    path.write_text(json.dumps(old))
    snapshot = listen_store.read_snapshot(bundle, path.name)
    assert snapshot["state"]["ratings"]["sample.wav"]["likeness"] == "5"
    assert "other.wav" not in snapshot["state"]["ratings"]
    assert listen_store.agent_review(bundle)["status"] == "RATED"
    assert json.loads(path.read_text()) == old  # migration read preserves original
    listen_store.save_snapshot(bundle, request_for(bundle))
    archived = list((path.parent / "history" / path.stem).glob("*.json"))
    assert any(json.loads(p.read_text()) == old for p in archived)


def test_changed_audio_blocks_old_ratings_and_archives_on_new_page(bundle):
    old_request = request_for(bundle)
    first = listen_store.save_snapshot(bundle, old_request)
    wav = bundle / "sample.wav"
    wav.write_bytes(wav.read_bytes() + b"changed")
    assert listen_store.agent_review(bundle)["status"] == "BLOCKED"
    with pytest.raises(listen_store.ReviewError, match="audio changed"):
        listen_store.save_snapshot(bundle, request_for(bundle, revision=first["revision"]))
    write_listen_page(bundle, "新音频")
    with pytest.raises(listen_store.ReviewError, match="experiment changed"):
        listen_store.save_snapshot(bundle, old_request)
    listen_store.save_snapshot(bundle, request_for(bundle, "2", first["revision"]))
    archives = list((bundle / "listen/history/listen-ratings").glob("*.json"))
    assert any(json.loads(p.read_text()).get("ratings", {}).get("sample.wav", {}).get("likeness") == "4" for p in archives)
    assert listen_store.agent_review(bundle)["status"] == "RATED"


def test_template_escapes_script_data_and_has_no_export_fallback(bundle):
    wav = bundle / "sample.wav"
    wav.with_suffix(".wav.json").write_text(json.dumps({"seed": "</script>"}))
    page = write_listen_page(bundle, "测试<script>").read_text()
    assert "exportRatings" not in page and "a.download" not in page
    assert "file:" in page and "serve_review.py" in page
    assert 'name="viewport"' in page
    assert "__DATA__" not in page
    assert page.count("</script>") == 1


def test_http_acknowledges_only_successful_disk_writes(bundle, monkeypatch):
    spec = importlib.util.spec_from_file_location("listen_http_test", REPO / "scripts/ingest/serve_review.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(bundle))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert "<h1>测试</h1>" in urllib.request.urlopen(url).read().decode()
        req = urllib.request.Request(url + "/api/listen-ratings", data=json.dumps(request_for(bundle)).encode(),
                                     headers={"Content-Type": "application/json"})
        result = json.loads(urllib.request.urlopen(req).read())
        assert result["ok"] and Path(result["path"]).is_file()
        def fail(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(mod, "save_snapshot", fail)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req)
        assert err.value.code == 500
        assert json.loads(err.value.read())["ok"] is False
    finally:
        server.shutdown()
        server.server_close()
