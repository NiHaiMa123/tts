"""Optional DOM + real HTTP integration; no browser download required."""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import threading
import wave
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from character_tts.web.listen_page import (
    write_listen_page, write_lora_pilot_listen_page,
)

REPO = Path(__file__).resolve().parents[2]


def test_dom_controls_save_and_restore_real_local_files(tmp_path):
    node = shutil.which("node")
    module = os.environ.get("TTS_JSDOM_MODULE")
    if not node or not module:
        pytest.skip("set TTS_JSDOM_MODULE to an installed jsdom module")
    spec = importlib.util.spec_from_file_location(
        "dom_listening_server", REPO / "scripts/ingest/serve_review.py")
    server_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server_module)
    servers = []
    env = dict(os.environ)
    try:
        for mode in ("basic", "pairs"):
            bundle = tmp_path / mode
            bundle.mkdir()
            names = ["sample.wav"] if mode == "basic" else [
                "a.wav", "b.wav", "reference_clean.wav", "prompt_P0.wav"]
            for name in names:
                with wave.open(str(bundle / name), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(16000)
                    f.writeframes(b"\0\0" * 1600)
            if mode == "basic":
                write_listen_page(bundle, "测试")
            else:
                cases = [{"case_id": name, "status": "ok", "condition": name,
                          "output": name + ".wav", "output_sha256": hashlib.sha256(
                              (bundle / (name + ".wav")).read_bytes()).hexdigest()}
                         for name in ("a", "b")]
                write_lora_pilot_listen_page(bundle, {
                    "experiment_id": "dom_pairs", "backend": "test",
                    "model": {"revision": "test"}, "cases": cases,
                    "packs": [{"pack_id": "pack1", "title": "第一组",
                               "pairs": ["p1", "p2"], "collapsed": False}],
                    "pairs": [{"pair_id": pid, "text_id": "中文台词",
                               "seed": 42, "cells": ["a", "b"]}
                              for pid in ("p1", "p2")]})
            server = ThreadingHTTPServer(("127.0.0.1", 0),
                                          server_module.make_handler(bundle))
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            prefix = "TTS_DOM_BASIC" if mode == "basic" else "TTS_DOM_PAIRED"
            env[prefix + "_URL"] = f"http://127.0.0.1:{server.server_address[1]}/"
            env[prefix + "_FILE"] = str(bundle / "listen" / (
                "listen-ratings.json" if mode == "basic" else
                "listen-ratings-lora-pilot.json"))
        result = subprocess.run([node, str(Path(__file__).with_name(
            "listen_dom_smoke.cjs"))], env=env, capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "DOM + live HTTP verified" in result.stdout
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
