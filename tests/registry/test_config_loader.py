from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from character_tts.registry import loader
from character_tts.registry.models import ConfigError


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return p


def test_env_expansion_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_ROOT", "/x")
    data = {"a": "${MY_ROOT}/f", "b": "${MISSING:fallback}", "c": ["${MY_ROOT}"]}
    out = loader.expand_env(data)
    assert out == {"a": "/x/f", "b": "fallback", "c": ["/x"]}


def test_env_expansion_missing_raises(monkeypatch):
    monkeypatch.delenv("NOPE_VAR", raising=False)
    with pytest.raises(ConfigError, match="NOPE_VAR"):
        loader.expand_env({"a": "${NOPE_VAR}"})


def test_env_file_loaded(tmp_path):
    env_file = tmp_path / ".env.local"
    env_file.write_text("FOO=bar\n# comment\nBAZ=\"qux\"\n", encoding="utf-8")
    assert loader.load_env_file(env_file) == {"FOO": "bar", "BAZ": "qux"}
    assert loader.load_env_file(tmp_path / "missing") == {}


def test_character_requires_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ROOT", "/x")
    path = _write(tmp_path, "c.yaml", {
        "character_id": "c", "display_name": "C",
        "dataset": {}, "reference": {},
    })
    with pytest.raises(ConfigError, match="reference.audio"):
        loader.load_character(str(path))


def test_character_load_ok(tmp_path):
    path = _write(tmp_path, "c.yaml", {
        "character_id": "c", "display_name": "C",
        "dataset": {"train_manifest": "t.jsonl"},
        "reference": {"audio": "a.wav", "text": "hi", "sha256": "0" * 64},
        "evaluation": {"anchor_texts": [{"id": "rain", "text": "t"}]},
    })
    c = loader.load_character(str(path))
    assert c.character_id == "c"
    assert c.anchor_texts[0].id == "rain"
    assert c.reference_sha256 == "0" * 64


def test_backend_validation(tmp_path):
    path = _write(tmp_path, "b.yaml", {"backend_id": "b", "worker": {}})
    with pytest.raises(ConfigError, match="python"):
        loader.load_backend(str(path))
    path = _write(tmp_path, "b.yaml", {
        "backend_id": "b",
        "worker": {"python": "p"},
        "model": {"path_or_id": "m"},
    })
    with pytest.raises(ConfigError, match="script"):
        loader.load_backend(str(path))


def test_evaluation_case_kinds(tmp_path):
    path = _write(tmp_path, "e.yaml", {
        "evaluation_id": "e", "character": "c",
        "cases": [{"backend": "b", "kind": "bogus", "output": "x.wav"}],
    })
    with pytest.raises(ConfigError, match="invalid kind"):
        loader.load_evaluation(str(path))


def test_repo_configs_parse(monkeypatch):
    """Shipped configs must validate with machine-local vars set."""
    monkeypatch.setenv("TTS_MODEL_ROOT", "hf_cache")
    for name in ("voxcpm2", "qwen3_tts"):
        b = loader.load_backend(name)
        assert b.backend_id == name
        assert b.worker_python
    ev = loader.load_evaluation("suoming_gate_v1")
    assert len(ev.cases) == 7
