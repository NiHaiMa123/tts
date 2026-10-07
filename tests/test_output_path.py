from __future__ import annotations

import pytest

from character_tts.audio.io import safe_output_path


def test_safe_output_path_ok(tmp_path):
    p = safe_output_path(tmp_path, "dots/zero.wav")
    assert p.parent.is_dir()
    assert str(p).startswith(str(tmp_path.resolve()))


def test_safe_output_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        safe_output_path(tmp_path, "../evil.wav")
    with pytest.raises(ValueError):
        safe_output_path(tmp_path, "a/../../evil.wav")


def test_safe_output_path_rejects_absolute(tmp_path):
    with pytest.raises(ValueError):
        safe_output_path(tmp_path, "C:/Windows/evil.wav")
