"""Unit tests for VoxCPM2 worker long-text chunking.

``_split_text`` is pure logic — importable without torch/voxcpm.
Long single-shot continuation drifts off the reference voice, so the
worker sentence-packs text into <= chunk_chars chunks and re-anchors
each one on the shared prompt cache.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
from voxcpm_worker import VoxCPMWorker  # noqa: E402

split = VoxCPMWorker._split_text


def test_short_text_returns_single_chunk():
    assert split("你好。世界。", 100) == ["你好。世界。"]


def test_packs_sentences_up_to_limit():
    text = "甲" * 30 + "。" + "乙" * 30 + "。" + "丙" * 30 + "。"
    chunks = split(text, 70)
    # 31+31=62 fits, adding 31 more exceeds 70 -> 2 chunks
    assert chunks == ["甲" * 30 + "。" + "乙" * 30 + "。", "丙" * 30 + "。"]
    assert all(len(c) <= 70 for c in chunks)


def test_oversized_sentence_falls_back_to_commas():
    piece = "甲" * 40 + "，" + "乙" * 40 + "，" + "丙" * 40 + "。"
    chunks = split(piece, 60)
    assert chunks == ["甲" * 40 + "，", "乙" * 40 + "，", "丙" * 40 + "。"]


def test_unbreakable_piece_hard_cuts():
    chunks = split("字" * 150, 60)
    assert chunks == ["字" * 60, "字" * 60, "字" * 30]


def test_all_punctuation_classes_split():
    text = "一！二？三；四…五。"
    assert len(split(text, 2)) == 5
    assert "".join(split(text, 2)) == text


def test_empty_pieces_are_dropped():
    chunks = split("好。。。", 10)
    assert chunks == ["好。。。"]


def test_roundtrip_preserves_text():
    text = "".join(f"第{i}句内容。" for i in range(20))
    assert "".join(split(text, 40)) == text


def test_whitespace_only_falls_back_to_original():
    assert split("   ", 10) == ["   "]
