"""Import legacy dotstts character assets (verifies hashes, never copies).

Returns a dict describing the result; status is "ok" or "blocked" with a
problems list — missing data is reported, never fabricated.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

GATE_ANCHOR_TEXT = (
    "午后凉风拂过，雨云渐聚，细雨敲在屋檐上。我坐在窗边听雨，"
    "拭去玄朱锁上的薄尘。这确实是有些凄迷的场景，但……我很喜欢。"
)

PROFILE_REL = "configs/voices/suoming_step500_v1.yaml"
DATASET_REL = "datasets/suoming/v1"

DEFAULT_CANDIDATES = (
    Path(r"E:\project\dotstts"),
    Path(r"D:\project\dotstts"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dotstts(candidates: tuple[Path, ...] = DEFAULT_CANDIDATES) -> Path | None:
    env = os.environ.get("DOTSTTS_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in candidates:
        if (cand / "src" / "dots_tts").is_dir():
            return cand
    return None


def _rel(root: Path, path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else root / p


def _cfg_path(root: Path, path: Path) -> str:
    try:
        r = path.resolve().relative_to(root.resolve())
        return "${DOTSTTS_ROOT}/" + r.as_posix()
    except ValueError:
        return str(path)


def import_suoming(root: Path, profile_rel: str = PROFILE_REL,
                   dataset_rel: str = DATASET_REL) -> dict[str, Any]:
    """Verify legacy assets and build the character config dict."""
    problems: list[str] = []
    profile_path = _rel(root, profile_rel)
    if not profile_path.is_file():
        return {"status": "blocked",
                "problems": [f"voice profile missing: {profile_path}"]}
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    prompt = profile.get("prompt") or {}
    prompt_audio = _rel(root, str(prompt.get("audio_path", "")))
    reference: dict[str, Any] = {"text": prompt.get("text", "")}
    if prompt_audio.is_file():
        actual = sha256_file(prompt_audio)
        expected = prompt.get("audio_sha256")
        if expected and actual != expected:
            problems.append(
                f"reference sha256 mismatch: {actual} != {expected}")
        reference["audio"] = _cfg_path(root, prompt_audio)
        reference["sha256"] = actual
    else:
        problems.append(f"reference audio missing: {prompt_audio}")

    ds_root = _rel(root, dataset_rel)
    dataset: dict[str, Any] = {"source": "legacy_dotstts"}
    for name in ("train", "validation", "test"):
        mpath = ds_root / f"{name}.jsonl"
        if mpath.is_file():
            dataset[f"{name}_manifest"] = _cfg_path(root, mpath)
        else:
            problems.append(f"dataset manifest missing: {mpath}")
    if (ds_root / "manifest.json").is_file():
        dataset["root"] = _cfg_path(root, ds_root)
        dataset["manifest"] = _cfg_path(root, ds_root / "manifest.json")

    adapter = profile.get("adapter") or {}
    adapter_dir = _rel(root, str(adapter.get("path", "")))
    adapter_info: dict[str, Any] = {}
    if adapter_dir.is_dir():
        adapter_info["path"] = _cfg_path(root, adapter_dir)
        adapter_info["format"] = adapter.get("format")
        adapter_info["training_step"] = adapter.get("training_step")
        for fname, key in (("trainable_model.safetensors", "weights_sha256"),
                           ("trainable_model.json", "metadata_sha256")):
            f = adapter_dir / fname
            if f.is_file():
                adapter_info[key] = sha256_file(f)
            else:
                problems.append(f"adapter file missing: {f}")
    else:
        problems.append(f"adapter dir missing: {adapter_dir}")

    validation_anchor = None
    val_manifest = ds_root / "validation.jsonl"
    if val_manifest.is_file():
        first = val_manifest.read_text(encoding="utf-8").splitlines()
        if first:
            try:
                rec = json.loads(first[0])
                validation_anchor = {"audio": rec.get("audio"),
                                     "text": rec.get("text")}
            except json.JSONDecodeError:
                pass

    if problems:
        return {"status": "blocked", "problems": problems}

    return {
        "status": "ok",
        "root": str(root),
        "character": {
            "character_id": "suoming",
            "display_name": "锁暝",
            "dataset": dataset,
            "reference": reference,
            "validation_anchor": validation_anchor,
            "legacy_voice_profile": {
                "path": _cfg_path(root, profile_path),
                "model_id": profile.get("model_id"),
                "base_model": {
                    "path": _cfg_path(
                        root,
                        _rel(root, str((profile.get("base_model") or {})
                                       .get("path", "")))),
                    "revision": (profile.get("base_model") or {})
                    .get("revision"),
                },
                "adapter": adapter_info,
                "generation": profile.get("generation") or {},
                "runtime": profile.get("runtime") or {},
            },
            "evaluation": {
                "anchor_texts": [{"id": "rain", "text": GATE_ANCHOR_TEXT}],
            },
            "provenance": {
                "imported_from": str(root),
                "imported_at": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
                "importer": "scripts/import_legacy_assets.py",
            },
        },
    }


def write_character(result: dict[str, Any], out_dir: Path) -> Path:
    if result.get("status") != "ok":
        raise ValueError(f"cannot write blocked import: {result.get('problems')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "suoming.yaml"
    out_path.write_text(
        yaml.safe_dump(result["character"], allow_unicode=True,
                       sort_keys=False),
        encoding="utf-8",
    )
    return out_path
