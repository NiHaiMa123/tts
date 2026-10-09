"""YAML config loading with ``${ENV_VAR}`` expansion and validation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .models import (
    AppConfig,
    BackendProfile,
    CharacterProfile,
    ConfigError,
    EvaluationConfig,
    GateCase,
)

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")
_ENV_FILE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")

_REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_root() -> Path:
    return _REPO_ROOT


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv-style file; returns {} when the file does not exist."""
    result: dict[str, str] = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_FILE_RE.match(line)
        if m:
            result[m.group(1)] = m.group(2).strip('"').strip("'")
    return result


def expand_env(value: Any, extra: dict[str, str] | None = None) -> Any:
    """Recursively expand ${VAR} in strings.

    Lookup order: ``extra`` (e.g. .env.local) then process environment.
    Unknown variables raise ConfigError — silent partial paths are worse
    than a loud failure.
    """
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if extra and name in extra:
                return extra[name]
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            raise ConfigError(f"environment variable '{name}' is not set")
        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: expand_env(v, extra) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v, extra) for v in value]
    return value


def _load_yaml(path: Path, env: dict[str, str]) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")
    env = {"TTS_ROOT": str(_REPO_ROOT), **env}
    return expand_env(data, env)


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def load_app_config(path: Path | None = None) -> AppConfig:
    path = path or _REPO_ROOT / "configs" / "app.yaml"
    env_file = _REPO_ROOT / ".env.local"
    env = {**load_env_file(env_file), **os.environ}
    data = _load_yaml(path, env)
    paths = data.get("paths") or {}
    outputs = _resolve(str(paths.get("outputs_root", "outputs")), _REPO_ROOT)
    logs = _resolve(str(paths.get("logs_dir", "logs")), _REPO_ROOT)
    defaults = data.get("defaults") or {}
    return AppConfig(
        outputs_root=outputs,
        logs_dir=logs,
        default_character=defaults.get("character"),
        default_backend=defaults.get("backend"),
    )


def load_character(path_or_id: str, env: dict[str, str] | None = None,
                   configs_dir: Path | None = None) -> CharacterProfile:
    env = env or {**load_env_file(_REPO_ROOT / ".env.local"), **os.environ}
    configs_dir = configs_dir or _REPO_ROOT / "configs" / "characters"
    path = Path(path_or_id)
    if not path.suffix:
        path = configs_dir / f"{path_or_id}.yaml"
    data = _load_yaml(path, env)
    profile = CharacterProfile(
        character_id=str(data.get("character_id") or path.stem),
        display_name=str(data.get("display_name") or path.stem),
        dataset=data.get("dataset") or {},
        reference=data.get("reference") or {},
        backend=data.get("backend"),
        adapters=data.get("adapters") or {},
        postprocess=data.get("postprocess") or {},
        evaluation=data.get("evaluation") or {},
        provenance=data.get("provenance") or {},
        config_path=path,
    )
    if not profile.reference_audio:
        raise ConfigError(f"{path}: character has no reference.audio")
    return profile


def load_backend(path_or_id: str, env: dict[str, str] | None = None,
                 configs_dir: Path | None = None) -> BackendProfile:
    env = env or {**load_env_file(_REPO_ROOT / ".env.local"), **os.environ}
    configs_dir = configs_dir or _REPO_ROOT / "configs" / "backends"
    path = Path(path_or_id)
    if not path.suffix:
        path = configs_dir / f"{path_or_id}.yaml"
    data = _load_yaml(path, env)
    profile = BackendProfile(
        backend_id=str(data.get("backend_id") or path.stem),
        family=str(data.get("family") or ""),
        enabled=bool(data.get("enabled", True)),
        worker=data.get("worker") or {},
        model=data.get("model") or {},
        cache=data.get("cache") or {},
        runtime=data.get("runtime") or {},
        generation=data.get("generation") or {},
        notes=str(data.get("notes") or ""),
        config_path=path,
    )
    _validate_backend(profile)
    return profile


def _validate_backend(profile: BackendProfile) -> None:
    profile.worker_python  # raises when missing
    if not profile.worker_script and not profile.worker.get("command"):
        raise ConfigError(
            f"backend {profile.backend_id}: worker needs 'script' or 'command'"
        )
    if not profile.model.get("path_or_id"):
        raise ConfigError(f"backend {profile.backend_id}: missing model.path_or_id")


def list_backends(configs_dir: Path | None = None) -> list[str]:
    configs_dir = configs_dir or _REPO_ROOT / "configs" / "backends"
    if not configs_dir.is_dir():
        return []
    return sorted(p.stem for p in configs_dir.glob("*.yaml"))


def list_characters(configs_dir: Path | None = None) -> list[str]:
    configs_dir = configs_dir or _REPO_ROOT / "configs" / "characters"
    if not configs_dir.is_dir():
        return []
    return sorted(p.stem for p in configs_dir.glob("*.yaml"))


def load_evaluation(path_or_id: str, env: dict[str, str] | None = None) -> EvaluationConfig:
    env = env or {**load_env_file(_REPO_ROOT / ".env.local"), **os.environ}
    path = Path(path_or_id)
    if not path.suffix:
        path = _REPO_ROOT / "configs" / "evaluations" / f"{path_or_id}.yaml"
    data = _load_yaml(path, env)
    cases: list[GateCase] = []
    valid_kinds = {"codec_roundtrip", "zero_shot", "generate"}
    for i, raw in enumerate(data.get("cases") or []):
        kind = str(raw.get("kind"))
        if kind not in valid_kinds:
            raise ConfigError(f"{path}: case {i} has invalid kind '{kind}'")
        cases.append(
            GateCase(
                backend=str(raw["backend"]),
                kind=kind,
                output=str(raw["output"]),
                options=raw.get("options") or {},
            )
        )
    if not cases:
        raise ConfigError(f"{path}: evaluation has no cases")
    output_dir = _resolve(str(data.get("output_dir", "outputs/gate")),
                          _REPO_ROOT)
    return EvaluationConfig(
        evaluation_id=str(data.get("evaluation_id") or path.stem),
        character=str(data.get("character") or ""),
        output_dir=output_dir,
        seed=int(data.get("seed", 42)),
        cases=cases,
        config_path=path,
    )
