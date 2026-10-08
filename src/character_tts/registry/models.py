"""Config schema for characters, backends, app and evaluations.

Plain dataclasses + explicit validation. Backend-specific sampling fields
live under ``generation``/``runtime`` free-form dicts and never enter the
common schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


def _req(data: dict[str, Any], key: str, ctx: str) -> Any:
    value = data.get(key)
    if value is None or value == "":
        raise ConfigError(f"{ctx}: missing required key '{key}'")
    return value


@dataclass
class AnchorText:
    id: str
    text: str
    audio: str | None = None
    sha256: str | None = None
    source: str | None = None


@dataclass
class CharacterProfile:
    character_id: str
    display_name: str
    dataset: dict[str, Any]
    reference: dict[str, Any]
    evaluation: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def anchor_texts(self) -> list[AnchorText]:
        out = []
        for item in self.evaluation.get("anchor_texts", []) or []:
            out.append(AnchorText(
                id=str(item["id"]),
                text=str(item["text"]),
                audio=item.get("audio"),
                sha256=item.get("sha256"),
                source=item.get("source"),
            ))
        return out

    @property
    def reference_audio(self) -> str | None:
        return self.reference.get("audio")

    @property
    def reference_text(self) -> str | None:
        return self.reference.get("text")

    @property
    def reference_sha256(self) -> str | None:
        return self.reference.get("sha256")

    @property
    def ground_truth(self) -> dict[str, Any] | None:
        """Real recorded utterance for the first anchor text that has one.

        Distinct from ``reference`` (the clone *prompt*): ground truth is the
        evaluation original — codec roundtrips reconstruct it and generated
        audio is compared against it. ``None`` when no anchor carries audio.
        """
        for anchor in self.anchor_texts:
            if anchor.audio:
                return {
                    "anchor_id": anchor.id,
                    "text": anchor.text,
                    "audio": anchor.audio,
                    "sha256": anchor.sha256,
                    "source": anchor.source,
                }
        return None


@dataclass
class BackendProfile:
    backend_id: str
    family: str
    enabled: bool
    worker: dict[str, Any]
    model: dict[str, Any]
    cache: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    config_path: Path | None = None

    @property
    def worker_python(self) -> str:
        return str(_req(self.worker, "python", f"backend {self.backend_id}"))

    @property
    def worker_script(self) -> str | None:
        script = self.worker.get("script")
        return str(script) if script else None

    @property
    def worker_cwd(self) -> str | None:
        cwd = self.worker.get("cwd")
        return str(cwd) if cwd else None

    @property
    def worker_env(self) -> dict[str, str]:
        return {str(k): str(v) for k, v in (self.worker.get("env") or {}).items()}

    @property
    def startup_timeout(self) -> float:
        return float(self.worker.get("startup_timeout_seconds", 300))


@dataclass
class AppConfig:
    outputs_root: Path
    logs_dir: Path
    default_character: str | None = None
    default_backend: str | None = None


@dataclass
class GateCase:
    backend: str
    kind: str  # "codec_roundtrip" | "zero_shot" | "generate"
    output: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    evaluation_id: str
    character: str
    output_dir: Path
    seed: int
    cases: list[GateCase]
    config_path: Path | None = None
