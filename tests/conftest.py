from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from character_tts.registry.models import BackendProfile  # noqa: E402

FAKE_WORKER = REPO_ROOT / "tests" / "fake_worker.py"


def make_profile(backend_id: str = "fake", env: dict | None = None,
                 startup_timeout: float = 30.0) -> BackendProfile:
    return BackendProfile(
        backend_id=backend_id,
        family="fake",
        enabled=True,
        worker={
            "python": sys.executable,
            "script": str(FAKE_WORKER),
            "cwd": str(REPO_ROOT),
            "startup_timeout_seconds": startup_timeout,
            "env": env or {},
        },
        model={"path_or_id": "fake/model", "revision": "abc123"},
    )


@pytest.fixture
def profile_factory():
    return make_profile
