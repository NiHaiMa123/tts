"""Download source selection policy (plan section 1).

Order: existing mirror config -> configured mirror -> 127.0.0.1:7897 proxy.
This module only *decides*; bootstrap scripts execute the plan so the
policy stays unit-testable without touching the network.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

MIRROR_ENV_KEYS = (
    "HF_ENDPOINT",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
    "UV_INDEX_URL",
)

DEFAULT_PROXY = "http://127.0.0.1:7897"


@dataclass
class SourceAttempt:
    kind: str  # "existing_mirror" | "mirror" | "proxy" | "local_cache"
    description: str
    env: dict[str, str] = field(default_factory=dict)


def existing_mirror_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the user's already-configured mirror/cache env vars."""
    env = env or dict(os.environ)
    return {k: env[k] for k in MIRROR_ENV_KEYS if env.get(k)}


def plan_download(*, env: dict[str, str] | None = None,
                  mirror_endpoint: str | None = "https://hf-mirror.com",
                  proxy_url: str = DEFAULT_PROXY,
                  cached: bool = False) -> list[SourceAttempt]:
    """Ordered download attempts for an HF-hosted asset.

    1. local cache hit -> no download at all
    2. user's existing mirror env (HF_ENDPOINT etc.) -> reuse verbatim
    3. bootstrap mirror endpoint (hf-mirror.com by default)
    4. 127.0.0.1:7897 proxy (only after mirror failure)
    """
    attempts: list[SourceAttempt] = []
    if cached:
        attempts.append(SourceAttempt(
            "local_cache", "asset already in local cache"))
        return attempts

    existing = existing_mirror_env(env)
    if existing.get("HF_ENDPOINT"):
        attempts.append(SourceAttempt(
            "existing_mirror",
            f"user-configured HF endpoint {existing['HF_ENDPOINT']}",
            env={},
        ))
    if mirror_endpoint:
        attempts.append(SourceAttempt(
            "mirror",
            f"hf mirror endpoint {mirror_endpoint}",
            env={"HF_ENDPOINT": mirror_endpoint},
        ))
    attempts.append(SourceAttempt(
        "proxy",
        f"local proxy {proxy_url} (mirror failed)",
        env={
            "HTTP_PROXY": proxy_url,
            "HTTPS_PROXY": proxy_url,
            "ALL_PROXY": proxy_url,
        },
    ))
    return attempts
