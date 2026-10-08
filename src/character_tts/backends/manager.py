"""Backend worker lifecycle manager.

Owns the single active worker subprocess (one large model resident at a
time). Switching follows the fixed sequence:

    reject concurrent generation -> shutdown -> wait exit -> terminate on
    timeout -> confirm dead -> start new worker -> health check.

Worker stderr is appended to ``logs_dir/<backend_id>.worker.log`` so model
libraries can print freely without corrupting the JSONL stdout channel.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from ..registry.loader import load_env_file, repo_root
from ..registry.models import BackendProfile
from .worker_client import WorkerClient

logger = logging.getLogger(__name__)

_STOP_GRACE_SECONDS = 15.0
_TERMINATE_GRACE_SECONDS = 10.0


class BackendBusyError(RuntimeError):
    pass


class WorkerStartError(RuntimeError):
    pass


class BackendManager:
    def __init__(self, logs_dir: Path,
                 on_event: Callable[[dict[str, Any]], None] | None = None):
        self._logs_dir = Path(logs_dir)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._on_event = on_event
        self._lock = threading.Lock()      # serializes manager-level ops
        self._generate_lock = threading.Lock()  # non-blocking generate gate
        self._generate_owner: int | None = None  # thread id holding the gate
        self._proc: subprocess.Popen[bytes] | None = None
        self._client: WorkerClient | None = None
        self._stderr_fh = None
        self._active_backend_id: str | None = None
        self._active_profile: BackendProfile | None = None

    # -- state ----------------------------------------------------------------
    @property
    def active_backend_id(self) -> str | None:
        return self._active_backend_id

    @property
    def profile(self) -> BackendProfile | None:
        return self._active_profile

    def is_alive(self) -> bool:
        # A client that hit EOF/IO-error means the worker is dead or dying
        # even if proc.poll() has not caught up yet.
        return (
            self._proc is not None
            and self._proc.poll() is None
            and self._client is not None
            and not self._client.closed
        )

    def worker_log_path(self, backend_id: str) -> Path:
        return self._logs_dir / f"{backend_id}.worker.log"

    # -- lifecycle --------------------------------------------------------------
    def _require_not_busy(self, op: str) -> None:
        if self.busy:
            raise BackendBusyError(
                f"cannot {op}: generation in progress"
            )

    def start(self, profile: BackendProfile) -> dict[str, Any]:
        """Launch a worker and block until health check passes."""
        self._require_not_busy("start worker")
        with self._lock:
            self._require_not_busy("start worker")
            self._stop_locked()
            return self._start_locked(profile)

    def _await_ready(self, profile: BackendProfile) -> dict[str, Any]:
        deadline = time.monotonic() + profile.startup_timeout
        last_exc: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is None:
                break
            rc = self._proc.poll()
            if rc is not None:
                tail = self._log_tail(profile.backend_id)
                raise WorkerStartError(
                    f"worker '{profile.backend_id}' exited during startup "
                    f"(rc={rc}). Log tail:\n{tail}"
                )
            try:
                return self._client.health(timeout=5)
            except Exception as exc:  # worker may still be loading model
                last_exc = exc
                time.sleep(2.0)
        tail = self._log_tail(profile.backend_id)
        raise WorkerStartError(
            f"worker '{profile.backend_id}' not ready within "
            f"{profile.startup_timeout}s (last error: {last_exc}). "
            f"Log tail:\n{tail}"
        )

    def _log_tail(self, backend_id: str, max_bytes: int = 4000) -> str:
        path = self.worker_log_path(backend_id)
        try:
            data = path.read_bytes()[-max_bytes:]
            return data.decode("utf-8", "replace")
        except OSError:
            return "<no worker log>"

    def ensure(self, profile: BackendProfile) -> WorkerClient:
        """Return a client for ``profile``, switching backend if needed.

        Internal worker switches are only allowed when no generation is in
        flight on *another* thread — the gate holder itself may ensure
        (that's how a job switches to its requested backend).
        """
        if (self._generate_lock.locked()
                and self._generate_owner != threading.get_ident()):
            raise BackendBusyError(
                "cannot ensure/switch: generation in progress")
        with self._lock:
            if self._active_backend_id != profile.backend_id or not self.is_alive():
                self._switch_locked(profile)
            assert self._client is not None
            return self._client

    def switch(self, profile: BackendProfile) -> dict[str, Any]:
        self._require_not_busy("switch backend")
        with self._lock:
            self._require_not_busy("switch backend")
            if self._active_backend_id == profile.backend_id and self.is_alive():
                return self._client.health(timeout=30)
            return self._switch_locked(profile)

    def _switch_locked(self, profile: BackendProfile) -> dict[str, Any]:
        self._stop_locked()
        return self._start_locked(profile)

    def _resolve_under_repo(self, value: str | None) -> str | None:
        if not value:
            return value
        p = Path(value)
        if p.is_absolute():
            return str(p)
        return str((repo_root() / p).resolve())

    def _resolve_model_path(self, value: str | None) -> str | None:
        """Resolve local model paths; keep HF repo ids verbatim."""
        if not value:
            return value
        p = Path(value)
        if p.is_absolute():
            return str(p)
        candidate = (repo_root() / value).resolve()
        return str(candidate) if candidate.exists() else value

    def _start_locked(self, profile: BackendProfile) -> dict[str, Any]:
        env = load_env_file(repo_root() / ".env.local")
        env.update(os.environ)
        env.update(profile.worker_env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")

        model_cfg = dict(profile.model)
        model_cfg["path_or_id"] = self._resolve_model_path(
            model_cfg.get("path_or_id"))
        for key in ("adapter", "tokenizer", "cache_dir"):
            if model_cfg.get(key):
                model_cfg[key] = self._resolve_model_path(str(model_cfg[key]))
        cache_cfg = dict(profile.cache)
        if cache_cfg.get("root"):
            cache_cfg["root"] = self._resolve_under_repo(str(cache_cfg["root"]))
        env["TTS_BACKEND_JSON"] = json.dumps(
            {
                "backend_id": profile.backend_id,
                "family": profile.family,
                "model": model_cfg,
                "cache": cache_cfg,
                "runtime": profile.runtime,
                "generation": profile.generation,
            },
            ensure_ascii=False,
        )

        script = profile.worker_script
        if script:
            script_path = Path(self._resolve_under_repo(script))
            argv = [self._resolve_under_repo(profile.worker_python),
                    str(script_path)]
        else:
            argv = str(profile.worker["command"]).split()
        cwd = self._resolve_under_repo(profile.worker_cwd) or str(repo_root())

        log_path = self.worker_log_path(profile.backend_id)
        self._stderr_fh = open(log_path, "ab")
        logger.info("starting worker %s: %s (cwd=%s)", profile.backend_id, argv, cwd)
        try:
            self._proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr_fh,
                cwd=cwd,
                env=env,
            )
        except OSError as exc:
            self._stderr_fh.close()
            self._stderr_fh = None
            raise WorkerStartError(
                f"failed to launch worker '{profile.backend_id}': {exc}"
            ) from exc

        self._client = WorkerClient(
            self._proc.stdin, self._proc.stdout, on_event=self._on_event
        )
        self._active_backend_id = profile.backend_id
        self._active_profile = profile
        try:
            return self._await_ready(profile)
        except Exception:
            self._stop_locked()
            raise

    def stop(self, wait_seconds: float = 0.0) -> None:
        """Stop the worker. Refuses while a generation is in flight;
        ``wait_seconds`` gives an in-flight job time to finish first."""
        deadline = time.monotonic() + wait_seconds
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.1)
        self._require_not_busy("stop worker")
        with self._lock:
            self._require_not_busy("stop worker")
            self._stop_locked()

    def force_stop(self) -> None:
        """Kill the worker even mid-generation — process teardown only."""
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        try:
            if proc.poll() is None and self._client is not None:
                try:
                    self._client.shutdown(timeout=_STOP_GRACE_SECONDS)
                except Exception as exc:
                    logger.warning("graceful shutdown failed: %s", exc)
            try:
                proc.wait(timeout=_STOP_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                logger.warning("worker did not exit; terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    logger.error("worker did not terminate; killing")
                    proc.kill()
                    proc.wait(timeout=10)
        finally:
            if self._stderr_fh is not None:
                try:
                    self._stderr_fh.close()
                except OSError:
                    pass
            self._proc = None
            self._client = None
            self._stderr_fh = None
            self._active_backend_id = None
            self._active_profile = None

    # -- generation gate ------------------------------------------------------
    def acquire_generate(self) -> bool:
        if not self._generate_lock.acquire(blocking=False):
            return False
        self._generate_owner = threading.get_ident()
        return True

    def release_generate(self) -> None:
        self._generate_owner = None
        self._generate_lock.release()

    @property
    def busy(self) -> bool:
        acquired = self._generate_lock.acquire(blocking=False)
        if acquired:
            self._generate_lock.release()
            return False
        return True

    def __enter__(self) -> "BackendManager":
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()
