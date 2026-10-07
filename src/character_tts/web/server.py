"""WebUI v1 — minimal character TTS front end.

Single-file HTML + a small JSON API on top of BackendManager:
choose character / backend -> auto-switch worker -> generate -> play.
No multi-user, no auth, no database (per plan section 13).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..backends.manager import BackendManager
from ..registry.loader import (
    list_backends,
    list_characters,
    load_app_config,
    load_backend,
    load_character,
    repo_root,
)
from ..registry.models import ConfigError

logger = logging.getLogger(__name__)

_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>character-tts</title>
<style>
 body { font-family: system-ui,"Segoe UI","Microsoft YaHei",sans-serif;
        background:#15171c; color:#e8e9eb; margin:0; }
 main { max-width:860px; margin:0 auto; padding:24px; }
 h1 { font-size:1.3rem; }
 .row { display:flex; gap:12px; margin:10px 0; align-items:center;
        flex-wrap:wrap; }
 label { color:#9aa0a8; font-size:.85rem; }
 select,input[type=number] { background:#0f1115; color:#e8e9eb;
   border:1px solid #2c313a; border-radius:6px; padding:6px 8px; }
 textarea { width:100%; min-height:110px; background:#0f1115;
   color:#e8e9eb; border:1px solid #2c313a; border-radius:8px;
   padding:10px; font-size:1rem; box-sizing:border-box; }
 button { background:#2d6cdf; color:#fff; border:0; border-radius:6px;
   padding:8px 16px; cursor:pointer; font-size:.9rem; }
 button:disabled { background:#333a45; cursor:not-allowed; }
 button.ghost { background:#333a45; }
 #status { font-size:.85rem; color:#9aa0a8; }
 #status .err { color:#ff7a7a; }
 #status .ok { color:#7adfa0; }
 audio { width:100%; margin-top:8px; }
 .card { background:#1e222a; border:1px solid #2c313a; border-radius:10px;
   padding:14px; margin-top:14px; }
 .pill { display:inline-block; padding:2px 8px; border-radius:10px;
   background:#0f1115; font-size:.75rem; color:#9aa0a8; margin-left:6px; }
 .jobs { font-size:.8rem; color:#9aa0a8; margin-top:8px; }
 a { color:#7aa8ff; }
</style>
</head>
<body>
<main>
<h1>character-tts <span class="pill" id="version">v1</span></h1>

<div class="card">
 <div class="row">
  <label>角色</label>
  <select id="character"></select>
  <label>后端</label>
  <select id="backend"></select>
  <button class="ghost" onclick="switchBackend()">切换后端</button>
 </div>
 <div class="row"><span id="status">loading…</span></div>
</div>

<div class="card">
 <label>参考音频</label>
 <div class="row"><span id="refinfo" style="font-size:.8rem;color:#9aa0a8"></span></div>
 <label style="margin-top:8px;display:block">文本</label>
 <textarea id="text" placeholder="输入要合成的文本"></textarea>
 <div class="row">
  <label>seed</label><input type="number" id="seed" value="42">
  <button id="genbtn" onclick="generate()">生成</button>
  <span class="jobs" id="jobinfo"></span>
 </div>
 <audio id="player" controls style="display:none"></audio>
</div>

<div class="card">
 <button class="ghost" onclick="openOutputs()">打开输出目录</button>
 <span class="jobs"><a href="/outputs/" target="_blank">浏览 outputs/</a></span>
</div>
</main>
<script>
let state = {};
async function api(path, opts) {
  const r = await fetch(path, opts ? {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(opts)}) : undefined);
  if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
  return r.json();
}
async function refresh() {
  try {
    state = await api("/api/state");
  } catch (e) {
    document.getElementById("status").innerHTML =
      `<span class="err">state error: ${e.message}</span>`;
    return;
  }
  const cs = document.getElementById("character");
  const bs = document.getElementById("backend");
  if (!cs.dataset.filled) {
    cs.innerHTML = state.characters.map(c =>
      `<option ${c===state.default_character?"selected":""}>${c}</option>`).join("");
    bs.innerHTML = state.backends.map(b =>
      `<option ${b===state.active_backend?"selected":""}>${b}</option>`).join("");
    cs.dataset.filled = bs.dataset.filled = "1";
    cs.onchange = loadRef; loadRef();
  }
  const h = state.health || {};
  document.getElementById("status").innerHTML = state.active_backend
    ? `<span class="ok">worker: ${state.active_backend}</span>` +
      ` <span class="pill">${h.backend_version||""}</span>` +
      ` <span class="pill">${h.device||""} ${h.dtype||""}</span>` +
      ` <span class="pill">model ${h.model_id||""}</span>` +
      (state.busy ? ' <span class="pill">BUSY</span>' : '')
    : '<span class="err">no worker running</span>';
}
async function loadRef() {
  const c = document.getElementById("character").value;
  const r = await api(`/api/character/${encodeURIComponent(c)}/reference`);
  document.getElementById("refinfo").textContent =
    r.audio ? `${r.audio.split(/[\\\\/]/).pop()} (${r.sha256?.slice(0,12)||""}…)` : "无";
}
async function switchBackend() {
  const b = document.getElementById("backend").value;
  document.getElementById("status").innerHTML = "switching…";
  try { await api("/api/select_backend", {backend: b}); }
  catch (e) { alert(e.message); }
  refresh();
}
async function generate() {
  const btn = document.getElementById("genbtn");
  btn.disabled = true;
  document.getElementById("jobinfo").textContent = "排队/生成中…";
  try {
    const r = await api("/api/generate", {
      character: document.getElementById("character").value,
      backend: document.getElementById("backend").value,
      text: document.getElementById("text").value,
      seed: parseInt(document.getElementById("seed").value),
    });
    pollJob(r.job_id);
  } catch (e) {
    document.getElementById("jobinfo").innerHTML =
      `<span class="err">${e.message}</span>`;
    btn.disabled = false;
  }
}
async function pollJob(id) {
  const j = await api(`/api/jobs/${id}`);
  if (j.status === "done") {
    document.getElementById("jobinfo").innerHTML =
      `<span class="ok">done ${j.wall_seconds?.toFixed(1)}s → ${j.output_path}</span>`;
    const p = document.getElementById("player");
    p.src = j.url + "?t=" + Date.now();
    p.style.display = "block"; p.play();
    document.getElementById("genbtn").disabled = false;
    refresh();
  } else if (j.status === "error") {
    document.getElementById("jobinfo").innerHTML =
      `<span class="err">${j.error}</span>`;
    document.getElementById("genbtn").disabled = false;
  } else {
    document.getElementById("jobinfo").textContent = j.status + "…";
    setTimeout(() => pollJob(id), 1500);
  }
}
async function openOutputs() { await api("/api/open_outputs", {}); }
refresh(); setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def create_app(manager: BackendManager | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="character-tts")
    app_config = load_app_config()
    app_config.outputs_root.mkdir(parents=True, exist_ok=True)
    manager = manager or BackendManager(logs_dir=app_config.logs_dir)
    jobs: dict[str, dict[str, Any]] = {}
    jobs_lock = threading.Lock()

    app.mount("/outputs", StaticFiles(directory=str(app_config.outputs_root)),
              name="outputs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        health = None
        if manager.is_alive() and manager._client is not None:
            try:
                health = manager._client.health(timeout=5)
            except Exception as exc:
                health = {"error": str(exc)}
        return {
            "characters": list_characters(),
            "backends": [b for b in list_backends()],
            "active_backend": manager.active_backend_id,
            "default_character": app_config.default_character,
            "default_backend": app_config.default_backend,
            "busy": manager.busy,
            "health": health,
        }

    @app.get("/api/character/{character_id}/reference")
    def reference(character_id: str) -> dict[str, Any]:
        try:
            c = load_character(character_id)
        except ConfigError as exc:
            raise HTTPException(404, str(exc))
        return {"audio": c.reference_audio, "text": c.reference_text,
                "sha256": c.reference_sha256}

    @app.post("/api/select_backend")
    def select_backend(payload: dict[str, str]) -> dict[str, Any]:
        backend_id = payload.get("backend", "")
        try:
            profile = load_backend(backend_id)
        except ConfigError as exc:
            raise HTTPException(404, str(exc))
        if not profile.enabled:
            raise HTTPException(400, f"backend {backend_id} disabled")
        try:
            health = manager.switch(profile)
        except Exception as exc:
            raise HTTPException(500, f"switch failed: {exc}")
        return {"active_backend": backend_id, "health": health}

    def _run_job(job_id: str, profile, character, text: str,
                 seed: int, options: dict[str, Any]) -> None:
        out_dir = app_config.outputs_root / "webui" / character.character_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{int(time.time())}_{job_id}.wav"
        try:
            client = manager.ensure(profile)
            if not manager.acquire_generate():
                raise RuntimeError("another generation is in progress")
            try:
                result = client.generate(
                    text=text, output_path=str(out_path),
                    reference_audio=character.reference_audio,
                    reference_text=character.reference_text,
                    seed=seed, options=options, timeout=1800,
                )
            finally:
                manager.release_generate()
            jobs[job_id].update({
                "status": "done",
                "output_path": str(out_path),
                "url": "/outputs/" + out_path.relative_to(
                    app_config.outputs_root).as_posix(),
                "wall_seconds": result.get("wall_seconds"),
            })
        except Exception as exc:
            jobs[job_id].update({"status": "error",
                                 "error": f"{type(exc).__name__}: {exc}"})

    @app.post("/api/generate")
    def generate(payload: dict[str, Any]) -> dict[str, str]:
        character_id = str(payload.get("character") or "")
        backend_id = str(payload.get("backend") or "")
        text = str(payload.get("text") or "").strip()
        if not text:
            raise HTTPException(400, "text is empty")
        try:
            character = load_character(character_id)
            profile = load_backend(backend_id)
        except ConfigError as exc:
            raise HTTPException(404, str(exc))
        if not profile.enabled:
            raise HTTPException(400, f"backend {backend_id} disabled")
        job_id = uuid.uuid4().hex[:8]
        with jobs_lock:
            jobs[job_id] = {"status": "queued"}
        options = payload.get("options") or {}
        seed = payload.get("seed")
        threading.Thread(
            target=_run_job,
            args=(job_id, profile, character, text, seed, options),
            daemon=True,
        ).start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]:
        with jobs_lock:
            j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "unknown job")
        return dict(j)

    @app.post("/api/open_outputs")
    def open_outputs() -> dict[str, str]:
        path = str(app_config.outputs_root)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - local desktop app
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"opened": path}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.stop()

    app.state.manager = manager
    return app


def main(host: str = "127.0.0.1", port: int = 7860) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(create_app(), host=host, port=port)
