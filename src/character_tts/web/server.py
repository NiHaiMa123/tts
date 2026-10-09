"""WebUI — character TTS front end (dotstts-style batch workbench).

Reads every ``inputs/*.txt`` (UTF-8), generates one WAV per file into
``outputs/webui/<character>/``. Single-card dark UI. The picker is the
character; each character locks its backend. Only closing the console
window stops the server. Launches Edge automatically after bind.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Any

from ..backends.manager import BackendBusyError, BackendManager
from ..registry.loader import (
    list_characters,
    load_app_config,
    load_backend,
    load_character,
    repo_root,
)
from ..registry.models import ConfigError

logger = logging.getLogger(__name__)

APP_ID = "character-tts-webui"
INPUTS_DIR = repo_root() / "inputs"

INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <title>角色 TTS</title>
  <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='16' fill='%23101522'/%3E%3Cpath d='M17 39c7-18 23-24 32-20-11 4-15 10-18 19 7-6 13-7 18-5-7 2-11 7-15 14-7 2-13-1-17-8Z' fill='%236ee7e0'/%3E%3C/svg%3E">
  <style>
    :root {
      font-family: Inter, "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif;
      color: #edf6f7; background: #080c13; font-synthesis: none;
      --panel: rgba(17, 24, 37, 0.92); --panel-soft: #151e2c;
      --line: rgba(172, 202, 211, 0.15); --muted: #9eafb9;
      --cyan: #73eee3; --cyan-deep: #39c9c1; --blue: #7aa2ff;
      --danger: #ff8f9b; --warning: #ffd18c;
      --shadow: 0 36px 100px rgba(0, 0, 0, 0.48);
    }
    * { box-sizing: border-box; }
    body {
      min-width: 320px; min-height: 100vh; margin: 0; overflow-x: hidden;
      background:
        radial-gradient(circle at 12% 18%, rgba(56, 208, 198, 0.13), transparent 30rem),
        radial-gradient(circle at 88% 76%, rgba(75, 112, 211, 0.13), transparent 34rem),
        linear-gradient(145deg, #070a10 0%, #0b111b 48%, #080c13 100%);
    }
    body::before {
      content: ""; position: fixed; inset: 0; pointer-events: none; opacity: 0.16;
      background-image:
        linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px);
      background-size: 44px 44px;
      mask-image: linear-gradient(to bottom, black, transparent 82%);
    }
    main { position: relative; display: grid; place-items: center;
      min-height: 100vh; padding: 32px 20px; }
    .shell { width: min(680px, 100%); border: 1px solid var(--line);
      border-radius: 24px; background: var(--panel); box-shadow: var(--shadow);
      backdrop-filter: blur(18px); overflow: hidden; }
    .accent { height: 3px; background: linear-gradient(90deg, transparent,
      var(--cyan), var(--blue), transparent); }
    .content { padding: clamp(26px, 5vw, 44px); }
    header { display: flex; align-items: flex-start; justify-content: space-between;
      gap: 18px; margin-bottom: 34px; }
    h1 { margin: 0; font-size: clamp(1.65rem, 5vw, 2.15rem); font-weight: 680;
      line-height: 1.15; letter-spacing: -0.035em; }
    .subtitle { margin: 9px 0 0; color: var(--muted); font-size: 0.94rem;
      line-height: 1.6; }
    .status-pill { flex: none; display: inline-flex; align-items: center; gap: 8px;
      min-height: 34px; padding: 7px 12px; border: 1px solid var(--line);
      border-radius: 999px; color: #cbd9df; background: rgba(255,255,255,0.035);
      font-size: 0.82rem; white-space: nowrap; }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted);
      box-shadow: 0 0 0 4px rgba(158, 175, 185, 0.08); }
    .status-pill.active .dot { background: var(--cyan); box-shadow:
      0 0 0 4px rgba(115,238,227,0.12), 0 0 18px var(--cyan);
      animation: pulse 1.4s ease-in-out infinite; }
    .status-pill.completed .dot { background: var(--cyan); }
    .status-pill.failed .dot, .status-pill.interrupted .dot { background: var(--danger); }
    @keyframes pulse { 50% { opacity: .46; transform: scale(.82); } }
    .field { margin-bottom: 18px; }
    label { display: block; margin-bottom: 9px; color: #d9e5e8;
      font-size: 0.9rem; font-weight: 620; }
    select, button { width: 100%; min-height: 52px; border-radius: 13px;
      font: inherit; }
    select { appearance: none; border: 1px solid rgba(175,207,216,0.2);
      padding: 0 48px 0 16px; color: #f1f7f8; background-color: var(--panel-soft);
      background-image:
        linear-gradient(45deg, transparent 50%, #a8bac1 50%),
        linear-gradient(135deg, #a8bac1 50%, transparent 50%);
      background-position: calc(100% - 20px) 23px, calc(100% - 15px) 23px;
      background-size: 5px 5px, 5px 5px; background-repeat: no-repeat;
      outline: none; transition: border-color .2s, box-shadow .2s; }
    select:focus-visible { border-color: var(--cyan-deep);
      box-shadow: 0 0 0 4px rgba(57,201,193,0.12); }
    select:disabled { opacity: .58; cursor: wait; }
    button { border: 0; padding: 0 20px; cursor: pointer; font-weight: 700;
      letter-spacing: .01em; transition: transform .16s, filter .16s, opacity .16s; }
    button:hover:not(:disabled) { transform: translateY(-1px); filter: brightness(1.06); }
    button:active:not(:disabled) { transform: translateY(0); }
    button:focus-visible { outline: 3px solid rgba(122,162,255,.5);
      outline-offset: 3px; }
    button:disabled { cursor: wait; opacity: .55; }
    #generate { color: #061312; background:
      linear-gradient(110deg, #73eee3, #7dd8f3 58%, #95aefc);
      box-shadow: 0 15px 32px rgba(55,202,195,.18); }
    .filelist { margin: -4px 0 18px; padding: 10px 14px;
      border: 1px dashed var(--line); border-radius: 10px;
      color: var(--muted); font-size: .82rem; line-height: 1.7;
      max-height: 120px; overflow-y: auto; }
    .filelist code { color: #cfe3e6; }
    .progress-panel { margin-top: 28px; padding-top: 24px;
      border-top: 1px solid var(--line); }
    .progress-head { display: flex; justify-content: space-between;
      align-items: baseline; gap: 16px; margin-bottom: 12px; }
    .progress-label { color: #dce8eb; font-size: .9rem; font-weight: 620; }
    .progress-value { color: var(--cyan); font-variant-numeric: tabular-nums;
      font-size: .9rem; }
    .progress-track { height: 8px; overflow: hidden; border-radius: 99px;
      background: rgba(255,255,255,.07); }
    .progress-bar { height: 100%; width: 0; border-radius: inherit;
      background: linear-gradient(90deg, var(--cyan-deep), var(--cyan), var(--blue));
      box-shadow: 0 0 20px rgba(115,238,227,.28); transition: width .35s ease; }
    .progress-bar.indeterminate { width: 32% !important;
      animation: indeterminate 1.25s ease-in-out infinite; }
    @keyframes indeterminate { from { transform: translateX(-110%); }
      to { transform: translateX(330%); } }
    #message { min-height: 48px; margin: 14px 0 0; color: var(--muted);
      font-size: .9rem; line-height: 1.65; overflow-wrap: anywhere; }
    #message.error { color: var(--danger); }
    #message.success { color: #a8f3e9; }
    .result { display: none; margin-top: 18px; padding: 16px;
      border: 1px solid rgba(115,238,227,.18); border-radius: 14px;
      background: rgba(79,222,210,.05); }
    .result.visible { display: block; }
    .result-copy { margin: 0 0 13px; color: #c9dcdf; font-size: .86rem;
      line-height: 1.55; }
    #open-output { min-height: 44px; color: #dff9f7;
      border: 1px solid rgba(115,238,227,.3); background: rgba(115,238,227,.09);
      box-shadow: none; }
    footer { margin-top: 26px; color: #748690; font-size: .78rem;
      line-height: 1.55; text-align: center; }
    @media (max-width: 560px) {
      main { padding: 14px; } .shell { border-radius: 18px; }
      .content { padding: 24px 20px 26px; }
      header { display: block; margin-bottom: 27px; }
      .status-pill { margin-top: 16px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .01ms !important;
        transition-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <main>
    <section class="shell" aria-labelledby="title">
      <div class="accent"></div>
      <div class="content">
        <header>
          <div>
            <h1 id="title">角色 TTS</h1>
            <p class="subtitle">自动读取 inputs 目录中的 TXT，每个文件生成一个最终音频。</p>
          </div>
          <div id="status-pill" class="status-pill">
            <span class="dot" aria-hidden="true"></span>
            <span id="status-name">连接中</span>
          </div>
        </header>

        <div class="field">
          <label for="model">角色</label>
          <select id="model" aria-describedby="model-description">
            <option value="">正在读取角色列表…</option>
          </select>
          <p id="model-description" class="subtitle">每个角色锁定自己的后端与参考音频。</p>
        </div>

        <div id="filelist" class="filelist">正在扫描 inputs/ …</div>

        <button id="generate" type="button">开始 TTS</button>

        <section class="progress-panel" aria-live="polite" aria-atomic="true">
          <div class="progress-head">
            <span class="progress-label">输出进度</span>
            <span id="progress-value" class="progress-value">0%</span>
          </div>
          <div class="progress-track" role="progressbar" aria-label="音频生成进度"
               aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
            <div id="progress-bar" class="progress-bar"></div>
          </div>
          <p id="message">正在连接本地服务…</p>
        </section>

        <section id="result" class="result">
          <p id="result-copy" class="result-copy"></p>
          <button id="open-output" type="button">打开输出目录</button>
        </section>

        <footer>关闭终端窗口即停止本地服务器；仅关闭页面不会中断任务。</footer>
      </div>
    </section>
  </main>

  <script>
    const activeStates = new Set(["preparing", "loading_model", "generating", "finalizing"]);
    const labels = {
      starting: "启动中", idle: "就绪", preparing: "准备中",
      loading_model: "加载模型", generating: "生成中", finalizing: "保存中",
      completed: "已完成", failed: "出错", interrupted: "已中断"
    };
    const model = document.querySelector("#model");
    const modelDescription = document.querySelector("#model-description");
    const generate = document.querySelector("#generate");
    const progressTrack = document.querySelector(".progress-track");
    const progressBar = document.querySelector("#progress-bar");
    const progressValue = document.querySelector("#progress-value");
    const message = document.querySelector("#message");
    const statusPill = document.querySelector("#status-pill");
    const statusName = document.querySelector("#status-name");
    const filelist = document.querySelector("#filelist");
    const result = document.querySelector("#result");
    const resultCopy = document.querySelector("#result-copy");
    const openOutput = document.querySelector("#open-output");
    let lastStatus = null, connected = true, audioCtx = null, sawActive = false;
    let modelSpecs = new Map();
    const BASE_TITLE = document.title || "角色 TTS";

    function tone(freq, startOffset, duration, peak = 0.14) {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = "sine"; osc.frequency.value = freq;
      const t0 = audioCtx.currentTime + startOffset;
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(peak, t0 + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + duration);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(t0); osc.stop(t0 + duration + 0.02);
    }
    function chime(kind) {
      try {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === "suspended") audioCtx.resume();
        if (kind === "done") { tone(880, 0, 0.09); tone(1174.66, 0.09, 0.22); }
        else if (kind === "fail") { tone(233.08, 0, 0.14); tone(174.61, 0.12, 0.26); }
      } catch (_) {}
    }

    async function request(path, options = {}) {
      const response = await fetch(path, { cache: "no-store",
        headers: { "Content-Type": "application/json" }, ...options });
      let payload = {};
      try { payload = await response.json(); } catch (_) {}
      if (!response.ok) throw new Error(payload.message || payload.detail || `请求失败：${response.status}`);
      return payload;
    }

    function updateModelDescription() {
      const item = modelSpecs.get(model.value);
      modelDescription.textContent = item
        ? item.description
        : "每个角色锁定自己的后端与参考音频。";
    }

    function setModels(models, selected, defaultModel) {
      if (!Array.isArray(models) || !models.length) return;
      const signature = models.map((item) => item.id).join("|");
      if (model.dataset.signature !== signature) {
        const previous = model.value;
        model.replaceChildren();
        modelSpecs = new Map();
        for (const item of models) {
          modelSpecs.set(item.id, item);
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = item.id === defaultModel ? `${item.label}（默认）` : item.label;
          model.appendChild(option);
        }
        const ids = new Set(models.map((item) => item.id));
        model.value = ids.has(previous) ? previous
          : (ids.has(selected) ? selected : defaultModel || models[0].id);
        model.dataset.signature = signature;
      }
      updateModelDescription();
    }

    function renderStatus(state) {
      connected = true;
      const prevStatus = lastStatus;
      lastStatus = state.status;
      const active = activeStates.has(state.status);
      const progress = Math.max(0, Math.min(100, Number(state.progress) || 0));
      const indeterminate = state.status === "loading_model";

      if (active) sawActive = true;
      else if (sawActive && prevStatus !== state.status) {
        if (state.status === "completed") chime("done");
        else if (state.status === "failed" || state.status === "interrupted") chime("fail");
      }
      const label = labels[state.status] || state.status || "未知";
      document.title = active
        ? `【${label} ${Math.round(progress)}%】${BASE_TITLE}`
        : `【${label}】${BASE_TITLE}`;

      setModels(state.characters,
        state.selected_character || state.default_character, state.default_character);
      model.disabled = active;
      generate.disabled = active || !state.can_generate;
      generate.textContent = active ? "正在生成…" : "开始 TTS";

      statusPill.className = `status-pill ${active ? "active" : state.status || ""}`;
      statusName.textContent = label;
      progressBar.classList.toggle("indeterminate", indeterminate);
      progressBar.style.width = `${progress}%`;
      progressValue.textContent = indeterminate ? "加载中" : `${Math.round(progress)}%`;
      progressTrack.setAttribute("aria-valuenow", String(Math.round(progress)));
      message.textContent = state.message || "";
      message.className = (state.status === "failed" || state.status === "interrupted")
        ? "error" : state.status === "completed" ? "success" : "";

      const files = Array.isArray(state.input_files) ? state.input_files : [];
      filelist.innerHTML = files.length
        ? `inputs/ 共 ${files.length} 个 txt：<br>` + files.map(f => `<code>${f}</code>`).join("、")
        : `inputs/ 目录没有 .txt 文件。把文本保存为 UTF-8 txt 放进去再点开始。`;
      generate.disabled = generate.disabled || !files.length;

      const canOpen = Boolean(state.can_open_output);
      result.classList.toggle("visible", canOpen);
      if (canOpen) {
        const count = Array.isArray(state.outputs) ? state.outputs.length : 0;
        resultCopy.textContent = `已生成 ${count} 个最终 WAV。`;
      }
    }

    async function pollStatus() {
      try {
        renderStatus(await request("/api/status"));
      } catch (error) {
        if (connected) {
          connected = false;
          statusPill.className = "status-pill interrupted";
          statusName.textContent = "连接已关闭";
          document.title = `【已断开】${BASE_TITLE}`;
          message.className = "error";
          message.textContent = "本地服务已退出。重新使用时请再次双击启动。";
          generate.disabled = true; model.disabled = true;
        }
      }
    }

    generate.addEventListener("click", async () => {
      generate.disabled = true;
      try {
        audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === "suspended") audioCtx.resume();
      } catch (_) {}
      result.classList.remove("visible");
      message.className = "";
      message.textContent = "正在创建任务…";
      try {
        await request("/api/generate", { method: "POST",
          body: JSON.stringify({ character: model.value }) });
        await pollStatus();
      } catch (error) {
        message.className = "error"; message.textContent = error.message;
        generate.disabled = false;
      }
    });

    model.addEventListener("change", updateModelDescription);

    openOutput.addEventListener("click", async () => {
      openOutput.disabled = true;
      try { await request("/api/open-output", { method: "POST", body: "{}" }); }
      catch (error) { message.className = "error"; message.textContent = error.message; }
      finally { openOutput.disabled = false; }
    });

    pollStatus();
    setInterval(pollStatus, 800);
  </script>
</body>
</html>
"""


class _Session:
    """Batch generation state machine."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"
        self.progress = 0.0
        self.message = ""
        self.outputs: list[str] = []
        self.errors: list[str] = []
        self.selected_character: str | None = None
        self.current_file: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status, "progress": self.progress,
                "message": self.message, "outputs": list(self.outputs),
                "errors": list(self.errors),
                "selected_character": self.selected_character,
                "current_file": self.current_file,
            }

    def transition(self, status: str, message: str = "", **kw: Any) -> None:
        with self.lock:
            self.status = status
            self.message = message
            for k, v in kw.items():
                setattr(self, k, v)


def _list_inputs() -> list[Path]:
    if not INPUTS_DIR.is_dir():
        return []
    return sorted(p for p in INPUTS_DIR.glob("*.txt")
                  if p.is_file() and not p.name.lower().startswith("readme"))


_BACKEND_LABELS = {
    "voxcpm2": "VoxCPM2 零样本 · 48kHz",
    "qwen3_tts": "Qwen3-TTS 零样本",
}


def _describe_character(character_id: str) -> str:
    """Label line for a character: its locked backend."""
    try:
        character = load_character(character_id)
    except ConfigError:
        return character_id
    backend = character.backend or load_app_config().default_backend or "-"
    return _BACKEND_LABELS.get(backend, backend) + "（锁定后端）"


def create_app(manager: BackendManager | None = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title=APP_ID)
    app_config = load_app_config()
    app_config.outputs_root.mkdir(parents=True, exist_ok=True)
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    manager = manager or BackendManager(logs_dir=app_config.logs_dir)
    session = _Session()

    app.mount("/outputs", StaticFiles(directory=str(app_config.outputs_root)),
              name="outputs")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        snap = session.snapshot()
        inputs = _list_inputs()
        characters = []
        for c in list_characters():
            try:
                load_character(c)
            except ConfigError:
                continue
            characters.append({
                "id": c,
                "label": load_character(c).display_name,
                "description": _describe_character(c),
            })
        return {
            "app_id": APP_ID,
            **snap,
            "characters": characters,
            "default_character": app_config.default_character,
            "input_files": [p.name for p in inputs],
            "can_generate": bool(inputs) and snap["status"] in (
                "idle", "completed", "failed", "interrupted"),
            "can_open_output": bool(snap["outputs"]),
        }

    def _batch(character_id: str) -> None:
        session.transition("preparing", "正在准备…")
        try:
            character = load_character(character_id)
        except ConfigError as exc:
            session.transition("failed", f"角色配置错误：{exc}")
            return
        backend_id = character.backend or app_config.default_backend
        try:
            profile = load_backend(backend_id)
        except ConfigError as exc:
            session.transition("failed",
                               f"角色 {character_id} 锁定的后端不可用：{exc}")
            return
        if not profile.enabled:
            session.transition("failed", f"角色 {character_id} 锁定的后端 "
                                         f"{backend_id} 未启用")
            return
        files = _list_inputs()
        if not files:
            session.transition("failed", "inputs/ 目录没有 .txt 文件")
            return
        out_dir = (app_config.outputs_root / "webui"
                   / character.character_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[str] = []
        try:
            if not manager.acquire_generate():
                raise BackendBusyError("另一个生成任务正在进行")
            try:
                session.transition("loading_model",
                                   f"正在加载后端 {backend_id}…")
                client = manager.ensure(profile)
                total = len(files)
                for i, txt_path in enumerate(files):
                    text = txt_path.read_text(encoding="utf-8").strip()
                    if not text:
                        continue
                    session.transition(
                        "generating",
                        f"[{i + 1}/{total}] {txt_path.name}："
                        f"{len(text)} 字，生成中…",
                        progress=100.0 * i / total,
                        current_file=txt_path.name)
                    stem = txt_path.stem or f"tts_{i}"
                    out_path = out_dir / f"{stem}.wav"
                    n = 1
                    while out_path.exists():
                        out_path = out_dir / f"{stem}-{n}.wav"
                        n += 1
                    client.generate(
                        text=text, output_path=str(out_path),
                        reference_audio=character.reference_audio,
                        reference_text=character.reference_text,
                        seed=42, options={}, timeout=1800)
                    outputs.append(out_path.name)
                    session.transition(
                        "generating",
                        f"[{i + 1}/{total}] {txt_path.name} 完成",
                        progress=100.0 * (i + 1) / total)
            finally:
                manager.release_generate()
        except Exception as exc:
            session.transition(
                "failed", f"{type(exc).__name__}: {exc}",
                errors=[f"{type(exc).__name__}: {exc}"])
            return
        session.transition("completed",
                           f"全部完成：{len(outputs)}/{len(files)} 个 WAV",
                           progress=100.0, outputs=outputs,
                           current_file=None)

    @app.post("/api/generate")
    def generate(payload: dict[str, Any]) -> dict[str, Any]:
        character_id = str(payload.get("character")
                           or app_config.default_character or "")
        try:
            character = load_character(character_id)
        except ConfigError:
            raise HTTPException(404, f"未知角色：{character_id}")
        backend_id = character.backend or app_config.default_backend
        try:
            if not load_backend(backend_id).enabled:
                raise HTTPException(
                    400, f"角色 {character_id} 锁定的后端 {backend_id} 未启用")
        except ConfigError:
            raise HTTPException(
                400, f"角色 {character_id} 锁定的后端 {backend_id} 不存在")
        if not _list_inputs():
            raise HTTPException(400, "inputs/ 目录没有 .txt 文件")
        if session.snapshot()["status"] in (
                "preparing", "loading_model", "generating", "finalizing"):
            raise HTTPException(409, "已有任务在进行中")
        session.selected_character = character_id
        session.outputs = []
        session.errors = []
        threading.Thread(target=_batch, args=(character_id,),
                         daemon=True).start()
        return {"started": True, "character": character_id,
                "backend": backend_id}

    @app.post("/api/open-output")
    def open_output() -> dict[str, str]:
        character_id = (session.snapshot()["selected_character"]
                        or app_config.default_character or "")
        path = str(app_config.outputs_root / "webui" / character_id)
        Path(path).mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606 - local desktop app
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        return {"opened": path}

    @app.on_event("shutdown")
    def _shutdown() -> None:
        try:
            manager.stop(wait_seconds=30)
        except BackendBusyError:
            logger.warning("shutdown during active generation; forcing "
                           "worker stop")
            manager.force_stop()

    app.state.manager = manager
    app.state.session = session
    return app


def _open_edge(url: str) -> None:
    """Open the page in Edge specifically; fall back to default browser."""
    if sys.platform == "win32":
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "msedge", url],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except OSError:
            pass
    webbrowser.open_new_tab(url)


def _existing_server_url(server_file: Path, timeout: float = 1.0) -> str | None:
    import json
    import urllib.request
    try:
        data = json.loads(server_file.read_text(encoding="utf-8"))
        url = str(data["url"])
        with urllib.request.urlopen(f"{url}/api/status",
                                    timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("app_id") == APP_ID:
            return url
    except Exception:
        return None
    return None


def main(host: str = "127.0.0.1", port: int = 7860,
         open_browser: bool = True) -> None:
    import json
    import uvicorn

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    server_dir = repo_root() / "outputs" / "webui"
    server_dir.mkdir(parents=True, exist_ok=True)
    server_file = server_dir / "server.json"

    existing = _existing_server_url(server_file) if server_file.is_file() else None
    if existing:
        logger.info("WebUI already running at %s — opening tab", existing)
        _open_edge(existing)
        return

    config = None
    server = None
    for p in range(port, port + 10):
        try:
            config = uvicorn.Config(create_app(), host=host, port=p,
                                    log_level="info")
            server = uvicorn.Server(config)
            # Pre-bind check happens inside server.run(); uvicorn binds in
            # startup. Cheap probe: try binding a raw socket first.
            import socket
            s = socket.socket()
            s.bind((host, p))
            s.close()
            break
        except OSError:
            continue
    if server is None:
        raise OSError(f"无法绑定端口 {port}-{port + 9}")

    url = f"http://{host}:{server.config.port}"
    server_file.write_text(json.dumps(
        {"app_id": APP_ID, "pid": os.getpid(), "url": url}, indent=2),
        encoding="utf-8")

    if open_browser:
        threading.Timer(0.8, _open_edge, args=(url,)).start()
    try:
        server.run()
    finally:
        try:
            registered = json.loads(server_file.read_text(encoding="utf-8"))
            if registered.get("pid") == os.getpid():
                server_file.unlink(missing_ok=True)
        except Exception:
            pass
