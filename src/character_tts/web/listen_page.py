"""Blind-listening page generator.

Emits a single self-contained index.html under the gate output dir. All
samples get an audio player plus subjective rating widgets (clarity /
grittiness / likeness / naturalness 1-5, stability and free notes).

Serve the gate dir with ``scripts/ingest/serve_review.py --bundle
outputs/gates/<exp>`` and every change POSTs straight to
``<exp>/listen/listen-ratings*.json`` (export format) plus a ``.state``
sidecar for restoring the page. Without the server the page falls back
to localStorage + manual JSON export. The platform never pre-fills
subjective scores.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 24px; background: #15171c; color: #e8e9eb; }
  h1 { font-size: 1.4rem; }
  .meta { color: #9aa0a8; font-size: .85rem; margin-bottom: 18px; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #2c313a; padding: 6px 8px; text-align: left;
           vertical-align: top; font-size: .88rem; }
  th { background: #1e222a; position: sticky; top: 0; }
  audio { width: 240px; }
  .sample-label { font-weight: 600; }
  .sub { color: #9aa0a8; font-size: .78rem; }
  select, input[type=text] { background: #0f1115; color: #e8e9eb;
    border: 1px solid #2c313a; border-radius: 4px; padding: 2px 4px;
    width: 56px; }
  input[type=text] { width: 120px; }
  .toolbar { margin: 16px 0; }
  button { background: #2d6cdf; color: #fff; border: 0; border-radius: 6px;
           padding: 8px 14px; cursor: pointer; font-size: .9rem; }
  button.secondary { background: #333a45; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="meta" id="meta"></div>
<div class="toolbar">
  <button onclick="exportRatings()">备用导出 JSON</button>
  <button class="secondary" onclick="clearRatings()">清空本页评分</button>
  <span id="saved" class="sub">连接本地服务…</span>
  <span class="sub">经 serve_review.py 打开时改动即写盘，无需导出。</span>
</div>
<table>
<thead><tr>
  <th>样本</th><th>播放</th><th>生成参数</th>
  <th>清澈度<br>1-5</th><th>磨砂/颗粒<br>1-5</th><th>像角色<br>1-5</th>
  <th>自然度<br>1-5</th><th>稳定性备注</th><th>其他备注</th>
</tr></thead>
<tbody id="rows"></tbody>
</table>
<script>
const SAMPLES = __SAMPLES__;
const KEY = "tts-listen-" + location.pathname;
const RATINGS_NAME = "listen-ratings.json";
let ratings = {};
let remoteOK = false, _pt = null;
function STATE() { return {ratings}; }
function adoptState(s) {
  if (s && s.ratings) {
    ratings = s.ratings;
    localStorage.setItem(KEY, JSON.stringify(ratings));
  }
}
function buildExport() {
  return {page: location.pathname,
          saved_at: new Date().toISOString(), ratings};
}
function setSaved(t) {
  const e = document.getElementById("saved");
  if (e) e.textContent = t;
}
function persist() {
  localStorage.setItem(KEY, JSON.stringify(STATE()));
  push();
}
function push() {
  if (!remoteOK) return;
  clearTimeout(_pt);
  _pt = setTimeout(async () => {
    try {
      await fetch("/api/listen-ratings", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: RATINGS_NAME,
          data: buildExport(), state: STATE()})});
      setSaved("已写入 listen/" + RATINGS_NAME + " ✓");
    } catch (e) {
      remoteOK = false;
      setSaved("服务断开，改动暂存浏览器");
    }
  }, 250);
}
function render() {
  document.getElementById("meta").textContent = SAMPLES.meta;
  const tbody = document.getElementById("rows");
  for (const s of SAMPLES.items) {
    const r = ratings[s.id] || {};
    const tr = document.createElement("tr");
    const ratingCell = (dim) =>
      `<select data-id="${s.id}" data-dim="${dim}" onchange="save(this)">
         <option value=""></option>` +
      [1,2,3,4,5].map(v =>
        `<option value="${v}" ${r[dim] == v ? "selected" : ""}>${v}</option>`
      ).join("") + `</select>`;
    const textCell = (dim) =>
      `<input type="text" data-id="${s.id}" data-dim="${dim}"
         value="${(r[dim] || "").replace(/"/g, "&quot;")}"
         onchange="save(this)">`;
    tr.innerHTML =
      `<td><div class="sample-label">${s.id}</div>
           <div class="sub">${s.backend} · ${s.kind}</div></td>
       <td><audio controls preload="none" src="${s.src}"></audio>
           <div class="sub">${s.duration}s · ${s.sample_rate}Hz</div></td>
       <td><div class="sub">${s.params}</div></td>
       <td>${ratingCell("clarity")}</td>
       <td>${ratingCell("grit")}</td>
       <td>${ratingCell("likeness")}</td>
       <td>${ratingCell("naturalness")}</td>
       <td>${textCell("stability")}</td>
       <td>${textCell("notes")}</td>`;
    tbody.appendChild(tr);
  }
}
async function boot() {
  try {
    const r = await fetch("/api/listen-ratings?name=" + RATINGS_NAME);
    remoteOK = true;
    if (r.ok) {
      const d = await r.json();
      if (d.state) adoptState(d.state);
    }
  } catch (e) {}
  if (!Object.keys(ratings).length)
    try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); }
    catch (e) {}
  setSaved(remoteOK ? "已连接 serve_review.py — 改动即写盘"
                    : "未连接 serve_review.py — 改动暂存浏览器");
  render();
}
function save(el) {
  const id = el.dataset.id, dim = el.dataset.dim;
  ratings[id] = ratings[id] || {};
  ratings[id][dim] = el.value;
  persist();
}
function exportRatings() {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = RATINGS_NAME;
  a.click();
}
function clearRatings() {
  if (!confirm("清除本页所有已保存评分？")) return;
  ratings = {};
  localStorage.removeItem(KEY);
  push();
  location.reload();
}
boot();
</script>
</body>
</html>
"""


def _fmt_params(sidecar: dict[str, Any] | None) -> str:
    if not sidecar:
        return ""
    parts: list[str] = []
    wr = sidecar.get("worker_result") or {}
    meta = wr.get("metadata") or {}
    gen = meta.get("generation") or {}
    for key in ("num_steps", "guidance_scale", "speaker_scale", "cfg_value",
                "inference_timesteps", "temperature", "top_k", "top_p",
                "ode_method", "variant", "language", "seed"):
        if key in gen:
            parts.append(f"{key}={gen[key]}")
    if sidecar.get("seed") is not None and "seed" not in gen:
        parts.append(f"seed={sidecar['seed']}")
    if meta.get("variant"):
        parts.append(f"variant={meta['variant']}")
    if meta.get("adapter"):
        parts.append(f"adapter={Path(str(meta['adapter'])).parent.name}")
    return html.escape(", ".join(dict.fromkeys(parts)))


def _top_level_role(name: str) -> tuple[str, str]:
    """(backend, kind) labels for wavs living at the output root."""
    if "ground_truth" in name:
        return "ground_truth", "原声(validation)"
    return "reference", "克隆参考音频"


def build_samples(output_dir: Path) -> dict[str, Any]:
    items = []
    wavs = sorted(
        Path(output_dir).rglob("*.wav"),
        # Originals first, then per-backend dirs — the listening flow
        # starts from the real utterances. Depth is judged on the path
        # *relative* to output_dir (p.parts counts absolute ancestors).
        key=lambda p: (len(p.relative_to(output_dir).parts) > 1,
                       p.as_posix()),
    )
    for wav in wavs:
        rel = wav.relative_to(output_dir)
        sidecar_path = wav.with_suffix(wav.suffix + ".json")
        sidecar = None
        if sidecar_path.is_file():
            try:
                sidecar = json.loads(
                    sidecar_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                sidecar = None
        if len(rel.parts) > 1:
            backend, kind = rel.parts[0], rel.stem
        else:
            backend, kind = _top_level_role(rel.stem)
        try:
            import soundfile as sf
            info = sf.info(str(wav))
            duration = round(info.duration, 2)
            sr = info.samplerate
        except Exception:
            duration, sr = "?", "?"
        items.append({
            "id": rel.as_posix(),
            "src": rel.as_posix(),
            "backend": backend,
            "kind": kind,
            "duration": duration,
            "sample_rate": sr,
            "params": _fmt_params(sidecar),
        })
    return {"items": items}


def write_listen_page(output_dir: str | Path, title: str,
                      meta_text: str = "") -> Path:
    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(output_dir)
    # Audio src paths are relative to output_dir; the page lives in
    # listen/, so prefix ../ for playback to work from the subdirectory.
    for item in samples["items"]:
        item["src"] = "../" + item["src"]
    samples["meta"] = meta_text
    page = _PAGE_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__SAMPLES__", json.dumps(samples, ensure_ascii=False))
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Phase 5A page: grouped, anonymized, sha256-bound ratings
# ---------------------------------------------------------------------------

_GROUP_TITLES = {
    "originals": "原声参考（非盲听）",
    "baseline": "基线（现用 Prompt P0）",
    "stability": "稳定性集（同 Prompt，多 seed）",
    "prompt_ab": "Prompt A/B（匿名条件）",
    "codec_diagnostics": "Codec 诊断链",
}

_PHASE5A_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 24px; background: #15171c; color: #e8e9eb; }
  h1 { font-size: 1.4rem; }
  h2 { font-size: 1.05rem; margin: 22px 0 6px; color: #9fc1ff; }
  .meta { color: #9aa0a8; font-size: .85rem; margin-bottom: 18px;
          white-space: pre-line; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #2c313a; padding: 6px 8px; text-align: left;
           vertical-align: top; font-size: .88rem; }
  th { background: #1e222a; position: sticky; top: 0; }
  audio { width: 230px; }
  .sample-label { font-weight: 600; }
  .sub { color: #9aa0a8; font-size: .78rem; }
  .fail { color: #ff8080; font-size: .82rem; }
  select, input[type=text] { background: #0f1115; color: #e8e9eb;
    border: 1px solid #2c313a; border-radius: 4px; padding: 2px 4px;
    width: 56px; }
  select.wide, input[type=text] { width: 110px; }
  .toolbar { margin: 16px 0; }
  button { background: #2d6cdf; color: #fff; border: 0; border-radius: 6px;
           padding: 8px 14px; cursor: pointer; font-size: .9rem; }
  button.secondary { background: #333a45; }
  .revealed .identity { display: inline; }
  .identity { display: none; color: #d8b45a; font-size: .78rem; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="meta" id="meta"></div>
<div class="toolbar">
  <button onclick="exportRatings()">备用导出 JSON</button>
  <button class="secondary" onclick="toggleReveal()">解盲/隐藏真实标签</button>
  <button class="secondary" onclick="clearRatings()">清空本页评分</button>
  <span id="saved" class="sub">连接本地服务…</span>
  <span class="sub">经 serve_review.py 打开时改动即写盘（含
    experiment_id + case_id + output_sha256，重新生成 WAV 后旧评分
    不会误套）；否则暂存 localStorage。</span>
</div>
<div id="groups"></div>
<script>
const DATA = __SAMPLES__;
const KEY = "tts-listen-" + location.pathname;
const RATINGS_NAME = "__RATINGS_NAME__";
let ratings = {};
let remoteOK = false, _pt = null;
function STATE() { return {ratings}; }
function adoptState(s) {
  if (s && s.ratings) {
    ratings = s.ratings;
    localStorage.setItem(KEY, JSON.stringify(ratings));
  }
}
function buildExport() {
  const sha = {};
  for (const g of DATA.groups)
    for (const s of g.items) sha[s.case_id] = s.sha256 || null;
  const out = {};
  for (const [id, r] of Object.entries(ratings))
    out[id] = {...r, output_sha256: sha[id] || null};
  return {experiment_id: DATA.experiment_id, page: location.pathname,
          saved_at: new Date().toISOString(), ratings: out};
}
function setSaved(t) {
  const e = document.getElementById("saved");
  if (e) e.textContent = t;
}
function persist() {
  localStorage.setItem(KEY, JSON.stringify(STATE()));
  push();
}
function push() {
  if (!remoteOK) return;
  clearTimeout(_pt);
  _pt = setTimeout(async () => {
    try {
      await fetch("/api/listen-ratings", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: RATINGS_NAME,
          data: buildExport(), state: STATE()})});
      setSaved("已写入 listen/" + RATINGS_NAME + " ✓");
    } catch (e) {
      remoteOK = false;
      setSaved("服务断开，改动暂存浏览器");
    }
  }, 250);
}
function render() {
  document.getElementById("meta").textContent = DATA.meta;
  const root = document.getElementById("groups");
  for (const g of DATA.groups) {
    const h = document.createElement("h2");
    h.textContent = g.title;
    root.appendChild(h);
    const table = document.createElement("table");
    table.innerHTML = `<thead><tr>
      <th>样本</th><th>播放</th>
      <th>清澈度<br>1-5</th><th>磨砂/颗粒<br>1-5</th><th>像角色<br>1-5</th>
      <th>自然度<br>1-5</th><th>文本完整</th><th>异常类型</th><th>备注</th>
    </tr></thead>`;
    const tbody = document.createElement("tbody");
    for (const s of g.items) {
      const r = ratings[s.case_id] || {};
      const tr = document.createElement("tr");
      const sel = (dim) =>
        `<select data-id="${s.case_id}" data-dim="${dim}" onchange="save(this)">
           <option value=""></option>` +
        [1,2,3,4,5].map(v =>
          `<option value="${v}" ${r[dim] == v ? "selected" : ""}>${v}</option>`
        ).join("") + `</select>`;
      const tc = (dim, opts) =>
        `<select class="wide" data-id="${s.case_id}" data-dim="${dim}"
           onchange="save(this)"><option value=""></option>` +
        opts.map(v =>
          `<option value="${v}" ${r[dim] == v ? "selected" : ""}>${v}</option>`
        ).join("") + `</select>`;
      const txt = (dim, ph) =>
        `<input type="text" data-id="${s.case_id}" data-dim="${dim}"
           placeholder="${ph || ''}"
           value="${(r[dim] || "").replace(/"/g, "&quot;")}"
           onchange="save(this)">`;
      const audioCell = s.src
        ? `<audio controls preload="none" src="${s.src}"></audio>
           <div class="sub">${s.duration}s · ${s.sample_rate}Hz</div>`
        : `<div class="fail">${s.error || "无输出"}</div>`;
      tr.innerHTML =
        `<td><div class="sample-label">${s.label}</div>
             <div class="identity">${s.case_id}</div></td>
         <td>${audioCell}</td>
         <td>${sel("clarity")}</td><td>${sel("grit")}</td>
         <td>${sel("likeness")}</td><td>${sel("naturalness")}</td>
         <td>${tc("text_complete", ["是", "否", "待审"])}</td>
         <td>${txt("artifact_type", "如:吞字/电流")}</td>
         <td>${txt("notes")}</td>`;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    root.appendChild(table);
  }
}
async function boot() {
  try {
    const r = await fetch("/api/listen-ratings?name=" + RATINGS_NAME);
    remoteOK = true;
    if (r.ok) {
      const d = await r.json();
      if (d.state) adoptState(d.state);
    }
  } catch (e) {}
  if (!Object.keys(ratings).length)
    try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); }
    catch (e) {}
  setSaved(remoteOK ? "已连接 serve_review.py — 改动即写盘"
                    : "未连接 serve_review.py — 改动暂存浏览器");
  render();
}
function save(el) {
  const id = el.dataset.id, dim = el.dataset.dim;
  ratings[id] = ratings[id] || {};
  ratings[id][dim] = el.value;
  persist();
}
function toggleReveal() {
  document.body.classList.toggle("revealed");
}
function exportRatings() {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = RATINGS_NAME;
  a.click();
}
function clearRatings() {
  if (!confirm("清除本页所有已保存评分？")) return;
  ratings = {};
  localStorage.removeItem(KEY);
  push();
  location.reload();
}
boot();
</script>
</body>
</html>
"""


_LORA_PILOT_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 24px; background: #15171c; color: #e8e9eb; }
  h1 { font-size: 1.4rem; }
  .meta { color: #9aa0a8; font-size: .85rem; margin-bottom: 14px;
          white-space: pre-line; }
  .calib { border: 1px solid #2c313a; border-radius: 8px; padding: 12px;
           margin-bottom: 18px; background: #191d24; }
  .calib h2 { font-size: 1rem; margin: 0 0 8px; color: #9fc1ff; }
  .pack { margin: 18px 0; border: 1px solid #2c313a; border-radius: 8px; }
  .pack-head { padding: 10px 14px; display: flex; gap: 12px;
               align-items: center; cursor: pointer; }
  .pack-head h2 { font-size: 1rem; margin: 0; color: #9fc1ff; flex: 1; }
  .pack-body { padding: 0 14px 14px; }
  .pair { border-top: 1px solid #242932; padding: 12px 0; }
  .pair-head { font-weight: 600; margin-bottom: 8px; }
  .side { display: inline-block; margin-right: 26px; vertical-align: top; }
  audio { width: 250px; }
  .sub { color: #9aa0a8; font-size: .78rem; }
  .choice label { margin-right: 14px; font-size: .9rem; }
  .tags label { margin-right: 10px; font-size: .82rem; color: #c9cdd4; }
  .fi { margin-top: 8px; font-size: .84rem; }
  select, input[type=text], input[type=number] { background: #0f1115;
    color: #e8e9eb; border: 1px solid #2c313a; border-radius: 4px;
    padding: 2px 4px; }
  input[type=number] { width: 60px; }
  input[type=text].wide { width: 140px; }
  button { background: #2d6cdf; color: #fff; border: 0; border-radius: 6px;
           padding: 8px 14px; cursor: pointer; font-size: .9rem;
           margin-right: 8px; }
  button.secondary { background: #333a45; }
  button.fatigue { background: #8a4b2a; }
  .toolbar { margin: 14px 0; }
  .dup { color: #d8b45a; font-size: .8rem; }
  .progress { color: #9aa0a8; font-size: .8rem; }
  details { margin-top: 6px; }
  .revealed .identity { display: inline; }
  .identity { display: none; color: #d8b45a; font-size: .78rem; }
  .stopped { opacity: .55; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="meta" id="meta"></div>
<div class="toolbar">
  <button onclick="exportRatings()">备用导出 JSON</button>
  <button class="secondary" onclick="toggleReveal()">解盲/隐藏真实标签</button>
  <button class="secondary" onclick="location.hash='calib'">回到参考校准</button>
  <button class="fatigue" onclick="markFatigue()">听麻木了/分不出来
    (UNCERTAIN_FATIGUE)</button>
  <button class="secondary" onclick="clearRatings()">清空</button>
  <span id="saved" class="sub">连接本地服务…</span>
  <span class="sub">每对先听哪边是随机且隐藏的；不要回放到听习惯为止——
    第一次听到就可以标记缺陷。疲劳请直接点橙色按钮停止，不要把
    「无差异」当疲劳。</span>
</div>
<div class="calib" id="calib">
  <h2>参考校准（先听原声，再开始配对）</h2>
  <div class="side"><div class="sub">干净原声（雨景，validation 真实录音）</div>
    <audio controls preload="auto" src="../reference_clean.wav"></audio></div>
  <div class="side"><div class="sub">Prompt P0 参考音频</div>
    <audio controls preload="none" src="../prompt_P0.wav"></audio></div>
</div>
<div id="packs"></div>
<script>
const DATA = __SAMPLES__;
const KEY = "tts-lora-pilot-" + location.pathname;
const RATINGS_NAME = "listen-ratings-lora-pilot.json";
let ratings = {}, session = {};
let remoteOK = false, _pt = null;
function STATE() { return {ratings, session}; }
function adoptState(s) {
  if (!s) return;
  ratings = s.ratings || {}; session = s.session || {};
  localStorage.setItem(KEY, JSON.stringify(STATE()));
}
function buildExport() {
  const pairs = {};
  for (const pack of DATA.packs)
    for (const pair of pack.pairs) {
      const r = ratings[pair.pair_id];
      const blind = {};
      pair.sides.forEach((s, i) => blind[i === 0 ? "A" : "B"] = s.case_id);
      pairs[pair.pair_id] = {
        batch: pack.pack_id,
        blind_order: blind,
        ...(r ? {...r,
          choice_real: r.choice === "A" ? blind.A :
                       r.choice === "B" ? blind.B : r.choice || null}
        : {status: session.fatigue ? "UNCERTAIN_FATIGUE"
                                   : "PENDING_USER_LISTENING"}),
      };
    }
  return {
    experiment_id: DATA.experiment_id,
    page: location.pathname,
    saved_at: new Date().toISOString(),
    session: {fatigue: !!session.fatigue,
              started_at: session.started_at || null,
              stopped_at: session.stopped_at || null},
    truth: DATA.truth,
    pairs,
  };
}
function setSaved(t) {
  const e = document.getElementById("saved");
  if (e) e.textContent = t;
}
function persist() {
  localStorage.setItem(KEY, JSON.stringify(STATE()));
  push();
}
function push() {
  if (!remoteOK) return;
  clearTimeout(_pt);
  _pt = setTimeout(async () => {
    try {
      await fetch("/api/listen-ratings", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({name: RATINGS_NAME,
          data: buildExport(), state: STATE()})});
      setSaved("已写入 listen/" + RATINGS_NAME + " ✓");
    } catch (e) {
      remoteOK = false;
      setSaved("服务断开，改动暂存浏览器");
    }
  }, 250);
}
function markFatigue() {
  session.fatigue = true;
  session.stopped_at = new Date().toISOString();
  persist();
  document.body.classList.add("stopped");
  alert("已记录 UNCERTAIN_FATIGUE。已自动写盘，可以停止本轮——疲劳样本不计入胜负。");
}

function render() {
document.getElementById("meta").textContent = DATA.meta;
if (session.fatigue) document.body.classList.add("stopped");
const root = document.getElementById("packs");
for (const pack of DATA.packs) {
  const wrap = document.createElement("div");
  wrap.className = "pack";
  const body = document.createElement("div");
  body.className = "pack-body";
  body.style.display = pack.collapsed ? "none" : "block";
  const head = document.createElement("div");
  head.className = "pack-head";
  head.innerHTML = `<h2>${pack.title}</h2>
    <span class="progress" data-prog="${pack.pack_id}"></span>
    <button class="secondary">${pack.collapsed ? "展开本组" : "收起"}</button>`;
  head.querySelector("button").onclick = (e) => {
    e.stopPropagation();
    body.style.display = body.style.display === "none" ? "block" : "none";
  };
  head.onclick = () => {
    body.style.display = body.style.display === "none" ? "block" : "none";
  };
  wrap.appendChild(head);

  for (const pair of pack.pairs) {
    const rated = ratings[pair.pair_id] || {};
    const card = document.createElement("div");
    card.className = "pair";
    const sidesHtml = pair.sides.map((s, i) => {
      const tag = i === 0 ? "甲" : "乙";
      return `<div class="side">
        <div><b>${tag}</b>
          <span class="identity">${s.case_id}</span>
          ${s.duplicate_sha ? '<span class="dup">同SHA重复样本</span>' : ""}
        </div>
        ${s.src ? `<audio controls preload="none" src="${s.src}"></audio>
          <div class="sub">${s.duration}s</div>`
        : `<div class="sub">缺失: ${s.error || "?"}</div>`}
      </div>`;
    }).join("");

    const choiceName = "c_" + pair.pair_id;
    const choices = [["A", "甲更好"], ["B", "乙更好"],
                     ["TIE", "无明显差异"], ["UNCERTAIN", "暂无法判断"]]
      .map(([v, lab]) =>
        `<label><input type="radio" name="${choiceName}" value="${v}"
          ${rated.choice === v ? "checked" : ""}
          onchange="saveChoice('${pair.pair_id}', this.value)"> ${lab}</label>`)
      .join("");

    const tags = DATA.defect_tags.map(t =>
      `<label><input type="checkbox" ${((rated.defect_tags || []).includes(t)) ? "checked" : ""}
        onchange="saveTags('${pair.pair_id}', this)"> ${t}</label>`)
      .join("");

    card.innerHTML = `
      <div class="pair-head">配对 ${pair.pair_id}
        <span class="sub">${pair.text_id} · seed ${pair.seed}</span>
        <span class="identity">${pair.cells.join(" vs ")}</span></div>
      ${sidesHtml}
      <div class="choice">${choices}</div>
      <div class="fi">
        <label><input type="checkbox" ${rated.first_impression ? "checked" : ""}
          onchange="saveFI('${pair.pair_id}', 'first_impression', this.checked)">
          第一次听到即标记缺陷（first_impression）</label>
        严重度 <select onchange="saveFI('${pair.pair_id}','severity',this.value)">
          <option value=""></option>
          ${["轻微","明显","严重"].map(v =>
            `<option ${rated.severity === v ? "selected" : ""}>${v}</option>`).join("")}
        </select>
        片段(选填) <input type="number" step="0.1" placeholder="起s"
          value="${rated.segment_start || ""}"
          onchange="saveFI('${pair.pair_id}','segment_start',this.value)">
        – <input type="number" step="0.1" placeholder="止s"
          value="${rated.segment_end || ""}"
          onchange="saveFI('${pair.pair_id}','segment_end',this.value)">
      </div>
      <div class="tags">最突出问题：${tags}</div>
      <details><summary class="sub">可选：四维 1-5 评分（非必填，不作主判据）</summary>
        ${["clarity","grit","likeness","naturalness"].map(d =>
          `<label class="sub">${d}
            <select onchange="saveFI('${pair.pair_id}','${d}',this.value)">
              <option value=""></option>
              ${[1,2,3,4,5].map(v =>
                `<option ${rated[d] == v ? "selected" : ""}>${v}</option>`).join("")}
            </select></label>`).join("")}
      </details>
      <div><input type="text" class="wide" placeholder="备注"
        value="${(rated.notes || "").replace(/"/g, "&quot;")}"
        onchange="saveFI('${pair.pair_id}','notes',this.value)"></div>`;
    body.appendChild(card);
  }
  wrap.appendChild(body);
  root.appendChild(wrap);
}
updateProgress();
}

function touch(pair_id) {
  ratings[pair_id] = ratings[pair_id] || {};
  ratings[pair_id].rated_at = new Date().toISOString();
}
function saveChoice(pair_id, v) { touch(pair_id); ratings[pair_id].choice = v; persist(); updateProgress(); }
function saveTags(pair_id, el) {
  touch(pair_id);
  const card = el.closest(".pair");
  ratings[pair_id].defect_tags = [...card.querySelectorAll(".tags input:checked")]
    .map(x => x.parentElement.textContent.trim());
  persist();
}
function saveFI(pair_id, dim, v) { touch(pair_id); ratings[pair_id][dim] = v; persist(); }
function updateProgress() {
  for (const pack of DATA.packs) {
    const done = pack.pairs.filter(p => (ratings[p.pair_id] || {}).choice).length;
    const el = document.querySelector(`[data-prog="${pack.pack_id}"]`);
    if (el) el.textContent = `${done}/${pack.pairs.length} 已评`;
  }
}
function toggleReveal() { document.body.classList.toggle("revealed"); }
function exportRatings() {
  const blob = new Blob([JSON.stringify(buildExport(), null, 2)],
                        {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = RATINGS_NAME;
  a.click();
}
function clearRatings() {
  if (!confirm("清除本页所有已保存评价？")) return;
  ratings = {}; session = {};
  localStorage.removeItem(KEY);
  push();
  location.reload();
}
async function boot() {
  try {
    const r = await fetch("/api/listen-ratings?name=" + RATINGS_NAME);
    remoteOK = true;
    if (r.ok) {
      const d = await r.json();
      if (d.state) adoptState(d.state);
    }
  } catch (e) {}
  if (!Object.keys(ratings).length && !Object.keys(session).length)
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || "{}");
      ratings = saved.ratings || {}; session = saved.session || {};
    } catch (e) {}
  session.started_at = session.started_at || new Date().toISOString();
  localStorage.setItem(KEY, JSON.stringify(STATE()));
  setSaved(remoteOK ? "已连接 serve_review.py — 改动即写盘"
                    : "未连接 serve_review.py — 改动暂存浏览器");
  render();
}
boot();
</script>
</body>
</html>
"""


def _duration_sr(path: Path) -> tuple[float | str, int | str]:
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return round(info.duration, 2), info.samplerate
    except Exception:
        return "?", "?"


def write_phase5a_listen_page(output_dir: str | Path,
                              manifest: dict[str, Any]) -> Path:
    """Grouped blind-listening page for the Phase 5A experiment.

    Samples are anonymized per group (S-xx); a reveal toggle and
    ``unblind_map.json`` expose the true case ids when the user wants
    them. Ratings export binds each entry to the output file's sha256.
    """
    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, Any]]] = {
        k: [] for k in _GROUP_TITLES}
    unblind: dict[str, Any] = {}

    def add(group: str, case_id: str, rel_src: str | None,
            label: str, extra: dict[str, Any] | None = None,
            error: str | None = None, sha256: str | None = None):
        item: dict[str, Any] = {
            "case_id": case_id, "label": label,
            "src": f"../{rel_src}" if rel_src else None,
            "sha256": sha256, "error": error,
            "duration": "", "sample_rate": "",
        }
        if rel_src:
            dur, sr = _duration_sr(output_dir / rel_src)
            item["duration"], item["sample_rate"] = dur, sr
        if extra:
            item.update(extra)
        groups[group].append(item)
        unblind[label] = {
            "case_id": case_id, "group": group, "src": rel_src,
            "output_sha256": sha256,
            **({} if extra is None else extra),
        }

    # originals (not blinded — they are the calibration anchors)
    gt = output_dir / "ground_truth_original.wav"
    ref = output_dir / "reference_original.wav"
    if gt.is_file():
        add("originals", "original/ground_truth_rain",
            "ground_truth_original.wav", "原声·validation 午后凉风",
            sha256=None)
    if ref.is_file():
        add("originals", "original/clone_reference_P0",
            "reference_original.wav", "克隆参考·谛天鉴(P0)",
            sha256=None)

    counters: dict[str, int] = {}
    cases = manifest.get("cases") or []
    for c in cases:
        group = "baseline" if (
            c.get("group") == "prompt_ab" and c.get("prompt_id") == "P0") \
            else c.get("group", "stability")
        counters[group] = counters.get(group, 0) + 1
        label = f"{group[0].upper()}-{counters[group]:02d}"
        rel = c.get("output")
        err = None if rel else (c.get("error") or "生成失败")
        add(group, c["case_id"], rel, label,
            extra={"text_id": c.get("text_id"),
                   "prompt_id": c.get("prompt_id"),
                   "seed": c.get("seed"),
                   "reused_from": c.get("reused_from")},
            error=err, sha256=c.get("output_sha256"))

    ccount = 0
    for iid, entry in (manifest.get("codec_diagnosis") or {}).items():
        arts = entry.get("artifacts") or {}
        enc = entry.get("encode_input") or {}
        for tag, node in (("encode_input", enc),
                          ("roundtrip48", arts.get("default") or {}),
                          ("roundtrip_cond16k",
                           arts.get("cond16000") or {})):
            p = node.get("path")
            if not p or not Path(p).is_file():
                continue
            ccount += 1
            rel = Path(p).relative_to(output_dir).as_posix()
            add("codec_diagnostics", f"codec/{iid}:{tag}", rel,
                f"C-{ccount:02d}",
                extra={"input_id": iid, "variant": tag},
                sha256=node.get("sha256"))

    group_list = [{"key": k, "title": _GROUP_TITLES[k], "items": v}
                  for k, v in groups.items() if v]
    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "标签默认匿名；点「解盲」查看真实 case_id。"
            "所有主观评分默认 PENDING_USER_LISTENING。"),
        "groups": group_list,
    }
    page = _PHASE5A_TEMPLATE.replace(
        "__TITLE__", html.escape(
            f"Phase 5A 盲听 — {manifest.get('experiment_id')}")).replace(
        "__RATINGS_NAME__", "listen-ratings-phase5a.json").replace(
        "__SAMPLES__", json.dumps(payload, ensure_ascii=False))
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


DEFECT_TAGS = ["磨砂/粗糙", "前顶/舌位", "空腔/口型大", "语速/停顿",
               "音色不符", "漏字/错读", "金属感", "其他"]


def write_lora_pilot_listen_page(output_dir: str | Path,
                                 manifest: dict[str, Any],
                                 rng_seed: int = 20261008) -> Path:
    """Low-fatigue paired A/B page for the LoRA pilot (PLAN 25.7).

    - pack1 shown by default (4 pairs), later packs collapsed
    - per pair: two blinded sides (甲/乙), deterministic shuffle per
      pair_id; side order balanced across pairs
    - choice + defect tags + first_impression + optional scores
    - UNCERTAIN_FATIGUE session flag; same-sha pairs flagged as
      duplicate evidence, never counted twice
    """
    import random

    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    cases = {c["case_id"]: c for c in manifest.get("cases") or []}
    sha_count: dict[str, int] = {}
    for c in cases.values():
        if c.get("output_sha256"):
            sha_count[c["output_sha256"]] = \
                sha_count.get(c["output_sha256"], 0) + 1

    def _side(case_id: str) -> dict[str, Any] | None:
        c = cases.get(case_id)
        if not c:
            return None
        rel = c.get("output")
        src = f"../{rel}" if rel else None
        dur, sr = _duration_sr(output_dir / rel) if rel else ("?", "?")
        return {"case_id": case_id, "condition": c.get("condition"),
                "src": src, "sha256": c.get("output_sha256"),
                "duration": dur, "sample_rate": sr,
                "error": None if c.get("status") == "ok"
                else (c.get("error") or c["status"]),
                "duplicate_sha": bool(c.get("output_sha256")
                                      and sha_count.get(
                                          c["output_sha256"], 0) > 1)}

    packs_payload, unblind = [], {}
    for pack in manifest.get("packs") or []:
        pairs = []
        for pid in pack.get("pairs") or []:
            pair = next((p for p in manifest.get("pairs") or []
                         if p["pair_id"] == pid), None)
            if not pair:
                continue
            cells = [s for s in (_side(c) for c in pair["cells"]) if s]
            rng = random.Random(f"{rng_seed}:{pid}")
            order = list(range(len(cells)))
            rng.shuffle(order)
            sides = [cells[i] for i in order]
            pairs.append({"pair_id": pid,
                          "text_id": pair["text_id"],
                          "seed": pair["seed"],
                          "cells": [s["case_id"] for s in cells],
                          "sides": sides})
            for i, s in enumerate(sides):
                unblind[f"{pid}:{'甲乙'[i]}"] = s["case_id"]
        packs_payload.append({"pack_id": pack["pack_id"],
                              "title": pack["title"],
                              "collapsed": pack.get("collapsed", False),
                              "pairs": pairs})

    truth = {c["case_id"]: {"condition": c.get("condition"),
                            "checkpoint": c.get("checkpoint"),
                            "output_sha256": c.get("output_sha256")}
             for c in cases.values()}
    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "defect_tags": DEFECT_TAGS,
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "先听参考原声校准 → 每组 3–5 对：甲/乙随机隐藏（同文同 "
            "seed，唯一变量是 checkpoint）。每对选「甲更好/乙更好/无"
            "明显差异/暂无法判断」+ 最突出问题标签；可选标首次印象。"
            "疲劳点橙色按钮停止，未听项保持 PENDING_USER_LISTENING。"),
        "packs": packs_payload,
        "truth": truth,
    }
    page = _LORA_PILOT_TEMPLATE.replace(
        "__TITLE__", html.escape(
            f"Phase 5B LoRA pilot 低疲劳 A/B — "
            f"{manifest.get('experiment_id')}")).replace(
        "__SAMPLES__", json.dumps(payload, ensure_ascii=False))
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def write_phase5a2_listen_page(output_dir: str | Path,
                               manifest: dict[str, Any],
                               rng_seed: int = 20261008) -> Path:
    """Paired blind-listening page for Phase 5A2.

    Each pair (same text + seed) is one section; the P0/P2 cells appear
    as anonymous 甲/乙 in a deterministic-but-randomized order so prompt
    identity stays hidden. Reveal toggle + ``unblind_map.json`` expose
    the truth; exports bind experiment_id + case_id + output_sha256.
    """
    import random

    output_dir = Path(output_dir)
    listen_dir = output_dir / "listen"
    listen_dir.mkdir(parents=True, exist_ok=True)

    cases = {c["case_id"]: c for c in manifest.get("cases") or []}
    groups: list[dict[str, Any]] = []
    unblind: dict[str, Any] = {}

    # Calibration originals — not blinded.
    orig_items = []
    for fname, label, cid in (
            ("ground_truth_original.wav", "原声·validation 午后凉风",
             "original/ground_truth_rain"),
            ("prompt_P0_original.wav", "Prompt 音频·甲来源",
             "original/prompt_P0"),
            ("prompt_P2_original.wav", "Prompt 音频·乙来源",
             "original/prompt_P2")):
        p = output_dir / fname
        if not p.is_file():
            continue
        dur, sr = _duration_sr(p)
        orig_items.append({
            "case_id": cid, "label": label, "src": f"../{fname}",
            "sha256": None, "error": None,
            "duration": dur, "sample_rate": sr,
        })
    if orig_items:
        groups.append({"key": "originals", "title": _GROUP_TITLES[
            "originals"], "items": orig_items})

    # One group per pair; side order shuffled deterministically.
    for i, pair in enumerate(manifest.get("pairs") or []):
        cells = [cases[cid] for cid in pair["cells"] if cid in cases]
        rng = random.Random(f"{rng_seed}:{pair['pair_id']}")
        order = list(range(len(cells)))
        rng.shuffle(order)
        items = []
        for side, idx in enumerate(order):
            c = cells[idx]
            label = ("甲", "乙")[side]
            rel = c.get("output")
            src = f"../{rel}" if rel else None
            dur, sr = _duration_sr(output_dir / rel) if rel else ("?", "?")
            items.append({
                "case_id": c["case_id"], "label": label,
                "src": src, "sha256": c.get("output_sha256"),
                "error": None if c.get("status") == "ok"
                else (c.get("error") or c["status"]),
                "duration": dur, "sample_rate": sr,
            })
            unblind[f"{pair['pair_id']}:{label}"] = {
                "case_id": c["case_id"], "prompt_id": c.get("prompt_id"),
                "text_id": pair["text_id"], "seed": pair["seed"],
                "src": rel, "output_sha256": c.get("output_sha256"),
                "reused_from": c.get("reused_from"),
            }
        groups.append({
            "key": pair["pair_id"],
            "title": f"配对 {i + 1}：{pair['text_id']} · "
                     f"seed {pair['seed']}",
            "items": items,
        })

    payload = {
        "experiment_id": manifest.get("experiment_id"),
        "meta": (
            f"实验 {manifest.get('experiment_id')} · "
            f"backend {manifest.get('backend')} · "
            f"rev {str(manifest.get('model', {}).get('revision'))[:12]}\n"
            "每个配对内 甲/乙 为匿名条件（同一文本同一 seed，仅 Prompt "
            "不同）；点「解盲」看真实 case_id。主观评分默认 "
            "PENDING_USER_LISTENING。"),
        "groups": groups,
    }
    page = _PHASE5A_TEMPLATE.replace(
        "__TITLE__", html.escape(
            f"Phase 5A2 配对盲听 — {manifest.get('experiment_id')}")).replace(
        "__RATINGS_NAME__", "listen-ratings-phase5a2.json").replace(
        "__SAMPLES__", json.dumps(payload, ensure_ascii=False))
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
