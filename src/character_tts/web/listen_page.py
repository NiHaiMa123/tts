"""Static blind-listening page generator.

Emits a single self-contained index.html under the gate output dir. All
samples get an audio player plus subjective rating widgets (clarity /
grittiness / likeness / naturalness 1-5, stability and free notes).
Ratings persist to localStorage and export as JSON — the platform never
pre-fills subjective scores.
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
  <button onclick="exportRatings()">导出评分 JSON</button>
  <button class="secondary" onclick="clearRatings()">清空本页评分</button>
  <span class="sub">评分仅保存在浏览器 localStorage，平台不自动填写。</span>
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
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}

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

function save(el) {
  const id = el.dataset.id, dim = el.dataset.dim;
  ratings[id] = ratings[id] || {};
  ratings[id][dim] = el.value;
  localStorage.setItem(KEY, JSON.stringify(ratings));
}
function exportRatings() {
  const blob = new Blob([JSON.stringify({
    page: location.pathname, exported_at: new Date().toISOString(),
    ratings,
  }, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "listen-ratings.json";
  a.click();
}
function clearRatings() {
  if (!confirm("清除本页所有已保存评分？")) return;
  ratings = {};
  localStorage.removeItem(KEY);
  location.reload();
}
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
  <button onclick="exportRatings()">导出评分 JSON</button>
  <button class="secondary" onclick="toggleReveal()">解盲/隐藏真实标签</button>
  <button class="secondary" onclick="clearRatings()">清空本页评分</button>
  <span class="sub">评分存 localStorage；导出 JSON 含 experiment_id + case_id +
    output_sha256，重新生成 WAV 后旧评分不会误套。</span>
</div>
<div id="groups"></div>
<script>
const DATA = __SAMPLES__;
const KEY = "tts-listen-" + location.pathname;
let ratings = {};
try { ratings = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}

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

function save(el) {
  const id = el.dataset.id, dim = el.dataset.dim;
  ratings[id] = ratings[id] || {};
  ratings[id][dim] = el.value;
  localStorage.setItem(KEY, JSON.stringify(ratings));
}
function toggleReveal() {
  document.body.classList.toggle("revealed");
}
function exportRatings() {
  const sha = {};
  for (const g of DATA.groups)
    for (const s of g.items) sha[s.case_id] = s.sha256 || null;
  const out = {};
  for (const [id, r] of Object.entries(ratings))
    out[id] = {...r, output_sha256: sha[id] || null};
  const blob = new Blob([JSON.stringify({
    experiment_id: DATA.experiment_id,
    page: location.pathname,
    exported_at: new Date().toISOString(),
    ratings: out,
  }, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "listen-ratings-phase5a.json";
  a.click();
}
function clearRatings() {
  if (!confirm("清除本页所有已保存评分？")) return;
  ratings = {};
  localStorage.removeItem(KEY);
  location.reload();
}
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
        "__SAMPLES__", json.dumps(payload, ensure_ascii=False))
    out = listen_dir / "index.html"
    out.write_text(page, encoding="utf-8")
    (listen_dir / "unblind_map.json").write_text(
        json.dumps(unblind, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
