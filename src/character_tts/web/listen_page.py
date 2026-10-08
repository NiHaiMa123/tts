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
