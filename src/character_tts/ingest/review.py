"""Minimal blind review page for pool entries: keep/drop -> decisions.json.

The page plays each clip and lets the reviewer mark keep/drop; export
produces {"fids_or_shas": [...] } consumed by freeze --exclude-file.
"""
from __future__ import annotations

import html
import json
from pathlib import Path


def build_review_page(index_path: Path, audio_root: Path,
                      out_dir: Path) -> Path:
    """Write a self-contained review bundle for an ingest index.

    ``audio_root`` is the pool dir on disk; clips are copied into
    ``out_dir/audio/`` so the bundle is portable — just open review.html.
    """
    import shutil
    rows = [json.loads(l) for l in
            Path(index_path).read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("status") == "ok"]
    out_dir = Path(out_dir)
    bundle_audio = out_dir / "audio"
    bundle_audio.mkdir(parents=True, exist_ok=True)

    items = []
    for i, r in enumerate(rows):
        src = Path(audio_root) / r["audio"]
        dst = bundle_audio / r["audio"]
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)
        items.append({
            "i": i, "file": f"audio/{r['audio']}", "sha": r["audio_sha256"],
            "label": r.get("label") or "", "text": r.get("text") or "",
            "dur": (r.get("metrics") or {}).get("duration_s", 0),
            "flags": r.get("flags") or [],
        })
    payload = json.dumps(items, ensure_ascii=False)

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>入库审核</title><style>
body{{font-family:system-ui,sans-serif;background:#0b111b;color:#e8f2f4;
max-width:900px;margin:24px auto;padding:0 16px}}
.row{{display:flex;gap:10px;align-items:center;padding:8px 10px;
border:1px solid #223;border-radius:10px;margin:6px 0;background:#111a28}}
.row.drop{{opacity:.45;border-color:#a44}}
audio{{height:32px;width:280px}}
.txt{{flex:1;font-size:.86rem;color:#cfe0e4}}
.badge{{font-size:.7rem;color:#8fb}}
button{{cursor:pointer;border:1px solid #355;border-radius:8px;
background:#16222f;color:#e8f2f4;padding:6px 12px}}
button.on{{background:#1c4; color:#04120a}}
#bar{{position:sticky;top:0;background:#0b111bee;padding:10px 0;
display:flex;gap:10px;align-items:center}}
</style></head><body>
<h2>入库审核 — keep/drop，导出给 freeze --exclude-file</h2>
<div id="bar">
  <button onclick="download()">导出 decisions.json</button>
  <span id="count"></span>
</div>
<div id="list"></div>
<script>
const items = {payload};
const decisions = new Map();
const list = document.getElementById('list');
for (const it of items) {{
  const div = document.createElement('div');
  div.className = 'row'; div.id = 'r'+it.i;
  div.innerHTML = `<audio controls preload="none"
    src="${{it.file}}"></audio>
    <span class="badge">${{it.label}}</span>
    <span class="txt">${{escapeHtml(it.text)}} <small>(${{it.dur}}s${{it.flags.length?' ⚠'+it.flags.join(','):''}})</small></span>
    <button onclick="mark(${{it.i}},'drop')">drop</button>`;
  list.appendChild(div);
}}
function escapeHtml(s){{return s.replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
function mark(i, v) {{
  decisions.set(items[i].sha, v);
  document.getElementById('r'+i).classList.toggle('drop', v==='drop');
  document.getElementById('count').textContent =
    `${{decisions.size}} 条标记 drop`;
}}
function download() {{
  const blob = new Blob([JSON.stringify({{drop:[...decisions.keys()]}},null,2)],
    {{type:'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'decisions.json'; a.click();
}}
</script></body></html>"""

    out = out_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out


def load_decisions(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data.get("drop") or data.get("exclude") or [])
