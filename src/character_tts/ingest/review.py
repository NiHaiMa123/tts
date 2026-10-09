"""Blind review bundle for pool entries: keep/drop + issue tags.

Generates a self-contained ``<out_dir>/review.html`` + copied audio so the
bundle can be moved/served anywhere. The reviewer listens, toggles issue
tags and/or marks drop, then exports ``decisions.json``::

    {"drop": ["<audio_sha256>", ...],
     "tags": {"<audio_sha256>": ["wrong_speaker", "noisy", ...]}}

``freeze_dataset --exclude-file`` consumes the drop list; tags are advisory
and recorded into the freeze manifest for provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

ISSUE_TAGS = ["wrong_speaker", "noisy", "wrong_text", "clipped",
              "emotion_off", "other_issue"]


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
            "text_asr": r.get("text_asr") or "",
            "dur": (r.get("metrics") or {}).get("duration_s", 0),
            "flags": r.get("flags") or [],
        })
    payload = json.dumps(items, ensure_ascii=False)
    tags_js = json.dumps([t.strip() for t in ISSUE_TAGS])

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>入库审核</title><style>
body{{font-family:system-ui,sans-serif;background:#0b111b;color:#e8f2f4;
max-width:960px;margin:24px auto;padding:0 16px}}
.row{{padding:8px 10px;border:1px solid #223;border-radius:10px;
margin:6px 0;background:#111a28}}
.row.drop{{opacity:.45;border-color:#a44}}
.top{{display:flex;gap:10px;align-items:center}}
audio{{height:32px;width:280px}}
.txt{{flex:1;font-size:.86rem;color:#cfe0e4}}
.badge{{font-size:.7rem;color:#8fb}}
.warn{{font-size:.7rem;color:#fc9}}
.tags{{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}}
button{{cursor:pointer;border:1px solid #355;border-radius:8px;
background:#16222f;color:#e8f2f4;padding:4px 10px;font-size:.78rem}}
button.on{{background:#1c4;color:#04120a}}
button.tag.on{{background:#e8a33d;color:#1a1206}}
button.drop.on{{background:#c33;color:#fff}}
#bar{{position:sticky;top:0;background:#0b111bee;padding:10px 0;
display:flex;gap:10px;align-items:center;z-index:9}}
small{{color:#7d94a0}}
</style></head><body>
<h2>入库审核 — keep/drop + 问题标签，导出给 freeze --exclude-file</h2>
<div id="bar">
  <button onclick="download()">导出 decisions.json</button>
  <span id="count"></span>
  <small>drop=不入库；tag=留库但记录问题</small>
</div>
<div id="list"></div>
<script>
const items = {payload};
const ISSUE_TAGS = {tags_js};
const decisions = new Map();   // sha -> {{drop:bool, tags:Set}}
const list = document.getElementById('list');
function esc(s){{return s.replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
for (const it of items) {{
  const div = document.createElement('div');
  div.className = 'row'; div.id = 'r'+it.i;
  const flagHtml = it.flags.length
    ? `<span class="warn">⚠ ${{esc(it.flags.join(' '))}}</span>` : '';
  const tagBtns = ISSUE_TAGS.map(t =>
    `<button class="tag" onclick="tag(${{it.i}},'${{t}}')">${{t}}</button>`)
    .join('');
  div.innerHTML = `<div class="top"><audio controls preload="none"
    src="${{it.file}}"></audio>
    <span class="badge">${{esc(it.label)}}</span>
    <span class="txt">${{esc(it.text) || (it.text_asr ? 'ASR: '+esc(it.text_asr) : '（无文本）')}} <small>(${{it.dur}}s)</small> ${{flagHtml}}</span>
    <button class="drop" onclick="drop(${{it.i}})">drop</button></div>
    <div class="tags">${{tagBtns}}</div>`;
  list.appendChild(div);
}}
function entry(i){{let d=decisions.get(items[i].sha);
  if(!d){{d={{drop:false,tags:new Set()}};decisions.set(items[i].sha,d)}}return d}}
function refresh(i){{const d=decisions.get(items[i].sha);
  const row=document.getElementById('r'+i);
  row.classList.toggle('drop',!!(d&&d.drop));
  row.querySelectorAll('.tag').forEach(b=>{{
    b.classList.toggle('on',!!(d&&d.tags.has(b.textContent)))}});
  row.querySelector('.drop').classList.toggle('on',!!(d&&d.drop));
  let nd=0,nt=0;decisions.forEach(d=>{{if(d.drop)nd++;nt+=d.tags.size}});
  document.getElementById('count').textContent=`${{nd}} drop / ${{nt}} tags`;}}
function drop(i){{const d=entry(i);d.drop=!d.drop;refresh(i)}}
function tag(i,t){{const d=entry(i);d.tags.has(t)?d.tags.delete(t):d.tags.add(t);refresh(i)}}
function download() {{
  const drop=[],tags={{}};
  decisions.forEach((d,sha)=>{{if(d.drop)drop.push(sha);
    if(d.tags.size)tags[sha]=[...d.tags]}});
  const blob=new Blob([JSON.stringify({{drop,tags}},null,2)],
    {{type:'application/json'}});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='decisions.json';a.click();
}}
</script></body></html>"""

    out = out_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out


def load_decisions(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data.get("drop") or data.get("exclude") or [])


def load_decision_tags(path: Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: list(v) for k, v in (data.get("tags") or {}).items()}
