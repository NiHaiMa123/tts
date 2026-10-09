"""Review / reference-pick bundles: self-contained html + audio dirs.

Pages are served by ``scripts/ingest/serve_review.py`` — every click POSTs
to the local server which writes straight to disk (no export step):

- review page   -> ``<bundle>/decisions.json``  {drop:[sha], tags:{sha:[..]}}
- refpick page  -> ``assets/characters/<id>/reference/ref.wav``
                 + ``ref_choice.json`` + configs/characters/<id>.yaml update
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ISSUE_TAGS = ["wrong_speaker", "noisy", "wrong_text", "clipped",
              "emotion_off", "other_issue"]

_BASE_CSS = """
body{font-family:system-ui,sans-serif;background:#0b111b;color:#e8f2f4;
max-width:960px;margin:24px auto;padding:0 16px}
.row{padding:8px 10px;border:1px solid #223;border-radius:10px;
margin:6px 0;background:#111a28}
.row.drop{opacity:.45;border-color:#a44}
.row.picked{border-color:#1c4}
.top{display:flex;gap:10px;align-items:center}
audio{height:32px;width:280px}
.txt{flex:1;font-size:.86rem;color:#cfe0e4}
.badge{font-size:.7rem;color:#8fb}
.warn{font-size:.7rem;color:#fc9}
.tags{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}
button{cursor:pointer;border:1px solid #355;border-radius:8px;
background:#16222f;color:#e8f2f4;padding:4px 10px;font-size:.78rem}
button.on{background:#1c4;color:#04120a}
button.tag.on{background:#e8a33d;color:#1a1206}
button.drop.on{background:#c33;color:#fff}
button.pick.on{background:#1c4;color:#04120a}
#bar{position:sticky;top:0;background:#0b111bee;padding:10px 0;
display:flex;gap:10px;align-items:center;z-index:9}
small{color:#7d94a0}
#saved{color:#8fb;font-size:.78rem}
"""


def _rows(index_path: Path) -> list[dict]:
    return [json.loads(l) for l in
            Path(index_path).read_text(encoding="utf-8").splitlines()
            if l.strip() and json.loads(l).get("status") == "ok"]


def _cos_of(r: dict) -> float | None:
    for f in r.get("flags") or []:
        if f.startswith("speaker_cos:"):
            try:
                return float(f.split(":", 1)[1])
            except ValueError:
                return None
    return None


def _copy_bundle_audio(rows: list[dict], audio_root: Path,
                       out_dir: Path) -> None:
    bundle_audio = out_dir / "audio"
    bundle_audio.mkdir(parents=True, exist_ok=True)
    for r in rows:
        src = Path(audio_root) / r["audio"]
        dst = bundle_audio / r["audio"]
        if not dst.exists() and src.exists():
            shutil.copy2(src, dst)


def build_review_page(index_path: Path, audio_root: Path,
                      out_dir: Path) -> Path:
    """Keep/drop + issue-tag page. Writes decisions.json via server."""
    rows = _rows(index_path)
    out_dir = Path(out_dir)
    _copy_bundle_audio(rows, Path(audio_root), out_dir)

    items = [{
        "i": i, "file": f"audio/{r['audio']}", "sha": r["audio_sha256"],
        "label": r.get("label") or "", "text": r.get("text") or "",
        "text_asr": r.get("text_asr") or "",
        "dur": (r.get("metrics") or {}).get("duration_s", 0),
        "flags": r.get("flags") or [],
    } for i, r in enumerate(rows)]
    payload = json.dumps(items, ensure_ascii=False)
    tags_js = json.dumps(ISSUE_TAGS)

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>入库审核</title><style>{_BASE_CSS}</style></head><body>
<h2>入库审核 — 点击即保存（无需导出）</h2>
<div id="bar"><span id="saved">连接本地服务…</span><span id="count"></span>
  <small>drop=不入库；tag=留库但记录问题。需先运行 serve_review.py</small></div>
<div id="list"></div>
<script>
const items = {payload};
const ISSUE_TAGS = {tags_js};
const decisions = new Map();   // sha -> {{drop:bool, tags:Set}}
const list = document.getElementById('list');
function esc(s){{return s.replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
async function save() {{
  const drop=[],tags={{}};
  decisions.forEach((d,sha)=>{{if(d.drop)drop.push(sha);
    if(d.tags.size)tags[sha]=[...d.tags]}});
  try {{
    await fetch('/api/decisions',{{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{drop,tags}})}});
    document.getElementById('saved').textContent='已保存 ✓';
  }} catch(e) {{
    document.getElementById('saved').textContent='未连接服务（改动只在页面里）';
  }}
}}
async function restore() {{
  try {{
    const r = await fetch('/api/decisions'); const d = await r.json();
    for (const sha of d.drop||[]) decisions.set(sha,{{drop:true,tags:new Set()}});
    for (const [sha,tags] of Object.entries(d.tags||{{}})) {{
      const e = decisions.get(sha)||{{drop:false,tags:new Set()}};
      tags.forEach(t=>e.tags.add(t)); decisions.set(sha,e);}}
    items.forEach((_,i)=>refresh(i));
    document.getElementById('saved').textContent='已连接，历史决策已载入';
  }} catch(e) {{
    document.getElementById('saved').textContent='未连接 serve_review.py';
  }}
}}
for (const it of items) {{
  const div = document.createElement('div');
  div.className='row'; div.id='r'+it.i;
  const flagHtml = it.flags.length
    ? `<span class="warn">⚠ ${{esc(it.flags.join(' '))}}</span>` : '';
  const tagBtns = ISSUE_TAGS.map(t =>
    `<button class="tag" onclick="tag(${{it.i}},'${{t}}')">${{t}}</button>`).join('');
  div.innerHTML = `<div class="top"><audio controls preload="none"
    src="${{it.file}}"></audio>
    <span class="badge">${{esc(it.label)}}</span>
    <span class="txt">${{esc(it.text) || (it.text_asr ? 'ASR: '+esc(it.text_asr) : '（无文本）')}}
      <small>(${{it.dur}}s)</small> ${{flagHtml}}</span>
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
function drop(i){{entry(i).drop=!entry(i).drop;refresh(i);save()}}
function tag(i,t){{const d=entry(i);
  d.tags.has(t)?d.tags.delete(t):d.tags.add(t);refresh(i);save()}}
restore();
</script></body></html>"""
    out = out_dir / "review.html"
    out.write_text(page, encoding="utf-8")
    return out


def build_refpick_page(index_path: Path, audio_root: Path, out_dir: Path,
                       char_id: str, top_n: int = 40,
                       min_cos: float | None = None) -> Path:
    """Reference-audio pick page. Click writes ref.wav via server."""
    rows = _rows(index_path)

    def key(r):
        c = _cos_of(r)
        return c if c is not None else -1.0

    cands = [r for r in rows if
             (min_cos is None or (key(r) >= min_cos))]
    cands.sort(key=lambda r: -key(r))
    cands = cands[:top_n] if top_n else cands
    out_dir = Path(out_dir)
    _copy_bundle_audio(cands, Path(audio_root), out_dir)

    items = [{
        "i": i, "file": f"audio/{r['audio']}", "sha": r["audio_sha256"],
        "label": r.get("label") or "", "text": r.get("text") or "",
        "cos": _cos_of(r), "dur": (r.get("metrics") or {}).get("duration_s", 0),
    } for i, r in enumerate(cands)]
    payload = json.dumps(items, ensure_ascii=False)

    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>选择参考音 — {char_id}</title><style>{_BASE_CSS}</style></head><body>
<h2>选择参考音 — {char_id}（点击直接写入 assets/characters/{char_id}/reference/）</h2>
<div id="bar"><span id="saved">连接本地服务…</span>
  <small>候选按声纹 cos 降序；需先运行 serve_review.py</small></div>
<div id="list"></div>
<script>
const items = {payload};
const CHAR = {json.dumps(char_id)};
let picked = null;
const list = document.getElementById('list');
function esc(s){{return s.replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]))}}
for (const it of items) {{
  const div=document.createElement('div');
  div.className='row'; div.id='r'+it.i;
  div.innerHTML=`<div class="top"><audio controls preload="none"
    src="${{it.file}}"></audio>
    <span class="badge">${{esc(it.label)}}</span>
    <span class="txt">${{esc(it.text)}} <small>(${{it.dur}}s${{it.cos!=null?' cos '+it.cos:''}})</small></span>
    <button class="pick" onclick="pick(${{it.i}})">选为参考音</button></div>`;
  list.appendChild(div);
}}
async function pick(i) {{
  const it=items[i];
  try {{
    const r=await fetch('/api/ref-pick',{{method:'POST',
      headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{char:CHAR,audio:it.file.split('/').pop(),
        sha:it.sha,text:it.text}})}});
    const d=await r.json();
    if(d.ok){{picked=it.sha;refresh();
      document.getElementById('saved').textContent=
        '已写入 '+d.ref_path;}}
    else document.getElementById('saved').textContent='写入失败: '+d.error;
  }} catch(e) {{
    document.getElementById('saved').textContent='未连接 serve_review.py';
  }}
}}
async function restore() {{
  try {{
    const r=await fetch('/api/ref-choice?char='+CHAR);const d=await r.json();
    if(d.audio_sha256){{picked=d.audio_sha256;refresh();
      document.getElementById('saved').textContent='已载入历史选择';}}
    else document.getElementById('saved').textContent='已连接（尚未选择）';
  }} catch(e) {{
    document.getElementById('saved').textContent='未连接 serve_review.py';
  }}
}}
function refresh(){{items.forEach((it,i)=>{{
  const row=document.getElementById('r'+i);
  const on=it.sha===picked;
  row.classList.toggle('picked',on);
  row.querySelector('.pick').classList.toggle('on',on);
  row.querySelector('.pick').textContent=on?'✓ 当前参考音':'选为参考音';}})}}
restore();
</script></body></html>"""
    out = out_dir / "refpick.html"
    out.write_text(page, encoding="utf-8")
    return out


def load_decisions(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return set(data.get("drop") or data.get("exclude") or [])


def load_decision_tags(path: Path) -> dict[str, list[str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: list(v) for k, v in (data.get("tags") or {}).items()}
