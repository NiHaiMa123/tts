"""One local-file review template for every listening experiment."""
from __future__ import annotations

import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote


def render_listen_page(output_dir: Path, title: str, data: dict[str, Any],
                       ratings_name: str, mode: str) -> str:
    """Bind page state to the experiment, audio bytes and blind order."""
    output_dir = Path(output_dir).resolve()
    data["experiment_id"] = data.get("experiment_id") or output_dir.name
    data["mode"] = mode
    data["ratings_name"] = ratings_name
    if mode == "pairs":
        data.setdefault("calibration", [
            {"case_id": "calibration/" + name, "src": "../" + name,
             "label": label, "duration": "", "sample_rate": ""}
            for name, label in (("reference_clean.wav", "干净原声"),
                                ("prompt_P0.wav", "克隆参考"))])
    assets: dict[str, str | None] = {}
    audio_paths: dict[str, str] = {}
    items = list(data.get("items", []))
    items.extend(data.get("calibration", []))
    for group in data.get("groups", []):
        items.extend(group["items"])
    for pack in data.get("packs", []):
        for pair in pack["pairs"]:
            items.extend(pair["sides"])
    for item in items:
        cid = item.get("case_id") or item["id"]
        item["case_id"] = cid
        src = item.get("src")
        sha = None
        if src:
            logical = Path(os.path.abspath(output_dir / "listen" / unquote(src)))
            path = logical.resolve()
            if not logical.is_relative_to(output_dir) and not item.get("sha256"):
                raise ValueError(f"listening audio outside bundle: {cid}")
            if path.is_file():
                sha = hashlib.sha256(path.read_bytes()).hexdigest()
                if item.get("sha256") and item["sha256"] != sha:
                    raise ValueError(f"listening audio hash mismatch: {cid}")
                # Existing experiments reuse earlier WAVs via symlinks. Make
                # the served bundle self-contained without exposing that tree.
                if not path.is_relative_to(output_dir):
                    local = output_dir / "listen" / "audio" / f"{sha}.wav"
                    local.parent.mkdir(parents=True, exist_ok=True)
                    if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != sha:
                        shutil.copy2(path, local)
                    path = local
                    item["src"] = f"audio/{sha}.wav"
                audio_paths[cid] = path.relative_to(output_dir).as_posix()
            else:
                item["src"] = None
                item["error"] = item.get("error") or "音频缺失"
        item["sha256"] = sha
        assets[cid] = sha
        if cid in data.get("truth", {}):
            data["truth"][cid]["output_sha256"] = sha
    data["asset_hashes"] = assets
    # Include blind order: changing A/B order must invalidate old choices.
    identity = hashlib.sha256(json.dumps(
        data, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    data["identity"] = identity
    contract = {"schema_version": "local-listening-v1",
                "experiment_id": data["experiment_id"],
                "identity": identity, "ratings_name": ratings_name,
                "mode": mode, "asset_hashes": assets,
                "audio_paths": audio_paths,
                "blind_orders": {p["pair_id"]: {
                    "A": p["sides"][0]["case_id"],
                    "B": p["sides"][1]["case_id"]}
                    for g in data.get("packs", []) for p in g["pairs"]
                    if len(p["sides"]) == 2},
                "review_ids": ([p["pair_id"] for g in data.get("packs", [])
                                for p in g["pairs"]] if mode == "pairs"
                               else list(assets))}
    (output_dir / "listen" / "review-contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    return TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", payload)


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#edf6f7;background:#0b111b;
 --muted:#a6b6c1;--line:#2b3a4c;--accent:#73eee3;--danger:#ff9aa7}
*{box-sizing:border-box}body{margin:0;padding:24px}main{max-width:1200px;margin:auto}
h1{font-size:1.5rem;margin:0 0 10px}h2{font-size:1.1rem;margin:0}
.meta,.sub{color:var(--muted);font-size:.85rem;white-space:pre-line}
.toolbar,.card,.calib{background:#151e2c;border:1px solid var(--line);border-radius:14px;padding:16px;margin:16px 0}
.toolbar{position:sticky;top:8px;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.saved{color:var(--accent)}.error{color:var(--danger)}button{border:1px solid var(--line);border-radius:8px;
 background:#22344a;color:#edf6f7;padding:8px 12px;cursor:pointer}button.fatigue{background:#744127}
button:disabled{opacity:.45;cursor:default}.side{display:inline-block;vertical-align:top;margin:12px 20px 12px 0}
audio{width:280px;max-width:100%}.identity{display:none;color:#ffd18c}.revealed .identity{display:block}
.fields,.choice,.tags{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0}
label{font-size:.9rem}select,input,textarea{background:#0d1622;color:#edf6f7;border:1px solid var(--line);
 border-radius:6px;padding:6px}textarea{width:100%;min-height:60px}input[type=number]{width:80px}
details{margin:16px 0}summary{cursor:pointer}.pair{border-top:1px solid var(--line);padding:16px 0}
.stopped #content{opacity:.7}.progress{color:var(--muted);margin-left:12px}
@media(max-width:600px){body{padding:12px}.toolbar{position:static}.side{display:block;margin-right:0}audio{width:100%}}
</style>
</head>
<body><main>
<h1>__TITLE__</h1><div id="meta" class="meta"></div>
<div class="toolbar">
 <span id="saved" role="status" aria-live="polite">连接本地服务…</span>
 <button id="retry" type="button">重试保存</button>
 <button id="reveal" type="button">解盲 / 隐藏标签</button>
 <button id="fatigue" class="fatigue" type="button">听累了 / 分不出来</button>
 <button id="clear" type="button">清空本轮评价</button>
 <span class="sub">选择即写入本地文件；显示“已保存”后，直接告诉 Agent“继续”。</span>
</div>
<section id="calibration" class="calib" hidden></section>
<div id="content"></div>
<p class="sub">清澈度、无磨砂/颗粒、像角色、自然度均为 1–5，5 分表示更好。主观评分不预填。</p>
</main>
<script>
const DATA = __DATA__;
const RATINGS_NAME = DATA.ratings_name;
const KEY = 'tts-local-review:' + DATA.experiment_id + ':' + DATA.identity;
let ratings = {}, session = {}, revision = null;
let pending = null, inflight = false, connected = false, retryTimer = null;
let dirty = false, serial = 0, ready = false;
const esc = v => String(v ?? '').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const state = () => ({ratings, session, identity:DATA.identity});
function status(text, error=false){const e=document.getElementById('saved');e.textContent=text;e.className=error?'error':'saved';}
function cache(){try{localStorage.setItem(KEY,JSON.stringify({...state(),dirty,base_revision:revision}));}catch(e){}}
function buildExport(){
 const common={schema_version:'local-listening-v1',experiment_id:DATA.experiment_id,
   identity:DATA.identity,page:location.pathname,saved_at:new Date().toISOString()};
 if(DATA.mode==='pairs'){
  const pairs={};
  for(const pack of DATA.packs)for(const pair of pack.pairs){
   const r=ratings[pair.pair_id], blind={};pair.sides.forEach((s,i)=>blind[i===0?'A':'B']=s.case_id);
   pairs[pair.pair_id]={batch:pack.pack_id,blind_order:blind,
    ...(r?{...r,choice_real:r.choice==='A'?blind.A:r.choice==='B'?blind.B:r.choice||null}:
       {status:session.fatigue?'UNCERTAIN_FATIGUE':'PENDING_USER_LISTENING'})};
  }
  return {...common,session:{...session,fatigue:!!session.fatigue},truth:DATA.truth,pairs};
 }
 const out={};for(const [id,r]of Object.entries(ratings))out[id]={...r,output_sha256:DATA.asset_hashes[id]||null};
 return {...common,session,ratings:out};
}
function persist(){
 session.draft_at=new Date().toISOString();
 dirty=true;cache();pending={name:RATINGS_NAME,identity:DATA.identity,data:buildExport(),state:state(),serial:++serial};
 status('正在写入本地文件…');flush();
}
async function flush(){
 if(inflight||!pending||!ready)return;
 clearTimeout(retryTimer);inflight=true;
 const snapshot=pending;pending=null;
 try{
  const response=await fetch('/api/listen-ratings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({...snapshot,expected_revision:revision})});
  const result=await response.json();
  if(!response.ok||result.ok!==true){const err=new Error(result.error||'保存失败');err.conflict=response.status===409;throw err;}
  revision=result.revision;connected=true;
  cache();
  if(!pending&&snapshot.serial===serial){dirty=false;cache();status('已保存到本地文件 ✓ 可以告诉 Agent“继续”');}
 }catch(e){
  if(!pending)pending=snapshot;
  connected=false;status(e.conflict?'文件或实验已变化，请刷新后继续；当前选择尚未写盘':
    '未写入本地文件：'+e.message+'。请保持本页并重试。',true);
  if(!e.conflict)retryTimer=setTimeout(flush,1800);
 }finally{inflight=false;if(pending&&connected)flush();}
}
function record(id,dim,value){
 ratings[id]=ratings[id]||{};ratings[id][dim]=value;ratings[id].rated_at=new Date().toISOString();
 persist();updateProgress();
}
function fields(id,r){
 return '<div class="fields">'+[['clarity','清澈度'],['grit','无磨砂/颗粒'],['likeness','像角色'],['naturalness','自然度']]
  .map(([key,label])=>`<label>${label} <select data-id="${esc(id)}" data-dim="${key}"><option value=""></option>`+
    [1,2,3,4,5].map(v=>`<option value="${v}" ${r[key]==v?'selected':''}>${v}</option>`).join('')+'</select></label>').join('')+
  `<label>文本完整 <select data-id="${esc(id)}" data-dim="text_complete"><option value=""></option>`+
    ['是','否','待审'].map(v=>`<option ${r.text_complete===v?'selected':''}>${v}</option>`).join('')+'</select></label></div>';
}
function noteFields(id,r){return `<div class="fields"><label>异常类型 <input data-id="${esc(id)}" data-dim="artifact_type" value="${esc(r.artifact_type)}"></label>`+
 `<label>稳定性 <input data-id="${esc(id)}" data-dim="stability" value="${esc(r.stability)}"></label></div>`+
 `<textarea aria-label="备注" placeholder="备注（选填）" data-id="${esc(id)}" data-dim="notes">${esc(r.notes)}</textarea>`;}
function audio(s,label){return `<div class="side"><b>${esc(label)}</b><span class="identity">${esc(s.case_id)}</span>`+
 (s.src?`<audio controls preload="none" src="${esc(s.src)}"></audio><div class="sub">${esc(s.duration)}s · ${esc(s.sample_rate)}Hz</div>`:
 `<div class="error">${esc(s.error||'无输出')}</div>`)+(s.duplicate_sha?'<div class="sub">同 SHA 重复证据</div>':'')+'</div>';}
function sample(s){const id=s.case_id,r=ratings[id]||{};return `<article class="card">${audio(s,s.label||s.kind||s.backend)}`+
 `<div class="identity">${esc(s.params)}</div>`+fields(id,r)+noteFields(id,r)+'</article>';}
function pairCard(pair){
 const id=pair.pair_id,r=ratings[id]||{};
 return `<article class="pair"><h2>配对 ${esc(id)} <span class="sub">${esc(pair.text_id)} · seed ${esc(pair.seed)}</span></h2>`+
 pair.sides.map((s,i)=>audio(s,i===0?'甲':'乙')).join('')+
 '<div class="choice">'+[['A','甲更好'],['B','乙更好'],['TIE','无明显差异'],['UNCERTAIN','暂无法判断']]
  .map(([v,label])=>`<label><input type="radio" name="choice_${esc(id)}" data-id="${esc(id)}" data-dim="choice" value="${v}" ${r.choice===v?'checked':''}>${label}</label>`).join('')+'</div>'+
 `<div class="tags">`+(DATA.defect_tags||[]).map(tag=>`<label><input type="checkbox" data-id="${esc(id)}" data-tag="${esc(tag)}" ${(r.defect_tags||[]).includes(tag)?'checked':''}>${esc(tag)}</label>`).join('')+'</div>'+
 `<label><input type="checkbox" data-id="${esc(id)}" data-dim="first_impression" ${r.first_impression?'checked':''}>首次印象即发现缺陷</label>`+
 `<div class="fields"><label>严重度 <select data-id="${esc(id)}" data-dim="severity"><option value=""></option>`+
 ['轻微','明显','严重'].map(v=>`<option ${r.severity===v?'selected':''}>${v}</option>`).join('')+'</select></label>'+
 ['segment_start','segment_end'].map((key,i)=>`<label>${i?'结束秒':'开始秒'} <input type="number" min="0" step="0.1" data-id="${esc(id)}" data-dim="${key}" value="${esc(r[key])}"></label>`).join('')+'</div>'+
 `<details><summary>可选：四维评分</summary>${fields(id,r)}</details>`+noteFields(id,r)+'</article>';
}
function render(){
 document.getElementById('meta').textContent=DATA.meta||'';
 const root=document.getElementById('content');root.replaceChildren();
 if(DATA.mode==='pairs'){
  const calib=document.getElementById('calibration');calib.hidden=false;
  calib.innerHTML='<h2>先听原声校准</h2>'+(DATA.calibration||[]).map(s=>audio(s,s.label)).join('');
  for(const pack of DATA.packs){const d=document.createElement('details');d.className='card';d.open=!pack.collapsed;
   d.innerHTML=`<summary>${esc(pack.title)}<span class="progress" data-pack="${esc(pack.pack_id)}"></span></summary>`+pack.pairs.map(pairCard).join('');root.appendChild(d);}
 }else if(DATA.groups){for(const g of DATA.groups){const section=document.createElement('section');
  section.innerHTML=`<h2>${esc(g.title)}</h2>`+g.items.map(sample).join('');root.appendChild(section);}}
 else root.innerHTML=DATA.items.map(sample).join('');
 document.body.classList.toggle('stopped',!!session.fatigue);updateProgress();
}
function updateProgress(){for(const pack of DATA.packs||[]){const done=pack.pairs.filter(p=>(ratings[p.pair_id]||{}).choice).length;
 const e=[...document.querySelectorAll('[data-pack]')].find(e=>e.dataset.pack===pack.pack_id);if(e)e.textContent=`${done}/${pack.pairs.length} 已评`;}}
document.getElementById('content').addEventListener('input',e=>{const el=e.target;
 if(el.matches('input:not([type=radio]):not([type=checkbox]),textarea'))record(el.dataset.id,el.dataset.dim,el.value);});
document.getElementById('content').addEventListener('change',e=>{const el=e.target;if(!el.dataset.id)return;
 if(el.dataset.tag){const r=ratings[el.dataset.id]||{};const tags=new Set(r.defect_tags||[]);el.checked?tags.add(el.dataset.tag):tags.delete(el.dataset.tag);record(el.dataset.id,'defect_tags',[...tags]);}
 else if(el.matches('select,input[type=radio],input[type=checkbox]'))record(el.dataset.id,el.dataset.dim,el.type==='checkbox'?el.checked:el.value);});
document.getElementById('reveal').onclick=()=>document.body.classList.toggle('revealed');
document.getElementById('retry').onclick=()=>flush();
document.getElementById('fatigue').onclick=()=>{session.fatigue=true;session.stopped_at=new Date().toISOString();persist();document.body.classList.add('stopped');};
document.getElementById('clear').onclick=()=>{if(!confirm('清空本轮评价？'))return;ratings={};session={started_at:new Date().toISOString()};persist();render();};
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='选择尚未写入本地文件';}});
document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden'&&pending)flush();});
async function boot(){
 if(location.protocol==='file:'){status('请用 serve_review.py 打开本地试听网址，才能直接写盘。',true);render();
  document.querySelectorAll('button,input,select,textarea').forEach(e=>e.disabled=true);return;}
 try{
  const response=await fetch('/api/listen-ratings?name='+encodeURIComponent(RATINGS_NAME),{cache:'no-store'});
  const d=await response.json();
  if(!response.ok)throw new Error(d.error||'本地服务未提供评分接口');
  if(!('state'in d))throw new Error('本地评分接口不兼容');
  revision=d.revision||null;
  if(d.state&&d.identity===DATA.identity){ratings=d.state.ratings||{};session=d.state.session||{};}
  let cached=null;try{cached=JSON.parse(localStorage.getItem(KEY)||'null');}catch(e){}
  // Disk is authoritative. Recover browser drafts only when newer than disk.
 if(cached?.dirty&&cached.identity===DATA.identity&&cached.base_revision===revision&&(!d.saved_at||Date.parse(cached.session?.draft_at)>=Date.parse(d.saved_at))){
   ratings=cached.ratings||{};session=cached.session||{};dirty=true;
  }
  session.started_at=session.started_at||new Date().toISOString();ready=true;connected=true;
  render();status(d.identity&&d.identity!==DATA.identity?'旧实验评分已隔离，本轮尚未评价':'已连接本地文件 · 选择即保存');
  if(dirty)persist();
 }catch(e){render();status(e.message+'；请确认本地服务和当前实验后刷新。',true);
  document.querySelectorAll('#content input,#content select,#content textarea,#clear,#fatigue').forEach(e=>e.disabled=true);}
}
boot();
</script></body></html>
'''
