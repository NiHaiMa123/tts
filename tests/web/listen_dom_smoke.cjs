const assert=require('node:assert/strict');
const fs=require('node:fs');
const {JSDOM,VirtualConsole}=require(process.env.TTS_JSDOM_MODULE||'jsdom');
let failWrites=false;
const errors=[];
async function page(url,file=false){
 const html=await (await fetch(url)).text();
 const console=new VirtualConsole();console.on('jsdomError',e=>errors.push(e.message));
 const dom=new JSDOM(html,{url:file?'file:///tmp/listen/index.html':url,runScripts:'dangerously',virtualConsole:console,
  beforeParse(win){win.confirm=()=>true;win.fetch=async(path,options)=>{
   if(failWrites&&options?.method==='POST')return new Response(JSON.stringify({ok:false,error:'simulated disk full'}),{status:500,headers:{'content-type':'application/json'}});
   if(options?.method==='POST')await new Promise(r=>setTimeout(r,15));
   return fetch(new URL(path,url),options);
  };}});
 await wait(()=>dom.window.document.querySelector('#saved').textContent!=='连接本地服务…');
 return dom;
}
async function wait(fn){const deadline=Date.now()+7000;while(Date.now()<deadline){if(fn())return;await new Promise(r=>setTimeout(r,20));}throw new Error('timeout');}
function choose(dom,key,value,id){
 const doc=dom.window.document;
 const el=[...doc.querySelectorAll('[data-dim]')].find(e=>e.dataset.dim===key&&(!id||e.dataset.id===id));
 assert(el,key);el.value=value;el.dispatchEvent(new dom.window.Event(el.tagName==='SELECT'?'change':'input',{bubbles:true}));
}
function saved(dom){return dom.window.document.querySelector('#saved').textContent.includes('已保存到本地文件');}
const disk=()=>JSON.parse(fs.readFileSync(process.env.TTS_DOM_BASIC_FILE,'utf8'));
(async()=>{
 let dom=await page(process.env.TTS_DOM_BASIC_URL);
 assert.equal(dom.window.document.querySelectorAll('a[download]').length,0);
 choose(dom,'likeness','1');choose(dom,'likeness','2');choose(dom,'likeness','5');
 choose(dom,'notes','备注无需离焦 <script>');
 await wait(()=>saved(dom));assert.equal(disk().ratings['sample.wav'].likeness,'5');
 assert.equal(disk().ratings['sample.wav'].notes,'备注无需离焦 <script>');dom.window.close();
 dom=await page(process.env.TTS_DOM_BASIC_URL);assert.equal(dom.window.document.querySelector('[data-dim=likeness]').value,'5');
 assert.equal(dom.window.document.querySelector('textarea').value,'备注无需离焦 <script>');
 failWrites=true;choose(dom,'likeness','1');await wait(()=>dom.window.document.querySelector('#saved').className==='error');
 assert(!saved(dom));assert.equal(disk().ratings['sample.wav'].likeness,'5');
 failWrites=false;dom.window.document.querySelector('#retry').click();await wait(()=>saved(dom));assert.equal(disk().ratings['sample.wav'].likeness,'1');dom.window.close();
 const first=await page(process.env.TTS_DOM_BASIC_URL),second=await page(process.env.TTS_DOM_BASIC_URL);
 choose(first,'clarity','5');await wait(()=>saved(first));choose(second,'clarity','1');
 await wait(()=>second.window.document.querySelector('#saved').className==='error');assert.equal(disk().ratings['sample.wav'].clarity,'5');
 first.window.close();second.window.close();
 dom=await page(process.env.TTS_DOM_BASIC_URL);dom.window.document.querySelector('#clear').click();await wait(()=>saved(dom));assert.deepEqual(disk().ratings,{});dom.window.close();
 dom=await page(process.env.TTS_DOM_PAIRED_URL);const choice=dom.window.document.querySelector('input[name=choice_p1][value=A]');choice.checked=true;
 choice.dispatchEvent(new dom.window.Event('change',{bubbles:true}));await wait(()=>saved(dom));
 let pairs=JSON.parse(fs.readFileSync(process.env.TTS_DOM_PAIRED_FILE,'utf8'));
 assert.equal(pairs.pairs.p1.choice_real,pairs.pairs.p1.blind_order.A);
 dom.window.document.querySelector('#fatigue').click();await wait(()=>saved(dom));
 pairs=JSON.parse(fs.readFileSync(process.env.TTS_DOM_PAIRED_FILE,'utf8'));
 assert(pairs._review.fatigue);assert.equal(pairs.pairs.p2.status,'UNCERTAIN_FATIGUE');dom.window.close();
 dom=await page(process.env.TTS_DOM_BASIC_URL,true);assert(dom.window.document.querySelector('[data-dim=likeness]').disabled);
 assert(dom.window.document.querySelector('#saved').textContent.includes('serve_review.py'));dom.window.close();
 assert.deepEqual(errors,[]);
 console.log('DOM + live HTTP verified: autosave, final edit, notes before blur, refresh, HTTP failure/retry, stale tab, clear, blind mapping, fatigue, file:// read-only');
})().catch(e=>{console.error(e);process.exitCode=1;});
