// Drives headless Edge via CDP to open the local web app, select the target
// session, wait for result, screenshot and extract DOM text.
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9229;
const OUTDIR = __dirname;
const TARGET_SID = '20260831_164529_1f5c';
const TARGET_NAME = 'SVID_20260506_171653_1.mp4';
const URL = 'http://127.0.0.1:8000';

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function launch() {
  const userData = path.join(OUTDIR, '_edge_profile_' + Date.now());
  const args = [
    '--headless=new',
    '--disable-gpu',
    '--no-sandbox',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${userData}`,
    '--window-size=1600,1200',
    'about:blank'
  ];
  const child = spawn(EDGE, args, { stdio: 'ignore' });
  child.on('error', e => console.error('SPAWN_ERR', e.message));
  return child;
}

async function getPageWs() {
  for (let i = 0; i < 40; i++) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json`);
      const list = await res.json();
      const page = list.find(t => t.type === 'page' && !t.url.startsWith('devtools'));
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch (e) { /* retry */ }
    await sleep(400);
  }
  throw new Error('no page ws target');
}

let msgId = 0;
function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  const pending = new Map();
  const events = [];
  ws.addEventListener('message', (ev) => {
    const data = JSON.parse(ev.data);
    if (data.id && pending.has(data.id)) {
      pending.get(data.id)(data);
      pending.delete(data.id);
    } else if (data.method) {
      events.push(data);
    }
  });
  function call(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++msgId;
      pending.set(id, (r) => r.error ? reject(new Error(r.error.message)) : resolve(r.result));
      ws.send(JSON.stringify({ id, method, params }));
    });
  }
  const ready = new Promise((res, rej) => {
    ws.addEventListener('open', res);
    ws.addEventListener('error', (e) => rej(new Error('ws error')));
  });
  return { ws, call, ready, events };
}

async function evaluate(cdp, expression) {
  const res = await cdp.call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (res.exceptionDetails) throw new Error('EVAL_ERR: ' + JSON.stringify(res.exceptionDetails));
  return res.result.value;
}

async function shot(cdp, file) {
  const res = await cdp.call('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(OUTDIR, file), Buffer.from(res.data, 'base64'));
  console.log('SCREENSHOT_SAVED', file, res.data.length, 'b64chars');
}

async function waitFor(cdp, expr, timeoutMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const v = await evaluate(cdp, expr);
    if (v) return v;
    await sleep(800);
  }
  return null;
}

(async () => {
  const child = launch();
  let wsUrl;
  try { wsUrl = await getPageWs(); }
  catch (e) { console.error('NO_WS', e.message); try { child.kill(); } catch{} process.exit(1); }

  const cdp = await connect(wsUrl);
  await cdp.ready;
  await cdp.call('Page.enable');
  await cdp.call('Runtime.enable');

  // navigate
  await cdp.call('Page.navigate', { url: URL });
  await sleep(2500);
  // wait for app + session list to render
  await waitFor(cdp, `!!document.querySelector('.session-item')`, 20000);

  // dump sessions
  const sessions = await evaluate(cdp, `JSON.stringify(Array.from(document.querySelectorAll('.session-item')).map(el=>({name:el.querySelector('.item-name')?.textContent?.trim(), active: el.classList.contains('active')})))`);
  console.log('SESSIONS_DOM', sessions);

  // click the target session (first / most recent has name SVID_20260506_171653_1.mp4)
  const clicked = await evaluate(cdp, `(()=>{const items=Array.from(document.querySelectorAll('.session-item'));const target=items.find(el=>el.querySelector('.item-name')?.textContent?.includes('SVID_20260506_171653_1'));if(!target)return 'NOT_FOUND';target.click();return 'CLICKED';})()`);
  console.log('CLICK_RESULT', clicked);

  if (clicked === 'CLICKED') {
    // wait for result panel content / voice collapse to appear
    await sleep(1500);
    await waitFor(cdp, `!!document.querySelector('.voice-collapse') || !!document.querySelector('.result-panel .chat-panel')`, 20000);
    await sleep(800);
  }

  // Screenshot: full page (top region shows panel) + viewport
  await shot(cdp, '01_overview.png');
  // full page
  try {
    const full = await cdp.call('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
    fs.writeFileSync(path.join(OUTDIR, '02_fullpage.png'), Buffer.from(full.data, 'base64'));
    console.log('SCREENSHOT_SAVED 02_fullpage.png', full.data.length);
  } catch (e) { console.log('fullpage err', e.message); }

  // Expand the voice collapse panel
  const exp = await evaluate(cdp, `(()=>{const h=Array.from(document.querySelectorAll('.voice-collapse .el-collapse-item__header'));if(h.length){h[0].click();return 'CLICKED_VOICE_HEADER '+h.length;}return 'NO_VOICE_HEADER';})()`);
  console.log('EXPAND_RESULT', exp);
  await sleep(900);

  // scroll the chat area down to reveal collapsed panels
  await evaluate(cdp, `document.querySelector('.el-main, .app-main, main, body')?.scrollTo? window.scrollTo(0,1200):0; true`);
  await shot(cdp, '03_voice_expanded.png');

  // Dump all text content from the result panel area
  const resultText = await evaluate(cdp, `JSON.stringify({
    voiceCollapse: (()=>{const h=document.querySelector('.voice-collapse .el-collapse-item__header');return h?h.textContent.replace(/\\s+/g,' ').trim():null})(),
    voiceItems: Array.from(document.querySelectorAll('.voice-collapse .voice-item')).map(el=>el.textContent.replace(/\\s+/g,' ').trim()).slice(0,26),
    voiceItemCount: document.querySelectorAll('.voice-collapse .voice-item').length,
    audioCollapse: (()=>{const h=document.querySelector('.audio-collapse .el-collapse-item__header');return h?h.textContent.replace(/\\s+/g,' ').trim():null})(),
    audioItemCount: document.querySelectorAll('.audio-collapse .audio-item').length,
    headerStats: (()=>{const h=document.querySelector('.result-panel .header-stats');return h?h.textContent.replace(/\\s+/g,' ').trim():null})(),
    summaryText: document.body.innerText
  })`);
  console.log('RESULT_DUMP_START');
  console.log(resultText);
  console.log('RESULT_DUMP_END');

  await sleep(300);
  try { child.kill(); } catch(e){}
  process.exit(0);
})().catch(e => { console.error('FATAL', e.message); try{process.exit(1)}catch(_){} });