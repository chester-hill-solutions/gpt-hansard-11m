"""GET / — Hansard Chat UI (self-contained, iframe-embeddable, no dependencies).

Harness: users type natural questions; the UI wraps them in the model's
fine-tuning scaffold (Q: …\nA:) automatically. "Continuation" mode passes
raw text through for open-ended prompts (Bill C-, Mr. Speaker, …).
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hansard Chat — GPT-Hansard-11M</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120'%3E%3Cpolygon points='25,96 95,96 88,72 32,72' fill='%238a5a2b'/%3E%3Ccircle cx='60' cy='48' r='24' fill='%236f4a23'/%3E%3Ccircle cx='40' cy='30' r='7' fill='%236f4a23'/%3E%3Ccircle cx='80' cy='30' r='7' fill='%236f4a23'/%3E%3Cellipse cx='60' cy='60' rx='13' ry='9' fill='%23a9773f'/%3E%3Crect x='55' y='63' width='4' height='9' rx='1' fill='%23fff'/%3E%3Crect x='61' y='63' width='4' height='9' rx='1' fill='%23fff'/%3E%3Cellipse cx='60' cy='53' rx='5' ry='3.5' fill='%232b2014'/%3E%3Ccircle cx='50' cy='44' r='2.6' fill='%232b2014'/%3E%3Ccircle cx='70' cy='44' r='2.6' fill='%232b2014'/%3E%3C/svg%3E">
<style>
  :root {
    --ink:#292d38; --ink-muted:#575e6d; --red:#c8102e; --bg:#f7f7f5; --card:#ffffff;
    --line:#e4e6ea; --ok:#1a7f4e; --fur:#6f4a23; --fur-light:#a9773f; --wood:#8a5a2b;
  }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e7e9ee; --ink-muted:#9aa1ad; --bg:#161920; --card:#1e222c; --line:#2c3140; }
  }
  * { box-sizing:border-box; }
  .sr-only { position:absolute; width:1px; height:1px; padding:0; margin:-1px; overflow:hidden;
             clip:rect(0 0 0 0); white-space:nowrap; border:0; }
  html, body { height:100%; }
  body { margin:0; font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         color:var(--ink); background:var(--bg);
         height:100vh; display:flex; flex-direction:column; }
  body::before { content:""; position:fixed; left:0; top:0; bottom:0; width:4px;
                 background:var(--red); z-index:9; }
  header { display:flex; align-items:center; gap:12px; padding:12px 18px 12px 26px;
           border-bottom:1px solid var(--line); background:var(--card); }
  .logo { width:36px; height:36px; flex-shrink:0; }
  header .who { flex:1; min-width:0; }
  header h1 { margin:0; font:750 17px/1.2 Georgia,'Times New Roman',serif; letter-spacing:-.01em; }
  header p { margin:2px 0 0; font-size:11.5px; color:var(--ink-muted); }
  header a.cardlink { font-size:11.5px; color:var(--ink-muted); text-decoration:none; white-space:nowrap; }
  header a.cardlink:hover { color:var(--red); text-decoration:underline; }
  :focus-visible { outline:2px solid var(--red); outline-offset:2px; }
  #log { flex:1; overflow-y:auto; padding:20px 20px 10px; scroll-behavior:smooth; }
  .wrap { max-width:860px; margin:0 auto; display:flex; flex-direction:column; gap:16px; }
  .welcome { padding:4px 2px 0; }
  .welcome h2 { margin:0 0 6px; font:650 26px/1.15 Georgia,'Times New Roman',serif; letter-spacing:-.01em; }
  .welcome p { margin:0 0 14px; font-size:13.5px; color:var(--ink-muted); max-width:62ch; }
  .examples { display:grid; grid-template-columns:1fr 1fr; gap:2px 22px; }
  .examples button { text-align:left; padding:7px 0 7px 12px; border:0; border-left:2px solid var(--line);
                     background:transparent; cursor:pointer; font:inherit; font-size:13px; color:var(--ink); }
  .examples button:hover { border-left-color:var(--red); background:color-mix(in srgb, var(--red) 4%, transparent); }
  .examples .tag { display:block; font-size:10px; font-weight:700; letter-spacing:.08em;
                   text-transform:uppercase; color:var(--ink-muted); margin-bottom:2px; }
  .row { max-width:860px; margin:0 auto; width:100%; display:flex; gap:10px; align-items:flex-start; }
  .row.user { justify-content:flex-end; }
  .msg { padding:0; border-radius:0; max-width:100%; white-space:pre-wrap;
         word-break:break-word; position:relative; }
  .msg.user { max-width:75%; padding:9px 13px; background:color-mix(in srgb, var(--card) 55%, var(--line));
              border:1px solid var(--line); border-radius:18px 18px 4px 18px; }
  .msg.model { font-size:14.5px; }
  .mwho { font-size:12px; font-weight:700; color:var(--ink-muted); margin-bottom:3px; }
  .meta { font-size:10.5px; color:var(--ink-muted); display:flex; gap:10px; align-items:center;
          margin:6px 2px 0 38px; max-width:860px; }
  .row.user + .meta { margin-left:0; justify-content:flex-end; margin-right:2px; }
  .meta button { background:none; border:0; padding:0; font:inherit; font-size:10.5px; color:var(--ink-muted);
                 cursor:pointer; }
  .meta button:hover { color:var(--red); }
  .dots span { display:inline-block; width:5px; height:5px; border-radius:99px; background:var(--ink-muted);
               margin-right:3px; animation:blink 1.2s infinite; }
  .dots span:nth-child(2) { animation-delay:.2s; } .dots span:nth-child(3) { animation-delay:.4s; }
  @keyframes blink { 0%,80%,100% { opacity:.25; } 40% { opacity:1; } }
  .composer { padding:12px 20px 14px; border-top:1px solid var(--line); background:var(--card); }
  .composer .box { max-width:860px; margin:0 auto; background:var(--card); border:1px solid var(--line);
                   border-radius:22px; padding:10px 14px 8px; }
  .composer .box:focus-within { border-color:color-mix(in srgb, var(--red) 45%, var(--line)); }
  textarea { width:100%; resize:none; border:0; background:transparent; font:inherit; color:var(--ink);
             max-height:120px; outline:none; }
  .under { display:flex; align-items:center; gap:8px; margin-top:6px; flex-wrap:wrap; }
  .seg { display:flex; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .seg button { border:0; background:transparent; font-size:11.5px; font-weight:650; padding:0 12px; min-height:44px;
                cursor:pointer; color:var(--ink-muted); }
  .seg button.on { background:color-mix(in srgb, var(--red) 10%, transparent); color:var(--red); }
  .hint { font-size:10.5px; color:var(--ink-muted); }
  .send { margin-left:auto; border:0; border-radius:10px; background:var(--red); color:#fff; font-weight:700;
          font-size:13px; padding:10px 22px; min-height:44px; cursor:pointer; }
  .send[disabled] { opacity:.45; cursor:default; }
  .stop { display:none; border:0; border-radius:10px; background:transparent; color:var(--ink-muted);
          font-size:12.5px; cursor:pointer; border:1px solid var(--line); padding:10px 16px; min-height:44px; }
  .errorbox { display:none; border:1px solid color-mix(in srgb, var(--red) 45%, var(--line));
              background:color-mix(in srgb, var(--red) 8%, var(--card)); color:var(--ink);
              font-size:12.5px; border-radius:10px; padding:10px 14px; margin:10px 0; gap:10px; align-items:center; }
  .errorbox button { border:0; border-radius:8px; background:var(--red); color:#fff; font-weight:700;
                     font-size:12px; padding:8px 16px; min-height:44px; cursor:pointer; white-space:nowrap; }
  footer { text-align:center; font-size:10.5px; color:var(--ink-muted); padding:0 16px 10px; }
  footer a { color:inherit; }
  .tabs { display:flex; gap:6px; }
  .tabs button { border:1px solid var(--line); background:transparent; color:var(--ink-muted); font-size:12px; font-weight:650; padding:0 12px; min-height:44px; border-radius:9px; cursor:pointer; }
  .tabs button.on { background:var(--ink); color:var(--bg); border-color:var(--ink); }
  .attbox { max-width:900px; margin:0 auto; }
  .attbox textarea { width:100%; border:1px solid var(--line); border-radius:10px; padding:10px 12px; font:inherit; color:var(--ink); background:var(--card); min-height:60px; resize:vertical; }
  .attrow { display:flex; gap:10px; align-items:center; margin:10px 0; flex-wrap:wrap; }
  .abtn { border:0; border-radius:10px; background:var(--red); color:#fff; font-weight:700; font-size:13px; padding:8px 14px; cursor:pointer; }
  .abtn[disabled] { opacity:.5; }
  #atok { display:flex; flex-wrap:wrap; gap:2px; margin:12px 0; }
  #atok span { font-family:ui-monospace,Menlo,monospace; font-size:12px; padding:3px 5px; border-radius:6px; cursor:pointer; border:1px solid transparent; white-space:pre; }
  #atok span.sel { border-color:var(--red); }
  #atok span:hover { border-color:var(--ink-muted); }
  .agrid { display:grid; grid-template-columns:repeat(6, 76px); gap:6px; margin:12px 0; }
  .agrid canvas { width:76px; height:76px; border:1px solid var(--line); border-radius:6px; cursor:pointer; background:var(--card); }
  .agrid canvas.on { outline:2px solid var(--red); }
  #abig { border:1px solid var(--line); border-radius:10px; background:var(--card); }
  .acap { font-size:12px; color:var(--ink-muted); margin:8px 0 14px; }
  .atop { font-size:12.5px; margin:6px 0 12px; }
  .atop ul { margin:4px 0 0; padding-left:18px; }
  .alab { font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; color:var(--ink-muted); margin:14px 0 6px; }
  .knobs { display:flex; gap:6px; }
  .aerr { display:none; border:1px solid color-mix(in srgb, var(--red) 45%, var(--line)); background:color-mix(in srgb, var(--red) 8%, var(--card)); color:var(--ink); font-size:12.5px; border-radius:10px; padding:10px 12px; margin:10px 0; }
  @media (prefers-reduced-motion: reduce) { .dots span { animation:none; opacity:.5; } }
  @media (max-width: 760px) { #modehint { display:none; } }
  @media (max-width: 520px) {
    header { padding:8px 12px 8px 20px; gap:8px; }
    header .logo { width:28px; height:28px; }
    header .who p { display:none; }
    header h1 { font-size:15px; }
    header a.cardlink { display:none; }
    footer { display:none; }
    #log { padding:12px 12px 6px; }
    .composer { padding:8px 10px 10px; }
  }
</style>
</head>
<body>
<header>
  <svg class="logo" viewBox="0 0 120 120" aria-label="Hansard Chat logo: a beaver at the dispatch box">
    <polygon points="25,96 95,96 88,72 32,72" fill="var(--wood)"/>
    <rect x="30" y="70" width="60" height="4" rx="2" fill="#6d451f"/>
    <line x1="86" y1="70" x2="86" y2="52" stroke="var(--ink)" stroke-width="3"/>
    <circle cx="86" cy="49" r="5" fill="var(--ink)"/>
    <circle cx="60" cy="48" r="24" fill="var(--fur)"/>
    <circle cx="40" cy="30" r="7" fill="var(--fur)"/>
    <circle cx="80" cy="30" r="7" fill="var(--fur)"/>
    <circle cx="40" cy="30" r="3.4" fill="#4a3016"/>
    <circle cx="80" cy="30" r="3.4" fill="#4a3016"/>
    <ellipse cx="60" cy="60" rx="13" ry="9" fill="var(--fur-light)"/>
    <rect x="55" y="63" width="4" height="9" rx="1.4" fill="#fff"/>
    <rect x="61" y="63" width="4" height="9" rx="1.4" fill="#fff"/>
    <ellipse cx="60" cy="53" rx="5" ry="3.5" fill="#2b2014"/>
    <circle cx="50" cy="44" r="2.6" fill="#2b2014"/>
    <circle cx="70" cy="44" r="2.6" fill="#2b2014"/>
    <polygon points="52,76 60,80 52,84" fill="var(--red)"/>
    <polygon points="68,76 60,80 68,84" fill="var(--red)"/>
  </svg>
  <div class="who">
    <h1>Hansard Chat</h1>
    <p>GPT-Hansard-11M · built from scratch on the Canadian House of Commons, 2006–2026</p>
  </div>
  <a class="cardlink" href="https://huggingface.co/NathanielArfin/gpt-hansard-11m" target="_blank" rel="noopener">Model card ↗</a>
</header>
<div id="chatview" style="display:flex;flex-direction:column;flex:1;min-height:0;">
<div id="log" role="log" aria-live="polite" aria-busy="false" aria-label="Chat transcript"><div class="wrap" id="feed"></div></div>
<div class="composer">
  <div class="box">
    <label class="sr-only" for="p">Ask Parliament a question</label>
    <textarea id="p" rows="1" placeholder="Ask Parliament a question…"></textarea>
    <div class="under">
      <div class="seg" id="mode" aria-label="Prompt mode">
        <button class="on" data-m="question" type="button" aria-pressed="true">Question</button>
        <button data-m="continue" type="button" aria-pressed="false">Continue</button>
      </div>
      <span class="knobs">
        <label class="pill" for="temp">temp <input id="temp" type="range" min="0.2" max="1.5" step="0.05" value="0.6"><b id="tempv">0.60</b></label>
        <label class="pill" for="len">max tokens <input id="len" type="range" min="40" max="200" step="10" value="120"><b id="lenv">120</b></label>
      </span>
      <span class="hint" id="modehint">the model sees Q:/A:, as in training</span>
      <button class="stop" id="stop" type="button">Stop</button>
      <button class="send" id="send" type="button">Ask ↵</button>
    </div>
  </div>
</div>
<div class="errorbox" id="errbox" role="alert">
  <span id="errmsg"></span>
  <button id="errretry" type="button">Try again</button>
</div>
<footer>11.33M parameters · closed-book: trained only on the official Debates of the House of Commons · <a href="https://huggingface.co/datasets/NathanielArfin/canadian-hansard-2006-now" target="_blank" rel="noopener">dataset</a> · <a href="https://huggingface.co/NathanielArfin/gpt-hansard-11m" target="_blank" rel="noopener">model card</a></footer>
<script>
const feed = document.getElementById('feed'), log = document.getElementById('log');
const ta = document.getElementById('p'), send = document.getElementById('send'), stopBtn = document.getElementById('stop');
const modeHint = document.getElementById('modehint');
const errbox = document.getElementById('errbox'), errmsg = document.getElementById('errmsg');
const errretry = document.getElementById('errretry');
const chatlog = document.getElementById('log');
let controller = null, mode = 'question', lastRaw = null, lastMode = 'question';

const EXAMPLES = [
  ["The excuse machine", "What is the excuse for this government's inaction on Faries?", "question"],
  ["Gas prices", "What will the Prime Minister do about energy prices?", "question"],
  ["Never in its training", "What is a filibuster?", "question"],
  ["Definition", "What is prorogation?", "question"],
  ["FR in → EN out (EN-locked)", "Qu'est-ce que le Hansard ?", "question"],
  ["Continue a speech", "Mr. Speaker, the cost of housing", "continue"],
];

function toPrompt(raw) {
  return mode === 'question' ? ('Q: ' + raw + '\\nA:') : raw;
}

function nearBottom() { return log.scrollHeight - log.scrollTop - log.clientHeight < 90; }
function scrollDown(force) { if (force || nearBottom()) log.scrollTop = log.scrollHeight; }

function setMode(m) {
  mode = m;
  document.querySelectorAll('#mode button').forEach(b => {
    const on = b.dataset.m === m;
    b.classList.toggle('on', on);
    b.setAttribute('aria-pressed', String(on));
  });
  modeHint.textContent = m === 'question' ? 'the model sees Q:/A:, as in training'
                                          : 'raw continuation, the model keeps talking';
}
document.querySelectorAll('#mode button').forEach(b => b.addEventListener('click', () => setMode(b.dataset.m)));

function welcome() {
  const el = document.createElement('div');
  el.className = 'welcome';
  el.innerHTML = '<h2>Ask the House.</h2><p>An 11.33M-parameter GPT, trained from scratch on twenty years of the ' +
    'official Debates of the Canadian House of Commons, then fine-tuned on 35,401 question→answer exchanges ' +
    'mined from the record (EN-locked). Just ask; the Q:/A: scaffolding is handled for you. This build answers ' +
    'in Parliament\u2019s register — whatever language the question arrives in.</p>';
  const grid = document.createElement('div');
  grid.className = 'examples';
  for (const [tag, p, m] of EXAMPLES) {
    const b = document.createElement('button');
    b.innerHTML = '<span class="tag">' + tag + '</span>' + p;
    b.addEventListener('click', () => { setMode(m); ta.value = p; ta.dispatchEvent(new Event('input')); autosize(); doSend(); });
    grid.appendChild(b);
  }
  el.appendChild(grid);
  feed.appendChild(el);
}

function addBubble(cls, text) {
  const row = document.createElement('div');
  row.className = 'row' + (cls === 'user' ? ' user' : '');
  if (cls === 'model') {
    const av = document.createElement('div');
    av.innerHTML = '<svg viewBox="0 0 120 120" width="28" height="28" style="margin-top:2px;border-radius:8px;background:var(--card)"><polygon points="25,96 95,96 88,72 32,72" fill="#8a5a2b"/><circle cx="60" cy="48" r="24" fill="#6f4a23"/><circle cx="40" cy="30" r="7" fill="#6f4a23"/><circle cx="80" cy="30" r="7" fill="#6f4a23"/><ellipse cx="60" cy="60" rx="13" ry="9" fill="#a9773f"/><rect x="55" y="63" width="4" height="9" rx="1.4" fill="#fff"/><rect x="61" y="63" width="4" height="9" rx="1.4" fill="#fff"/><ellipse cx="60" cy="53" rx="5" ry="3.5" fill="#2b2014"/><circle cx="50" cy="44" r="2.6" fill="#2b2014"/><circle cx="70" cy="44" r="2.6" fill="#2b2014"/><polygon points="52,76 60,80 52,84" fill="#c8102e"/><polygon points="68,76 60,80 68,84" fill="#c8102e"/></svg>';
    row.appendChild(av);
  }
  const msg = document.createElement('div');
  msg.className = 'msg ' + cls;
  msg.textContent = text;
  row.appendChild(msg);
  feed.appendChild(row);
  scrollDown(true);
  return { row, msg };
}

function addMeta(text, copyText) {
  const m = document.createElement('div');
  m.className = 'meta';
  const s = document.createElement('span'); s.textContent = text;
  m.appendChild(s);
  if (copyText) {
    const c = document.createElement('button'); c.textContent = 'copy';
    c.addEventListener('click', () => navigator.clipboard.writeText(copyText).then(() => { c.textContent = 'copied ✓'; setTimeout(() => c.textContent = 'copy', 1200); }));
    m.appendChild(c);
  }
  feed.appendChild(m); scrollDown();
  return m;
}

const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
function typewriter(el, text) {
  return new Promise((done) => {
    if (REDUCED) { el.textContent = text; scrollDown(); done(); return; }
    let i = 0;
    const step = Math.max(1, Math.round(text.length / 40));
    const t = setInterval(() => {
      i = Math.min(text.length, i + step);
      el.textContent = text.slice(0, i);
      scrollDown();
      if (i >= text.length) { clearInterval(t); done(); }
    }, 16);
  });
}

function autosize() {
  ta.style.height = 'auto';
  ta.style.height = Math.min(ta.scrollHeight, 120) + 'px';
}
ta.addEventListener('input', autosize);
ta.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); doSend(); } });
document.getElementById('temp').addEventListener('input', (e) => document.getElementById('tempv').textContent = (+e.target.value).toFixed(2));
document.getElementById('len').addEventListener('input', (e) => document.getElementById('lenv').textContent = e.target.value);

async function doSend() {
  const raw = ta.value.trim();
  if (!raw || controller) return;
  lastRaw = raw; lastMode = mode;
  document.querySelector('.welcome')?.remove();
  hideError();
  addBubble('user', raw);
  const pending = addBubble('model pending', '');
  pending.msg.innerHTML = '<span class="dots"><span></span><span></span><span></span></span>';
  send.disabled = true; stopBtn.style.display = 'inline-block';
  chatlog.setAttribute('aria-busy', 'true');
  controller = new AbortController();
  const t0 = performance.now();
  try {
    const r = await fetch('/api/generate', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt: toPrompt(raw),
        temperature: parseFloat(document.getElementById('temp').value),
        max_new: parseInt(document.getElementById('len').value) }),
      signal: controller.signal,
    });
    const d = await r.json();
    pending.msg.classList.remove('pending');
    if (!d.text) throw new Error(d.error || 'empty response');
    const am = d.text.match(/^((?:Right )?(?:Hon\\.|L['’]hon\\.|Mr\\.|Ms\\.|Mrs\\.)[^:]{4,120}?):\\s*([\\s\\S]*)$/);
    let body = d.text;
    pending.msg.innerHTML = '';   // clear the loading dots — we append now, not overwrite
    if (am) {
      const who = document.createElement('div');
      who.className = 'mwho';
      who.textContent = am[1];
      pending.msg.appendChild(who);
      body = am[2];
    }
    const mbody = document.createElement('div');
    mbody.className = 'mbody';
    pending.msg.appendChild(mbody);
    await typewriter(mbody, body);
    // keep the question and the answer on one screen: align the exchange to the top
    pending.row.scrollIntoView({ block: 'start', behavior: REDUCED ? 'auto' : 'smooth' });
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    addMeta('GPT-Hansard-11M · ' + dt + 's' + (mode === 'continue' ? ' · continuation' : ''), raw + d.text);
  } catch (err) {
    pending.msg.classList.remove('pending');
    pending.msg.textContent = '';
    pending.row.remove();
    if (err.name === 'AbortError') {
      addBubble('model', '— stopped —');
    } else {
      showError(err.message === 'Failed to fetch'
        ? 'Can\\'t reach the model. It may be restarting — try again.'
        : 'Error: ' + err.message);
    }
  }
  send.disabled = false; stopBtn.style.display = 'none'; controller = null;
  chatlog.setAttribute('aria-busy', 'false');
  scrollDown();
}
function showError(msg) {
  errmsg.textContent = msg;
  errbox.style.display = 'flex';
}
function hideError() {
  errbox.style.display = 'none';
}
errretry.addEventListener('click', () => {
  hideError();
  if (lastRaw) { ta.value = lastRaw; setMode(lastMode); ta.dispatchEvent(new Event('input')); autosize(); doSend(); }
});
stopBtn.addEventListener('click', () => controller?.abort());
send.addEventListener('click', doSend);
ta.focus();
welcome();

</script>
</body>
</html>
"""
