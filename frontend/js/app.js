import { get } from './api.js';
import { el, debounce, controls, codeLines, fmt } from './ui.js';
import ch01 from './panels/ch01_intro.js';
import tok from './panels/ch02_tokenize.js';
import win from './panels/ch02_windows.js';
import emb from './panels/ch02_embed.js';
import simple from './panels/ch03_simple.js';
import selfa from './panels/ch03_self.js';
import causal from './panels/ch03_causal.js';
import mha from './panels/ch03_mha.js';

const PANELS = [ch01, tok, win, emb, simple, selfa, causal, mha];
const BOOK_SENTENCE = 'Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.';
export const G = { text: BOOK_SENTENCE, seed: 123, weights: 'random', numbers: false, gpt2: false };

const $ = (s) => document.querySelector(s);
let active = null, renderId = 0, built = null, step = 0, playing = null;
const visited = new Set();

// ---------------------------------------------------------------- path nav
function buildPath() {
  const nav = $('#path');
  PANELS.forEach(p => nav.append(el('button', { class: 'node', 'data-id': p.id, onClick: () => show(p.id) },
    el('span', { class: 'dot' }), el('span', { class: 'sec', text: p.section }), el('span', { class: 'ttl', text: p.title }))));
}
function markPath() {
  const i = PANELS.indexOf(active);
  document.querySelectorAll('#path .node').forEach((b, k) => { b.classList.toggle('on', k === i); b.classList.toggle('done', k < i); });
}

// ---------------------------------------------------------------- inspector
function probe(info) {
  const box = $('#probe');
  if (!info) { box.className = 'probe empty'; box.textContent = 'hover anything in the scene'; return; }
  box.className = 'probe';
  box.replaceChildren(el('div', {},
    el('div', { class: 't', html: info.title }),
    ...(info.rows || []).map(([k, v]) => el('div', { class: 'row' }, el('span', { text: k }), el('b', { text: typeof v === 'number' ? fmt(v) : String(v) }))),
    info.f ? el('div', { class: 'f', text: info.f }) : null));
}

// ---------------------------------------------------------------- scrubber
function setStep(n, fromPlay = false) {
  if (!built) return;
  step = Math.max(0, Math.min(built.steps.length - 1, n));
  built.scene.setStep(step);
  $('#s-range').value = step;
  const s = built.steps[step];
  $('#s-label').textContent = `${step + 1}/${built.steps.length} · ${s.label}`;
  built.hl(new Set([].concat(s.line ?? [])), Math.max(...built.steps.slice(0, step + 1).flatMap(x => [].concat(x.line ?? -1))));
  if (s.note) $('#guide').querySelector('.stepnote')?.replaceChildren(el('p', { html: s.note }));
  if (!fromPlay) stopPlay();
}
function stopPlay() { if (playing) { clearInterval(playing); playing = null; $('#s-play').textContent = '▶'; } }
function play() {
  if (playing) return stopPlay();
  if (step >= built.steps.length - 1) setStep(0, true);
  $('#s-play').textContent = '❚❚';
  playing = setInterval(() => { if (step >= built.steps.length - 1) stopPlay(); else setStep(step + 1, true); }, 900);
}
function wireScrub() {
  $('#s-prev').onclick = () => setStep(step - 1);
  $('#s-next').onclick = () => setStep(step + 1);
  $('#s-play').onclick = play;
  $('#s-range').oninput = (e) => setStep(+e.target.value);
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, select, textarea')) return;
    if (e.key === 'ArrowRight') setStep(step + 1);
    else if (e.key === 'ArrowLeft') setStep(step - 1);
    else if (e.key === ' ') { e.preventDefault(); play(); }
  });
}

// ---------------------------------------------------------------- render
async function show(id) {
  const p = PANELS.find(x => x.id === id) || PANELS[0];
  const changed = p !== active;
  active = p; location.hash = p.id; markPath();
  $('#title').replaceChildren(el('div', { class: 'eyebrow', text: `Book ${p.section}` }), el('h1', { text: p.title }));
  await render(changed);
}

export async function render(fresh = false) {
  if (!active) return;
  const my = ++renderId;
  stopPlay();
  const wrap = $('#scene-wrap');
  $('#controls').replaceChildren(controls(active.controlSpec ? active.controlSpec(G) : [], active.S || {}, () => render()));
  wrap.replaceChildren(el('div', { class: 'loading', text: 'running the book code…' }));
  try {
    const data = await active.load(G);
    if (my !== renderId) return;
    built = active.build(data, G);
    built.scene.onInspect = probe;
    wrap.replaceChildren(built.scene.svg);
    const b = built.scene.svg.getBBox();
    built.size = { w: Math.ceil(b.width + 60), h: Math.ceil(b.height + 60) };
    built.scene.svg.setAttribute('viewBox', `${b.x - 30} ${b.y - 30} ${built.size.w} ${built.size.h}`);
    applyZoom();
    $('#guide').replaceChildren(el('div', { html: built.guide || '' }), el('div', { class: 'stepnote' }));
    built.hl = codeLines($('#code'), built.code || []);
    $('#s-range').max = built.steps.length - 1;
    const first = fresh && !visited.has(active.id);
    visited.add(active.id);
    if (first) { setStep(0); play(); } else setStep(fresh ? built.steps.length - 1 : Math.min(step, built.steps.length - 1));
  } catch (e) {
    if (my === renderId) wrap.replaceChildren(el('div', { class: 'error', text: String(e.message || e) }));
    console.error(e);
  }
}

// ---------------------------------------------------------------- zoom
let zoom = 'fit';
function applyZoom() {
  if (!built || !built.size) return;
  const stage = $('#stage');
  const fit = Math.max(.6, Math.min(1, (stage.clientWidth - 56) / built.size.w));   // below 0.6 it is unreadable; scroll instead
  const s = zoom === 'fit' ? fit : zoom;
  built.scene.svg.setAttribute('width', Math.round(built.size.w * s));
  built.scene.svg.setAttribute('height', Math.round(built.size.h * s));
  document.querySelectorAll('#zoom button').forEach(b => b.classList.toggle('on', b.dataset.z === (zoom === 'fit' ? 'fit' : zoom === 1 ? '1' : '')));
}
function wireZoom() {
  $('#zoom').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    const cur = zoom === 'fit' ? +$('#stage').querySelector('svg')?.getAttribute('width') / built.size.w : zoom;
    zoom = b.dataset.z === 'fit' ? 'fit' : b.dataset.z === '1' ? 1 : Math.max(.25, Math.min(3, cur * (b.dataset.z === '+' ? 1.25 : .8)));
    applyZoom();
  });
  window.addEventListener('resize', debounce(applyZoom, 150));
}

// ---------------------------------------------------------------- toolbar
function wireBar() {
  const text = $('#g-text'), seed = $('#g-seed');
  text.value = G.text; seed.value = G.seed;
  text.addEventListener('input', debounce(() => { G.text = text.value; render(); }, 450));
  seed.addEventListener('change', () => { G.seed = +seed.value || 0; render(); });
  const seg = (id, key, coerce) => $(id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    if (key === 'weights' && b.dataset.v === 'gpt2' && !G.gpt2) { alert('GPT-2 weights are not in the HuggingFace cache yet.'); return; }
    G[key] = coerce(b.dataset.v);
    $(id).querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
    render();
  });
  seg('#g-weights', 'weights', v => v);
  seg('#g-numbers', 'numbers', v => v === '1');
}

async function init() {
  buildPath(); wireBar(); wireScrub(); wireZoom();
  try {
    const h = await get('/health');
    G.gpt2 = h.gpt2_cached;
    $('#health').innerHTML = `torch ${h.torch} · <span class="ok">cpu</span> · gpt2 ${h.gpt2_cached ? '<span class="ok">ready</span>' : '<span class="bad">missing</span>'}`;
  } catch { $('#health').innerHTML = '<span class="bad">backend offline</span>'; }
  document.addEventListener('rerender', () => render());
  window.addEventListener('hashchange', () => { const id = location.hash.slice(1); if (active && id !== active.id) show(id); });
  show(location.hash.slice(1) || PANELS[0].id);
}
init();
