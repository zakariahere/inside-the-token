// SVG scene primitives: strips (vectors), grids (weight matrices), the attention
// "loom" (dot matrix), ribbons, rulers (embedding tables), ops, frames.
// Every primitive can carry data-row (token index) for hover lighting and
// data-step for the code scrubber.

const NS = 'http://www.w3.org/2000/svg';

export const ROLE = { x: '#94a3b8', q: '#22d3ee', k: '#fb7185', v: '#a3e635', a: '#c084fc', o: '#fbbf24', id: '#f8fafc', pos: '#f472b6', neg: '#60a5fa' };
export const HEADS = ['#22d3ee', '#fb7185', '#a3e635', '#c084fc', '#fbbf24', '#f472b6', '#34d399', '#f97316', '#818cf8', '#e879f9', '#2dd4bf', '#facc15'];

const NEG = [59, 130, 246], MID = [30, 41, 59], POS = [249, 115, 22];
const S0 = [30, 41, 59], S1 = [168, 85, 247], S2 = [245, 208, 254];
const mix = (a, b, t) => `rgb(${a.map((c, i) => Math.round(c + (b[i] - c) * t)).join(',')})`;
export function diverging(v, m = 1) { const t = Math.max(-1, Math.min(1, v / (m || 1))); return t < 0 ? mix(MID, NEG, -t) : mix(MID, POS, t); }
export function sequential(v, hi = 1) { const t = Math.max(0, Math.min(1, v / (hi || 1))); return t < .5 ? mix(S0, S1, t * 2) : mix(S1, S2, (t - .5) * 2); }
export const fmt = (v, d = 3) => v === '-inf' ? '−∞' : v == null ? 'nan' : typeof v !== 'number' ? String(v) : Number.isInteger(v) ? String(v) : v.toFixed(d);
export const vis = (s) => String(s).replace(/ /g, '␣').replace(/\n/g, '⏎');

export class Scene {
  constructor({ showValues = false } = {}) {
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('class', 'scene');
    this.root = this.el('g', {}, this.svg);
    this.info = new Map();
    this.showValues = showValues;
    this.onInspect = () => {};
    this.maxStep = 0;
    this._wire();
  }

  el(tag, attrs = {}, parent = this.root) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
    parent.append(n);
    return n;
  }
  txt(x, y, text, attrs = {}, parent = this.root) {
    const t = this.el('text', { x, y, ...attrs }, parent);
    t.textContent = text;
    return t;
  }
  _step(s) { this.maxStep = Math.max(this.maxStep, s || 0); return s; }

  // ----- vector strip ------------------------------------------------------
  strip({ x, y, values, role = 'x', row = null, step = 0, cellH = 18, absmax = null, hi = 1, kind = 'div',
          label = null, maxW = 170, cellW = null, info = null, id = null, segments = null, segColors = HEADS, until = null }) {
    const n = values.length;
    cellW = cellW ?? Math.max(4, Math.min(22, Math.floor(maxW / n)));
    const g = this.el('g', { class: 'strip', 'data-step': this._step(step), 'data-until': until, 'data-row': row, 'data-role': role, transform: `translate(${x},${y})`, id });
    const finite = values.filter(v => typeof v === 'number');
    const m = absmax ?? (Math.max(...finite.map(Math.abs)) || 1);
    values.forEach((v, i) => {
      const fill = typeof v !== 'number' ? '#000' : kind === 'seq' ? sequential(v, hi) : diverging(v, m);
      this.el('rect', { x: i * cellW, y: 0, width: Math.max(1, cellW - 1), height: cellH, rx: 2, fill }, g);
      if (this.showValues && cellW >= 30) this.txt(i * cellW + cellW / 2, cellH / 2, fmt(v, 2), { class: 'cell' }, g);
    });
    const w = n * cellW - 1;
    if (segments) {
      const sw = w / segments;
      for (let s = 0; s < segments; s++) this.el('rect', { x: s * sw, y: cellH + 2, width: Math.max(1, sw - 1), height: 3, fill: segColors[s % segColors.length], opacity: .95 }, g);
    } else {
      this.el('rect', { x: 0, y: cellH + 2, width: w, height: 2, fill: ROLE[role] || role, opacity: .9 }, g);
    }
    if (label) this.txt(-8, cellH / 2, label, { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'middle' }, g);
    if (info) this.info.set(g, info);
    return { x, y, w, h: cellH, left: [x, y + cellH / 2], right: [x + w, y + cellH / 2], top: [x + w / 2, y], bottom: [x + w / 2, y + cellH + 4], cellW };
  }

  // ----- weight matrix grid ------------------------------------------------
  grid({ x, y, data, role = 'x', step = 0, cell = 14, kind = 'div', absmax = null, title = null, rowLabels = null, colLabels = null, info = null, infoFn = null }) {
    const R = data.length, C = data[0].length;
    const g = this.el('g', { class: 'grid', 'data-step': this._step(step), 'data-role': role, transform: `translate(${x},${y})` });
    const finite = data.flat().filter(v => typeof v === 'number');
    const m = absmax ?? (Math.max(...finite.map(Math.abs)) || 1);
    this.el('rect', { x: -2, y: -2, width: C * cell + 3, height: R * cell + 3, rx: 4, stroke: ROLE[role], fill: 'none', opacity: .6 }, g);
    data.forEach((row, i) => row.forEach((v, j) => {
      const r = this.el('rect', { x: j * cell, y: i * cell, width: cell - 1, height: cell - 1, rx: 1.5, fill: kind === 'seq' ? sequential(v, m) : diverging(v, m) }, g);
      if (this.showValues && cell >= 30) this.txt(j * cell + cell / 2, i * cell + cell / 2, fmt(v, 2), { class: 'cell' }, g);
      if (infoFn) this.info.set(r, infoFn(i, j, v));
    }));
    if (title) this.txt(0, -8, title, { class: 'lbl', fill: ROLE[role] }, g);
    if (rowLabels) rowLabels.forEach((l, i) => this.txt(-6, i * cell + cell / 2, vis(l), { class: 'sm', 'text-anchor': 'end', 'dominant-baseline': 'middle' }, g));
    if (colLabels) colLabels.forEach((l, j) => this.txt(j * cell + cell / 2, R * cell + 10, vis(l), { class: 'sm', 'text-anchor': 'middle' }, g));
    if (info) this.info.set(g, info);
    const w = C * cell - 1, h = R * cell - 1;
    return { x, y, w, h, left: [x, y + h / 2], right: [x + w, y + h / 2], top: [x + w / 2, y], bottom: [x + w / 2, y + h] };
  }

  // ----- the attention loom: rows = queries, cols = keys, dot radius = value -
  loom({ x, y, data, step = 0, cell = 22, rowLabels = null, colLabels = null, mask = null, dropped = null, hi = null,
         rowRole = 'q', colRole = 'k', signed = false, infoFn = null, title = null, colStep = null }) {
    const R = data.length, C = data[0].length;
    const g = this.el('g', { class: 'loom', 'data-step': this._step(step), transform: `translate(${x},${y})` });
    const finite = data.flat().filter(v => typeof v === 'number');
    const max = hi ?? (Math.max(...finite.map(Math.abs)) || 1);
    this.el('rect', { x: 0, y: 0, width: C * cell, height: R * cell, rx: 6, class: 'gridbg' }, g);
    for (let i = 0; i < R; i++) for (let j = 0; j < C; j++) {
      const v = data[i][j], cx = j * cell + cell / 2, cy = i * cell + cell / 2;
      const d = this.el('g', { class: 'dot', 'data-row': i, 'data-col': j }, g);
      const masked = v === '-inf' || (mask && mask[i][j]);
      if (masked) {
        this.el('circle', { cx, cy, r: Math.max(2, cell * .13), class: 'masked' }, d);
      } else {
        const t = Math.min(1, Math.abs(v) / max);
        const r = 1.5 + t * (cell / 2 - 2.5);
        const fill = signed && v < 0 ? ROLE.neg : ROLE.a;
        this.el('circle', { cx, cy, r, fill, opacity: .3 + .7 * t }, d);
        if (this.showValues && cell >= 30) this.txt(cx, cy, fmt(v, 2), { class: 'cell' }, d);
        if (dropped && dropped[i][j]) {
          const k = cell * .28;
          this.el('line', { x1: cx - k, y1: cy - k, x2: cx + k, y2: cy + k, class: 'x' }, d);
          this.el('line', { x1: cx - k, y1: cy + k, x2: cx + k, y2: cy - k, class: 'x' }, d);
        }
      }
      if (infoFn) this.info.set(d, infoFn(i, j, v));
    }
    if (rowLabels && cell >= 9) rowLabels.forEach((l, i) => this.txt(-7, i * cell + cell / 2, vis(l), { class: 'sm', fill: ROLE[rowRole], 'text-anchor': 'end', 'dominant-baseline': 'middle', 'data-row': i, 'data-role': rowRole }, g));
    if (colLabels && cell >= 9) colLabels.forEach((l, j) => {
      const t = this.txt(0, 0, vis(l), { class: 'sm', fill: ROLE[colRole], 'text-anchor': 'start', 'data-row': j, 'data-role': colRole, transform: `translate(${j * cell + cell / 2 + 3},-6) rotate(-55)` }, g);
      if (colStep != null) t.setAttribute('data-step', colStep);
    });
    if (title) this.txt(0, R * cell + 16, title, { class: 'lbl' }, g);
    const w = C * cell, h = R * cell;
    return { x, y, w, h, cell, rowLeft: i => [x, y + i * cell + cell / 2], rowRight: i => [x + w, y + i * cell + cell / 2], colTop: j => [x + j * cell + cell / 2, y], left: [x, y + h / 2], right: [x + w, y + h / 2], bottom: [x + w / 2, y + h] };
  }

  // ----- ribbon (bezier) --------------------------------------------------
  ribbon({ from, to, role = 'x', width = 1.5, step = 0, row = null, opacity = .55, vertical = false, color = null, until = null }) {
    const [x1, y1] = from, [x2, y2] = to;
    const d = vertical
      ? `M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`
      : `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`;
    return this.el('path', { d, class: 'ribbon', stroke: color || ROLE[role] || role, 'stroke-width': width, opacity, pathLength: 1, 'data-step': this._step(step), 'data-until': until, 'data-row': row, 'data-role': role, style: `color:${color || ROLE[role]}` });
  }

  // ----- small operator pill ---------------------------------------------
  op({ x, y, text, sub = null, step = 0, w = null, role = null }) {
    w = w ?? Math.max(40, text.length * 7 + 18);
    const h = sub ? 30 : 22;
    const g = this.el('g', { class: 'op', 'data-step': this._step(step), transform: `translate(${x - w / 2},${y - h / 2})` });
    const r = this.el('rect', { width: w, height: h, rx: h / 2 }, g);
    if (role) r.setAttribute('stroke', ROLE[role]);
    this.txt(w / 2, sub ? 10 : h / 2, text, {}, g);
    if (sub) this.txt(w / 2, 22, sub, { class: 'sub' }, g);
    return { x: x - w / 2, y: y - h / 2, w, h, left: [x - w / 2, y], right: [x + w / 2, y] };
  }

  // ----- token pill --------------------------------------------------------
  pill({ x, y, text, row = null, step = 0, role = 'id', w = null, h = 20, fill = null, info = null, sub = null }) {
    w = w ?? Math.max(28, vis(text).length * 7.2 + 14);
    const g = this.el('g', { class: 'pill', 'data-step': this._step(step), 'data-row': row, transform: `translate(${x},${y - h / 2})` });
    const r = this.el('rect', { width: w, height: h, stroke: ROLE[role] }, g);
    if (fill) r.setAttribute('fill', fill);
    this.txt(w / 2, h / 2, vis(text), { class: 'tok' }, g);
    if (sub != null) this.txt(w / 2, h + 10, String(sub), { class: 'sm', 'text-anchor': 'middle' }, g);
    if (info) this.info.set(g, info);
    return { x, y, w, h, left: [x, y], right: [x + w, y] };
  }

  // ----- embedding table "ruler": a tall table with marks at looked-up rows --
  ruler({ x, y, w = 34, h = 300, n, marks, step = 0, title = null, role = 'x', sub = null }) {
    const g = this.el('g', { class: 'ruler', 'data-step': this._step(step), transform: `translate(${x},${y})` });
    this.el('rect', { class: 'body', width: w, height: h, rx: 5 }, g);
    for (let k = 1; k < 8; k++) this.el('line', { x1: 0, y1: h * k / 8, x2: w, y2: h * k / 8, stroke: '#1c2532' }, g);
    if (title) this.txt(w / 2, -10, title, { class: 'lbl', 'text-anchor': 'middle', fill: ROLE[role] }, g);
    if (sub) this.txt(w / 2, h + 14, sub, { class: 'sm', 'text-anchor': 'middle' }, g);
    const pos = {};
    marks.forEach(mk => {
      const yy = Math.round((mk.idx + .5) / n * h);
      const t = this.el('g', { 'data-row': mk.row, 'data-role': role }, g);
      this.el('line', { x1: 0, y1: yy, x2: w, y2: yy, class: 'tick', stroke: mk.color || ROLE[role] }, t);
      if (mk.info) this.info.set(t, mk.info);
      pos[mk.row] = [x + w, y + yy];
    });
    return { x, y, w, h, at: r => pos[r], left: [x, y + h / 2], right: [x + w, y + h / 2] };
  }

  frame({ x, y, w, h, title = null, step = 0, role = null }) {
    const g = this.el('g', { 'data-step': this._step(step) });
    const r = this.el('rect', { x, y, width: w, height: h, class: 'frame', rx: 10 }, g);
    if (role) r.setAttribute('stroke', ROLE[role]);
    if (title) this.txt(x + 12, y - 8, title, { class: 'hd', fill: role ? ROLE[role] : null }, g);
    return { x, y, w, h };
  }

  label(x, y, text, attrs = {}) { return this.txt(x, y, text, { class: 'lbl', ...attrs }); }

  // ----- step reveal ------------------------------------------------------
  setStep(n) {
    this.svg.querySelectorAll('[data-step]').forEach(e => {
      const s = +e.getAttribute('data-step'), u = e.getAttribute('data-until');
      e.classList.toggle('future', s > n || (u != null && n > +u));
    });
  }

  // ----- hover lighting + inspector -------------------------------------
  _wire() {
    const svg = this.svg;
    svg.addEventListener('mouseover', (e) => {
      let t = e.target, infoEl = null, rowEl = null;
      while (t && t !== svg) {
        if (!infoEl && this.info.has(t)) infoEl = t;
        if (!rowEl && t.hasAttribute && t.hasAttribute('data-row')) rowEl = t;
        t = t.parentNode;
      }
      if (infoEl) this.onInspect(this.info.get(infoEl));
      if (rowEl) {
        const row = rowEl.getAttribute('data-row'), col = rowEl.getAttribute('data-col');
        const role = rowEl.getAttribute('data-role');
        this.light(row, col, role);
      }
    });
    svg.addEventListener('mouseout', (e) => {
      if (!svg.contains(e.relatedTarget)) { this.unlight(); this.onInspect(null); }
    });
    svg.addEventListener('mouseleave', () => { this.unlight(); this.onInspect(null); });
  }
  light(row, col, role) {
    this.svg.classList.add('hot');
    this.svg.querySelectorAll('[data-row]').forEach(e => {
      const r = e.getAttribute('data-row'), c = e.getAttribute('data-col'), er = e.getAttribute('data-role');
      let lit;
      if (col != null) {
        // a loom dot (query row, key col): light query-side row and key-side col
        lit = (c != null) ? (r === row && c === col) || (r === row && c == null) : (er === 'k' ? r === col : r === row);
      } else {
        lit = r === row || (c != null && c === row && role === 'k');
      }
      e.classList.toggle('lit', lit);
    });
  }
  unlight() { this.svg.classList.remove('hot'); this.svg.querySelectorAll('.lit').forEach(e => e.classList.remove('lit')); }
}

/** helpers for panel layouts */
export const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
export const info = (title, rows = [], f = null, roleKey = null) => ({ title, rows, f, role: roleKey });
