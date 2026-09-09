// Small HTML helpers (DOM building, controls, code rendering).

export function el(tag, props = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props || {})) {
    if (v == null) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else if (k.startsWith('on')) n.addEventListener(k.slice(2).toLowerCase(), v);
    else n.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    n.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return n;
}

export const fmt = (v, d = 4) => {
  if (v === '-inf') return '−∞';
  if (v === 'inf') return '∞';
  if (v == null) return 'nan';
  if (typeof v !== 'number') return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(d);
};
export const vis = (s) => String(s).replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
export const shape = (arr, label) => `<span class="shape">${label ? `<b>${label}</b> ` : ''}[${arr.join(', ')}]</span>`;
export const role = (r, txt) => `<span class="role role-${r}">${txt || r}</span>`;
export function debounce(fn, ms = 250) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

/** controls bound to `values`; calls onChange() after any edit.
 *  spec: [{key,label,type:'range'|'select'|'check'|'number',min,max,step,options:[[v,label]],disabled}] */
export function controls(spec, values, onChange) {
  const frag = document.createDocumentFragment();
  for (const s of spec) {
    if (s.type === 'check') {
      const inp = el('input', { type: 'checkbox' });
      inp.checked = !!values[s.key]; inp.disabled = !!s.disabled;
      inp.addEventListener('change', () => { values[s.key] = inp.checked; onChange(); });
      frag.append(el('label', { class: 'ctl check' }, inp, el('span', { text: s.label })));
      continue;
    }
    const lab = el('label', { class: 'ctl' }, el('span', { text: s.label }));
    if (s.type === 'select') {
      const sel = el('select');
      for (const [v, l] of s.options) sel.append(el('option', { value: v, text: l ?? v }));
      sel.value = String(values[s.key]); sel.disabled = !!s.disabled;
      sel.addEventListener('change', () => { values[s.key] = isNaN(+sel.value) ? sel.value : +sel.value; onChange(); });
      lab.append(sel);
    } else if (s.type === 'number') {
      const inp = el('input', { type: 'number', min: s.min, max: s.max, step: s.step ?? 1, value: values[s.key] });
      inp.disabled = !!s.disabled;
      inp.addEventListener('change', () => { values[s.key] = +inp.value; onChange(); });
      lab.append(inp);
    } else {
      const val = el('span', { class: 'val', text: values[s.key] });
      const inp = el('input', { type: 'range', min: s.min, max: s.max, step: s.step ?? 1, value: values[s.key] });
      inp.disabled = !!s.disabled;
      inp.addEventListener('input', () => { val.textContent = inp.value; });
      inp.addEventListener('change', () => { values[s.key] = +inp.value; onChange(); });
      lab.append(el('span', {}, inp, val));
    }
    frag.append(lab);
  }
  return frag;
}

const TOK = /("[^"\n]*"|'[^'\n]*')|(#[^\n]*)|\b(\d+(?:\.\d+)?)\b|\b(class|def|return|for|in|import|from|if|else|not|None|True|False|super|self|assert|as|with)\b|\b([A-Za-z_]\w*)(?=\()/g;
function hl(line) {
  return line.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(TOK, (m, st, cm, nb, kw, fn) =>
    st ? `<span class="st">${st}</span>` : cm ? `<span class="cm">${cm}</span>` : nb ? `<span class="nb">${nb}</span>`
      : kw ? `<span class="kw">${kw}</span>` : `<span class="fn">${fn}</span>`);
}
/** render code lines into `pre`; returns a function(activeLineSet) that updates highlighting */
export function codeLines(pre, lines) {
  pre.innerHTML = lines.map((l, i) => `<span class="ln" data-i="${i}">${hl(l)}</span>`).join('');
  const spans = [...pre.querySelectorAll('.ln')];
  return (onSet, maxSeen) => {
    spans.forEach((s, i) => {
      s.className = 'ln ' + (onSet.has(i) ? 'on' : i <= maxSeen ? 'past' : 'future');
    });
    const first = spans.find(s => s.classList.contains('on'));
    if (first) first.scrollIntoView({ block: 'nearest' });
  };
}
