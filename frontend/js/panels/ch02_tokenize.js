import { post } from '../api.js';
import { Scene, ROLE, info, vis } from '../scene.js';
import { legend } from './common.js';

const hue = (id, a = 1) => `hsla(${(id * 137.508) % 360}, 75%, 65%, ${a})`;

export default {
  id: 'ch02-tokenize', chapter: 2, section: '§2.5', title: 'Byte pair encoding: text → integers',
  S: {},
  async load(G) { return post('/ch02/tokenize', { text: G.text }); },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const T = r.count, text = r.decoded;

    // ---- the string, character by character, with token cuts ----------------
    const charW = 8.6, perLine = 96, lineH = 30, x0 = 20, y0 = 40;
    const pos = (ci) => [x0 + (ci % perLine) * charW, y0 + Math.floor(ci / perLine) * lineH];
    sc.txt(x0, 18, 'raw text', { class: 'hd' });
    [...text].forEach((ch, ci) => { const [x, y] = pos(ci); sc.txt(x, y, ch === ' ' ? '·' : ch, { class: 'tok', fill: '#8b97a5', 'data-step': 0 }); });
    let ci = 0;
    const spans = r.tokens.map((t, i) => {
      const len = t.text.length, start = ci; ci += len;
      // coloured overlay characters + bracket under the span (step 1)
      const g = sc.el('g', { 'data-step': 1, 'data-row': i });
      for (let k = start; k < start + len; k++) { const [x, y] = pos(k); const ch = text[k]; sc.txt(x, y, ch === ' ' ? '·' : ch, { class: 'tok', fill: hue(t.id) }, g); }
      const [ax, ay] = pos(start), [bx] = pos(start + len - 1);
      sc.el('path', { d: `M${ax - 1},${ay + 5} v4 H${bx + charW - 1} v-4`, stroke: hue(t.id), fill: 'none', 'stroke-width': 1.5 }, g);
      sc.info.set(g, info(`token ${i} <i style="background:${hue(t.id)};color:#000">${vis(t.text)}</i>`, [['id', t.id], ['chars', len], ['bytes', `[${t.bytes.join(', ')}]`]]));
      return { start, len, mid: [(ax + bx + charW) / 2, ay + 10] };
    });

    // ---- token rows -----------------------------------------------------------
    const lines = Math.ceil(text.length / perLine);
    const top = y0 + lines * lineH + 40, rowH = T > 40 ? 16 : 24, y = i => top + i * rowH + rowH / 2;
    sc.txt(x0, top - 16, 'tokens', { class: 'hd' });
    sc.txt(260, top - 16, 'integers', { class: 'hd' });
    sc.txt(400, top - 16, 'vocabulary · 50,257 rows', { class: 'hd' });
    const rulerH = Math.max(240, Math.min(T * rowH, 520));
    const ruler = sc.ruler({ x: 420, y: top, h: rulerH, n: r.vocab_size, step: 3, role: 'id', sub: 'row = id',
      marks: r.tokens.map((t, i) => ({ idx: t.id, row: i, color: hue(t.id), info: info(`vocab row ${t.id}`, [['token', vis(t.text)], ['position in table', `${(t.id / r.vocab_size * 100).toFixed(1)} %`]]) })) });
    r.tokens.forEach((t, i) => {
      const p = sc.pill({ x: x0, y: y(i), text: t.text, row: i, step: 1, fill: hue(t.id, .16), w: Math.max(36, vis(t.text).length * 7.4 + 14), h: rowH - 5,
        info: info(`token ${i} <i style="background:${hue(t.id)};color:#000">${vis(t.text)}</i>`, [['id', t.id], ['bytes', `[${t.bytes.join(', ')}]`], ['special', t.text === '<|endoftext|>' ? 'yes' : 'no']]) });
      sc.ribbon({ from: spans[i].mid, to: [p.x + p.w / 2, y(i) - rowH / 2 + 2], role: 'x', step: 1, row: i, opacity: .25, vertical: true, color: hue(t.id) });
      sc.ribbon({ from: p.right, to: [255, y(i)], role: 'id', step: 2, row: i, color: hue(t.id), opacity: .5 });
      const g = sc.el('g', { 'data-step': 2, 'data-row': i });
      sc.txt(260, y(i), String(t.id), { class: 'tok', fill: hue(t.id), 'dominant-baseline': 'middle' }, g);
      sc.ribbon({ from: [300, y(i)], to: [ruler.x, ruler.at(i)[1]], role: 'id', step: 3, row: i, color: hue(t.id), opacity: .5 });
    });

    // ---- decode ------------------------------------------------------------------
    const dy = Math.max(top + T * rowH, ruler.y + rulerH) + 50;
    sc.txt(x0, dy - 16, 'tokenizer.decode(integers)', { class: 'hd', 'data-step': 4 });
    sc.txt(x0, dy + 4, r.decoded, { class: 'tok', 'data-step': 4, fill: ROLE.o });

    return {
      scene: sc,
      code: [
        'text = ' + JSON.stringify(G.text.length > 60 ? G.text.slice(0, 57) + '…' : G.text),
        'tokenizer = tiktoken.get_encoding("gpt2")               # BPE, 50,257 tokens, no <|unk|>',
        'integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})',
        `# len(integers) = ${T}   each integer is a row number in the vocabulary`,
        'strings = tokenizer.decode(integers)                     # lossless round trip',
      ],
      steps: [
        { label: 'raw text', line: 0, note: 'The model never sees characters. Everything starts by cutting the string.' },
        { label: 'BPE cuts it into tokens', line: [1, 2], note: 'Common words are one token, rare words split into pieces. A leading space belongs to the token (<code>␣cat</code> ≠ <code>cat</code>).' },
        { label: 'each token → integer', line: 2, note: `${T} integers. That is the whole input to the model.` },
        { label: 'ids are rows of the vocabulary', line: 3, note: 'The ruler is the 50,257-row table; each tick is where a token lives. Chapter 2.7 turns those rows into vectors.' },
        { label: 'decode reverses it', line: 4, note: 'Byte-level BPE can encode any string, so decoding gives the text back exactly.' },
      ],
      guide: `<h3>What to try</h3><p>Type <code>someunknownPlace</code> (splits into pieces), an emoji (several byte tokens), or a capitalised word vs lowercase.</p>
        <h3>Why not the regex tokenizer?</h3><p>§2.3–2.4 build a toy tokenizer with an <code>&lt;|unk|&gt;</code> token to explain the idea. The book then switches to GPT-2's real BPE (§2.5), which is what runs here.</p>
        ${legend([['id', 'token ids']])}`,
    };
  },
};
