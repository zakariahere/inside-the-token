import { post } from '../api.js';
import { Scene, info, ROLE, clamp, vis } from '../scene.js';
import { rows, SOURCE, DIN, legend, shapes, headColor } from './common.js';
import { xColumn, vecColumn, loomBlock, contextColumn, scoreInfo, weightInfo } from './attn.js';

const S = { source: 'book', emb_dim: 3, d_out: 2, num_heads: 2, dropout: 0.0, train: false, qkv_bias: false, apply_ln1: true, head: 0, show_dims: 16 };

export default {
  id: 'ch03-mha', chapter: 3, section: '§3.6', title: 'Many heads, one output',
  S,
  controlSpec(G) {
    const gpt2 = G.weights === 'gpt2';
    return [SOURCE(gpt2), DIN(S, G),
      { key: 'd_out', label: 'd_out', type: 'select', disabled: gpt2, options: [[2, '2 · book'], [4, '4'], [8, '8'], [12, '12'], [64, '64'], [768, '768']] },
      { key: 'num_heads', label: 'num_heads', type: 'select', disabled: gpt2, options: [[1, '1'], [2, '2 · book'], [3, '3'], [4, '4'], [6, '6'], [8, '8'], [12, '12 · GPT-2']] },
      { key: 'dropout', label: 'dropout', type: 'range', min: 0, max: 0.9, step: 0.1, disabled: gpt2 },
      { key: 'train', label: 'training mode', type: 'check', disabled: gpt2 },
      { key: 'qkv_bias', label: 'qkv_bias', type: 'check', disabled: gpt2 },
      { key: 'apply_ln1', label: 'GPT-2 ln_1 first (ch. 4)', type: 'check', disabled: !gpt2 },
      { key: 'show_dims', label: 'dims drawn', type: 'range', min: 4, max: 64, step: 4 }];
  },
  async load(G) {
    const gpt2 = G.weights === 'gpt2';
    if (gpt2 && S.source === 'book') S.source = 'text';
    const r = await post('/ch03/mha', { ...S, text: G.text, seed: G.seed, weights: G.weights });
    S.head = Math.min(S.head, r.config.num_heads - 1);
    return r;
  },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const c = r.config, toks = r.tokens, T = toks.length, H = c.num_heads, h = S.head, hc = headColor(h);
    const miniCell = clamp(Math.floor(120 / T), 2, 10), miniH = T * miniCell;
    const R = rows(T, { top: miniH + 90 });
    const hd = r.heads[h];

    // ---- head gallery (click to select) ---------------------------------------
    sc.txt(110, 18, `attention pattern of every head · click to inspect`, { class: 'hd' });
    r.heads.forEach((hh, i) => {
      const gx = 110 + i * (T * miniCell + 18);
      const g = sc.el('g', { 'data-step': 5, style: 'cursor:pointer' });
      const lm = sc.loom({ x: gx, y: 32, data: hh.attn_weights_dropped.data, cell: miniCell, step: 5, hi: 1 });
      sc.el('rect', { x: gx - 3, y: 29, width: T * miniCell + 6, height: miniH + 6, rx: 5, fill: 'none', stroke: i === h ? headColor(i) : 'transparent', 'stroke-width': 2 }, g);
      sc.txt(gx, 32 + miniH + 14, `head ${i}`, { class: 'sm', fill: headColor(i) }, g);
      g.addEventListener('click', () => { S.head = i; document.dispatchEvent(new CustomEvent('rerender')); });
      sc.info.set(g, info(`head ${i}`, [['dims', `${i * c.head_dim}–${(i + 1) * c.head_dim - 1}`], ['click', 'to inspect']]));
      sc.el('rect', { x: gx, y: 32, width: T * miniCell, height: miniH, fill: 'transparent' }, g);   // click target
    });

    // ---- x and the three flat projections -------------------------------------
    const xs = xColumn(sc, { toks, R, data: r.x.data, x: 110, step: 0, maxW: 120 });
    const fx = 290, fw = 96;
    const qf = vecColumn(sc, { toks, R, data: r.queries_flat.data, x: fx, step: 1, role: 'q', title: 'W_query(x)', from: xs, maxW: fw, segments: H });
    const kf = vecColumn(sc, { toks, R, data: r.keys_flat.data, x: fx + fw + 30, step: 1, role: 'k', title: 'W_key(x)', from: xs, maxW: fw, segments: H, ribbonOpacity: .2 });
    const vf = vecColumn(sc, { toks, R, data: r.values_flat.data, x: fx + 2 * (fw + 30), step: 1, role: 'v', title: 'W_value(x)', from: xs, maxW: fw, segments: H, ribbonOpacity: .2 });
    sc.txt(fx, R.bottom + 18, `colour bars under each strip = which head owns those ${c.head_dim} dims`, { class: 'sm', 'data-step': 1 });

    // ---- selected head's slices ------------------------------------------------
    const hx = fx + 3 * (fw + 30) + 30, hw = 56;
    sc.frame({ x: hx - 14, y: R.top - 36, w: 3 * (hw + 26) + 10, h: R.bottom - R.top + 60, title: `head ${h} · view → transpose`, step: 2, role: null });
    const segFrom = (col, i) => [col[i].x + (h + .5) / H * col[i].w, col[i].bottom[1]];
    const qh = vecColumn(sc, { toks, R, data: hd.queries.data, x: hx, step: 2, role: 'q', title: `q[${h}]`, maxW: hw });
    const kh = vecColumn(sc, { toks, R, data: hd.keys.data, x: hx + hw + 26, step: 2, role: 'k', title: `k[${h}]`, maxW: hw });
    const vh = vecColumn(sc, { toks, R, data: hd.values.data, x: hx + 2 * (hw + 26), step: 2, role: 'v', title: `v[${h}]`, maxW: hw });
    toks.forEach((_, i) => {
      sc.ribbon({ from: qf[i].right, to: qh[i].left, color: hc, step: 2, row: i, opacity: .35, width: 1 });
      sc.ribbon({ from: kf[i].right, to: kh[i].left, color: hc, step: 2, row: i, opacity: .2, width: 1 });
      sc.ribbon({ from: vf[i].right, to: vh[i].left, color: hc, step: 2, row: i, opacity: .2, width: 1 });
    });

    // ---- loom sequence ---------------------------------------------------------
    const lx = hx + 3 * (hw + 26) + 50;
    const scores = loomBlock(sc, { toks, R, data: hd.attn_scores.data, x: lx, step: 3, title: `scores · head ${h}`, from: qh, keys: kh, signed: true, infoFn: scoreInfo(toks, hd.queries.data, hd.keys.data) });
    sc.op({ x: scores.x + scores.w + 45, y: R.mid, text: 'mask', sub: '−∞', step: 4, role: 'k' });
    const masked = loomBlock(sc, { toks, R, data: hd.masked_scores.data, x: scores.x + scores.w + 90, step: 4, title: 'masked', prev: scores, signed: true, infoFn: scoreInfo(toks, hd.queries.data, hd.keys.data) });
    const active = c.training && c.dropout > 0;
    const dropped = hd.keep_mask.data.map(row => row.map(v => v === 0));
    sc.op({ x: masked.x + masked.w + 45, y: R.mid, text: `÷ √${c.d_k}`, sub: active ? 'softmax · dropout' : 'softmax', step: 5 });
    const weights = loomBlock(sc, { toks, R, data: hd.attn_weights_dropped.data, x: masked.x + masked.w + 90, step: 5, title: `attn_weights · head ${h}`, prev: masked, hi: active ? null : 1, dropped: active ? dropped : null,
      infoFn: weightInfo(toks, hd.attn_weights_dropped.data) });
    sc.op({ x: weights.x + weights.w + 45, y: R.mid, text: '@ v', step: 6, role: 'v' });
    const ctx = contextColumn(sc, { toks, R, data: hd.context_heads.data, x: weights.x + weights.w + 90, step: 6, from: weights, title: `context[${h}]`, role: 'a' });

    // ---- concat + out_proj ------------------------------------------------------
    const cx = ctx[0].x + 140;
    const cat = vecColumn(sc, { toks, R, data: r.context_concat.data, x: cx, step: 7, role: 'a', title: 'concat heads', maxW: fw, segments: H });
    toks.forEach((_, i) => sc.ribbon({ from: ctx[i].right, to: [cat[i].x + (h + .5) / H * cat[i].w, cat[i].top[1]], color: hc, step: 7, row: i, opacity: .35, width: 1 }));
    const gcell = Math.max(5, Math.min(14, Math.floor(130 / Math.max(r.out_proj_weight.shown_shape[0], r.out_proj_weight.shown_shape[1]))));
    const opg = sc.grid({ x: cx + fw + 50, y: R.top + 8, data: r.out_proj_weight.data, role: 'o', step: 8, cell: gcell, title: `out_proj.weight [${r.out_proj_weight.shape.join(', ')}]`, infoFn: (i, j, val) => info(`out_proj.weight[${i}, ${j}]`, [['value', val]]) });
    sc.op({ x: opg.x + opg.w / 2, y: opg.y + opg.h + 24, text: 'x @ Wᵀ + b', step: 8, role: 'o' });
    const outs = vecColumn(sc, { toks, R, data: r.out.data, x: opg.x + opg.w + 60, step: 9, role: 'o', title: 'context_vecs (out)', maxW: fw });
    toks.forEach((_, i) => sc.ribbon({ from: cat[i].right, to: outs[i].left, role: 'o', step: 9, row: i, opacity: .4 }));

    const gpt2 = c.weights === 'gpt2';
    return {
      scene: sc,
      code: [
        'def forward(self, x):',
        '    b, num_tokens, d_in = x.shape',
        '    keys = self.W_key(x)                     # (b, T, d_out)',
        '    queries = self.W_query(x)',
        '    values = self.W_value(x)',
        '    # split d_out into (num_heads, head_dim), then move heads forward',
        '    keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)',
        '    queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)',
        '    values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)',
        '    attn_scores = queries @ keys.transpose(2, 3)                 # (b, h, T, T)',
        '    mask_bool = self.mask.bool()[:num_tokens, :num_tokens]',
        '    attn_scores.masked_fill_(mask_bool, -torch.inf)',
        '    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)',
        '    attn_weights = self.dropout(attn_weights)',
        '    context_vec = (attn_weights @ values).transpose(1, 2)        # (b, T, h, head_dim)',
        '    context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)   # concat',
        '    context_vec = self.out_proj(context_vec)',
        '    return context_vec',
      ],
      steps: [
        { label: 'x', line: [0, 1], note: gpt2 ? `Real GPT-2 block 0. ${c.ln1_applied ? 'x has been through <code>ln_1</code> first, as in the real model.' : '<b>ln_1 is off</b>: the trained weights see inputs at the wrong scale.'}` : `x is [${c.batch}, ${T}, ${c.d_in}].` },
        { label: 'one big projection each', line: [2, 3, 4], note: `W_query is one Linear of width d_out=${c.d_out}. The coloured bars show which slice belongs to which head. No per-head matrices exist.` },
        { label: 'view + transpose', line: [5, 6, 7, 8], note: `Free reshapes: [T, ${c.d_out}] → [T, ${H}, ${c.head_dim}] → [${H}, T, ${c.head_dim}]. Head ${h} is now ${c.head_dim} dims wide.` },
        { label: 'scores per head', line: 9, note: 'One matmul does all heads at once thanks to the head axis being in front.' },
        { label: 'mask', line: [10, 11], note: 'Same causal buffer as §3.5, broadcast across heads.' },
        { label: 'softmax' + (active ? ' + dropout' : ''), line: [12, 13], note: 'Each head gets its own attention pattern. Click any head in the gallery to inspect it.' },
        { label: 'context per head', line: 14, note: `[${H}, T, ${c.head_dim}] → transpose back to [T, ${H}, ${c.head_dim}].` },
        { label: 'concat', line: 15, note: `<code>contiguous().view</code> lays the heads side by side: [T, ${c.d_out}]. The coloured bar shows where head ${h} landed.` },
        { label: 'out_proj', line: 16, note: 'A final Linear mixes information across heads. The only place heads talk to each other.' },
        { label: 'output', line: 17, note: gpt2 ? 'Verified against HuggingFace GPT-2 to 1e-4.' : `Book check with seed 123, d_out=2, num_heads=2: first row [0.3190, 0.4858].` },
      ],
      guide: `${shapes([['W_query(x)', r.shapes.queries_flat], ['queries', r.shapes.queries], ['attn_scores', r.shapes.attn_scores], ['context', r.shapes.context_heads], ['concat', r.shapes.context_concat], ['out', r.shapes.out]])}
        ${legend([['x', 'x'], ['q', 'query'], ['k', 'key'], ['v', 'value'], ['a', 'attention'], ['o', 'out']])}
        <p>head_dim = d_out / num_heads = ${c.head_dim}. Heads are colour-coded: <span style="color:${hc}">■</span> head ${h} is selected.</p>
        ${gpt2 ? '<h3>What to look for</h3><p>Heads that attend to the <b>previous token</b> (bright sub-diagonal), heads that park attention on the <b>first token</b> (an attention sink), heads tracking a repeated word.</p>' : '<h3>Try</h3><p>Switch <b>weights</b> to GPT-2 small and type a sentence with a repeated word. Real heads have jobs.</p>'}`,
    };
  },
};
