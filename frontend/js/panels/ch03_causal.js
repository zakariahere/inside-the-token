import { post } from '../api.js';
import { Scene, ROLE } from '../scene.js';
import { rows, SOURCE, DIN, legend, shapes } from './common.js';
import { vecColumn, loomBlock, contextColumn, scoreInfo, weightInfo } from './attn.js';

const S = { source: 'book', emb_dim: 3, d_out: 2, dropout: 0.5, train: true, qkv_bias: false };

export default {
  id: 'ch03-causal', chapter: 3, section: '§3.5', title: 'Hide the future: causal mask and dropout',
  S,
  controlSpec(G) {
    return [SOURCE(), DIN(S, G),
      { key: 'd_out', label: 'd_out', type: 'range', min: 1, max: 16 },
      { key: 'dropout', label: 'dropout p', type: 'range', min: 0, max: 0.9, step: 0.1 },
      { key: 'train', label: 'training mode', type: 'check' },
      { key: 'qkv_bias', label: 'qkv_bias', type: 'check' }];
  },
  async load(G) { return post('/ch03/causal', { ...S, text: G.text, seed: G.seed, weights: G.weights, show_dims: 16 }); },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const toks = r.tokens, T = toks.length, R = rows(T);
    const q = r.queries.data, k = r.keys.data, v = r.values.data;
    const keep = r.keep_mask.data, dropped = keep.map(row => row.map(c => c === 0));
    const active = r.training && r.dropout > 0;

    // this endpoint returns q/k/v directly (x was shown in §3.4); the scene starts there
    const qx = 110, kx = qx + 110, vx = kx + 110;
    const qs = vecColumn(sc, { toks, R, data: q, x: qx, step: 0, role: 'q', title: 'queries', maxW: 80 });
    toks.forEach((t, i) => sc.txt(qx - 8, R.y(i), t, { class: 'lbl', 'text-anchor': 'end', 'dominant-baseline': 'middle', 'data-row': i, 'data-step': 0 }));
    const ks = vecColumn(sc, { toks, R, data: k, x: kx, step: 0, role: 'k', title: 'keys', maxW: 80 });
    const vs = vecColumn(sc, { toks, R, data: v, x: vx, step: 0, role: 'v', title: 'values', maxW: 80 });

    const lx = vx + 150;
    const scores = loomBlock(sc, { toks, R, data: r.attn_scores.data, x: lx, step: 1, title: 'attn_scores', from: qs, keys: ks, signed: true, infoFn: scoreInfo(toks, q, k) });
    sc.op({ x: scores.x + scores.w + 45, y: R.mid, text: 'masked_fill_', sub: 'triu → −∞', step: 2, role: 'k' });
    const masked = loomBlock(sc, { toks, R, data: r.masked_scores.data, x: scores.x + scores.w + 90, step: 2, title: 'future = −∞', prev: scores, signed: true, infoFn: scoreInfo(toks, q, k) });
    sc.op({ x: masked.x + masked.w + 45, y: R.mid, text: `÷ √${r.d_k}`, sub: 'softmax', step: 3 });
    const weights = loomBlock(sc, { toks, R, data: r.attn_weights.data, x: masked.x + masked.w + 90, step: 3, title: 'attn_weights', prev: masked, hi: 1,
      infoFn: weightInfo(toks, r.attn_weights.data, (i, j) => [['§3.5.1 renormalised', r.masked_simple_norm.data[i][j]]]) });
    sc.op({ x: weights.x + weights.w + 45, y: R.mid, text: 'dropout', sub: `p=${r.dropout} · ${r.training ? 'train' : 'eval'}`, step: 4 });
    const drop = loomBlock(sc, { toks, R, data: r.attn_weights_dropped.data, x: weights.x + weights.w + 90, step: 4, title: active ? `after dropout · kept ×${(1 / (1 - r.dropout)).toFixed(1)}` : 'after dropout (no-op)', prev: weights, dropped: active ? dropped : null, hi: null,
      infoFn: (i, j, val) => weightInfo(toks, r.attn_weights_dropped.data, () => [['dropped', dropped[i][j] && j <= i ? 'yes → 0' : 'no'], ['scale', active ? `1/(1−${r.dropout}) = ${(1 / (1 - r.dropout)).toFixed(2)}` : '1']])(i, j, val) });
    sc.op({ x: drop.x + drop.w + 45, y: R.mid, text: '@ values', step: 5, role: 'v' });
    contextColumn(sc, { toks, R, data: r.context_vec.data, x: drop.x + drop.w + 90, step: 5, from: drop, title: 'context_vec' });
    vs.forEach((g, i) => sc.ribbon({ from: g.right, to: [drop.x + drop.w + 60, R.y(i)], role: 'v', step: 5, row: i, opacity: .1, width: 1 }));

    return {
      scene: sc,
      code: [
        'class CausalAttention(nn.Module):',
        '    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):',
        '        ...',
        '        self.dropout = nn.Dropout(dropout)',
        "        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))",
        '',
        '    def forward(self, x):',
        '        b, num_tokens, d_in = x.shape',
        '        keys, queries, values = self.W_key(x), self.W_query(x), self.W_value(x)',
        '        attn_scores = queries @ keys.transpose(1, 2)',
        '        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)',
        '        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)',
        '        attn_weights = self.dropout(attn_weights)',
        '        context_vec = attn_weights @ values',
        '        return context_vec',
      ],
      steps: [
        { label: 'q, k, v', line: 8, note: `Batched now: x is [2, ${T}, ${r.d_in}] (the book stacks the same inputs twice). Item 0 is drawn.` },
        { label: 'scores', line: 9, note: 'As in §3.4, but with <code>transpose(1, 2)</code> because of the batch dimension.' },
        { label: 'mask the future', line: [4, 10], note: 'The buffer is an upper-triangular matrix of ones. Where it is 1 the score becomes −∞ (hollow rings). A buffer is saved with the model but never trained.' },
        { label: 'softmax', line: 11, note: 'e<sup>−∞</sup> = 0, so future tokens get exactly zero weight and rows still sum to 1. Identical to §3.5.1\'s "zero out then renormalise" (hover a weight to compare).' },
        { label: 'dropout', line: [3, 12], note: active ? `Training only: each weight is dropped with p = ${r.dropout}; survivors are scaled by 1/(1−p) so the expected sum stays 1. Rows no longer sum to 1 exactly.` : 'p = 0 or eval mode: dropout is the identity. Enable <b>training mode</b> and raise <b>p</b>.' },
        { label: 'context', line: 13, note: 'Each context row now only mixes values from itself and earlier tokens.' },
      ],
      guide: `${shapes([['batch', r.batch_shape], ['queries', r.queries.shape], ['mask', r.mask.shape], ['context_vec', r.context_shape]])}
        ${legend([['q', 'query'], ['k', 'key'], ['v', 'value'], ['a', 'attention'], ['o', 'context']])}
        <h3>Reading the loom</h3><p>Hollow ring = masked. Crossed dot = dropped by dropout. Dot size = weight.</p>
        <h3>Why −∞ and not 0?</h3><p>Zeroing <em>after</em> softmax leaks future information into the denominator; −∞ <em>before</em> softmax removes it from the normalisation entirely (§3.5.2).</p>`,
    };
  },
};
