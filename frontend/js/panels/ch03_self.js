import { post } from '../api.js';
import { Scene, info, fmt, ROLE } from '../scene.js';
import { rows, SOURCE, DIN, legend, shapes } from './common.js';
import { xColumn, vecColumn, loomBlock, contextColumn, scoreInfo, weightInfo } from './attn.js';

const S = { source: 'book', emb_dim: 3, d_out: 2, qkv_bias: false, seed: 789 };

export default {
  id: 'ch03-self', chapter: 3, section: '§3.4', title: 'Trainable queries, keys, values',
  S,
  controlSpec(G) {
    return [SOURCE(), DIN(S, G),
      { key: 'd_out', label: 'd_out', type: 'range', min: 1, max: 16 },
      { key: 'seed', label: 'seed · book 789', type: 'number', min: 0, max: 99999 },
      { key: 'qkv_bias', label: 'qkv_bias', type: 'check' }];
  },
  async load(G) { return post('/ch03/self', { ...S, text: G.text, weights: G.weights, show_dims: 16 }); },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const toks = r.tokens, T = toks.length, R = rows(T), dIn = r.d_in, dOut = r.d_out;
    const q = r.queries.data, k = r.keys.data, v = r.values.data;

    const xs = xColumn(sc, { toks, R, data: r.inputs.data, x: 110, step: 0, maxW: 150 });

    // weight matrices, stacked
    const wx = 330, cell = Math.max(6, Math.min(16, Math.floor(150 / Math.max(r.W_query.shown_shape[0], r.W_query.shown_shape[1]))));
    const gh = r.W_query.shown_shape[0] * cell + 34;
    sc.txt(wx, R.top - 14, 'nn.Linear weights', { class: 'hd' });
    const grids = ['W_query', 'W_key', 'W_value'].map((n, gi) => sc.grid({ x: wx, y: R.top + 12 + gi * gh, data: r[n].data, role: 'qkv'[gi], step: 1, cell, title: `${n}  [${r[n].shape.join(', ')}]${r[n].truncated ? ' (cut)' : ''}`,
      infoFn: (i, j, val) => info(`${n}[${i}, ${j}]`, [['value', val], ['maps input dim', i], ['to output dim', j]]) }));

    // q k v columns
    const qx = wx + 190, kx = qx + 110, vx = kx + 110;
    const qs = vecColumn(sc, { toks, R, data: q, x: qx, step: 2, role: 'q', title: 'queries', from: xs, maxW: 80 });
    const ks = vecColumn(sc, { toks, R, data: k, x: kx, step: 2, role: 'k', title: 'keys', from: xs, maxW: 80, ribbonOpacity: .25 });
    const vs = vecColumn(sc, { toks, R, data: v, x: vx, step: 2, role: 'v', title: 'values', from: xs, maxW: 80, ribbonOpacity: .25 });

    // loom
    const lx = vx + 150;
    const scores = loomBlock(sc, { toks, R, data: r.attn_scores.data, x: lx, step: 3, title: 'attn_scores = q @ kᵀ', from: qs, keys: ks, signed: true, infoFn: scoreInfo(toks, q, k) });
    sc.op({ x: scores.x + scores.w + 45, y: R.mid, text: `÷ √${r.d_k}`, sub: 'softmax', step: 4 });
    const weights = loomBlock(sc, { toks, R, data: r.attn_weights.data, x: scores.x + scores.w + 90, step: 5, title: 'attn_weights', prev: scores, hi: 1,
      infoFn: weightInfo(toks, r.attn_weights.data, (i, j) => [['without ÷√d_k', fmt(r.attn_weights_unscaled.data[i][j], 3)]]) });
    sc.op({ x: weights.x + weights.w + 45, y: R.mid, text: '@ values', step: 6, role: 'v' });
    contextColumn(sc, { toks, R, data: r.context_vec.data, x: weights.x + weights.w + 90, step: 6, from: weights, title: 'context_vec' });
    vs.forEach((g, i) => sc.ribbon({ from: g.right, to: [weights.x + weights.w + 60, R.y(i)], role: 'v', step: 6, row: i, opacity: .12, width: 1 }));

    return {
      scene: sc,
      code: [
        'class SelfAttention_v2(nn.Module):',
        '    def __init__(self, d_in, d_out, qkv_bias=False):',
        '        super().__init__()',
        '        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)',
        '        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)',
        '        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)',
        '',
        '    def forward(self, x):',
        '        keys = self.W_key(x)',
        '        queries = self.W_query(x)',
        '        values = self.W_value(x)',
        '        attn_scores = queries @ keys.T',
        '        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)',
        '        context_vec = attn_weights @ values',
        '        return context_vec',
        '',
        `torch.manual_seed(${S.seed}); sa_v2 = SelfAttention_v2(d_in=${dIn}, d_out=${dOut})`,
      ],
      steps: [
        { label: 'x', line: 7, note: `${T} tokens × ${dIn} dims in.` },
        { label: 'three weight matrices', line: [3, 4, 5, 16], note: `Each is [d_in=${dIn}, d_out=${dOut}]. They are the only things training changes. Shown in the book's [in, out] orientation; nn.Linear stores the transpose.` },
        { label: 'project to q, k, v', line: [8, 9, 10], note: 'Same x, three different linear maps. Query = "what am I looking for", key = "what do I contain", value = "what do I pass on".' },
        { label: 'scores = q · k', line: 11, note: 'Rows are queries, columns are keys. Cyan ribbons feed rows, rose ribbons feed columns.' },
        { label: 'scale by √d_k', line: 12, note: `Dot products grow with dimension; dividing by √${r.d_k} keeps softmax from saturating (§3.4.1). Hover a weight to compare with the unscaled value.` },
        { label: 'softmax', line: 12, note: 'Each row sums to 1 again.' },
        { label: 'mix the values', line: 13, note: 'Context = Σ α · value. Queries and keys only decide the proportions; values decide what is mixed.' },
      ],
      guide: `${shapes([['x', r.inputs.shape], ['W_query', r.W_query.shape], ['queries', r.queries.shape], ['attn_scores', r.attn_scores.shape], ['context_vec', r.context_vec.shape]])}
        ${legend([['x', 'x'], ['q', 'query'], ['k', 'key'], ['v', 'value'], ['a', 'attention'], ['o', 'context']])}
        <h3>Book check</h3><p>With seed 789 the first context row is [−0.0739, 0.0713]. Change <b>d_out</b> or <b>d_in</b> to 768 and watch the unscaled softmax collapse to one-hot.</p>`,
    };
  },
};
