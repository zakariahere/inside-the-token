import { post } from '../api.js';
import { Scene, info, fmt, vis, ROLE } from '../scene.js';
import { rows, SOURCE, DIN, legend, shapes } from './common.js';
import { xColumn, loomBlock, contextColumn, weightInfo } from './attn.js';

const S = { source: 'book', emb_dim: 3 };

export default {
  id: 'ch03-simple', chapter: 3, section: '§3.3', title: 'Attention with no weights to learn',
  S,
  controlSpec(G) { return [SOURCE(), DIN(S, G)]; },
  async load(G) { return post('/ch03/simple', { ...S, text: G.text, seed: G.seed, weights: G.weights, query_index: 1, show_dims: 16 }); },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const toks = r.tokens, T = toks.length, R = rows(T), X = r.inputs.data, dIn = r.inputs.shape[1];

    const xs = xColumn(sc, { toks, R, data: X, x: 110, step: 0, maxW: 150 });
    const l1x = 110 + 150 + 70;
    const scores = loomBlock(sc, { toks, R, data: r.attn_scores.data, x: l1x, step: 1, title: 'attn_scores = x @ xᵀ', from: xs, signed: true, rowRole: 'x', colRole: 'x',
      infoFn: (i, j, v) => info(`ω · ${vis(toks[i])} · ${vis(toks[j])}`, [['dot product', v], ...(dIn <= 8 ? [['terms', X[i].map((a, d) => `${fmt(a, 2)}·${fmt(X[j][d], 2)}`).join(' + ')]] : [])]) });
    sc.op({ x: scores.x + scores.w + 40, y: R.mid, text: 'softmax', sub: 'dim=-1', step: 2 });
    const weights = loomBlock(sc, { toks, R, data: r.attn_weights.data, x: scores.x + scores.w + 80, step: 2, title: 'attn_weights · rows sum to 1', prev: scores, hi: 1, rowRole: 'x', colRole: 'x',
      infoFn: weightInfo(toks, r.attn_weights.data) });
    sc.op({ x: weights.x + weights.w + 45, y: R.mid, text: '@ inputs', step: 3 });
    contextColumn(sc, { toks, R, data: r.context_vecs.data, x: weights.x + weights.w + 90, step: 3, from: weights, title: 'context_vecs', absmax: Math.max(...X.flat().map(Math.abs)) });

    return {
      scene: sc,
      code: [
        `inputs = torch.tensor([[0.43, 0.15, 0.89], ...])      # [${T}, ${dIn}]  one row per token`,
        'attn_scores = inputs @ inputs.T                        # [T, T]  every pair, dot product',
        'attn_weights = torch.softmax(attn_scores, dim=-1)      # each row sums to 1',
        `all_context_vecs = attn_weights @ inputs               # [${T}, ${dIn}]  weighted mix of the inputs`,
      ],
      steps: [
        { label: 'inputs', line: 0, note: `The book's six 3-number vectors for <i>Your journey starts with one step</i>. Switch inputs to feed your own text through the chapter 2 embeddings.` },
        { label: 'scores = dot products', line: 1, note: 'Row i, column j is x<sub>i</sub>·x<sub>j</sub>: how similar two tokens are. Blue dots are negative. Hover a dot for the arithmetic.' },
        { label: 'softmax per row', line: 2, note: 'Each row becomes a probability distribution. The book first tries plain ω/Σω, then keeps softmax: always positive, well-behaved gradients (§3.3.1).' },
        { label: 'context = weighted sum', line: 3, note: 'Token i\'s new vector is Σ<sub>j</sub> α<sub>ij</sub> x<sub>j</sub>. Note every token can see every other, including later ones — fixed in §3.5.' },
      ],
      guide: `${shapes([['inputs', r.inputs.shape], ['attn_scores', r.attn_scores.shape], ['attn_weights', r.attn_weights.shape], ['context_vecs', r.context_vecs.shape]])}
        ${legend([['x', 'inputs'], ['a', 'attention']])}
        <h3>The one-query walkthrough (§3.3.1)</h3><p>Hover the <b>journey</b> row of the loom. The book's numbers: ω = [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865], α = [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581], z = [0.4419, 0.6515, 0.5683].</p>
        <h3>Why the diagonal is bright</h3><p>A vector is most similar to itself, so without learned projections each token mostly attends to itself. That is one reason for §3.4.</p>`,
    };
  },
};
