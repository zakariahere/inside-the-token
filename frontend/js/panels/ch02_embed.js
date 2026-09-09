import { post } from '../api.js';
import { Scene, ROLE, info, vis, fmt } from '../scene.js';
import { legend, shapes } from './common.js';

const S = { source: 'text', max_length: 4, stride: 4, batch_size: 8, emb_dim: 256, show_dims: 24, b: 0 };

export default {
  id: 'ch02-embed', chapter: 2, section: '§2.7–2.8', title: 'Token + positional embeddings',
  S,
  controlSpec(G) {
    const gpt2 = G.weights === 'gpt2';
    return [
      { key: 'source', label: 'text', type: 'select', options: [['text', 'toolbar text'], ['verdict', 'the-verdict.txt (book)']] },
      { key: 'max_length', label: 'max_length', type: 'range', min: 1, max: 12 },
      { key: 'stride', label: 'stride', type: 'range', min: 1, max: 12 },
      { key: 'batch_size', label: 'batch_size', type: 'range', min: 1, max: 8 },
      { key: 'emb_dim', label: 'output_dim', type: 'select', disabled: gpt2, options: [[3, '3'], [16, '16'], [256, '256 · book'], [768, '768 · GPT-2']] },
      { key: 'show_dims', label: 'dims drawn', type: 'range', min: 4, max: 64, step: 4 },
      { key: 'b', label: 'batch item', type: 'range', min: 0, max: S.batch_size - 1 },
    ];
  },
  async load(G) {
    const gpt2 = G.weights === 'gpt2';
    const r = await post('/ch02/embed', { ...S, text: G.text, weights: G.weights, seed: G.seed, emb_dim: gpt2 ? 768 : S.emb_dim });
    S.b = Math.min(S.b, r.inputs.length - 1);
    return r;
  },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const b = S.b, ids = r.inputs[b], toks = r.input_tokens[b], T = ids.length, D = r.emb_dim;
    const rowH = 36, top = 70, y = i => top + i * rowH + rowH / 2;
    const absmax = Math.max(Math.abs(r.input_embeddings.stats.min), Math.abs(r.input_embeddings.stats.max));
    const rh = Math.max(220, T * rowH);
    const stripW = 190;

    sc.txt(20, 22, `inputs[${b}]`, { class: 'hd' });
    const tokRuler = sc.ruler({ x: 150, y: top, h: rh, n: r.vocab_size, step: 1, role: 'id', title: 'token table', sub: `[${r.vocab_size}, ${D}]`,
      marks: ids.map((id, i) => ({ idx: id, row: i, info: info(`row ${id} of the token table`, [['token', vis(toks[i])], ['row width', D]]) })) });
    const posRuler = sc.ruler({ x: 150 + 34 + stripW + 110, y: top, h: rh, n: T, step: 2, role: 'pos', title: 'position table', sub: `[${T}, ${D}]`,
      marks: ids.map((_, i) => ({ idx: i, row: i, info: info(`row ${i} of the position table`, [['position', i], ['same for every batch item', 'yes']]) })) });

    const xTok = tokRuler.x + tokRuler.w + 60, xPos = posRuler.x + posRuler.w + 60, xSum = xPos + stripW + 90;
    sc.txt(xTok, 22, 'token_embeddings[b]', { class: 'hd', fill: ROLE.x });
    sc.txt(xPos, 22, 'pos_embeddings', { class: 'hd', fill: ROLE.pos });
    sc.txt(xSum, 22, 'input_embeddings[b]', { class: 'hd', fill: ROLE.o });

    ids.forEach((id, i) => {
      const p = sc.pill({ x: 20, y: y(i), text: id, w: 58, row: i, step: 0, sub: vis(toks[i]).slice(0, 8), info: info(`inputs[${b}][${i}]`, [['id', id], ['token', vis(toks[i])]]) });
      sc.ribbon({ from: p.right, to: [tokRuler.x, tokRuler.at(i)[1]], role: 'id', step: 1, row: i, opacity: .5 });
      const te = r.token_embeddings.data[b][i], pe = r.pos_embeddings.data[i], ie = r.input_embeddings.data[b][i];
      const st1 = sc.strip({ x: xTok, y: y(i) - 9, values: te, role: 'x', row: i, step: 1, maxW: stripW, absmax, info: info(`token_embeddings[${b}][${i}]`, [['token', vis(toks[i])], ['shape', `[${D}]`], ['first', te.slice(0, 4).map(v => fmt(v, 2)).join(' ')]]) });
      sc.ribbon({ from: tokRuler.at(i), to: st1.left, role: 'x', step: 1, row: i, opacity: .5 });
      sc.ribbon({ from: st1.right, to: [posRuler.x, posRuler.at(i)[1]], role: 'pos', step: 2, row: i, opacity: .25 });
      const st2 = sc.strip({ x: xPos, y: y(i) - 9, values: pe, role: 'pos', row: i, step: 2, maxW: stripW, absmax, info: info(`pos_embeddings[${i}]`, [['position', i], ['shape', `[${D}]`], ['first', pe.slice(0, 4).map(v => fmt(v, 2)).join(' ')]]) });
      sc.ribbon({ from: posRuler.at(i), to: st2.left, role: 'pos', step: 2, row: i, opacity: .5 });
      const st3 = sc.strip({ x: xSum, y: y(i) - 9, values: ie, role: 'o', row: i, step: 3, maxW: stripW, absmax, info: info(`input_embeddings[${b}][${i}]`, [['= token + position', ''], ['first', ie.slice(0, 4).map(v => fmt(v, 2)).join(' ')]]) });
      sc.ribbon({ from: st2.right, to: st3.left, role: 'o', step: 3, row: i, opacity: .5 });
    });
    sc.op({ x: posRuler.x - 30, y: top + T * rowH / 2, text: '+', step: 2, w: 28 });
    sc.op({ x: xSum - 45, y: top + T * rowH / 2, text: '=', step: 3, w: 28 });
    sc.txt(xTok, top + T * rowH + 18, `first ${Math.min(S.show_dims, D)} of ${D} dims drawn`, { class: 'sm' });

    return {
      scene: sc,
      code: [
        `token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)   # [${r.vocab_size}, ${D}]`,
        `pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)  # [${T}, ${D}]`,
        '',
        `dataloader = create_dataloader_v1(text, batch_size=${S.batch_size}, max_length=${S.max_length}, stride=${S.stride}, shuffle=False)`,
        `inputs, targets = next(iter(dataloader))                               # [${r.inputs.length}, ${T}]`,
        `token_embeddings = token_embedding_layer(inputs)                       # [${r.inputs.length}, ${T}, ${D}]`,
        `pos_embeddings = pos_embedding_layer(torch.arange(max_length))        # [${T}, ${D}]`,
        `input_embeddings = token_embeddings + pos_embeddings                   # [${r.inputs.length}, ${T}, ${D}]`,
      ],
      steps: [
        { label: 'ids', line: [3, 4], note: `Batch item ${b}: ${T} integers. Change <b>batch item</b> to see other rows of the batch.` },
        { label: 'token lookup', line: [0, 5], note: `<code>nn.Embedding</code> is a lookup table: id ${ids[0]} pulls out row ${ids[0]}. Same id anywhere → same row.` },
        { label: 'position lookup', line: [1, 6], note: 'A second table indexed by position 0…' + (T - 1) + '. Without it attention could not tell word order.' },
        { label: 'add', line: 7, note: 'Element-wise sum, broadcast over the batch. This is the <b>x</b> every chapter 3 scene starts from.' },
      ],
      guide: `${shapes([['inputs', [r.inputs.length, T]], ['token_embeddings', r.token_embeddings.shape], ['pos_embeddings', r.pos_embeddings.shape], ['input_embeddings', r.input_embeddings.shape]])}
        ${legend([['id', 'ids'], ['x', 'token vectors'], ['pos', 'position vectors'], ['o', 'sum']])}
        <p>${G.weights === 'gpt2' ? 'These are GPT-2\'s trained <code>wte</code> / <code>wpe</code> tables. Hover the position ruler: nearby positions get similar rows.' : `Random init with <code>torch.manual_seed(${G.seed})</code>, as in the book. Only the shapes and the lookup mechanics matter yet.`}</p>
        <h3>Try</h3><p>Repeat a word (<code>the cat and the dog</code>): identical token rows, different sums.</p>`,
    };
  },
};
