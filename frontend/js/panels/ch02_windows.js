import { post } from '../api.js';
import { Scene, ROLE, info, vis } from '../scene.js';
import { legend } from './common.js';

const S = { source: 'text', max_length: 4, stride: 1, batch_size: 2 };

export default {
  id: 'ch02-windows', chapter: 2, section: '§2.6', title: 'Sliding windows: inputs and targets',
  S,
  controlSpec() {
    return [
      { key: 'source', label: 'text', type: 'select', options: [['text', 'toolbar text'], ['verdict', 'the-verdict.txt (book)']] },
      { key: 'max_length', label: 'max_length', type: 'range', min: 1, max: 10 },
      { key: 'stride', label: 'stride', type: 'range', min: 1, max: 10 },
      { key: 'batch_size', label: 'batch_size', type: 'range', min: 1, max: 8 },
    ];
  },
  async load(G) { return post('/ch02/windows', { ...S, text: G.text, max_pairs: 64 }); },
  build(r, G) {
    const sc = new Scene({ showValues: G.numbers });
    const L = S.max_length, N = Math.min(r.tokens.length, 36), K = Math.min(r.pairs.length, 10);
    const pw = 46, x0 = 30, ys = 50;

    // ---- the id stream --------------------------------------------------------
    sc.txt(x0, 18, `token id stream · ${r.num_tokens} ids`, { class: 'hd' });
    const px = i => x0 + i * pw;
    for (let i = 0; i < N; i++) sc.pill({ x: px(i), y: ys, text: r.ids[i], w: pw - 6, row: i, step: 0, sub: vis(r.tokens[i]).slice(0, 6),
      info: info(`position ${i}`, [['id', r.ids[i]], ['token', vis(r.tokens[i])]]) });
    if (r.tokens.length > N) sc.txt(px(N) + 4, ys + 4, `… +${r.num_tokens - N}`, { class: 'sm' });

    // ---- windows sliding over the stream, one per step ----------------------
    const pairs = r.pairs.slice(0, K);
    const top = 130, rowH = 44;
    sc.txt(x0, top - 22, 'input x  →  target y (x shifted by one)', { class: 'hd' });
    pairs.forEach((p, k) => {
      const st = k + 1;
      if (p.start + L < N + 1) {
        sc.el('rect', { x: px(p.start) - 4, y: ys - 16, width: L * pw + 2, height: 32, rx: 8, fill: 'none', stroke: ROLE.q, 'stroke-width': 2, 'data-step': st, 'data-until': st });
        sc.el('rect', { x: px(p.start + 1) - 2, y: ys - 12, width: L * pw - 2, height: 40, rx: 8, fill: 'none', stroke: ROLE.o, 'stroke-width': 2, 'stroke-dasharray': '4 3', 'data-step': st, 'data-until': st });
      }
      const ry = top + k * rowH;
      const g = sc.el('g', { 'data-step': st });
      sc.txt(x0, ry + 4, `i=${p.start}`, { class: 'sm' }, g);
      p.input_ids.forEach((id, j) => {
        sc.pill({ x: x0 + 50 + j * pw, y: ry, text: id, w: pw - 6, step: st, role: 'q', fill: 'rgba(34,211,238,.10)', info: info(`x[${k}][${j}]`, [['id', id], ['token', vis(p.input_tokens[j])], ['stream position', p.start + j]]) });
        if (p.start + j < N) sc.ribbon({ from: [px(p.start + j) + pw / 2 - 3, ys + 22], to: [x0 + 50 + j * pw + pw / 2 - 3, ry - 10], role: 'q', step: st, until: st, vertical: true, opacity: .35 });
      });
      sc.txt(x0 + 50 + L * pw + 8, ry + 4, '→', { class: 'lbl', fill: ROLE.o }, g);
      p.target_ids.forEach((id, j) => sc.pill({ x: x0 + 50 + (L + 1) * pw + (j) * pw, y: ry, text: id, w: pw - 6, step: st, role: 'o', fill: 'rgba(251,191,36,.10)', info: info(`y[${k}][${j}]`, [['id', id], ['token', vis(p.target_tokens[j])], ['= x shifted by', 1]]) }));
    });
    if (r.total_pairs > K) sc.txt(x0, top + K * rowH, `… ${r.total_pairs - K} more pairs (stride ${S.stride})`, { class: 'sm', 'data-step': K });

    // ---- the DataLoader batch --------------------------------------------------
    const fb = r.first_batch, by = top + K * rowH + 50, bst = K + 1;
    if (fb) {
      sc.txt(x0, by - 20, `first batch · inputs [${fb.shape.join(', ')}] · targets [${fb.shape.join(', ')}]`, { class: 'hd', 'data-step': bst });
      fb.inputs.forEach((row, b) => {
        sc.txt(x0, by + b * 30 + 4, `b${b}`, { class: 'sm', 'data-step': bst });
        row.forEach((id, j) => sc.pill({ x: x0 + 40 + j * pw, y: by + b * 30, text: id, w: pw - 6, step: bst, role: 'q', fill: 'rgba(34,211,238,.10)', info: info(`inputs[${b}][${j}]`, [['id', id], ['token', vis(fb.input_tokens[b][j])]]) }));
        fb.targets[b].forEach((id, j) => sc.pill({ x: x0 + 70 + (L + j) * pw, y: by + b * 30, text: id, w: pw - 6, step: bst, role: 'o', fill: 'rgba(251,191,36,.10)', info: info(`targets[${b}][${j}]`, [['id', id], ['token', vis(fb.target_tokens[b][j])]]) }));
      });
      sc.frame({ x: x0 + 30, y: by - 18, w: L * pw + 8, h: fb.inputs.length * 30 + 10, step: bst, role: 'q' });
      sc.frame({ x: x0 + 62 + L * pw, y: by - 18, w: L * pw + 8, h: fb.inputs.length * 30 + 10, step: bst, role: 'o' });
    } else {
      sc.txt(x0, by, 'no full batch: text shorter than max_length + 1', { class: 'sm', 'data-step': bst });
    }

    const steps = [{ label: 'token ids', line: 2, note: 'The whole corpus is one long list of integers. No labels needed: the next id is the label.' }];
    pairs.forEach((p, k) => steps.push({ label: `window i=${p.start}`, line: [3, 4, 5], note: k === 0
      ? `Take ${L} ids as <b>x</b>; the same ${L} ids one step later are <b>y</b>. One window gives ${L} training examples: each prefix of x predicts the matching y.`
      : `stride = ${S.stride}: the window moves ${S.stride} id${S.stride > 1 ? 's' : ''}. ${S.stride === L ? 'No overlap between windows.' : S.stride === 1 ? 'Every id starts a window; heavy overlap.' : 'Partial overlap.'}` }));
    steps.push({ label: 'DataLoader batch', line: [8, 9], note: `<code>batch_size=${S.batch_size}</code> stacks windows into a tensor of shape [${S.batch_size}, ${L}]. This is exactly what the embedding layer receives next.` });

    return {
      scene: sc,
      code: [
        'class GPTDatasetV1(Dataset):',
        '    def __init__(self, txt, tokenizer, max_length, stride):',
        '        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})',
        '        for i in range(0, len(token_ids) - max_length, stride):',
        '            input_chunk = token_ids[i:i + max_length]',
        '            target_chunk = token_ids[i + 1: i + max_length + 1]',
        '            self.input_ids.append(torch.tensor(input_chunk))',
        '            self.target_ids.append(torch.tensor(target_chunk))',
        `dataloader = create_dataloader_v1(text, batch_size=${S.batch_size}, max_length=${L}, stride=${S.stride}, shuffle=False)`,
        'inputs, targets = next(iter(dataloader))',
      ],
      steps,
      guide: `<h3>The trick</h3><p>Next-word prediction is self-supervised: the target is the input shifted one position right. Nothing is hand-labelled.</p>
        ${legend([['q', 'input x'], ['o', 'target y']])}
        <h3>Try</h3><p>Set <b>stride = max_length</b> for non-overlapping windows (the book's GPT-2 setting is 256 / 128). Switch to <b>the-verdict.txt</b> to see the book's first batch <code>[40, 367, 2885, 1464]</code>.</p>`,
    };
  },
};
