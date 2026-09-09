import { Scene, ROLE, info } from '../scene.js';
import { legend } from './common.js';

const TOKS = ['Your', 'journey', 'starts', 'with', 'one', 'step'];
// illustrative causal weights for the mini loom (not computed)
const W = TOKS.map((_, i) => TOKS.map((_, j) => j > i ? '-inf' : +(1 / (i + 1) * (j === i ? 1.4 : 0.8)).toFixed(2)));

export default {
  id: 'ch01-intro', chapter: 1, section: '§1', title: 'What a GPT does, and which parts we build',
  S: {},
  async load() { return {}; },
  build(_, G) {
    const sc = new Scene({ showValues: G.numbers });
    const y0 = 40, rowH = 30, y = i => y0 + 60 + i * rowH + rowH / 2;

    // stage 0: text
    sc.txt(20, y0, 'text', { class: 'hd' });
    TOKS.forEach((t, i) => sc.pill({ x: 20, y: y(i), text: t, row: i, step: 0, info: info(`token <i style="background:${ROLE.id};color:#000">${t}</i>`, [['chapter', '2']]) }));

    // stage 1: ids
    sc.txt(150, y0, 'token ids', { class: 'hd' });
    const ids = [7120, 7002, 4940, 351, 530, 2239];
    TOKS.forEach((t, i) => { sc.ribbon({ from: [90, y(i)], to: [150, y(i)], role: 'id', step: 1, row: i }); sc.pill({ x: 150, y: y(i), text: ids[i], row: i, step: 1, w: 52 }); });

    // stage 2: embeddings
    sc.txt(260, y0, 'embeddings', { class: 'hd' });
    TOKS.forEach((t, i) => {
      sc.ribbon({ from: [202, y(i)], to: [260, y(i)], role: 'x', step: 2, row: i });
      const vals = Array.from({ length: 12 }, (_, k) => Math.sin((i + 1) * (k + 1) * 0.7) * 0.8);
      sc.strip({ x: 260, y: y(i) - 9, values: vals, role: 'x', row: i, step: 2, maxW: 120, absmax: 1 });
    });

    // stage 3: attention loom
    sc.txt(430, y0, 'causal attention', { class: 'hd' });
    const lm = sc.loom({ x: 460, y: y(0) - rowH / 2, data: W, cell: rowH, step: 3, colLabels: TOKS, colStep: 3,
      infoFn: (i, j, v) => info(`α · ${TOKS[i]} → ${TOKS[j]}`, [['weight', v === '-inf' ? 'masked' : v]]) });
    TOKS.forEach((_, i) => sc.ribbon({ from: [379, y(i)], to: lm.rowLeft(i), role: 'q', step: 3, row: i }));
    TOKS.forEach((_, i) => sc.ribbon({ from: lm.rowRight(i), to: [lm.x + lm.w + 60, y(i)], role: 'a', step: 3, row: i }));
    TOKS.forEach((_, i) => sc.strip({ x: lm.x + lm.w + 60, y: y(i) - 9, values: Array.from({ length: 12 }, (_, k) => Math.cos((i + 2) * (k + 1) * 0.5) * 0.7), role: 'o', row: i, step: 3, maxW: 120, absmax: 1 }));

    // stage 4-5 (faded, later chapters)
    const fx = lm.x + lm.w + 220;
    sc.frame({ x: fx, y: y(0) - 26, w: 150, h: TOKS.length * rowH + 30, title: 'MLP · LayerNorm · ×12', step: 4 });
    sc.txt(fx + 12, y(2), 'chapter 4', { class: 'sm' });
    sc.txt(fx + 12, y(3), '(not in this app)', { class: 'sm' });
    sc.frame({ x: fx + 180, y: y(0) - 26, w: 150, h: TOKS.length * rowH + 30, title: 'logits → next token', step: 5 });
    sc.txt(fx + 192, y(2), 'chapter 5', { class: 'sm' });
    ['forward', 'step', 'path', 'day'].forEach((t, i) => sc.txt(fx + 192, y(3) + i * 14, `${t}  ${['31%', '22%', '9%', '4%'][i]}`, { class: 'sm', fill: i === 0 ? ROLE.o : null }));
    sc.ribbon({ from: [fx - 10, y(2)], to: [fx, y(2)], role: 'x', step: 4, opacity: .3 });
    sc.ribbon({ from: [fx + 150, y(2)], to: [fx + 180, y(2)], role: 'x', step: 5, opacity: .3 });

    return {
      scene: sc,
      code: [
        '# stage 1 - building (this app)',
        'token_ids = tokenizer.encode(text)                     # chapter 2  (§2.5)',
        'x = tok_emb(token_ids) + pos_emb(positions)            # chapter 2  (§2.7-2.8)',
        'z = MultiHeadAttention(x)                              # chapter 3  (§3.3-3.6)',
        '# chapter 4: LayerNorm, GELU MLP, residuals, 12 stacked blocks',
        '# chapter 5: pretraining on next-token prediction, or loading GPT-2 weights',
        '# chapters 6-7: fine-tuning for classification / instructions',
      ],
      steps: [
        { label: 'text in', line: 0, note: 'A GPT is a decoder-only transformer trained on one task: guess the next token (§1.4).' },
        { label: 'tokenize', line: 1, note: 'Text becomes integers with byte pair encoding. GPT-2 has 50,257 of them.' },
        { label: 'embed', line: 2, note: 'Each id selects a 768-number row; a position row is added so order survives.' },
        { label: 'attend', line: 3, note: 'Every token mixes information from earlier tokens only. The rest of this app is this box.' },
        { label: 'refine (ch. 4)', line: 4, note: 'MLP, LayerNorm, residuals, stacked 12 times. Not built here.' },
        { label: 'predict (ch. 5)', line: 5, note: 'The last vector becomes 50,257 scores and one is sampled.' },
      ],
      guide: `<h3>Reading the scenes</h3>
        <p>Every section is one scene. Tokens are rows; time flows left to right. Colours never change meaning:</p>
        ${legend([['x', 'input x'], ['q', 'query'], ['k', 'key'], ['v', 'value'], ['a', 'attention'], ['o', 'output']])}
        <p>Dots in a loom grow with the attention weight. A hollow ring is a masked (future) position.</p>
        <h3>Scrubber</h3><p>The bar below steps through the book's code one line at a time: <kbd>←</kbd> <kbd>→</kbd> or <kbd>space</kbd>.</p>
        <h3>Weights</h3><p><b>random · book</b> reproduces the printed numbers with <code>torch.manual_seed</code>. <b>GPT-2 small</b> loads the trained tables and block-0 attention into the same classes.</p>`,
    };
  },
};
