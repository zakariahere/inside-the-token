import { clamp, HEADS } from '../scene.js';
import { shape, role } from '../ui.js';

export const SOURCE = (disabled = false) => ({ key: 'source', label: 'inputs', type: 'select', disabled,
  options: [['book', 'book · "Your journey starts with one step"'], ['text', 'toolbar text → chapter 2 embeddings']] });
export const DIN = (S, G) => ({ key: 'emb_dim', label: 'd_in (text mode)', type: 'select', disabled: S.source === 'book' || G.weights === 'gpt2',
  options: [[3, '3'], [8, '8'], [64, '64'], [768, '768']] });

/** row geometry: rows aligned with loom cells so token i lines up everywhere */
export function rows(T, { top = 60, max = 30, min = 9 } = {}) {
  const rowH = T <= 10 ? max : clamp(Math.floor(520 / T), min, max);
  return { rowH, top, y: i => top + i * rowH + rowH / 2, cellH: Math.max(5, rowH - 10), bottom: top + T * rowH, mid: top + T * rowH / 2 };
}

export const legend = (items) => `<div class="shapes">${items.map(([r, t]) => role(r, t)).join(' ')}</div>`;
export const shapes = (list) => `<div class="shapes">${list.map(([l, s]) => shape(s, l)).join('')}</div>`;
export const headColor = (h) => HEADS[h % HEADS.length];
