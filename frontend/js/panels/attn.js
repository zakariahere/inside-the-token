// Shared building blocks for the chapter 3 scenes: token rows, x column,
// q/k/v columns, loom blocks, context column. All aligned on the same rows.
import { info, fmt, vis, ROLE } from '../scene.js';

export function xColumn(sc, { toks, R, data, x = 110, step = 0, absmax = null, maxW = 180, label = true, role = 'x', title = 'x' }) {
  sc.txt(x, R.top - 14, title, { class: 'hd', fill: ROLE[role] });
  return data.map((row, i) => sc.strip({ x, y: R.y(i) - R.cellH / 2, values: row, role, row: i, step, absmax, maxW, cellH: R.cellH,
    label: label ? toks[i] : null,
    info: info(`${title}[${i}] <i style="background:${ROLE[role]};color:#000">${vis(toks[i])}</i>`, [['shape', `[${row.length}]`], ['values', row.slice(0, 6).map(v => fmt(v, 2)).join(' ') + (row.length > 6 ? ' …' : '')]]) }));
}

export function vecColumn(sc, { toks, R, data, x, step, role, title, from = null, maxW = 90, absmax = null, segments = null, ribbonRole = null, ribbonOpacity = .45, infoExtra = () => [] }) {
  sc.txt(x, R.top - 14, title, { class: 'hd', fill: ROLE[role] || role });
  const m = absmax ?? (Math.max(...data.flat().map(Math.abs)) || 1);
  return data.map((row, i) => {
    const g = sc.strip({ x, y: R.y(i) - R.cellH / 2, values: row, role, row: i, step, absmax: m, maxW, cellH: R.cellH, segments,
      info: info(`${title}[${i}] <i style="background:${ROLE[role] || role};color:#000">${vis(toks[i])}</i>`, [['shape', `[${row.length}]`], ['values', row.slice(0, 6).map(v => fmt(v, 3)).join(' ') + (row.length > 6 ? ' …' : '')], ...infoExtra(i)]) });
    if (from) sc.ribbon({ from: from[i].right, to: g.left, role: ribbonRole || role, step, row: i, opacity: ribbonOpacity });
    return g;
  });
}

/** loom aligned to rows; ribbons from `from[i].right` into row i (role rowRole);
 *  optional key ribbons from `keys[j].right` up to column j. */
export function loomBlock(sc, { toks, R, data, x, step, title, from = null, keys = null, prev = null, mask = null, dropped = null, signed = false, hi = null, infoFn = null, rowRole = 'q', colRole = 'k', colStep = null }) {
  const lm = sc.loom({ x, y: R.top, data, cell: R.rowH, step, colLabels: R.rowH >= 9 ? toks : null, colStep: colStep ?? step, mask, dropped, signed, hi, infoFn, rowRole, colRole });
  sc.txt(x, R.top - 14, title, { class: 'hd', fill: ROLE.a });
  if (from) toks.forEach((_, i) => sc.ribbon({ from: from[i].right, to: lm.rowLeft(i), role: rowRole, step, row: i, opacity: .5 }));
  if (prev) toks.forEach((_, i) => sc.ribbon({ from: prev.rowRight(i), to: lm.rowLeft(i), role: 'a', step, row: i, opacity: .35, width: 1 }));
  if (keys) toks.forEach((_, j) => sc.ribbon({ from: keys[j].right, to: [lm.colTop(j)[0], lm.y - 2], role: colRole, step, row: j, opacity: .22, width: 1 }));
  return lm;
}

export function contextColumn(sc, { toks, R, data, x, step, from, title = 'context', role = 'o', absmax = null, segments = null }) {
  return vecColumn(sc, { toks, R, data, x, step, role, title, maxW: 120, absmax, segments,
    infoExtra: () => [['= Σ α · value', '']] }).map((g, i) => { sc.ribbon({ from: from.rowRight(i), to: g.left, role: 'a', step, row: i, opacity: .5 }); return g; });
}

export const scoreInfo = (toks, q, k, name = 'ω') => (i, j, v) =>
  info(`${name} · <i style="background:${ROLE.q};color:#000">${vis(toks[i])}</i> → <i style="background:${ROLE.k};color:#000">${vis(toks[j])}</i>`,
    v === '-inf' ? [['masked', 'future token']] : [['score', v], ...(q && q[i].length <= 8 ? [['q · k', q[i].map((a, d) => `${fmt(a, 2)}·${fmt(k[j][d], 2)}`).join(' + ')]] : [])]);

export const weightInfo = (toks, rows, extra = () => []) => (i, j, v) =>
  info(`α · <i style="background:${ROLE.q};color:#000">${vis(toks[i])}</i> → <i style="background:${ROLE.k};color:#000">${vis(toks[j])}</i>`,
    v === '-inf' || v === 0 && j > i ? [['weight', 0], ['reason', 'future token, masked']] : [['weight', v], ['row sum', fmt(rows[i].reduce((a, b) => a + (typeof b === 'number' ? b : 0), 0), 3)], ...extra(i, j, v)]);
