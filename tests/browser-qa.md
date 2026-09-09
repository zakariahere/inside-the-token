# Browser acceptance record

Verified on 2026-09-09 using the Codex in-app browser against the local FastAPI app.

- All 10 lessons: navigated through all 44 guided steps; no error panels.
- All 9 applicable Explore views: rendered actual API results.
- Q/K/V: selecting query component 1 for bank showed `(0 × 0) + (2 × 1) + (0 × 1) + (3 × 0) = 2`; switching to river changed the result to 4.
- Scaling: toggling division by sqrt(d_k) changed the weights and the selected output calculation.
- Causal/dropout: blocked future entries; resampling changed the mask; evaluation removed dropout.
- Multi-head: invalid 8-output/3-head configuration displayed an actionable error; returning to 2 heads recovered. GPT-2 displayed 12 heads and head 11 could be selected.
- Tokenizer: `A café 🌊 中文 <b>bank</b>` rendered as literal text, with exact decoding and honest byte-fragment previews.
- BPE: advancing a merge kept its disclosure open and changed `l / o / w` to `lo / w`.
- Full PyTorch class disclosure loaded actual CausalAttention source.
- Lesson step survived reload. Existing panel hashes are mapped in the lesson registry.
- Desktop screenshot: 1440 × 1000, Q/K/V projection.
- Tablet screenshot: 820 × 1180, causal attention. Adjusted two-column cards to stack at this width so attention bars remain readable; rechecked the screenshot.
- Mobile screenshot: 390 × 844, Unicode tokenization. Checked every lesson's final step at this width: no page-wide horizontal overflow and no error panels.
- Motion is manual; static review confirms the reduced-motion media rule disables transitions and animations. OS preference emulation was not available through the browser tool.

Automated numeric and API coverage is in `test_lessons.py` plus the existing suite. GPT-2 availability and missing data are exercised with explicit error tests, without deleting local caches. The browser pass is a recorded acceptance check, not a headless test-runner dependency.
