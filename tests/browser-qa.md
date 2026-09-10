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

## Multi-head lesson adaptation · 2026-09-10

- Rechecked the revised six-step lesson against the latest single-head learning notes.
- The recap preserves `The / river / bank / was` and identifies `muddy` as the last shifted target before switching fixtures.
- Verified the `d_out = 3` to divisible `d_out = 2` handoff, the project/view/transpose shapes, both head selections, allowed and blocked key calculations, per-head Value sums, concatenation, and output-projection arithmetic.
- Explore mode still renders custom random inputs and dropout results without stray placeholder text.
- At 390 × 844, the recap and shape ladder stack cleanly; measured document width stays within the viewport with no page-wide horizontal overflow.

## Zakaria visual identity · 2026-09-10

- Matched the portfolio’s graphite surfaces, electric-blue emphasis, ice-blue secondary accents, technical mono labels, and geometric display type.
- Loaded the approved 368 × 560 transparent WebP mascot at its native aspect ratio with descriptive alternative text.
- Inspected the overview and multi-head lessons at 1280 × 720: the mascot remains clear beside the lesson introduction, cards retain strong contrast, and document width remains inside the viewport.
- Rechecked the narrow-screen rules: navigation collapses, the mascot reduces to 106px and moves beside the title, lesson copy returns to full width, and large matrices keep their existing local scrollers.
