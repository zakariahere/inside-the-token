# Inside the Token

A local, interactive companion to chapters 1–3 of Sebastian Raschka’s *Build a Large Language Model (From Scratch)*. Follow one token through a calculation, then expand to all rows and the actual PyTorch code.

The alternative lives on `codex/inside-the-token`; `master` retains the original app at `e28dc90`.

## Run locally

```powershell
uv sync --group dev
uv run python scripts/fetch_data.py
uv run uvicorn app.main:app --port 8001
```

Open <http://127.0.0.1:8001>. Add `--reload` while developing. Python 3.13 and CPU-only PyTorch are configured in `pyproject.toml`. The frontend is plain JavaScript and CSS: no Node install or frontend build is required.

The sample text is optional; choose your own text if it has not been downloaded. GPT-2 mode requires a cached checkpoint. To download it deliberately:

```powershell
uv run python -c "from huggingface_hub import hf_hub_download; hf_hub_download('gpt2', 'model.safetensors')"
```

## Learning path

1. **The bigger picture** — prediction, training versus generation, and where attention fits.
2. **Text → tokens** — actual GPT-2 IDs, raw bytes, Unicode, and a separate illustrative BPE merge.
3. **Inputs & targets** — move a training window, shift targets, change stride, inspect a batch.
4. **Embeddings & position** — lookup rows and inspect individual component sums.
5. **Why attention?** — the book’s simplified example, one query at a time.
6. **Queries, keys & values** — the hand-picked `The / bank / river` example from the learning notes.
7. **Scores → softmax** — scaling comparisons, stable exponentiation, and weighted contributions.
8. **Causal attention** — mask future positions before softmax and rebuild the context vector.
9. **Dropout & batches** — seeded independent masks, survivor scaling, evaluation mode, tensor axes.
10. **Multiple heads** — the book’s small example, slices, concatenation, projection, and GPT-2 exploration.

Use **Next step / Back** at your own pace; no autoplay or mandatory quizzes. Select tokens, matrix cells, or output components to inspect a calculation. Math and code disclosures provide more depth. **Explore** exposes custom inputs and model controls. Lesson positions are stored locally in this browser; the current text is session-only. Old `#ch03-self` style links map to the new lessons.

## Calculations and boundaries

- The book classes and existing API shapes remain intact. Original endpoint tensor values are rounded to four decimals and displayed calculations are marked approximate.
- `POST /api/lessons/bank-attention` accepts `scaling`, `causal`, `dropout` (0–0.9), `training`, and `seed`. It returns the fixed inputs and projection matrices, Q/K/V, raw/scaled/masked scores, stable-softmax intermediates, pre/post-dropout weights, masks, weighted contributions, row sums, and outputs at full precision.
- `GET /api/lessons/code` returns the actual source of `SelfAttention_v2`, `CausalAttention`, and `MultiHeadAttention` for the full-class disclosures.
- The bank fixture uses the existing `SelfAttention_v2` and trace helper. Unscaled comparisons and explicit dropout masks are teaching logic outside the book classes. Each dropout request uses a local seeded generator.
- Hand-picked and random examples illustrate arithmetic, not learned semantics. GPT-2 exploration loads trained embeddings and block-0 attention; it does not run a complete text generator. The overview’s probabilities and BPE merge illustration are labeled illustrations.
- Masked values serialize as `"-inf"`. Attention inputs are capped at 64 tokens; expanded large matrices scroll locally. GPT-2 and other large vectors show a labeled subset of components while the backend computes all dimensions.
- All user text is inserted with text nodes. Byte fragments are shown honestly rather than mapped to incorrect character offsets.

## Layout

```text
frontend/js/textbook/content.js      Lesson prose, steps, math, code excerpts, checks
frontend/js/textbook/components.js   Accessible text-safe UI and numeric components
frontend/js/textbook/scenes.js       API data flow and lesson-specific interactions
frontend/js/textbook/app.js          Navigation, progress, state, disclosures, errors
frontend/css/textbook.css            Responsive light textbook theme
app/lessons.py                      Hand-picked fixture and source-code endpoints
llm_from_scratch/                   Original book code, traces, GPT-2 mappings
```

The original scene modules remain in the tree for reference; the new entrypoint does not import them.

## Verification

```powershell
uv run pytest -q
```

The suite covers book numbers, API shapes, trace/forward agreement, GPT-2/HuggingFace agreement when cached, and the teaching fixture’s projections, softmax, masking, dropout, precision, and error handling. The GPT-2 reference tests may need HuggingFace tokenizer/config assets on first use.

Browser acceptance: walk every guided step; exercise Explore; inspect a projection component; toggle scaling; resample dropout; switch evaluation mode; select GPT-2 head 11; test Unicode and literal HTML text; try invalid head dimensions and short text; reload to check progress; inspect desktop, tablet, and mobile layouts. See `tests/browser-qa.md` for the recorded pass.
