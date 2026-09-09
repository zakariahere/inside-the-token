# LLM From Scratch Explainer

An interactive, local web app in the spirit of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/),
scoped to chapters 1–3 of Sebastian Raschka's *Build a Large Language Model (From Scratch)*, ending at
causal multi-head attention.

The backend runs the book's **final** code verbatim (`tiktoken` BPE, `GPTDatasetV1`, `create_dataloader_v1`,
`SelfAttention_v2`, `CausalAttention`, `MultiHeadAttention`) on the CPU and returns every intermediate tensor.

The frontend draws each section as one SVG **scene**: tokens are rows, time flows left to right, vectors are
cell strips, weight matrices are grids, and attention is a *loom* of dots whose radius is the weight (hollow ring =
masked, crossed = dropped). Colours never change meaning: input slate, query cyan, key rose, value lime, attention
violet, output amber.

- **Code scrubber** (bottom): the book's own `forward()` lines are the steps. Drag, press `←`/`→`, or `space` to
  reveal the scene line by line; ribbons draw in as their line executes.
- **Hover** any token, strip, or dot: its whole path lights up and the inspector (right) shows the exact values and
  the arithmetic behind that number.
- **Weights** toggle: `random · book` reproduces the printed numbers; `GPT-2 small` loads the trained tables and
  block-0 attention into the same classes (12 real heads, labelled with your tokens).
- **cells** toggle prints numbers inside cells when they are large enough; **zoom** fits the scene or scrolls it 1:1.

## Run

```powershell
uv sync --group dev; uv run python scripts/fetch_data.py; uv run uvicorn app.main:app --port 8000 --reload
```

Then open <http://localhost:8000>.

- Python 3.13, CPU-only PyTorch (pinned to the `download.pytorch.org/whl/cpu` index in `pyproject.toml`).
- `scripts/fetch_data.py` downloads `the-verdict.txt` (book §2.2) into `data/`.
- The **GPT-2 small** weights toggle needs `model.safetensors` from the `gpt2` HuggingFace repo. It is loaded via
  `huggingface_hub` and cached on first use.

## Tests

```powershell
uv run pytest -q
```

The suite asserts the book's printed values (§2.5 token IDs, §2.6 first batch, §3.3 weights/context vector,
§3.4 seed-789 output, §3.6.2 seed-123 output), that every traced intermediate equals the verbatim forward pass,
and that the GPT-2 weight mapping reproduces HuggingFace's block-0 attention to 1e-4.

## Layout

```
llm_from_scratch/   book code, one module per chapter (ch02_data, ch03_attention), plus
                    trace.py (step-by-step re-execution) and gpt2_weights.py (ch5 weight mapping)
app/                FastAPI: /api/ch02/{tokenize,windows,embed}, /api/ch03/{simple,self,causal,mha}
frontend/           no-build vanilla JS: js/scene.js (SVG primitives: strip, grid, loom, ribbon, ruler),
                    js/app.js (path nav, scrubber, inspector, zoom), one scene per book section in js/panels/
tests/
```

## Panels

| Panel | Book | What you see |
|---|---|---|
| Understanding LLMs | §1 | pipeline overview, GPT-2 config |
| Byte pair encoding | §2.5 | tokens, IDs, bytes, decode round-trip |
| Sliding-window sampling | §2.6 | input/target windows, `max_length`/`stride`, first DataLoader batch |
| Token + positional embeddings | §2.7–2.8 | `token_embeddings + pos_embeddings` heatmaps, shapes |
| Simplified self-attention | §3.3 | dot products, softmax, context vectors, single-query walkthrough |
| Trainable Q, K, V | §3.4 | `W_query/W_key/W_value`, scaled scores, √d_k comparison |
| Causal mask + dropout | §3.5 | renormalise vs. −∞ masking, dropout keep mask and 1/(1−p) scaling |
| Multi-head attention | §3.6 | per-head Q/K/V, attention maps, concat, `out_proj`; 12 real GPT-2 heads in GPT-2 mode |
