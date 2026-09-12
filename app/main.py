"""FastAPI app: runs the book's chapter 2-4 code and returns every intermediate."""
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from llm_from_scratch import gpt2_weights
from llm_from_scratch.ch02_data import (build_embeddings, create_dataloader_v1,
                                        embed_batch, get_tokenizer)
from llm_from_scratch.ch03_attention import (CausalAttention, MultiHeadAttention,
                                             SelfAttention_v2)
from llm_from_scratch.ch04_gpt import (GPT_CONFIG_124M, TINY_GPT_CONFIG,
                                      GPTModel, parameter_count)
from llm_from_scratch.ch04_trace import (trace_feed_forward, trace_gpt,
                                         trace_layer_norm, trace_shortcuts,
                                         trace_transformer_block)
from llm_from_scratch.trace import (trace_causal, trace_multihead,
                                    trace_self_attention, trace_simple)

from . import registry
from .lessons import router as lesson_router
from .schemas import (MAX_TOKENS, MHA_FIELDS, CausalRequest, Ch04BlockRequest,
                      Ch04GenerateRequest, Ch04ModelRequest, Ch04SeedRequest,
                      EmbedRequest, MHARequest, SelfAttnRequest, SimpleRequest,
                      TokenizeRequest, WindowsRequest)
from .serialize import tensor_to_json as tj

assert not torch.cuda.is_available(), "this app is CPU-only by design"
torch.set_grad_enabled(False)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app = FastAPI(title="LLM From Scratch Explainer", version="0.1.0")
app.include_router(lesson_router)


@app.exception_handler(FileNotFoundError)
async def missing_data(_request, _exc):
    return JSONResponse(status_code=503, content={"detail": "Sample data is missing. Run: uv run python scripts/fetch_data.py, or choose your own text."})


# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {
        "torch": torch.__version__,
        "device": "cpu",
        "cuda": torch.cuda.is_available(),
        "gpt2_cached": gpt2_weights.gpt2_cached(),
        "verdict_present": registry.VERDICT.exists(),
        "max_tokens": MAX_TOKENS,
    }


# ----------------------------- chapter 2 ----------------------------------- #
@app.post("/api/ch02/tokenize")
def tokenize(req: TokenizeRequest):
    ids = registry.encode(req.text)
    tok = get_tokenizer()
    tokens = [{"id": i, "text": tok.decode([i]), "bytes": list(tok.decode_single_token_bytes(i))}
              for i in ids]
    return {"ids": ids, "tokens": tokens, "count": len(ids),
            "vocab_size": tok.n_vocab, "decoded": tok.decode(ids)}


@app.post("/api/ch02/windows")
def windows(req: WindowsRequest):
    text = registry.resolve_text(req.source, req.text)
    ids = registry.encode(text)
    tok = get_tokenizer()
    pairs = []
    # exactly the loop inside GPTDatasetV1.__init__
    for i in range(0, len(ids) - req.max_length, req.stride):
        inp = ids[i:i + req.max_length]
        tgt = ids[i + 1: i + req.max_length + 1]
        pairs.append({"start": i, "input_ids": inp, "target_ids": tgt,
                      "input_tokens": registry.token_strings(inp),
                      "target_tokens": registry.token_strings(tgt)})
    total_pairs = len(pairs)
    pairs = pairs[:req.max_pairs]

    first_batch = None
    dl = create_dataloader_v1(text, batch_size=req.batch_size, max_length=req.max_length,
                              stride=req.stride, shuffle=False, drop_last=False)
    if len(dl) > 0:
        x, y = next(iter(dl))
        first_batch = {"inputs": x.tolist(), "targets": y.tolist(), "shape": list(x.shape),
                       "input_tokens": [registry.token_strings(r) for r in x.tolist()],
                       "target_tokens": [registry.token_strings(r) for r in y.tolist()],
                       "num_batches": len(dl)}
    return {"num_tokens": len(ids), "tokens": registry.token_strings(ids[:512]),
            "ids": ids[:512], "pairs": pairs, "total_pairs": total_pairs,
            "first_batch": first_batch}


@app.post("/api/ch02/embed")
def embed(req: EmbedRequest):
    if req.weights == "gpt2" and not gpt2_weights.gpt2_cached():
        raise HTTPException(503, "GPT-2 weights are not cached. Choose random weights, or download gpt2/model.safetensors with huggingface_hub first.")
    text = registry.resolve_text(req.source, req.text)
    dl = create_dataloader_v1(text, batch_size=req.batch_size, max_length=req.max_length,
                              stride=req.stride, shuffle=False, drop_last=False)
    if len(dl) == 0:
        raise HTTPException(400, "text is too short for this max_length; add more text")
    inputs, targets = next(iter(dl))

    if req.weights == "gpt2":
        emb_dim = 768
        tok, pos = gpt2_weights.build_gpt2_embeddings(req.max_length)
    else:
        emb_dim = req.emb_dim
        tok, pos = build_embeddings(output_dim=emb_dim, context_length=req.max_length,
                                    seed=req.seed)
    token_embeddings, pos_embeddings, input_embeddings = embed_batch(inputs, tok, pos)
    cols = req.show_dims
    return {
        "emb_dim": emb_dim, "vocab_size": tok.weight.shape[0],
        "inputs": inputs.tolist(),
        "input_tokens": [registry.token_strings(r) for r in inputs.tolist()],
        "weight_shapes": {"token_embedding_layer.weight": list(tok.weight.shape),
                          "pos_embedding_layer.weight": list(pos.weight.shape)},
        "token_embeddings": tj(token_embeddings, max_cols=cols),
        "pos_embeddings": tj(pos_embeddings, max_cols=cols),
        "input_embeddings": tj(input_embeddings, max_cols=cols),
    }


# ----------------------------- chapter 3 ----------------------------------- #
def _inputs(req):
    if req.weights == "gpt2" and req.source != "book" and not gpt2_weights.gpt2_cached():
        raise HTTPException(503, "GPT-2 weights are not cached. Choose random weights, or download gpt2/model.safetensors with huggingface_hub first.")
    try:
        return registry.make_inputs(req.source, req.text, req.emb_dim, req.seed, req.weights)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/ch03/simple")
def simple(req: SimpleRequest):
    x, tokens, meta = _inputs(req)
    if req.query_index >= x.shape[0]:
        raise HTTPException(400, "query_index out of range")
    t = trace_simple(x, req.query_index)
    c = req.show_dims
    return {
        "tokens": tokens, "meta": meta, "query_index": req.query_index,
        "inputs": tj(t["inputs"], max_cols=c),
        "single": {k: tj(v, max_cols=c) for k, v in t["single"].items()},
        "attn_scores": tj(t["attn_scores"]),
        "attn_weights": tj(t["attn_weights"]),
        "context_vecs": tj(t["context_vecs"], max_cols=c),
    }


@app.post("/api/ch03/self")
def self_attention(req: SelfAttnRequest):
    x, tokens, meta = _inputs(req)
    torch.manual_seed(req.seed)
    sa = SelfAttention_v2(d_in=x.shape[1], d_out=req.d_out, qkv_bias=req.qkv_bias)
    t = trace_self_attention(sa, x)
    c = req.show_dims
    return {
        "tokens": tokens, "meta": meta, "d_in": x.shape[1], "d_out": req.d_out, "d_k": t["d_k"],
        "inputs": tj(x, max_cols=c),
        "W_query": tj(t["W_query"], max_rows=c, max_cols=c),
        "W_key": tj(t["W_key"], max_rows=c, max_cols=c),
        "W_value": tj(t["W_value"], max_rows=c, max_cols=c),
        "queries": tj(t["queries"], max_cols=c), "keys": tj(t["keys"], max_cols=c),
        "values": tj(t["values"], max_cols=c),
        "attn_scores": tj(t["attn_scores"]), "scaled_scores": tj(t["scaled_scores"]),
        "attn_weights": tj(t["attn_weights"]),
        "attn_weights_unscaled": tj(t["attn_weights_unscaled"]),
        "context_vec": tj(t["context_vec"], max_cols=c),
    }


@app.post("/api/ch03/causal")
def causal(req: CausalRequest):
    x, tokens, meta = _inputs(req)
    batch = torch.stack((x, x), dim=0)          # the book's batch of two identical inputs
    torch.manual_seed(req.seed)
    ca = CausalAttention(d_in=x.shape[1], d_out=req.d_out, context_length=batch.shape[1],
                         dropout=req.dropout, qkv_bias=req.qkv_bias)
    ca.train(req.train)
    t = trace_causal(ca, batch, seed=req.seed)
    c = req.show_dims
    b0 = lambda k, **kw: tj(t[k][0], **kw)   # show batch item 0
    return {
        "tokens": tokens, "meta": meta, "batch_shape": list(batch.shape),
        "d_in": x.shape[1], "d_out": req.d_out, "d_k": t["d_k"],
        "dropout": t["dropout_p"], "training": t["training"],
        "queries": b0("queries", max_cols=c), "keys": b0("keys", max_cols=c),
        "values": b0("values", max_cols=c),
        "attn_scores": b0("attn_scores"),
        "attn_weights_full": b0("attn_weights_full"),
        "mask_simple": tj(t["mask_simple"]), "masked_simple": b0("masked_simple"),
        "row_sums": b0("row_sums"), "masked_simple_norm": b0("masked_simple_norm"),
        "mask": tj(t["mask"]), "masked_scores": b0("masked_scores"),
        "attn_weights": b0("attn_weights"),
        # kept=1, dropped=0, causally masked=-inf (drawn hatched)
        "keep_mask": tj(t["keep_mask"][0].float().masked_fill(t["mask"].bool(), -torch.inf)),
        "attn_weights_dropped": b0("attn_weights_dropped"),
        "context_vec": b0("context_vec", max_cols=c),
        "context_shape": list(t["context_vec"].shape),
    }


@app.post("/api/ch03/mha")
def mha(req: MHARequest):
    if req.weights == "gpt2":
        if not gpt2_weights.gpt2_cached():
            raise HTTPException(503, "GPT-2 weights are not cached. Choose random weights, or download gpt2/model.safetensors with huggingface_hub first.")
        x, tokens, meta = _inputs(req)
        module = gpt2_weights.build_gpt2_mha()
        ln_applied = bool(req.apply_ln1)
        if ln_applied:
            x = gpt2_weights.ln_1(x)
        batch = x.unsqueeze(0)
        module.train(False)
    else:
        x, tokens, meta = _inputs(req)
        batch = torch.stack((x, x), dim=0) if req.source == "book" else x.unsqueeze(0)
        ctx = req.context_length or batch.shape[1]
        if ctx < batch.shape[1]:
            raise HTTPException(400, "context_length smaller than number of tokens")
        if req.d_out % req.num_heads:
            raise HTTPException(400, "d_out must be divisible by num_heads")
        torch.manual_seed(req.seed)
        module = MultiHeadAttention(d_in=x.shape[1], d_out=req.d_out, context_length=ctx,
                                    dropout=req.dropout, num_heads=req.num_heads,
                                    qkv_bias=req.qkv_bias)
        module.train(req.train)
        ln_applied = False

    if req.batch_index >= batch.shape[0]:
        raise HTTPException(400, "batch_index out of range")
    t = trace_multihead(module, batch, seed=req.seed)
    cfg = dict(t["config"], weights=req.weights, ln1_applied=ln_applied,
               context_length=int(module.mask.shape[0]), seed=req.seed)
    include = set(req.include) if req.include else set(MHA_FIELDS)
    c = req.show_dims
    bi = req.batch_index

    out = {"tokens": tokens, "meta": meta, "config": cfg,
           "shapes": {k: list(v.shape) for k, v in t.items() if torch.is_tensor(v)}}
    if "x" in include:
        out["x"] = tj(batch[bi], max_cols=c)
    for k in ("queries_flat", "keys_flat", "values_flat", "context_concat", "out"):
        if k in include:
            out[k] = tj(t[k][bi], max_cols=c)
    if "out_proj_weight" in include:
        out["out_proj_weight"] = tj(t["out_proj_weight"], max_rows=c, max_cols=c)
        out["out_proj_bias"] = tj(t["out_proj_bias"], max_cols=c)
    if "mask" in include:
        out["mask"] = tj(t["mask"])
    heads = []
    for h in range(module.num_heads):
        hd = {"head": h}
        for k in ("queries", "keys", "values", "context_heads"):
            if k in include:
                hd[k] = tj(t[k][bi, h], max_cols=c)
        for k in ("attn_scores", "masked_scores", "attn_weights", "attn_weights_dropped"):
            if k in include:
                hd[k] = tj(t[k][bi, h])
        if "keep_mask" in include:
            hd["keep_mask"] = tj(t["keep_mask"][bi, h].float().masked_fill(t["mask"].bool(), -torch.inf))
        heads.append(hd)
    out["heads"] = heads
    return out


# ----------------------------- chapter 4 ----------------------------------- #
def _ch04_config(dropout=0.0):
    cfg = dict(TINY_GPT_CONFIG)
    cfg["drop_rate"] = dropout
    return cfg


def _ch04_tokens(text, context_length=8):
    ids = registry.encode(text)
    if not ids:
        raise HTTPException(400, "text must contain at least one token")
    ids = ids[:context_length]
    return torch.tensor([ids]), registry.token_strings(ids)


def _chapter_map(tiny_model=None):
    tiny_count = (sum(p.numel() for p in tiny_model.parameters())
                  if tiny_model is not None else parameter_count(TINY_GPT_CONFIG))
    return {
        "tiny_config": TINY_GPT_CONFIG,
        "gpt2_124m_config": GPT_CONFIG_124M,
        "parameter_counts": {
            "tiny_untied": tiny_count,
            "gpt2_book_untied": parameter_count(GPT_CONFIG_124M),
            "gpt2_with_weight_tying": parameter_count(GPT_CONFIG_124M, tied_output=True),
        },
        "shape_map": [
            {"name": "embedding width", "tiny": 4, "gpt2_124m": 768},
            {"name": "attention heads", "tiny": 2, "gpt2_124m": 12},
            {"name": "head width", "tiny": 2, "gpt2_124m": 64},
            {"name": "feed-forward width", "tiny": 16, "gpt2_124m": 3072},
            {"name": "transformer blocks", "tiny": 2, "gpt2_124m": 12},
            {"name": "context length", "tiny": 8, "gpt2_124m": 1024},
            {"name": "vocabulary logits", "tiny": 50257, "gpt2_124m": 50257},
        ],
    }


@app.post("/api/ch04/layernorm")
@torch.inference_mode()
def ch04_layernorm(req: Ch04SeedRequest):
    t = trace_layer_norm(req.seed)
    return {k: (tj(v) if torch.is_tensor(v) else v) for k, v in t.items()}


@app.post("/api/ch04/feedforward")
@torch.inference_mode()
def ch04_feedforward(req: Ch04SeedRequest):
    t = trace_feed_forward(TINY_GPT_CONFIG, req.seed)
    return {
        "config": TINY_GPT_CONFIG,
        **{k: tj(v, max_rows=16, max_cols=16) for k, v in t.items()},
    }


@app.post("/api/ch04/shortcuts")
def ch04_shortcuts(req: Ch04SeedRequest):
    t = trace_shortcuts(req.seed)
    out = {}
    for key, value in t.items():
        if key.endswith("_stages"):
            out[key] = [
                {k: tj(v) if torch.is_tensor(v) else v for k, v in stage.items()}
                for stage in value
            ]
        else:
            out[key] = tj(value) if torch.is_tensor(value) else value
    return out


@app.post("/api/ch04/block")
@torch.inference_mode()
def ch04_block(req: Ch04BlockRequest):
    cfg = _ch04_config(req.dropout)
    torch.manual_seed(req.seed)
    x = torch.randn(1, 4, cfg["emb_dim"])
    _module, t = trace_transformer_block(cfg, x, req.seed, req.train)
    return {
        "tokens": ["Every", " effort", " moves", " you"],
        "config": cfg,
        "training": req.train,
        "stages": {k: tj(v, max_cols=16) for k, v in t.items()},
    }


@app.post("/api/ch04/model")
@torch.inference_mode()
def ch04_model(req: Ch04ModelRequest):
    cfg = _ch04_config()
    token_ids, tokens = _ch04_tokens(req.text, cfg["context_length"])
    torch.manual_seed(req.seed)
    model = GPTModel(cfg)
    model.eval()
    t = trace_gpt(model, token_ids)
    last_logits = t["logits"][0, -1]
    top_logits, top_ids = torch.topk(last_logits, 5)
    grouped = {}
    for name, parameter in model.named_parameters():
        group = name.split(".")[0]
        grouped[group] = grouped.get(group, 0) + parameter.numel()
    return {
        "text": req.text,
        "token_ids": token_ids[0].tolist(),
        "tokens": tokens,
        "architecture": _chapter_map(model),
        "parameter_groups": grouped,
        "token_embeddings": tj(t["token_embeddings"], max_cols=16),
        "position_embeddings": tj(t["position_embeddings"], max_cols=16),
        "combined_embeddings": tj(t["combined_embeddings"], max_cols=16),
        "after_embedding_dropout": tj(t["after_embedding_dropout"], max_cols=16),
        "block_outputs": [tj(v, max_cols=16) for v in t["block_outputs"]],
        "final_norm": tj(t["final_norm"], max_cols=16),
        "logits": tj(t["logits"], max_cols=12),
        "top_last_position": [
            {"id": int(i), "token": registry.token_strings([int(i)])[0],
             "logit": round(float(logit), 4)}
            for logit, i in zip(top_logits, top_ids)
        ],
    }


@app.post("/api/ch04/generate")
@torch.inference_mode()
def ch04_generate(req: Ch04GenerateRequest):
    cfg = _ch04_config()
    input_ids = registry.encode(req.text)
    if not input_ids:
        raise HTTPException(400, "text must contain at least one token")
    torch.manual_seed(req.seed)
    model = GPTModel(cfg)
    model.eval()
    idx = torch.tensor([input_ids])
    steps = []
    for step in range(req.max_new_tokens):
        idx_cond = idx[:, -cfg["context_length"]:]
        logits = model(idx_cond)[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        top_probas, top_ids = torch.topk(probas[0], 3)
        chosen = int(top_ids[0])
        steps.append({
            "step": step + 1,
            "context_ids": idx_cond[0].tolist(),
            "context_tokens": registry.token_strings(idx_cond[0].tolist()),
            "logits_shape": list(logits.shape),
            "top_candidates": [
                {"id": int(i), "token": registry.token_strings([int(i)])[0],
                 "probability": round(float(prob), 8)}
                for prob, i in zip(top_probas, top_ids)
            ],
            "chosen_id": chosen,
            "chosen_token": registry.token_strings([chosen])[0],
        })
        idx = torch.cat((idx, torch.tensor([[chosen]])), dim=1)
    tokenizer = get_tokenizer()
    return {
        "warning": "Random weights: this demonstrates mechanics, not learned language.",
        "config": cfg,
        "prompt": req.text,
        "input_ids": input_ids,
        "steps": steps,
        "output_ids": idx[0].tolist(),
        "generated_text": tokenizer.decode(idx[0].tolist()),
    }


# static frontend last so /api/* wins
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
