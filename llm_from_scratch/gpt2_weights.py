"""Load real GPT-2 small (124M) weights into the chapter 2/3 modules.

This is chapter 5 material (section 5.5, ``load_weights_into_gpt``) brought
forward so the chapter 3 attention maps can be viewed with trained weights.
The mapping mirrors the book's ``assign`` helper and its handling of the
HuggingFace/OpenAI checkpoint layout:

* ``wte``/``wpe`` are plain embedding tables.
* ``h.<i>.attn.c_attn`` is a Conv1D with weight ``[768, 2304]`` = ``[in, out]``,
  holding Q, K, V side by side. ``nn.Linear`` wants ``[out, in]``, so we split
  along the last axis and transpose.
* ``h.<i>.attn.c_proj`` maps to ``out_proj`` the same way.
* GPT-2 applies LayerNorm (``ln_1``) *before* attention (chapter 4). It is
  exposed here so the UI can show it as an explicit, optional extra step.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .ch03_attention import MultiHeadAttention

GPT2_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True,
}

_NEEDED_PREFIXES = ("wte.", "wpe.", "h.0.attn.", "h.0.ln_1.")


def gpt2_cached() -> bool:
    try:
        hf_hub_download("gpt2", "model.safetensors", local_files_only=True)
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def load_gpt2_tensors(block: int = 0) -> dict:
    """Return only the tensors we need (embeddings + one attention block)."""
    path = hf_hub_download("gpt2", "model.safetensors")
    raw = load_file(path)
    prefixes = ("wte.", "wpe.", f"h.{block}.attn.", f"h.{block}.ln_1.")
    out = {}
    for k, v in raw.items():
        key = k[len("transformer."):] if k.startswith("transformer.") else k
        if key.startswith(prefixes):
            out[key] = v.float()
    return out


def assign(left, right):
    """Book listing 5.5 helper: shape-checked parameter assignment."""
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right.clone().detach())


def load_embeddings(token_embedding_layer, pos_embedding_layer, block: int = 0):
    p = load_gpt2_tensors(block)
    ctx = pos_embedding_layer.weight.shape[0]
    token_embedding_layer.weight = assign(token_embedding_layer.weight, p["wte.weight"])
    pos_embedding_layer.weight = assign(pos_embedding_layer.weight, p["wpe.weight"][:ctx])
    return token_embedding_layer, pos_embedding_layer


def load_into_mha(mha: MultiHeadAttention, block: int = 0):
    p = load_gpt2_tensors(block)
    pre = f"h.{block}.attn."
    q_w, k_w, v_w = torch.chunk(p[pre + "c_attn.weight"], 3, dim=-1)   # each [768, 768] (in, out)
    mha.W_query.weight = assign(mha.W_query.weight, q_w.T)
    mha.W_key.weight = assign(mha.W_key.weight, k_w.T)
    mha.W_value.weight = assign(mha.W_value.weight, v_w.T)

    q_b, k_b, v_b = torch.chunk(p[pre + "c_attn.bias"], 3, dim=-1)
    mha.W_query.bias = assign(mha.W_query.bias, q_b)
    mha.W_key.bias = assign(mha.W_key.bias, k_b)
    mha.W_value.bias = assign(mha.W_value.bias, v_b)

    mha.out_proj.weight = assign(mha.out_proj.weight, p[pre + "c_proj.weight"].T)
    mha.out_proj.bias = assign(mha.out_proj.bias, p[pre + "c_proj.bias"])
    return mha


def ln_1(x, block: int = 0):
    """GPT-2's pre-attention LayerNorm (chapter 4, section 4.2). eps=1e-5 as in HF."""
    p = load_gpt2_tensors(block)
    return F.layer_norm(x, (x.shape[-1],), p[f"h.{block}.ln_1.weight"],
                        p[f"h.{block}.ln_1.bias"], eps=1e-5)


@lru_cache(maxsize=1)
def build_gpt2_mha(block: int = 0) -> MultiHeadAttention:
    cfg = GPT2_CONFIG_124M
    mha = MultiHeadAttention(
        d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
        context_length=cfg["context_length"], dropout=cfg["drop_rate"],
        num_heads=cfg["n_heads"], qkv_bias=cfg["qkv_bias"],
    )
    load_into_mha(mha, block)
    mha.eval()
    return mha


@lru_cache(maxsize=4)
def build_gpt2_embeddings(context_length: int = 1024):
    cfg = GPT2_CONFIG_124M
    tok = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
    pos = torch.nn.Embedding(context_length, cfg["emb_dim"])
    load_embeddings(tok, pos)
    return tok, pos
