"""Step-by-step re-execution of the chapter 3 forward passes.

Each ``trace_*`` function repeats the exact operations of the corresponding
book code, but keeps every intermediate tensor so the UI can display it.
Every trace ends by asserting that its result equals the verbatim module's
output, so the visualisation can never drift from the real code.
"""
import torch

from .ch03_attention import (BOOK_INPUTS, CausalAttention, MultiHeadAttention,
                             SelfAttention_v2)


def trace_simple(inputs=BOOK_INPUTS, query_index=1):
    """Section 3.3 - simplified self-attention without trainable weights.

    3.3.1 walks through a single query (the book uses x^2 = "journey");
    3.3.2 does all queries at once with ``inputs @ inputs.T``.
    """
    # --- 3.3.1: one query -------------------------------------------------
    query = inputs[query_index]
    attn_scores_q = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_q[i] = torch.dot(x_i, query)
    attn_weights_q_naive = attn_scores_q / attn_scores_q.sum()
    attn_weights_q = torch.softmax(attn_scores_q, dim=0)
    context_vec_q = torch.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vec_q += attn_weights_q[i] * x_i

    # --- 3.3.2: all queries -----------------------------------------------
    attn_scores = inputs @ inputs.T
    attn_weights = torch.softmax(attn_scores, dim=-1)
    all_context_vecs = attn_weights @ inputs

    assert torch.allclose(all_context_vecs[query_index], context_vec_q, atol=1e-6)
    return {
        "inputs": inputs,
        "query_index": query_index,
        "single": {
            "query": query,
            "attn_scores": attn_scores_q,
            "attn_weights_naive": attn_weights_q_naive,
            "attn_weights": attn_weights_q,
            "context_vec": context_vec_q,
        },
        "attn_scores": attn_scores,
        "attn_weights": attn_weights,
        "context_vecs": all_context_vecs,
    }


def trace_self_attention(sa: SelfAttention_v2, x):
    """Section 3.4 - trainable Q/K/V. ``x`` is 2-D: (num_tokens, d_in)."""
    keys = sa.W_key(x)
    queries = sa.W_query(x)
    values = sa.W_value(x)
    attn_scores = queries @ keys.T
    d_k = keys.shape[-1]
    scaled_scores = attn_scores / d_k**0.5
    attn_weights = torch.softmax(scaled_scores, dim=-1)
    attn_weights_unscaled = torch.softmax(attn_scores, dim=-1)  # 3.4.1 sidebar
    context_vec = attn_weights @ values

    assert torch.allclose(context_vec, sa(x), atol=1e-6)
    return {
        "queries": queries, "keys": keys, "values": values,
        "attn_scores": attn_scores, "scaled_scores": scaled_scores,
        "attn_weights": attn_weights, "attn_weights_unscaled": attn_weights_unscaled,
        "context_vec": context_vec, "d_k": d_k,
        "W_query": sa.W_query.weight.T, "W_key": sa.W_key.weight.T, "W_value": sa.W_value.weight.T,
    }


def trace_causal(ca: CausalAttention, x, seed=123):
    """Section 3.5 - causal mask + dropout. ``x`` is 3-D: (b, num_tokens, d_in).

    Shows both the 3.5.1 approach (softmax, zero out, renormalise) and the
    3.5.2/3.5.3 approach (-inf before softmax), which give identical weights.
    """
    b, num_tokens, d_in = x.shape
    keys = ca.W_key(x)
    queries = ca.W_query(x)
    values = ca.W_value(x)
    d_k = keys.shape[-1]

    attn_scores = queries @ keys.transpose(1, 2)
    # 3.5.1 - mask *after* softmax, then renormalise
    attn_weights_full = torch.softmax(attn_scores / d_k**0.5, dim=-1)
    mask_simple = torch.tril(torch.ones(num_tokens, num_tokens))
    masked_simple = attn_weights_full * mask_simple
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums

    # 3.5.2 - mask with -inf *before* softmax (what the class actually does)
    mask_bool = ca.mask.bool()[:num_tokens, :num_tokens]
    masked_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
    attn_weights = torch.softmax(masked_scores / d_k**0.5, dim=-1)
    assert torch.allclose(attn_weights, masked_simple_norm, atol=1e-6)

    # 3.5.3 - dropout on the attention weights
    torch.manual_seed(seed)
    attn_weights_dropped = ca.dropout(attn_weights)
    keep_mask = (attn_weights_dropped != 0) | (attn_weights == 0)
    context_vec = attn_weights_dropped @ values

    torch.manual_seed(seed)
    assert torch.allclose(context_vec, ca(x), atol=1e-6)
    return {
        "queries": queries, "keys": keys, "values": values,
        "attn_scores": attn_scores,
        "attn_weights_full": attn_weights_full,
        "mask_simple": mask_simple, "masked_simple": masked_simple,
        "row_sums": row_sums, "masked_simple_norm": masked_simple_norm,
        "mask": ca.mask[:num_tokens, :num_tokens], "masked_scores": masked_scores,
        "attn_weights": attn_weights,
        "dropout_p": ca.dropout.p, "training": ca.training,
        "keep_mask": keep_mask, "attn_weights_dropped": attn_weights_dropped,
        "context_vec": context_vec, "d_k": d_k,
    }


def trace_multihead(mha: MultiHeadAttention, x, seed=123):
    """Section 3.6.2 - multi-head attention with weight splits.

    ``x`` is 3-D: (b, num_tokens, d_in). Returns per-head intermediates in the
    (b, num_heads, num_tokens, head_dim) layout used by the book's forward.
    """
    b, num_tokens, d_in = x.shape
    keys_flat = mha.W_key(x)
    queries_flat = mha.W_query(x)
    values_flat = mha.W_value(x)

    keys = keys_flat.view(b, num_tokens, mha.num_heads, mha.head_dim).transpose(1, 2)
    values = values_flat.view(b, num_tokens, mha.num_heads, mha.head_dim).transpose(1, 2)
    queries = queries_flat.view(b, num_tokens, mha.num_heads, mha.head_dim).transpose(1, 2)

    attn_scores = queries @ keys.transpose(2, 3)
    mask_bool = mha.mask.bool()[:num_tokens, :num_tokens]
    masked_scores = attn_scores.masked_fill(mask_bool, -torch.inf)
    d_k = keys.shape[-1]
    attn_weights = torch.softmax(masked_scores / d_k**0.5, dim=-1)

    torch.manual_seed(seed)
    attn_weights_dropped = mha.dropout(attn_weights)
    keep_mask = (attn_weights_dropped != 0) | (attn_weights == 0)

    context_heads = attn_weights_dropped @ values                 # (b, h, T, head_dim)
    context_concat = context_heads.transpose(1, 2).contiguous().view(b, num_tokens, mha.d_out)
    out = mha.out_proj(context_concat)

    torch.manual_seed(seed)
    assert torch.allclose(out, mha(x), atol=1e-5)
    return {
        "config": {
            "d_in": d_in, "d_out": mha.d_out, "num_heads": mha.num_heads,
            "head_dim": mha.head_dim, "num_tokens": num_tokens, "batch": b,
            "dropout": mha.dropout.p, "training": mha.training, "d_k": d_k,
            "qkv_bias": mha.W_query.bias is not None,
        },
        "queries_flat": queries_flat, "keys_flat": keys_flat, "values_flat": values_flat,
        "queries": queries, "keys": keys, "values": values,
        "attn_scores": attn_scores, "mask": mha.mask[:num_tokens, :num_tokens],
        "masked_scores": masked_scores, "attn_weights": attn_weights,
        "keep_mask": keep_mask, "attn_weights_dropped": attn_weights_dropped,
        "context_heads": context_heads, "context_concat": context_concat,
        "out_proj_weight": mha.out_proj.weight, "out_proj_bias": mha.out_proj.bias,
        "out": out,
    }
