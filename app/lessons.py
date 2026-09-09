"""Small, explicit teaching fixtures; the book's implementations stay untouched."""
import inspect

import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from llm_from_scratch.ch03_attention import CausalAttention, MultiHeadAttention, SelfAttention_v2
from llm_from_scratch.trace import trace_self_attention

router = APIRouter(prefix="/api/lessons")


class BankRequest(BaseModel):
    scaling: bool = True
    causal: bool = False
    dropout: float = Field(default=0, ge=0, le=0.9)
    training: bool = False
    seed: int = Field(default=42, ge=0, le=2147483647)


@router.post("/bank-attention")
def bank_attention(req: BankRequest):
    # Row-vector orientation: [input dimension, output dimension].
    x = torch.tensor([[1., 0., 2., 0.], [0., 2., 0., 3.], [0., 1., 3., 1.]], dtype=torch.float64)
    wq = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [0., 0.]], dtype=x.dtype)
    wk = torch.tensor([[0., 1.], [1., 0.], [0., 1.], [1., 1.]], dtype=x.dtype)
    wv = torch.tensor([[1., 1.], [0., 0.], [1., 0.], [0., 1.]], dtype=x.dtype)
    # fork_rng isolates the fixed fixture from other requests' random state.
    with torch.random.fork_rng():
        sa = SelfAttention_v2(4, 2).double()
        with torch.no_grad():
            for layer, weight in [(sa.W_query, wq), (sa.W_key, wk), (sa.W_value, wv)]:
                layer.weight.copy_(weight.T)
            traced = trace_self_attention(sa, x)
    q, k, v = (traced[n] for n in ("queries", "keys", "values"))
    scores = traced["attn_scores"]
    scaled = scores / (2 ** 0.5 if req.scaling else 1)
    mask = torch.triu(torch.ones(3, 3, dtype=torch.bool), diagonal=1) if req.causal else torch.zeros(3, 3, dtype=torch.bool)
    masked = scaled.masked_fill(mask, -torch.inf)
    # Stable softmax intermediates also make the displayed arithmetic safe.
    shifted = masked - masked.max(dim=-1, keepdim=True).values
    exponents = shifted.exp()
    weights = torch.softmax(masked, dim=-1)
    generator = torch.Generator().manual_seed(req.seed)
    keep = torch.rand(weights.shape, generator=generator) >= req.dropout if req.training else torch.ones_like(mask)
    multiplier = 1 / (1 - req.dropout) if req.training else 1
    dropped = weights * keep * multiplier
    contributions = dropped.unsqueeze(-1) * v.unsqueeze(0)
    out = dropped @ v

    def json_tensor(t):
        if t.dtype == torch.bool:
            return t.tolist()
        # JSON has no infinity; retain full floating point precision elsewhere.
        return [["-inf" if val == float("-inf") else val for val in row] for row in t.tolist()]

    tensors = dict(x=x, W_query=wq, W_key=wk, W_value=wv, queries=q, keys=k, values=v,
                   weights_unscaled=torch.softmax(scores, dim=-1),
                   weights_scaled=torch.softmax(scores / 2 ** 0.5, dim=-1),
                   scores=scores, scaled_scores=scaled, masked_scores=masked, shifted_scores=shifted,
                   exponents=exponents, weights=weights, keep_mask=keep, causal_mask=mask,
                   dropped_weights=dropped, out=out)
    return {"tokens": ["The", "bank", "river"], "kind": "hand-picked teaching example",
            "config": req.model_dump(), "d_k": 2, "dropout_multiplier": multiplier,
            **{name: json_tensor(t) for name, t in tensors.items()},
            "contributions": contributions.tolist(), "row_sums": dropped.sum(-1).tolist()}


@router.get("/code")
def lesson_code():
    return {cls.__name__: inspect.getsource(cls) for cls in (SelfAttention_v2, CausalAttention, MultiHeadAttention)}
