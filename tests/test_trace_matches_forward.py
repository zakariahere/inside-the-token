import pytest
import torch

from llm_from_scratch.ch03_attention import CausalAttention, MultiHeadAttention
from llm_from_scratch.trace import trace_causal, trace_multihead


@pytest.mark.parametrize("d_in,d_out,heads,dropout,train,bias", [
    (3, 2, 2, 0.0, False, False),
    (8, 8, 4, 0.5, True, True),
    (16, 12, 3, 0.1, True, False),
    (768, 768, 12, 0.0, False, True),
])
def test_trace_multihead_equals_forward(d_in, d_out, heads, dropout, train, bias):
    torch.manual_seed(7)
    x = torch.randn(2, 5, d_in)
    mha = MultiHeadAttention(d_in, d_out, 16, dropout, heads, qkv_bias=bias)
    mha.train(train)
    t = trace_multihead(mha, x, seed=42)   # asserts internally too
    torch.manual_seed(42)
    assert torch.allclose(t["out"], mha(x), atol=1e-5)
    if train and dropout > 0:
        assert not t["keep_mask"].all()


def test_trace_causal_equals_forward_with_dropout():
    torch.manual_seed(1)
    x = torch.randn(3, 6, 4)
    ca = CausalAttention(4, 4, 6, 0.5)
    ca.train()
    t = trace_causal(ca, x, seed=99)
    torch.manual_seed(99)
    assert torch.allclose(t["context_vec"], ca(x), atol=1e-6)
