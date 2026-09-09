import torch

from llm_from_scratch.ch03_attention import (BOOK_INPUTS, CausalAttention,
                                             MultiHeadAttention, SelfAttention_v2)
from llm_from_scratch.trace import (trace_causal, trace_multihead,
                                    trace_self_attention, trace_simple)


def test_section_3_3_simplified_attention():
    t = trace_simple(BOOK_INPUTS, query_index=1)
    assert torch.allclose(t["single"]["attn_scores"],
                          torch.tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865]), atol=1e-4)
    assert torch.allclose(t["single"]["attn_weights"],
                          torch.tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581]), atol=1e-4)
    assert torch.allclose(t["single"]["context_vec"],
                          torch.tensor([0.4419, 0.6515, 0.5683]), atol=1e-4)
    assert torch.allclose(t["attn_weights"].sum(dim=-1), torch.ones(6))


def test_section_3_4_self_attention_v2_seed_789():
    torch.manual_seed(789)
    sa = SelfAttention_v2(d_in=3, d_out=2)
    out = sa(BOOK_INPUTS)
    assert out.shape == (6, 2)
    assert torch.allclose(out[0], torch.tensor([-0.0739, 0.0713]), atol=1e-4)
    t = trace_self_attention(sa, BOOK_INPUTS)
    assert torch.allclose(t["context_vec"], out)


def test_section_3_5_causal_attention():
    batch = torch.stack((BOOK_INPUTS, BOOK_INPUTS), dim=0)
    torch.manual_seed(123)
    ca = CausalAttention(d_in=3, d_out=2, context_length=batch.shape[1], dropout=0.0)
    out = ca(batch)
    assert out.shape == (2, 6, 2)
    t = trace_causal(ca, batch)
    w = t["attn_weights"][0]
    assert torch.all(w.triu(diagonal=1) == 0)
    assert torch.allclose(w.sum(dim=-1), torch.ones(6))
    assert (t["masked_scores"][0] == -torch.inf).sum() == 15


def test_section_3_6_2_multihead_seed_123():
    batch = torch.stack((BOOK_INPUTS, BOOK_INPUTS), dim=0)
    torch.manual_seed(123)
    mha = MultiHeadAttention(d_in=3, d_out=2, context_length=batch.shape[1],
                             dropout=0.0, num_heads=2)
    out = mha(batch)
    assert out.shape == torch.Size([2, 6, 2])
    assert torch.allclose(out[0], out[1])
    assert torch.allclose(out[0, 0], torch.tensor([0.3190, 0.4858]), atol=1e-4)
    assert torch.allclose(out[0, -1], torch.tensor([0.2575, 0.4028]), atol=1e-4)
    t = trace_multihead(mha, batch)
    assert t["queries"].shape == (2, 2, 6, 1)
    assert t["context_concat"].shape == (2, 6, 2)
