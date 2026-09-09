"""Verify our Conv1D -> nn.Linear mapping against HuggingFace's GPT-2 block 0."""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from llm_from_scratch import gpt2_weights  # noqa: E402
from llm_from_scratch.ch02_data import get_tokenizer  # noqa: E402

if not gpt2_weights.gpt2_cached():
    pytest.skip("GPT-2 weights not in HF cache", allow_module_level=True)

TEXT = "The cat sat on the mat because it was tired"


@pytest.fixture(scope="module")
def hf():
    m = transformers.GPT2Model.from_pretrained("gpt2", attn_implementation="eager")
    m.eval()
    return m


def test_tiktoken_ids_match_hf_tokenizer():
    ids = get_tokenizer().encode(TEXT)
    hf_tok = transformers.AutoTokenizer.from_pretrained("gpt2")
    assert ids == hf_tok(TEXT).input_ids


def test_embeddings_match(hf):
    ids = torch.tensor([get_tokenizer().encode(TEXT)])
    tok, pos = gpt2_weights.build_gpt2_embeddings()
    ours = tok(ids) + pos(torch.arange(ids.shape[1]))
    theirs = hf.wte(ids) + hf.wpe(torch.arange(ids.shape[1]))
    assert torch.allclose(ours, theirs, atol=1e-5)


def test_block0_attention_matches_hf(hf):
    ids = torch.tensor([get_tokenizer().encode(TEXT)])
    captured = {}

    def hook(_mod, args, output):
        captured["out"] = output[0] if isinstance(output, tuple) else output

    handle = hf.h[0].attn.register_forward_hook(hook)
    with torch.no_grad():
        res = hf(ids, output_attentions=True)
    handle.remove()

    tok, pos = gpt2_weights.build_gpt2_embeddings()
    x = tok(ids) + pos(torch.arange(ids.shape[1]))
    x = gpt2_weights.ln_1(x)
    mha = gpt2_weights.build_gpt2_mha()
    assert mha.W_query.bias is not None

    from llm_from_scratch.trace import trace_multihead
    t = trace_multihead(mha, x)
    assert t["attn_weights"].shape == (1, 12, ids.shape[1], ids.shape[1])
    assert torch.allclose(t["attn_weights"], res.attentions[0], atol=1e-4)
    assert torch.allclose(t["out"], captured["out"], atol=1e-4)


def test_assign_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        gpt2_weights.assign(torch.zeros(2, 3), torch.zeros(3, 2))
