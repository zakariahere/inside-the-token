import torch
import torch.nn.functional as F
from fastapi.testclient import TestClient

from app.main import app
from llm_from_scratch.ch04_gpt import (GPT_CONFIG_124M, TINY_GPT_CONFIG,
                                      GELU, LayerNorm, parameter_count)
from llm_from_scratch.ch04_trace import (trace_feed_forward, trace_layer_norm,
                                         trace_shortcuts,
                                         trace_transformer_block)


client = TestClient(app)


def test_layer_norm_matches_pytorch_and_normalizes_each_row():
    trace = trace_layer_norm()
    layer = LayerNorm(trace["linear_relu"].shape[-1])
    expected = F.layer_norm(
        trace["linear_relu"],
        (trace["linear_relu"].shape[-1],),
        layer.scale,
        layer.shift,
        layer.eps,
    )
    assert torch.allclose(trace["output"], expected, atol=1e-6)
    assert torch.allclose(trace["output_mean"], torch.zeros(2, 1), atol=1e-6)
    assert torch.all(trace["output_variance"] > 0.999)
    assert torch.all(trace["output_variance"] < 1)


def test_gelu_and_feed_forward_match_reference_operations():
    trace = trace_feed_forward(TINY_GPT_CONFIG)
    assert torch.allclose(
        GELU()(trace["curve_x"]),
        F.gelu(trace["curve_x"], approximate="tanh"),
        atol=1e-6,
    )
    assert trace["input"].shape == (1, 2, 4)
    assert trace["expanded"].shape == (1, 2, 16)
    assert trace["output"].shape == trace["input"].shape
    assert trace["gelu_curve"][5] < 0 < trace["relu_curve"][7]


def test_shortcuts_use_real_backward_gradients():
    trace = trace_shortcuts()
    assert trace["plain_gradients"].shape == (5,)
    assert trace["shortcut_gradients"].shape == (5,)
    assert torch.all(trace["shortcut_gradients"] > trace["plain_gradients"])
    assert all(stage["shortcut"] for stage in trace["shortcut_stages"][:4])
    assert trace["shortcut_stages"][-1]["shortcut"] is False


def test_transformer_trace_preserves_shape_and_both_residual_sums():
    torch.manual_seed(9)
    x = torch.randn(1, 4, 4)
    _module, trace = trace_transformer_block(TINY_GPT_CONFIG, x)
    assert trace["output"].shape == x.shape
    assert torch.allclose(
        trace["after_attention"],
        trace["shortcut1"] + trace["attention_dropped"],
    )
    assert torch.allclose(
        trace["output"],
        trace["shortcut2"] + trace["feed_forward_dropped"],
    )


def test_book_parameter_counts_include_weight_tying_explanation():
    assert parameter_count(GPT_CONFIG_124M) == 163_009_536
    assert parameter_count(GPT_CONFIG_124M, tied_output=True) == 124_412_160


def test_ch04_model_api_exposes_real_shapes_and_mapping():
    response = client.post("/api/ch04/model", json={"text": "Every effort moves you"})
    assert response.status_code == 200
    result = response.json()
    assert result["combined_embeddings"]["shape"] == [1, 4, 4]
    assert len(result["block_outputs"]) == 2
    assert result["logits"]["shape"] == [1, 4, 50_257]
    assert result["architecture"]["parameter_counts"]["gpt2_book_untied"] == 163_009_536


def test_greedy_generation_is_deterministic_and_crops_context():
    body = {"text": "Every effort moves you", "seed": 123, "max_new_tokens": 8}
    first = client.post("/api/ch04/generate", json=body)
    second = client.post("/api/ch04/generate", json=body)
    assert first.status_code == 200
    assert first.json() == second.json()
    result = first.json()
    assert len(result["steps"]) == 8
    assert all(step["chosen_id"] == step["top_candidates"][0]["id"] for step in result["steps"])
    assert len(result["steps"][-1]["context_ids"]) == TINY_GPT_CONFIG["context_length"]
    assert len(result["output_ids"]) == len(result["input_ids"]) + 8


def test_ch04_api_validation_and_source_disclosure():
    assert client.post("/api/ch04/generate", json={"text": "", "max_new_tokens": 2}).status_code == 422
    assert client.post("/api/ch04/generate", json={"max_new_tokens": 9}).status_code == 422
    source = client.get("/api/lessons/code").json()
    for name in ("LayerNorm", "GELU", "FeedForward", "TransformerBlock", "GPTModel", "generate_text_simple"):
        assert name in source


def test_frontend_advertises_chapter_four():
    page = client.get("/")
    assert "Raschka, chapters 1–4" in page.text
    assert "js/textbook/app.js?v=3" in page.text
