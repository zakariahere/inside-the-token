"""Inspectable Chapter 4 calculations kept separate from the book classes."""
from copy import deepcopy

import torch
import torch.nn as nn

from .ch04_gpt import (FeedForward, GELU, GPTModel, LayerNorm,
                       TransformerBlock)


def trace_layer_norm(seed=123):
    torch.manual_seed(seed)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    activations = layer(batch_example)
    norm = LayerNorm(activations.shape[-1])
    mean = activations.mean(dim=-1, keepdim=True)
    variance = activations.var(dim=-1, keepdim=True, unbiased=False)
    centered = activations - mean
    normalized = centered / torch.sqrt(variance + norm.eps)
    output = norm(activations)
    assert torch.allclose(output, norm.scale * normalized + norm.shift)
    return {
        "batch_input": batch_example,
        "linear_relu": activations,
        "mean": mean,
        "variance": variance,
        "centered": centered,
        "normalized": normalized,
        "scale": norm.scale,
        "shift": norm.shift,
        "output": output,
        "output_mean": output.mean(dim=-1, keepdim=True),
        "output_variance": output.var(dim=-1, keepdim=True, unbiased=False),
        "eps": norm.eps,
    }


def trace_feed_forward(cfg, seed=123):
    x_curve = torch.linspace(-3, 3, 13)
    gelu = GELU()
    relu_curve = torch.relu(x_curve)
    gelu_curve = gelu(x_curve)

    torch.manual_seed(seed)
    module = FeedForward(cfg)
    x = torch.tensor(
        [[[0.25, -0.50, 0.75, 1.00],
          [-0.30, 0.20, 0.60, -0.10]]],
        dtype=torch.float32,
    )
    expanded = module.layers[0](x)
    activated = module.layers[1](expanded)
    output = module.layers[2](activated)
    assert torch.allclose(output, module(x))
    return {
        "curve_x": x_curve,
        "relu_curve": relu_curve,
        "gelu_curve": gelu_curve,
        "input": x,
        "linear1_weight": module.layers[0].weight,
        "linear1_bias": module.layers[0].bias,
        "expanded": expanded,
        "activated": activated,
        "linear2_weight": module.layers[2].weight,
        "linear2_bias": module.layers[2].bias,
        "output": output,
    }


class _ShortcutNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x, collect=False):
        stages = []
        for layer in self.layers:
            layer_output = layer(x)
            shortcut_used = self.use_shortcut and x.shape == layer_output.shape
            x = x + layer_output if shortcut_used else layer_output
            if collect:
                stages.append((layer_output, x, shortcut_used))
        return (x, stages) if collect else x


def _shortcut_run(model, sample_input):
    output, stages = model(sample_input, collect=True)
    target = torch.tensor([[0.0]])
    loss = nn.MSELoss()(output, target)
    loss.backward()
    gradients = torch.stack([
        layer[0].weight.grad.abs().mean() for layer in model.layers
    ])
    return output, loss, gradients, stages


def trace_shortcuts(seed=123):
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1.0, 0.0, -1.0]])
    with torch.enable_grad():
        torch.manual_seed(seed)
        plain = _ShortcutNetwork(layer_sizes, False)
        shortcut = _ShortcutNetwork(layer_sizes, True)
        shortcut.load_state_dict(deepcopy(plain.state_dict()))
        p_out, p_loss, p_grad, p_stages = _shortcut_run(plain, sample_input)
        s_out, s_loss, s_grad, s_stages = _shortcut_run(shortcut, sample_input)
    return {
        "input": sample_input.detach(),
        "plain_output": p_out.detach(),
        "plain_loss": p_loss.detach(),
        "plain_gradients": p_grad.detach(),
        "shortcut_output": s_out.detach(),
        "shortcut_loss": s_loss.detach(),
        "shortcut_gradients": s_grad.detach(),
        "plain_stages": [
            {"layer_output": a.detach(), "output": b.detach(), "shortcut": c}
            for a, b, c in p_stages
        ],
        "shortcut_stages": [
            {"layer_output": a.detach(), "output": b.detach(), "shortcut": c}
            for a, b, c in s_stages
        ],
    }


def trace_transformer_block(cfg, x, seed=123, training=False):
    torch.manual_seed(seed)
    module = TransformerBlock(cfg)
    module.train(training)

    torch.manual_seed(seed)
    shortcut1 = x
    norm1 = module.norm1(x)
    attention = module.att(norm1)
    attention_dropped = module.drop_shortcut(attention)
    after_attention = attention_dropped + shortcut1
    shortcut2 = after_attention
    norm2 = module.norm2(after_attention)
    feed_forward = module.ff(norm2)
    feed_forward_dropped = module.drop_shortcut(feed_forward)
    output = feed_forward_dropped + shortcut2

    torch.manual_seed(seed)
    assert torch.allclose(output, module(x), atol=1e-6)
    return module, {
        "input": x,
        "shortcut1": shortcut1,
        "norm1": norm1,
        "attention": attention,
        "attention_dropped": attention_dropped,
        "after_attention": after_attention,
        "shortcut2": shortcut2,
        "norm2": norm2,
        "feed_forward": feed_forward,
        "feed_forward_dropped": feed_forward_dropped,
        "output": output,
    }


def trace_gpt(model: GPTModel, token_ids):
    _batch_size, seq_len = token_ids.shape
    token_embeddings = model.tok_emb(token_ids)
    positions = torch.arange(seq_len, device=token_ids.device)
    position_embeddings = model.pos_emb(positions)
    x = token_embeddings + position_embeddings
    after_embedding_dropout = model.drop_emb(x)
    dropped_embeddings = after_embedding_dropout
    block_outputs = []
    for block in model.trf_blocks:
        after_embedding_dropout = block(after_embedding_dropout)
        block_outputs.append(after_embedding_dropout)
    final_norm = model.final_norm(after_embedding_dropout)
    logits = model.out_head(final_norm)
    assert torch.allclose(logits, model(token_ids), atol=1e-6)
    return {
        "token_embeddings": token_embeddings,
        "position_embeddings": position_embeddings,
        "combined_embeddings": x,
        "after_embedding_dropout": dropped_embeddings,
        "block_outputs": block_outputs,
        "final_norm": final_norm,
        "logits": logits,
    }
