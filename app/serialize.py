"""Tensor -> JSON helpers. Values are rounded, optionally truncated, and
accompanied by the *full* shape and summary statistics so the UI can show
"[8, 4, 256]" even when only 32 columns are transmitted."""
import math

import torch


def _round_val(v, decimals):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isinf(v):
            return "-inf" if v < 0 else "inf"
        if math.isnan(v):
            return None
        return round(v, decimals)
    return v


def _round_nested(x, decimals):
    if isinstance(x, list):
        return [_round_nested(i, decimals) for i in x]
    return _round_val(x, decimals)


def tensor_to_json(t, max_rows=None, max_cols=None, decimals=4, name=None):
    """Serialise a tensor. ``max_rows`` limits the second-to-last dim,
    ``max_cols`` the last dim (only applied when the tensor has that many dims)."""
    t = t.detach()
    full_shape = list(t.shape)
    view = t
    truncated = False
    if max_cols is not None and view.dim() >= 1 and view.shape[-1] > max_cols:
        view = view[..., :max_cols]
        truncated = True
    if max_rows is not None and view.dim() >= 2 and view.shape[-2] > max_rows:
        view = view[..., :max_rows, :]
        truncated = True

    if t.dtype == torch.bool:
        stats = {"true": int(t.sum()), "total": t.numel()}
        data = view.int().tolist()
    else:
        finite = t[torch.isfinite(t)] if t.is_floating_point() else t.float()
        if finite.numel():
            stats = {
                "min": round(finite.min().item(), decimals),
                "max": round(finite.max().item(), decimals),
                "mean": round(finite.mean().item(), decimals),
                "std": round(finite.std().item(), decimals) if finite.numel() > 1 else 0.0,
            }
        else:
            stats = {"min": None, "max": None, "mean": None, "std": None}
        data = _round_nested(view.tolist(), decimals)

    out = {
        "shape": full_shape,
        "shown_shape": list(view.shape),
        "truncated": truncated,
        "dtype": str(t.dtype).replace("torch.", ""),
        "data": data,
        "stats": stats,
    }
    if name:
        out["name"] = name
    return out
