"""Cached, expensive-to-build objects and the shared 'text -> inputs' helper."""
from functools import lru_cache
from pathlib import Path

import torch

from llm_from_scratch import gpt2_weights
from llm_from_scratch.ch02_data import build_embeddings, get_tokenizer
from llm_from_scratch.ch03_attention import BOOK_INPUTS, BOOK_TOKENS

from .schemas import MAX_TOKENS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VERDICT = DATA_DIR / "the-verdict.txt"


@lru_cache(maxsize=1)
def verdict_text() -> str:
    if not VERDICT.exists():
        raise FileNotFoundError("data/the-verdict.txt missing - run scripts/fetch_data.py")
    return VERDICT.read_text(encoding="utf-8")


def encode(text: str):
    return get_tokenizer().encode(text, allowed_special={"<|endoftext|>"})


def token_strings(ids):
    tok = get_tokenizer()
    return [tok.decode([i]) for i in ids]


def resolve_text(source: str, text: str) -> str:
    return verdict_text() if source == "verdict" else text


def make_inputs(source, text, emb_dim, seed, weights):
    """Return ``(x, tokens, meta)`` where ``x`` is 2-D (num_tokens, d_in).

    * ``book``  -> the 6x3 tensor from section 3.3.
    * ``text``  -> tokenize, then token + positional embeddings (chapter 2),
                   with either seed-random tables or real GPT-2 tables.
    """
    if source == "book":
        return BOOK_INPUTS.clone(), list(BOOK_TOKENS), {"d_in": 3, "embedding": "book §3.3 constants"}

    ids = encode(resolve_text(source, text))[:MAX_TOKENS]
    if not ids:
        raise ValueError("text produced no tokens")
    if weights == "gpt2":
        emb_dim = 768
        tok, pos = gpt2_weights.build_gpt2_embeddings()
        meta_emb = "GPT-2 wte + wpe"
    else:
        tok, pos = build_embeddings(output_dim=emb_dim, context_length=MAX_TOKENS, seed=seed)
        meta_emb = f"nn.Embedding random (seed {seed})"
    id_t = torch.tensor(ids)
    x = tok(id_t) + pos(torch.arange(len(ids)))
    return x, token_strings(ids), {"d_in": emb_dim, "embedding": meta_emb, "ids": ids}
