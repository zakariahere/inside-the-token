"""Chapter 2 - Working with text data (final code, Raschka sections 2.5-2.8).

Only the book's final code lives here. The regex-based SimpleTokenizerV1/V2 from
sections 2.3-2.4 are intentionally omitted: GPT-2 uses byte-pair encoding via tiktoken.
"""
from functools import lru_cache

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@lru_cache(maxsize=1)
def get_tokenizer():
    """Section 2.5 - the GPT-2 BPE tokenizer (50,257 token vocabulary)."""
    return tiktoken.get_encoding("gpt2")


class GPTDatasetV1(Dataset):
    """Listing 2.5 - sliding-window dataset producing (input, target) pairs."""

    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    """Listing 2.6 - a data loader to generate batches with input-target pairs."""
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader


def build_embeddings(output_dim, context_length, vocab_size=50257, seed=123):
    """Sections 2.7-2.8 - token and absolute positional embedding layers.

    The book creates these with ``torch.manual_seed(123)`` right before
    ``torch.nn.Embedding(...)``; we mirror that order so the numbers match.
    """
    torch.manual_seed(seed)
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    pos_embedding_layer = nn.Embedding(context_length, output_dim)
    return token_embedding_layer, pos_embedding_layer


def embed_batch(inputs, token_embedding_layer, pos_embedding_layer):
    """Section 2.8 - ``input_embeddings = token_embeddings + pos_embeddings``.

    Returns (token_embeddings, pos_embeddings, input_embeddings).
    """
    max_length = inputs.shape[1]
    token_embeddings = token_embedding_layer(inputs)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings = token_embeddings + pos_embeddings
    return token_embeddings, pos_embeddings, input_embeddings
