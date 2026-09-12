from typing import Literal, Optional

from pydantic import BaseModel, Field

MAX_TOKENS = 64
BOOK_SENTENCE = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
                 " of someunknownPlace.")

Weights = Literal["random", "gpt2"]
Source = Literal["book", "text", "verdict"]


class TokenizeRequest(BaseModel):
    text: str = Field(default=BOOK_SENTENCE, max_length=4000)


class WindowsRequest(BaseModel):
    source: Literal["text", "verdict"] = "text"
    text: str = Field(default=BOOK_SENTENCE, max_length=4000)
    max_length: int = Field(default=4, ge=1, le=64)
    stride: int = Field(default=1, ge=1, le=64)
    batch_size: int = Field(default=1, ge=1, le=8)
    max_pairs: int = Field(default=32, ge=1, le=256)


class EmbedRequest(BaseModel):
    source: Literal["text", "verdict"] = "text"
    text: str = Field(default=BOOK_SENTENCE, max_length=4000)
    max_length: int = Field(default=4, ge=1, le=64)
    stride: int = Field(default=4, ge=1, le=64)
    batch_size: int = Field(default=8, ge=1, le=8)
    emb_dim: int = Field(default=256, ge=2, le=768)
    weights: Weights = "random"
    seed: int = 123
    show_dims: int = Field(default=32, ge=1, le=768)


class _Ch3Base(BaseModel):
    source: Source = "book"
    text: str = Field(default="Your journey starts with one step", max_length=2000)
    emb_dim: int = Field(default=3, ge=1, le=768)   # d_in when source != book
    seed: int = 123
    weights: Weights = "random"
    show_dims: int = Field(default=16, ge=1, le=768)


class SimpleRequest(_Ch3Base):
    query_index: int = Field(default=1, ge=0)


class SelfAttnRequest(_Ch3Base):
    seed: int = 789
    d_out: int = Field(default=2, ge=1, le=768)
    qkv_bias: bool = False


class CausalRequest(_Ch3Base):
    d_out: int = Field(default=2, ge=1, le=768)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    train: bool = True
    qkv_bias: bool = False


class MHARequest(_Ch3Base):
    d_out: int = Field(default=2, ge=1, le=768)
    num_heads: int = Field(default=2, ge=1, le=12)
    context_length: Optional[int] = Field(default=None, ge=1, le=1024)
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    train: bool = False
    qkv_bias: bool = False
    apply_ln1: bool = True          # only meaningful when weights == "gpt2"
    batch_index: int = Field(default=0, ge=0)
    include: Optional[list[str]] = None   # subset of MHA_FIELDS; None = all


MHA_FIELDS = ["x", "queries_flat", "keys_flat", "values_flat", "queries", "keys",
              "values", "attn_scores", "mask", "masked_scores", "attn_weights",
              "keep_mask", "attn_weights_dropped", "context_heads",
              "context_concat", "out_proj_weight", "out"]


class Ch04SeedRequest(BaseModel):
    seed: int = Field(default=123, ge=0, le=1_000_000)


class Ch04BlockRequest(Ch04SeedRequest):
    dropout: float = Field(default=0.0, ge=0.0, le=0.9)
    train: bool = False


class Ch04ModelRequest(Ch04SeedRequest):
    text: str = Field(default="Every effort moves you", min_length=1, max_length=2000)


class Ch04GenerateRequest(Ch04ModelRequest):
    max_new_tokens: int = Field(default=4, ge=1, le=8)
