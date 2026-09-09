import torch

from llm_from_scratch.ch02_data import (build_embeddings, create_dataloader_v1,
                                        embed_batch, get_tokenizer)

BOOK_SENTENCE = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces"
                 " of someunknownPlace.")


def test_tokenizer_matches_book_section_2_5():
    ids = get_tokenizer().encode(BOOK_SENTENCE, allowed_special={"<|endoftext|>"})
    assert ids[:10] == [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554]
    assert get_tokenizer().decode(ids) == BOOK_SENTENCE


def test_first_batch_matches_book_section_2_6(verdict_text):
    dl = create_dataloader_v1(verdict_text, batch_size=1, max_length=4,
                              stride=1, shuffle=False)
    inputs, targets = next(iter(dl))
    assert inputs.tolist() == [[40, 367, 2885, 1464]]
    assert targets.tolist() == [[367, 2885, 1464, 1807]]


def test_input_embeddings_shape_section_2_8(verdict_text):
    dl = create_dataloader_v1(verdict_text, batch_size=8, max_length=4,
                              stride=4, shuffle=False)
    inputs, _ = next(iter(dl))
    tok, pos = build_embeddings(output_dim=256, context_length=4)
    token_embeddings, pos_embeddings, input_embeddings = embed_batch(inputs, tok, pos)
    assert token_embeddings.shape == torch.Size([8, 4, 256])
    assert pos_embeddings.shape == torch.Size([4, 256])
    assert input_embeddings.shape == torch.Size([8, 4, 256])
