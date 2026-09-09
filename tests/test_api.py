import pytest
from fastapi.testclient import TestClient

from app.main import app
from llm_from_scratch import gpt2_weights

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["cuda"] is False


def test_tokenize_book_sentence():
    r = client.post("/api/ch02/tokenize", json={})
    assert r.status_code == 200
    assert r.json()["ids"][:3] == [15496, 11, 466]
    assert r.json()["tokens"][8]["text"] == "<|endoftext|>"


def test_windows_verdict(verdict_text):
    r = client.post("/api/ch02/windows", json={"source": "verdict", "max_length": 4, "stride": 1})
    j = r.json()
    assert r.status_code == 200
    assert j["pairs"][0]["input_ids"] == [40, 367, 2885, 1464]
    assert j["first_batch"]["targets"] == [[367, 2885, 1464, 1807]]


def test_embed_shapes(verdict_text):
    r = client.post("/api/ch02/embed", json={"source": "verdict", "batch_size": 8,
                                             "max_length": 4, "stride": 4, "emb_dim": 256})
    j = r.json()
    assert r.status_code == 200
    assert j["input_embeddings"]["shape"] == [8, 4, 256]
    assert j["input_embeddings"]["shown_shape"] == [8, 4, 32]


def test_simple_book():
    j = client.post("/api/ch03/simple", json={}).json()
    assert j["attn_weights"]["shape"] == [6, 6]
    assert abs(j["single"]["attn_weights"]["data"][1] - 0.2379) < 1e-3


def test_causal_has_minus_inf_above_diagonal():
    j = client.post("/api/ch03/causal", json={"dropout": 0.5}).json()
    ms = j["masked_scores"]["data"]
    assert all(ms[i][k] == "-inf" for i in range(6) for k in range(6) if k > i)
    assert all(ms[i][k] != "-inf" for i in range(6) for k in range(6) if k <= i)
    assert j["keep_mask"]["shape"] == [6, 6]


def test_mha_book_config():
    j = client.post("/api/ch03/mha", json={"d_out": 2, "num_heads": 2}).json()
    assert j["shapes"]["out"] == [2, 6, 2]
    assert len(j["heads"]) == 2
    assert j["heads"][0]["queries"]["shape"] == [6, 1]
    assert abs(j["out"]["data"][0][0] - 0.3190) < 1e-3


def test_mha_text_random():
    j = client.post("/api/ch03/mha", json={"source": "text", "text": "The cat sat",
                                           "emb_dim": 8, "d_out": 8, "num_heads": 2,
                                           "include": ["attn_weights"]}).json()
    assert j["config"]["num_tokens"] == 3
    assert "queries" not in j["heads"][0]
    assert j["heads"][0]["attn_weights"]["shape"] == [3, 3]


@pytest.mark.skipif(not gpt2_weights.gpt2_cached(), reason="no GPT-2 cache")
def test_mha_gpt2():
    j = client.post("/api/ch03/mha", json={"source": "text", "weights": "gpt2",
                                           "text": "The cat sat on the mat because it",
                                           "include": ["attn_weights"]}).json()
    assert j["config"]["num_heads"] == 12 and j["config"]["ln1_applied"] is True
    assert len(j["heads"]) == 12
    assert j["tokens"][1] == " cat"
