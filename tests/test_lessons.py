"""Numerical contracts for the independent teaching fixture."""
import math

import pytest
import torch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def bank(**options):
    response = client.post('/api/lessons/bank-attention', json=options)
    assert response.status_code == 200, response.text
    return response.json()


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def test_projections_and_bank_scores():
    r = bank()
    assert r['queries'] == [[3., 2.], [0., 2.], [3., 4.]]
    assert r['keys'] == [[0., 3.], [5., 3.], [2., 4.]]
    assert r['values'] == [[3., 1.], [0., 3.], [3., 1.]]
    assert r['scores'][1] == [6., 6., 8.]
    for weight, output in [('W_query','queries'),('W_key','keys'),('W_value','values')]:
        assert torch.equal(tensor(r['x']) @ tensor(r[weight]), tensor(r[output]))


@pytest.mark.parametrize('scaling', [True, False])
def test_softmax_and_output_full_precision(scaling):
    r = bank(scaling=scaling)
    expected = torch.softmax(tensor([[6.,6.,8.]]) / (math.sqrt(2) if scaling else 1), dim=-1)[0]
    assert torch.allclose(tensor(r['weights'][1]), expected, atol=1e-12)
    assert torch.allclose(tensor(r['out'][1]), expected @ tensor(r['values']), atol=1e-12)
    assert abs(sum(r['weights'][1])-1) < 1e-12
    assert torch.allclose(tensor(r['exponents'][1]) / sum(r['exponents'][1]), expected, atol=1e-12)


@pytest.mark.parametrize('seed', [0, 42, 123])
@pytest.mark.parametrize('training', [True, False])
def test_causal_dropout_and_single_query_equivalence(seed, training):
    r = bank(causal=True, dropout=.5, training=training, seed=seed)
    assert r['weights'][1] == [.5, .5, 0.]
    assert r == bank(causal=True, dropout=.5, training=training, seed=seed)
    for i in range(3):
        for j in range(i+1,3):
            assert r['masked_scores'][i][j] == '-inf'
            assert r['dropped_weights'][i][j] == 0
        single = tensor(r['dropped_weights'][i]) @ tensor(r['values'])
        assert torch.allclose(single, tensor(r['out'][i]), atol=1e-12)
        assert torch.allclose(tensor(r['contributions'][i]).sum(0),single,atol=1e-12)
    if not training:
        assert r['weights'] == r['dropped_weights']
        assert r['dropout_multiplier'] == 1
    else:
        assert torch.equal(tensor(r['dropped_weights']),tensor(r['weights'])*tensor(r['keep_mask'])*2)


def test_dropout_is_independent_not_fixed_count_or_renormalized():
    results = [bank(causal=True,dropout=.5,training=True,seed=seed) for seed in range(8)]
    sums = {r['row_sums'][1] for r in results}
    assert 0. in sums and 2. in sums
    assert len({str(r['keep_mask']) for r in results}) > 1


@pytest.mark.parametrize('options', [{'dropout':1}, {'dropout':-.1}, {'seed':-1}])
def test_invalid_controls_are_rejected(options):
    assert client.post('/api/lessons/bank-attention',json=options).status_code == 422


def test_full_classes_are_actual_source():
    r = client.get('/api/lessons/code')
    assert r.status_code == 200
    assert 'register_buffer' in r.json()['CausalAttention']
    assert 'class MultiHeadAttention' in r.json()['MultiHeadAttention']


def test_unicode_bytes_round_trip():
    text='A café 🌊 中文 <b>bank</b>'
    r=client.post('/api/ch02/tokenize',json={'text':text}).json()
    assert r['decoded']==text
    assert bytes(b for token in r['tokens'] for b in token['bytes']).decode('utf-8')==text


def test_missing_sample_data_has_actionable_response(monkeypatch):
    from app import registry
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(registry,'verdict_text',missing)
    r=client.post('/api/ch02/windows',json={'source':'verdict'})
    assert r.status_code==503
    assert 'fetch_data.py' in r.json()['detail']

@pytest.mark.parametrize('path', ['/api/ch02/embed', '/api/ch03/self', '/api/ch03/mha'])
def test_uncached_gpt2_has_actionable_response(monkeypatch,path):
    from llm_from_scratch import gpt2_weights
    monkeypatch.setattr(gpt2_weights,'gpt2_cached',lambda:False)
    r=client.post(path,json={'weights':'gpt2','source':'text','text':'The cat sat on a mat.'})
    assert r.status_code==503
    assert 'not cached' in r.json()['detail']
