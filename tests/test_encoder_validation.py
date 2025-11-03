import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minilm import Encoder


@pytest.fixture()
def encoder():
    return Encoder(
        vocab_size=32,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_seq_length=8,
    )


def test_forward_rejects_negative_ids(encoder):
    token_ids = torch.tensor([[-1, 0, 1]], dtype=torch.long)
    with pytest.raises(ValueError):
        encoder(token_ids)


def test_forward_rejects_ids_outside_vocab(encoder):
    token_ids = torch.tensor([[0, 1, 100]], dtype=torch.long)
    with pytest.raises(ValueError):
        encoder(token_ids)


def test_forward_rejects_sequences_longer_than_max_length(encoder):
    token_ids = torch.arange(10, dtype=torch.long).unsqueeze(0)
    with pytest.raises(ValueError):
        encoder(token_ids)


def test_forward_requires_matching_attention_mask_shape(encoder):
    token_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1]], dtype=torch.long)
    with pytest.raises(ValueError):
        encoder(token_ids, attention_mask=attention_mask)


def test_encode_requires_either_text_or_token_ids(encoder):
    with pytest.raises(ValueError):
        encoder.encode(text=None, token_ids=None)

    with pytest.raises(ValueError):
        encoder.encode(text="hello", token_ids=[0, 1])


def test_encode_rejects_invalid_token_ids(encoder):
    with pytest.raises(ValueError):
        encoder.encode(token_ids=[-1, 0, 1])

    with pytest.raises(ValueError):
        encoder.encode(token_ids=[0, 1, 999])


def test_get_embeddings_validates_input_length(encoder):
    with pytest.raises(ValueError):
        encoder.get_embeddings([])

    with pytest.raises(ValueError):
        encoder.get_embeddings([-1, 0])
