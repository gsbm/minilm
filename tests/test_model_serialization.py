from pathlib import Path
import tempfile

import torch

from minilm import Encoder, LanguageModel, Tokenizer


def build_trained_model():
    tokenizer = Tokenizer(vocab_size=32)
    tokenizer.fit(["hello world", "goodbye world"])
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_seq_length=16,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)
    bos_id = tokenizer.word_to_id[tokenizer.bos_token]
    eos_id = tokenizer.word_to_id[tokenizer.eos_token]
    hello_id = tokenizer.word_to_id.get("hello", tokenizer.word_to_id[tokenizer.unk_token])
    dummy_input = torch.tensor([[bos_id, hello_id]], dtype=torch.long)
    dummy_target = torch.tensor([[hello_id, eos_id]], dtype=torch.long)
    model.compute_loss(dummy_input, dummy_target)
    model.is_trained = True
    model.training_history = [{"epoch": 1, "train_loss": 0.1, "val_loss": None}]
    return model


def test_model_save_and_load_round_trip():
    model = build_trained_model()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pt"
        model.save(path.as_posix())
        loaded = LanguageModel.load(path.as_posix())

    assert loaded.is_trained is True
    assert loaded.training_history == model.training_history

    tokenizer = loaded.tokenizer
    assert tokenizer.is_fitted is True
    assert tokenizer.word_to_id == model.tokenizer.word_to_id

    sample_ids = [
        tokenizer.word_to_id[tokenizer.bos_token],
        tokenizer.word_to_id.get("hello", tokenizer.word_to_id[tokenizer.unk_token]),
        tokenizer.word_to_id[tokenizer.eos_token],
    ]
    with torch.no_grad():
        original_logits = model(torch.tensor([sample_ids]))
        reloaded_logits = loaded(torch.tensor([sample_ids]))
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-6)
