import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from minilm import Encoder, LanguageModel, Tokenizer, Trainer


def test_end_to_end_training_and_generation(tmp_path):
    dataset_lines = [
        "hello world",
        "hello there",
        "hello world again",
        "hi there",
        "goodbye world",
    ]
    dataset_path = tmp_path / "dataset.txt"
    dataset_path.write_text("\n".join(dataset_lines), encoding="utf-8")

    trainer = Trainer(num_epochs=1, warmup_steps=0, save_dir=str(tmp_path / "checkpoints"))
    dataset = trainer.load_dataset(dataset_path, file_format="text")

    tokenizer = Tokenizer(vocab_size=128)
    tokenizer.fit(dataset)

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=16,
        hidden_dim=32,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_seq_length=32,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)

    train_loader = trainer.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=2,
        sequence_length=16,
        shuffle=False,
    )

    trainer.train(model=model, train_loader=train_loader, validation_split=0.0)

    assert model.is_trained is True
    assert len(model.training_history) == 1
    first_epoch = model.training_history[0]
    assert isinstance(first_epoch, dict)
    assert "train_loss" in first_epoch
    assert first_epoch["train_loss"] >= 0.0

    perplexity = trainer.evaluate_perplexity(model=model, test_dataset=dataset, batch_size=2)
    assert perplexity > 0
    assert math.isfinite(perplexity)

    generated = model.generate(
        prompt="hello",
        max_length=10,
        temperature=1.0,
        top_k=3,
        top_p=0.9,
        do_sample=False,
        num_beams=1,
    )
    assert isinstance(generated, str)
    assert generated

    model_path = tmp_path / "model_state.pt"
    model.save(str(model_path))
    assert model_path.exists()

    loaded_model = LanguageModel.load(str(model_path))
    assert loaded_model.is_trained is True
    assert loaded_model.training_history == model.training_history

    regenerated = loaded_model.generate(
        prompt="hello",
        max_length=8,
        temperature=1.0,
        top_k=3,
        top_p=0.9,
        do_sample=False,
        num_beams=1,
    )
    assert isinstance(regenerated, str)
    assert regenerated

    # Ensure loaded model produces identical logits for the prompt
    model.eval()
    loaded_model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode("hello", add_special_tokens=True)], dtype=torch.long)
        original_logits = model(input_ids)
        reloaded_logits = loaded_model(input_ids)
    assert torch.allclose(original_logits, reloaded_logits, atol=1e-5)
