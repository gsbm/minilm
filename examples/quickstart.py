"""Minimal quickstart demo: train on a tiny in-memory dataset and generate text."""

from __future__ import annotations

from pathlib import Path

import torch

from minilm import Encoder, LanguageModel, Tokenizer, Trainer


def main() -> None:
    texts = [
        "hello world",
        "hello there",
        "general kenobi",
        "goodbye world",
    ]

    tokenizer = Tokenizer(vocab_size=256)
    tokenizer.fit(texts)

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=2,
        dropout=0.0,
        max_seq_length=32,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)

    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=5e-4,
        num_epochs=1,
        warmup_steps=0,
        save_dir="./checkpoints_quickstart",
        gradient_clip=1.0,
        weight_decay=0.01,
        accumulation_steps=1,
    )

    loader = trainer.prepare_dataloader(
        dataset=texts,
        tokenizer=tokenizer,
        batch_size=4,
        sequence_length=32,
        shuffle=True,
    )

    trainer.train(model=model, train_loader=loader, validation_split=0.0, checkpoint_every=1, save_latest=True)

    # Save and generate
    out = Path("./models/quickstart_model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.as_posix())

    model.is_trained = True
    print(model.generate("hello", max_length=16, temperature=1.0, top_k=5, top_p=0.9, do_sample=False, num_beams=1))


if __name__ == "__main__":
    main()
