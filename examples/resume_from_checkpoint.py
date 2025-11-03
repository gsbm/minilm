"""Resume training from a checkpoint directory or file."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minilm import Encoder, LanguageModel, Tokenizer, Trainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--resume-from", required=True, help="Path to checkpoint file or directory (uses latest.pt)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    args = p.parse_args()

    # Toy data just to drive a quick resume step
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
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        warmup_steps=0,
    )
    loader = trainer.prepare_dataloader(dataset=texts, tokenizer=tokenizer, batch_size=4, sequence_length=32, shuffle=True)

    trainer.train(
        model=model,
        train_loader=loader,
        validation_split=0.0,
        resume_from=args.resume_from,
        checkpoint_every=1,
    )

    out = Path("./models/resumed_model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.as_posix())


if __name__ == "__main__":
    main()
