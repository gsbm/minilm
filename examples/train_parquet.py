"""Train a model on Parquet shards with checkpointing enabled."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import torch

from minilm import Encoder, LanguageModel, Tokenizer, Trainer


def load_parquet_shards(shards: Iterable[str], limit: Optional[int] = None) -> list[str]:
    trainer = Trainer(num_epochs=1)
    samples: list[str] = []
    for shard in shards:
        path = Path(shard)
        if not path.exists():
            raise FileNotFoundError(f"Dataset shard not found: {path}")
        samples.extend(trainer.load_dataset(path, file_format="parquet"))
        if limit is not None and len(samples) >= limit:
            return samples[:limit]
    if not samples:
        raise ValueError("No training samples found in shards.")
    return samples


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", action="append", required=True, help="Path(s) to parquet file(s) or directory shards")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--no-save-latest", action="store_false", dest="save_latest")
    args = p.parse_args()

    dataset = load_parquet_shards(args.shard, limit=args.limit)

    tokenizer = Tokenizer(vocab_size=4096)
    tokenizer.fit(dataset)

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_length=128,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)

    trainer = Trainer(
        device="cuda" if torch.cuda.is_available() else "cpu",
        learning_rate=3e-4,
        num_epochs=args.epochs,
        warmup_steps=0,
        save_dir=args.save_dir,
        gradient_clip=1.0,
        weight_decay=0.01,
        accumulation_steps=2,
    )

    loader = trainer.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=16,
        sequence_length=128,
        shuffle=True,
    )

    trainer.train(
        model=model,
        train_loader=loader,
        validation_split=0.1,
        checkpoint_every=args.checkpoint_every,
        save_latest=args.save_latest,
    )

    out = Path("./models/parquet_model.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out.as_posix())


if __name__ == "__main__":
    main()
