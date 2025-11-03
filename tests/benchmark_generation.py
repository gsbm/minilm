"""Benchmark utilities for profiling MiniLM training and generation."""

import time

import torch

from minilm import Encoder, LanguageModel, Tokenizer, Trainer


def synthetic_dataset(num_samples: int, sentence_length: int) -> list[str]:
    base = ["token" + str(i) for i in range(sentence_length)]
    sentence = " ".join(base)
    return [sentence for _ in range(num_samples)]


def benchmark_training(
    num_samples: int = 1024,
    sentence_length: int = 64,
    batch_size: int = 32,
    sequence_length: int = 128,
    device: str | None = None,
) -> float:
    """Benchmark the duration of a single training epoch on synthetic data."""
    dataset = synthetic_dataset(num_samples, sentence_length)

    tokenizer = Tokenizer(vocab_size=2048)
    tokenizer.fit(dataset)

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_length=sequence_length,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)

    trainer = Trainer(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_epochs=1,
        warmup_steps=0,
        save_dir="./benchmarks",
    )

    dataloader = trainer.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        shuffle=False,
    )

    start = time.perf_counter()
    trainer.train(model=model, train_loader=dataloader, validation_split=0.0)
    return time.perf_counter() - start


def benchmark_generation(
    prompt: str = "benchmark prompt",
    max_length: int = 256,
    iterations: int = 32,
    device: torch.device | None = None,
) -> float:
    """Benchmark average generation latency after minimal training."""
    dataset = synthetic_dataset(256, 32)
    tokenizer = Tokenizer(vocab_size=4096)
    tokenizer.fit(dataset)

    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_seq_length=max_length,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)
    trainer = Trainer(num_epochs=1, warmup_steps=0)
    dataloader = trainer.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=32,
        sequence_length=max_length,
        shuffle=False,
    )
    trainer.train(model=model, train_loader=dataloader, validation_split=0.0)

    model.eval()
    model_device = device or next(model.parameters()).device

    start = time.perf_counter()
    for _ in range(iterations):
        model.generate(prompt=prompt, max_length=max_length, device=model_device)
    duration = time.perf_counter() - start
    return duration / iterations


if __name__ == "__main__":
    train_time = benchmark_training()
    print(f"Training benchmark duration: {train_time:.2f}s")

    gen_time = benchmark_generation()
    print(f"Average generation latency: {gen_time:.4f}s")
