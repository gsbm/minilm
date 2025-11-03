import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minilm import Tokenizer, Encoder, LanguageModel, Trainer


def build_small_setup():
    texts = ["hello world", "hello there", "general kenobi"]
    tokenizer = Tokenizer(vocab_size=32)
    tokenizer.fit(texts)
    encoder = Encoder(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        max_seq_length=16,
    )
    model = LanguageModel(encoder=encoder, tokenizer=tokenizer)
    trainer = Trainer(
        device="cpu",
        num_epochs=1,
        scheduler_type="none",
        use_amp=False,
    )
    loader = trainer.prepare_dataloader(dataset=texts, tokenizer=tokenizer, batch_size=2, sequence_length=12)
    return model, trainer, loader


def test_checkpoint_and_resume_from_dir():
    model, trainer, loader = build_small_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        # initial train with checkpoint save
        trainer.save_dir = tmpdir
        trainer.train(model=model, train_loader=loader, validation_split=0.0, checkpoint_every=1, save_latest=True)

        latest = Path(tmpdir) / "latest.pt"
        assert latest.exists()

        # resume for one more epoch
        model2, trainer2, loader2 = build_small_setup()
        trainer2.save_dir = tmpdir
        trainer2.train(
            model=model2,
            train_loader=loader2,
            validation_split=0.0,
            resume_from=tmpdir,
            checkpoint_every=1,
            save_latest=True,
        )

        # history should contain two epochs: 1 then 2
        assert isinstance(model2.training_history, list)
        epochs = [e.get("epoch") for e in model2.training_history]
        assert epochs == [1, 2]


def test_resume_from_file_path():
    model, trainer, loader = build_small_setup()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_dir = tmpdir
        trainer.train(model=model, train_loader=loader, validation_split=0.0, checkpoint_every=1, save_latest=True)
        latest = Path(tmpdir) / "latest.pt"
        assert latest.exists()

        # resume using explicit file path
        model2, trainer2, loader2 = build_small_setup()
        trainer2.save_dir = tmpdir
        trainer2.train(
            model=model2,
            train_loader=loader2,
            validation_split=0.0,
            resume_from=latest.as_posix(),
            checkpoint_every=1,
            save_latest=True,
        )
        epochs = [e.get("epoch") for e in model2.training_history]
        assert epochs == [1, 2]
