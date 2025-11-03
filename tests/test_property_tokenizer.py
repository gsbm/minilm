from __future__ import annotations

import torch
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from minilm import Tokenizer, Trainer

_TOKEN = st.sampled_from(["alpha", "beta", "gamma", "delta", "epsilon"])
_SENTENCE = st.lists(_TOKEN, min_size=1, max_size=6).map(" ".join)
_DATASET = st.lists(_SENTENCE, min_size=1, max_size=20)


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(dataset=_DATASET, sample=_SENTENCE)
def test_encode_decode_round_trip(dataset: list[str], sample: str):
    tokenizer = Tokenizer(vocab_size=128)
    fit_set = dataset + [sample]
    tokenizer.fit(fit_set)

    token_ids = tokenizer.encode(sample, add_special_tokens=True)
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert decoded.split() == sample.split()


@given(
    dataset=_DATASET,
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=4, max_value=16),
)
def test_dataloader_masks_align_with_padding(dataset: list[str], batch_size: int, sequence_length: int):
    tokenizer = Tokenizer(vocab_size=256)
    tokenizer.fit(dataset)

    trainer = Trainer(num_epochs=0)
    dataloader = trainer.prepare_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        sequence_length=sequence_length,
        shuffle=False,
    )

    try:
        batch = next(iter(dataloader))
    except StopIteration:
        assume(False)
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    assert input_ids.shape == attention_mask.shape
    assert input_ids.size(1) == sequence_length

    pad_id = tokenizer.word_to_id[tokenizer.pad_token]
    pad_positions = input_ids == pad_id
    assert torch.equal((attention_mask == 0), pad_positions)
