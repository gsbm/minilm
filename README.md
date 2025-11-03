# minilm

minilm is a lightweight toolkit for experimenting with compact language models. It offers a complete pipeline from tokenisation and encoding through training, evaluation, and text generation while staying dependency-light and easy to extend.

## Installation

```bash
pip install -e .
```

For development (tests, linting, property-based checks):

```bash
pip install -r requirements-dev.txt
```

## Quick start

```python
from minilm import Tokenizer, Encoder, LanguageModel, Trainer

texts = ["hello world", "hello there", "general kenobi"]

# 1. Tokenise and fit vocabulary
tokenizer = Tokenizer(vocab_size=64)
tokenizer.fit(texts)

# 2. Build encoder and language model
encoder = Encoder(vocab_size=tokenizer.vocab_size, embedding_dim=32, hidden_dim=64, num_layers=2, num_heads=2)
model = LanguageModel(encoder=encoder, tokenizer=tokenizer)

# 3. Prepare data loader and trainer
trainer = Trainer(device="cpu", num_epochs=2, use_amp=False)
loader = trainer.prepare_dataloader(dataset=texts, tokenizer=tokenizer, batch_size=2, sequence_length=32)
trainer.train(model=model, train_loader=loader, validation_split=0.0)

model.is_trained = True
print(model.generate("hello", max_length=10))
```

---

## Package overview

| Module | Description |
| ------ | ----------- |
| `minilm.tokenizer` | Word-level tokenizer with padding, truncation, and serialisation helpers. |
| `minilm.encoder`   | Transformer-based encoder turning token IDs into contextual embeddings. |
| `minilm.model`     | Decoder-only language model with sampling utilities and checkpointing. |
| `minilm.trainer`   | Training orchestration with dataloader prep, schedulers, AMP, and evaluation. |

Each module exports a primary class with well-defined public methods, detailed below with usage examples.

### `Tokenizer`

#### `Tokenizer.__init__(vocab_size=10000, pad_token="<PAD>", unk_token="<UNK>", bos_token="<BOS>", eos_token="<EOS>")`
Creates a tokenizer with configurable vocabulary size and special tokens. Raises `ValueError` if parameters are invalid.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| vocab_size | int | 10000 | Maximum vocabulary size including special tokens. Must be > 0. |
| pad_token | str | "<PAD>" | Padding token; must be non-empty and distinct from other specials. |
| unk_token | str | "<UNK>" | Unknown token for out-of-vocabulary items; must be distinct. |
| bos_token | str | "<BOS>" | Beginning-of-sequence token; must be distinct. |
| eos_token | str | "<EOS>" | End-of-sequence token; must be distinct. |

Example:
```python
tokenizer = Tokenizer(vocab_size=512, pad_token="<pad>")
```

#### `Tokenizer.fit(texts)`
Builds the vocabulary from a non-empty list of strings. Strips whitespace and counts token frequency.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| texts | list[str] | required | Non-empty list of non-blank strings. Raises `ValueError` if empty/invalid. |

Example:
```python
tokenizer.fit(["a small step", "a giant leap"])
```

#### `Tokenizer.encode(text, add_special_tokens=True, max_length=None, padding=False, truncation=False)`
Converts text to token IDs, optionally adding BOS/EOS, padding, and truncation behaviour. Requires prior call to `fit`.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| text | str | required | Input text, must be non-empty. |
| add_special_tokens | bool | True | If True, wraps tokens with BOS/EOS. |
| max_length | Optional[int] | None | If provided, must be > 0. Enforces max output length. |
| padding | bool | False | If True, pads up to `max_length` (or to current length when `max_length` is None). |
| truncation | bool | False | If False and sequence exceeds `max_length`, raises `ValueError`. |

Example:
```python
ids = tokenizer.encode("a small step", max_length=8, padding=True, truncation=True)
```

#### `Tokenizer.decode(token_ids, skip_special_tokens=True)`
Reconstructs a string from token IDs, optionally removing special tokens.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| token_ids | list[int] | required | Non-empty sequence of non-negative IDs. |
| skip_special_tokens | bool | True | If True, removes PAD/BOS/EOS from output. |

Example:
```python
text = tokenizer.decode(ids, skip_special_tokens=True)
```

#### `Tokenizer.tokenize(text)`
Splits input text into whitespace-delimited tokens.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| text | str | required | Input text; must be non-empty. |

Example:
```python
words = tokenizer.tokenize("tokenise this")
```

### `Encoder`

#### `Encoder.__init__(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=4, num_heads=4, dropout=0.1, max_seq_length=512)`
Initialises token/position embeddings and a transformer encoder stack. Validates dimensions and dropout parameters.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| vocab_size | int | required | Size of tokenizer vocabulary; > 0. |
| embedding_dim | int | 128 | Token/position embedding size; > 0. |
| hidden_dim | int | 256 | Feed-forward dimension; > 0. |
| num_layers | int | 4 | Number of encoder layers; > 0. |
| num_heads | int | 4 | Attention heads per layer; > 0. |
| dropout | float | 0.1 | Must satisfy 0.0 ≤ dropout < 1.0. |
| max_seq_length | int | 512 | Max supported sequence length; > 0. |

Example:
```python
encoder = Encoder(vocab_size=tokenizer.vocab_size, embedding_dim=64, hidden_dim=128, num_layers=2, num_heads=4)
```

#### `Encoder.encode(text=None, token_ids=None, return_embeddings=True, attention_mask=None, device=None)`
Produces contextual embeddings for either raw text or pre-tokenised IDs. Mutual exclusivity between `text` and `token_ids` is enforced.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| text | Optional[str] | None | Raw text to encode. Cannot be provided together with `token_ids`. |
| token_ids | Optional[Sequence[int]] | None | Pre-tokenised IDs. Cannot be provided together with `text`. |
| return_embeddings | bool | True | Present for API parity; embeddings are always returned. |
| attention_mask | Optional[Sequence[int]] | None | 1 for valid, 0 for padding; must match sequence length. |
| device | Optional[torch.device] | None | Target device for returned tensors. |

Example:
```python
embeddings = encoder.encode(token_ids=ids, device=torch.device("cpu"))
```

#### `Encoder.forward(token_ids, attention_mask=None)`
Performs the transformer forward pass, returning contextual hidden states. Automatically expands 1D tensors to batch form.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| token_ids | torch.Tensor | required | Shape (batch, seq_len) or (seq_len,). Must be non-negative and < vocab size. |
| attention_mask | Optional[torch.Tensor] | None | Same shape as `token_ids`; 1=valid, 0=pad. Defaults to ones. |

Example:
```python
hidden = encoder(torch.tensor([ids]))
```

#### `Encoder.get_embeddings(token_ids, device=None)`
Returns raw embedding vectors without contextualisation (useful for inspections or weight tying).

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| token_ids | Sequence[int] | required | Non-empty sequence of non-negative IDs within vocab range. |
| device | Optional[torch.device] | None | Device for output tensor. |

Example:
```python
static_vectors = encoder.get_embeddings([tokenizer.word_to_id["a"], tokenizer.word_to_id["small"]])
```

### `LanguageModel`

#### `LanguageModel.__init__(encoder, tokenizer, model_type="decoder-only")`
Wraps an encoder with a linear output head sized to the tokenizer vocabulary. Only `decoder-only` is currently supported.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| encoder | nn.Module | required | Backbone providing `embedding_dim`. |
| tokenizer | Tokenizer | required | Tokenizer instance with vocabulary metadata. |
| model_type | str | "decoder-only" | Must be exactly "decoder-only". Stored in checkpoints. |

Example:
```python
model = LanguageModel(encoder=encoder, tokenizer=tokenizer)
```

#### `LanguageModel.forward(input_ids, attention_mask=None)`
Returns next-token logits for a batch of token IDs.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| input_ids | torch.Tensor | required | Token IDs of shape (batch, seq_len); cast to long if needed. |
| attention_mask | Optional[torch.Tensor] | None | Mask passed to encoder. |

Example:
```python
logits = model(torch.tensor([ids]))
```

#### `LanguageModel.generate(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.95, do_sample=True, num_beams=1, device=None)`
Performs autoregressive text generation with sampling controls. Requires the model to have been trained.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| prompt | str | required | Non-empty seed text. |
| max_length | int | 100 | Maximum total tokens (prompt + generated). > 0. |
| temperature | float | 1.0 | Softmax temperature; must be > 0. |
| top_k | int | 50 | Keep top-k tokens for sampling; > 0. |
| top_p | float | 0.95 | Nucleus sampling threshold in (0.0, 1.0]. |
| do_sample | bool | True | If False, greedily select argmax. |
| num_beams | int | 1 | Must be 1; other values raise `NotImplementedError`. |
| device | Optional[torch.device] | None | Overrides device; defaults to model parameter device. |

Example:
```python
model.is_trained = True
text = model.generate("Once upon", max_length=30, temperature=0.8)
```

#### `LanguageModel.compute_loss(input_ids, target_ids, attention_mask=None)`
Computes cross-entropy loss with optional masking, accepting tensors or lists.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| input_ids | Tensor or Sequence[int] | required | Teacher-forced inputs. |
| target_ids | Tensor or Sequence[int] | required | Targets aligned with `input_ids`. |
| attention_mask | Optional[Tensor or Sequence[int]] | None | Optional mask aligned with inputs. |

Example:
```python
loss = model.compute_loss(inputs[:, :-1], inputs[:, 1:], attention_mask=masks[:, :-1])
```

#### `LanguageModel.save(path)` / `LanguageModel.load(path)`
Persist and restore model, encoder configuration, tokenizer state, and training history.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| path | str | required | Filesystem path for saving or loading checkpoints. |

Example:
```python
model.save("models/minilm.pt")
reloaded = LanguageModel.load("models/minilm.pt")
```

### `Trainer`

#### `Trainer.__init__(..., scheduler_type="linear", scheduler_kwargs=None, use_amp=False, amp_dtype=None)`
Configures training hyperparameters, learning-rate scheduler choice, and optional CUDA automatic mixed precision.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| device | str | "cpu" | "cpu" or "cuda". Required for AMP (`cuda`). |
| learning_rate | float | 1e-4 | Optimizer LR; must be > 0. |
| num_epochs | int | 10 | Number of epochs; ≥ 0. |
| warmup_steps | int | 1000 | Steps for warmup; ≥ 0. |
| save_dir | str | "./checkpoints" | Directory for checkpoints. |
| gradient_clip | float | 1.0 | Max gradient norm; > 0. |
| weight_decay | float | 0.01 | L2 regularisation; ≥ 0. |
| accumulation_steps | int | 1 | Gradient accumulation steps; > 0. |
| scheduler_type | str | "linear" | One of {"linear","cosine","cosine_warmup","plateau","none"}. |
| scheduler_kwargs | Optional[dict] | None | Extra args for scheduler constructors (e.g., `t_max`, `warmup_iters`, `patience`). |
| use_amp | bool | False | Enables automatic mixed precision (CUDA only). |
| amp_dtype | Optional[str] | None | "float16" or "bfloat16"; optional override for autocast. |

Example:
```python
trainer = Trainer(
    device="cuda",
    num_epochs=3,
    learning_rate=5e-4,
    scheduler_type="cosine_warmup",
    scheduler_kwargs={"warmup_iters": 10, "t_max": 90},
    use_amp=True,
    amp_dtype="float16",
)
```

#### `Trainer.load_dataset(path, format="text")`
Loads datasets from text, JSON, CSV, or Parquet sources. Returns a list of non-empty strings.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| path | str or Path | required | Input file or directory. Must exist. |
| format | str | "text" | One of {"text","json","csv","parquet"}. |

Example:
```python
records = trainer.load_dataset("data/train.json", format="json")
```

#### `Trainer.prepare_dataloader(dataset, tokenizer, batch_size=32, sequence_length=128, shuffle=True)`
Tokenises dataset entries and creates a PyTorch `DataLoader` yielding `input_ids` and `attention_mask` tensors.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| dataset | list[str] | required | Non-empty list of strings. |
| tokenizer | Tokenizer | required | Fitted tokenizer with special tokens. |
| batch_size | int | 32 | Batch size; > 0. |
| sequence_length | int | 128 | Max length after padding/truncation; > 0. |
| shuffle | bool | True | Use random sampler when True. |

Example:
```python
loader = trainer.prepare_dataloader(records, tokenizer, batch_size=8, sequence_length=64)
```

#### `Trainer.train(model, train_loader, validation_split=0.1, resume_from=None, checkpoint_every=1, save_latest=True)`
Runs training with gradient accumulation, configurable scheduler, optional AMP, validation split, and checkpointing/resume support.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| model | nn.Module | required | Must implement `compute_loss` and accept `input_ids`/`attention_mask`. |
| train_loader | Iterable[dict] | required | Yields `{"input_ids": Tensor, "attention_mask": Tensor}`. |
| validation_split | float | 0.1 | Fraction in [0.0,1.0) of batches held out for validation. |
| resume_from | Optional[str] | None | Path to a checkpoint file or directory. If a directory is provided, `latest.pt` inside it is used. Restores model, optimizer, scheduler, AMP scaler states, last epoch and history. |
| checkpoint_every | int | 1 | Save an intermediate checkpoint every N epochs (saved as `checkpoint-epoch{N}.pt`). |
| save_latest | bool | True | When True, also writes/updates `latest.pt` checkpoint in `save_dir`. |

Example:
```python
trainer.train(model=model, train_loader=loader, validation_split=0.2, checkpoint_every=1)
print(model.training_history)
```

##### Checkpointing & Resuming

Trainer saves checkpoints into `save_dir` containing:

- `model_state_dict`, `optimizer_state_dict`, optional `scheduler_state_dict`, optional AMP `scaler_state_dict`.
- `epoch`, `global_step`, and accumulated `training_history`.

Save a checkpoint every epoch and keep a moving latest pointer:

```python
trainer = Trainer(save_dir="./checkpoints", num_epochs=2)
trainer.train(model, loader, validation_split=0.0, checkpoint_every=1, save_latest=True)
# Resume later from directory (uses latest.pt) or explicit file path
trainer = Trainer(save_dir="./checkpoints", num_epochs=1)
trainer.train(model, loader, validation_split=0.0, resume_from="./checkpoints")
```

#### `Trainer.evaluate_perplexity(model, test_dataset, batch_size=32)`
Computes perplexity on held-out text using teacher forcing.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| model | LanguageModel | required | Must be trained and expose a tokenizer and `compute_loss`. |
| test_dataset | list[str] | required | Non-empty held-out dataset. |
| batch_size | int | 32 | Evaluation batch size; > 0. |

Example:
```python
ppl = trainer.evaluate_perplexity(model, ["hello world", "general kenobi"], batch_size=1)
```

---

## Testing

Run the full suite (unit, property-based, regression tests):

```bash
pytest
```

Key test categories include tokenizer edge cases, encoder validation, serialization round-trips, scheduler/AMP configuration, and property-based encode-decode checks.

## Contributing

Issues and pull requests are welcome. Please run `pytest` and ensure style/docstrings remain consistent with the existing codebase before submitting changes.
