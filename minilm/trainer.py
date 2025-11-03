"""Module for training language models."""

import csv
import json
import math
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


class Trainer:
    """Trainer for language models."""

    def __init__(
        self,
        device: str = "cpu",
        learning_rate: float = 1e-4,
        num_epochs: int = 10,
        warmup_steps: int = 1000,
        save_dir: str = "./checkpoints",
        gradient_clip: float = 1.0,
        weight_decay: float = 0.01,
        accumulation_steps: int = 1,
        scheduler_type: str = "linear",
        scheduler_kwargs: Optional[dict] = None,
        *,
        use_amp: bool = False,
        amp_dtype: Optional[str] = None,
    ):
        """Configure the training loop hyperparameters and device placement.

        Args:
            device: Execution device identifier (``"cpu"`` or ``"cuda"``).
            learning_rate: Optimiser learning rate.
            num_epochs: Number of passes over the training dataset.
            warmup_steps: Steps allocated for linear learning-rate warmup.
            save_dir: Directory where checkpoints may be written.
            gradient_clip: Maximum gradient norm during optimisation.
            weight_decay: L2 regularisation strength.
            accumulation_steps: Number of batches to accumulate before each
                optimiser step.
            scheduler_type: Name of the learning-rate scheduler to apply
                (``"linear"``, ``"cosine"``, ``"cosine_warmup"``, ``"plateau"``,
                or ``"none"``).
            scheduler_kwargs: Optional keyword arguments forwarded to the
                scheduler constructor.
            use_amp: Enable automatic mixed precision during training.
            amp_dtype: Optional dtype string (``"float16"`` or ``"bfloat16"``)
                for autocast when AMP is enabled.

        Raises:
            ValueError: If supplied arguments are outside acceptable ranges.
        """
        if not isinstance(device, str) or not device:
            raise ValueError("Trainer requires 'device' to be a non-empty string (e.g., 'cpu' or 'cuda').")
        allowed_devices = {"cpu", "cuda"}
        if device not in allowed_devices:
            raise ValueError("Trainer 'device' must be exactly one of: " + ", ".join(sorted(allowed_devices)))
        if not isinstance(learning_rate, (int, float)) or float(learning_rate) <= 0.0:
            raise ValueError("Trainer requires 'learning_rate' to be a positive number.")
        for name, value in {
            "num_epochs": num_epochs,
            "warmup_steps": warmup_steps,
            "accumulation_steps": accumulation_steps,
        }.items():
            if not isinstance(value, int) or value < 0:
                qualifier = "non-negative" if name == "warmup_steps" else "positive"
                raise ValueError(f"Trainer requires '{name}' to be a {qualifier} integer.")
        if accumulation_steps == 0:
            raise ValueError("Trainer requires 'accumulation_steps' to be greater than zero.")
        if not isinstance(save_dir, str) or not save_dir:
            raise ValueError("Trainer requires 'save_dir' to be a non-empty string path.")
        if not isinstance(gradient_clip, (int, float)) or float(gradient_clip) <= 0.0:
            raise ValueError("Trainer requires 'gradient_clip' to be a positive number.")
        if not isinstance(weight_decay, (int, float)) or float(weight_decay) < 0.0:
            raise ValueError("Trainer requires 'weight_decay' to be a non-negative number.")
        if not isinstance(scheduler_type, str) or not scheduler_type:
            raise ValueError("Trainer requires 'scheduler_type' to be a non-empty string.")
        allowed_schedulers = {"linear", "cosine", "cosine_warmup", "plateau", "none"}
        if scheduler_type.lower() not in allowed_schedulers:
            raise ValueError(
                "Trainer 'scheduler_type' must be one of: " + ", ".join(sorted(allowed_schedulers)),
            )
        if scheduler_kwargs is None:
            scheduler_kwargs = {}
        if not isinstance(scheduler_kwargs, dict):
            raise ValueError("Trainer 'scheduler_kwargs' must be provided as a dictionary when supplied.")
        if not isinstance(use_amp, bool):
            raise ValueError("Trainer requires 'use_amp' to be a boolean flag.")
        valid_amp_dtypes = {None, "float16", "bfloat16"}
        if amp_dtype not in valid_amp_dtypes:
            raise ValueError("Trainer 'amp_dtype' must be one of {None, 'float16', 'bfloat16'}.")
        if use_amp and device != "cuda":
            raise ValueError("Automatic mixed precision is only supported when training on a CUDA device.")
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = save_dir
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.scheduler_type = scheduler_type.lower()
        self.scheduler_kwargs = dict(scheduler_kwargs)
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

    def load_dataset(
        self,
        path: Union[str, Path],
        file_format: str = "text",
    ) -> list:
        """Load a text dataset from disk.

        Args:
            path: Location of the dataset file or directory to read.
            file_format: Input format. One of ``"text"``, ``"json"``, ``"csv"``,
                or ``"parquet"``.

        Returns:
            List[str]: Non-empty text samples suitable for tokenization.

        Raises:
            ValueError: If arguments are malformed or the file contents cannot be
                interpreted as text samples.
            FileNotFoundError: If ``path`` does not exist.
            ImportError: If ``file_format=\"parquet\"`` is requested without
                ``pyarrow`` installed.
        """
        if not isinstance(path, (str, Path)):
            raise ValueError("Trainer.load_dataset expects 'path' to be a string or pathlib.Path.")
        if isinstance(path, str) and not path:
            raise ValueError("Trainer.load_dataset requires 'path' to be a non-empty string when provided as text.")
        if not isinstance(file_format, str) or not file_format:
            raise ValueError("Trainer.load_dataset expects 'file_format' to be a non-empty string.")
        allowed_formats = {"text", "json", "csv", "parquet"}
        if file_format not in allowed_formats:
            raise ValueError("Trainer.load_dataset 'file_format' must be exactly one of: " + ", ".join(sorted(allowed_formats)))
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path_obj}")
        if file_format == "text":
            return self._load_text_dataset(path_obj)
        if file_format == "json":
            return self._load_json_dataset(path_obj)
        if file_format == "parquet":
            return self._load_parquet_dataset(path_obj)
        return self._load_csv_dataset(path_obj)

    def prepare_dataloader(
        self,
        dataset: list,
        tokenizer,
        batch_size: int = 32,
        sequence_length: int = 128,
        *,
        shuffle: bool = True,
    ) -> DataLoader:
        """Tokenize dataset entries and return a PyTorch ``DataLoader``.

        Args:
            dataset: Sequence of input text records.
            tokenizer: Tokenizer instance exposing an ``encode`` method and
                special-token vocabulary.
            batch_size: Number of sequences per batch.
            sequence_length: Maximum tokenized sequence length (padding and
                truncation are applied when necessary).
            shuffle: Whether to shuffle samples (``RandomSampler``) or preserve
                order (``SequentialSampler``).

        Returns:
            DataLoader: Batches of ``input_ids`` and ``attention_mask`` tensors
            suitable for language-model training.

        Raises:
            ValueError: If inputs are malformed or dataset entries are empty.
        """

        if not isinstance(dataset, list) or not dataset:
            raise ValueError("Trainer.prepare_dataloader expects 'dataset' to be a non-empty list.")
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError("Trainer.prepare_dataloader requires 'tokenizer' with an 'encode' method.")
        for name, value in {"batch_size": batch_size, "sequence_length": sequence_length}.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Trainer.prepare_dataloader expects '{name}' to be a positive integer.")
        if not isinstance(shuffle, bool):
            raise ValueError("Trainer.prepare_dataloader expects 'shuffle' to be a boolean.")

        pad_id = tokenizer.word_to_id[tokenizer.pad_token]

        class _TokenDataset(Dataset):
            def __init__(self, records: Sequence[str]) -> None:
                self.records = records

            def __len__(self) -> int:
                return len(self.records)

            def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
                entry = self.records[idx]
                if not isinstance(entry, str) or not entry.strip():
                    raise ValueError("Dataset entries must be non-empty strings.")
                token_ids = tokenizer.encode(
                    entry,
                    add_special_tokens=True,
                    max_length=sequence_length,
                    padding=True,
                    truncation=True,
                )
                input_tensor = torch.tensor(token_ids, dtype=torch.long)
                attention_mask = torch.tensor([0 if token == pad_id else 1 for token in token_ids], dtype=torch.long)
                return input_tensor, attention_mask

        def _collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> dict:
            inputs, masks = zip(*batch)
            input_tensor = torch.stack(inputs, dim=0)
            mask_tensor = torch.stack(masks, dim=0)
            return {"input_ids": input_tensor, "attention_mask": mask_tensor}

        dataset_obj = _TokenDataset(dataset)
        sampler = RandomSampler(dataset_obj) if shuffle else SequentialSampler(dataset_obj)
        return DataLoader(dataset_obj, batch_size=batch_size, sampler=sampler, collate_fn=_collate_fn)

    def train(
        self,
        model,
        train_loader,
        *,
        validation_split: float = 0.1,
        resume_from: Optional[str] = None,
        checkpoint_every: int = 1,
        save_latest: bool = True,
    ):
        """Train a language model using teacher-forcing next-token prediction.

        Args:
            model: ``torch.nn.Module`` implementing ``compute_loss`` and
                receiving ``input_ids``/``attention_mask``.
            train_loader: DataLoader yielding dictionaries with
                ``input_ids`` and ``attention_mask`` tensors.
            validation_split: Fraction of batches to hold out for validation at
                the beginning of ``train_loader``.

        Raises:
            ValueError: If ``model``/``train_loader``/``validation_split`` are
                invalid or yield empty batches.
            RuntimeError: If training encounters non-finite losses.
        """
        if model is None or not isinstance(model, torch.nn.Module):
            raise ValueError("Trainer.train requires 'model' to be a torch.nn.Module instance.")
        if train_loader is None:
            raise ValueError("Trainer.train requires 'train_loader' to be provided.")
        if not isinstance(validation_split, (int, float)):
            raise ValueError("Trainer.train expects 'validation_split' to be numeric.")
        if not 0.0 <= float(validation_split) < 1.0:
            raise ValueError("Trainer.train requires 'validation_split' to be in the range [0.0, 1.0).")

        device = torch.device(self.device)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler, scheduler_requires_metric = self._create_scheduler(optimizer)

        amp_enabled = self.use_amp
        if amp_enabled:
            device_type = device.type
            amp_dtype = self._resolve_amp_dtype(device_type)
            autocast_kwargs = {"device_type": device_type, "dtype": amp_dtype}
            scaler = torch.cuda.amp.GradScaler(enabled=device_type == "cuda")
        else:
            autocast_kwargs = None
            scaler = None

        train_batches = list(train_loader)
        if not train_batches:
            raise ValueError("Trainer.train requires 'train_loader' to yield at least one batch.")
        val_size = int(len(train_batches) * float(validation_split))
        val_batches = train_batches[:val_size]
        train_batches = train_batches[val_size:] or train_batches

        # Handle resume
        start_epoch_number = 1
        global_step = 0
        history = []
        if resume_from:
            ckpt_path = self._resolve_checkpoint_path(resume_from)
            if ckpt_path is None:
                raise FileNotFoundError(f"Could not find checkpoint to resume from: {resume_from}")
            loaded = torch.load(ckpt_path, map_location=device)
            # Restore model and trainer state
            model.load_state_dict(loaded.get("model_state_dict", {}))
            optimizer.load_state_dict(loaded.get("optimizer_state_dict", {}))
            if scheduler is not None and "scheduler_state_dict" in loaded:
                scheduler.load_state_dict(loaded["scheduler_state_dict"])
            if amp_enabled and scaler is not None and "scaler_state_dict" in loaded:
                scaler.load_state_dict(loaded["scaler_state_dict"])
            start_epoch_number = int(loaded.get("epoch", 0)) + 1
            global_step = int(loaded.get("global_step", 0))
            prev_history = loaded.get("training_history", [])
            if isinstance(prev_history, list):
                history.extend(prev_history)

        # Ensure checkpoint directory
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for local_epoch_idx in range(self.num_epochs):
            epoch = start_epoch_number + local_epoch_idx
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()
            pending_update = False
            for batch_idx, batch in enumerate(train_batches):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                if input_ids.size(1) < 2:
                    continue
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                mask = attention_mask[:, :-1]

                context_manager = torch.autocast(**autocast_kwargs) if amp_enabled else nullcontext()
                with context_manager:
                    loss = model.compute_loss(inputs, targets, attention_mask=mask)
                if not torch.isfinite(loss):
                    raise RuntimeError("Encountered non-finite loss during training.")
                loss = loss / self.accumulation_steps
                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                epoch_loss += loss.item()
                pending_update = True
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if amp_enabled and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), self.gradient_clip)
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    pending_update = False
                global_step += 1

            if pending_update:
                if amp_enabled and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), self.gradient_clip)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            train_loss_avg = epoch_loss / max(len(train_batches), 1)

            val_loss = None
            if val_batches:
                model.eval()
                losses = []
                with torch.no_grad():
                    for batch in val_batches:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        if input_ids.size(1) < 2:
                            continue
                        inputs = input_ids[:, :-1]
                        targets = input_ids[:, 1:]
                        mask = attention_mask[:, :-1]
                        context_manager = torch.autocast(**autocast_kwargs) if amp_enabled else nullcontext()
                        with context_manager:
                            batch_loss = model.compute_loss(inputs, targets, attention_mask=mask)
                        losses.append(batch_loss.detach().cpu().item())
                val_loss = sum(losses) / len(losses) if losses else None

            if scheduler is not None:
                if scheduler_requires_metric:
                    metric_value = val_loss if val_loss is not None else train_loss_avg
                    scheduler.step(metric_value)
                else:
                    scheduler.step()

            history.append({
                "epoch": epoch,
                "train_loss": train_loss_avg,
                "val_loss": val_loss,
            })

            # Save intermediate checkpoints
            if checkpoint_every > 0 and (epoch % checkpoint_every == 0):
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if amp_enabled else None,
                    epoch=epoch,
                    global_step=global_step,
                    history=history,
                    save_dir=save_dir,
                    save_latest=save_latest,
                )

        model.is_trained = True
        model.training_history = history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------
    def _resolve_checkpoint_path(self, resume_from: str) -> Optional[Path]:
        """Resolve a checkpoint path from a file or directory hint.

        If a directory is provided, looks for 'latest.pt' inside it.
        If a file path is provided, returns it if present.
        """
        hint = Path(resume_from)
        if hint.is_dir():
            latest = hint / "latest.pt"
            return latest if latest.exists() else None
        return hint if hint.exists() else None

    def _save_checkpoint(
        self,
        *,
        model,
        optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        scaler: Optional[torch.cuda.amp.GradScaler],
        epoch: int,
        global_step: int,
        history: List[Dict[str, object]],
        save_dir: Path,
        save_latest: bool,
    ) -> None:
        """Persist a training checkpoint including optimizer/scheduler/scaler state.

        Saves 'checkpoint-epoch{N}.pt' and, if requested, also updates 'latest.pt'.
        """
        state: Dict[str, object] = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_history": history,
            "trainer_config": {
                "learning_rate": self.learning_rate,
                "scheduler_type": self.scheduler_type,
                "scheduler_kwargs": dict(self.scheduler_kwargs),
                "accumulation_steps": self.accumulation_steps,
                "gradient_clip": self.gradient_clip,
                "weight_decay": self.weight_decay,
            },
        }
        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None:
            state["scaler_state_dict"] = scaler.state_dict()

        epoch_path = save_dir / f"checkpoint-epoch{epoch}.pt"
        torch.save(state, epoch_path)
        if save_latest:
            latest_path = save_dir / "latest.pt"
            torch.save(state, latest_path)

    def evaluate_perplexity(
        self,
        model,
        test_dataset: list,
        batch_size: int = 32,
    ) -> float:
        """Compute perplexity on held-out text samples.

        Args:
            model: Trained language model exposing ``tokenizer`` and
                ``compute_loss``.
            test_dataset: Sequence of test strings.
            batch_size: Number of samples processed per evaluation batch.

        Returns:
            float: Perplexity computed as ``exp(mean(loss))`` over valid
            samples.

        Raises:
            ValueError: If inputs are malformed or no valid samples are
                available.
            RuntimeError: If the model has not been trained yet.
        """
        if model is None or not hasattr(model, "compute_loss"):
            raise ValueError("Trainer.evaluate_perplexity requires 'model' to provide a 'compute_loss' method.")
        if not isinstance(test_dataset, list) or not test_dataset:
            raise ValueError("Trainer.evaluate_perplexity expects 'test_dataset' to be a non-empty list.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Trainer.evaluate_perplexity expects 'batch_size' to be a positive integer.")
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError("Trainer.evaluate_perplexity requires the model to expose a tokenizer with an 'encode' method.")
        if not getattr(model, "is_trained", False):
            raise RuntimeError("Trainer.evaluate_perplexity requires the model to be trained before evaluation.")
        encoder = getattr(model, "encoder", None)
        max_length = getattr(encoder, "max_seq_length", None) if encoder is not None else None
        encode_kwargs = {"add_special_tokens": True}
        if isinstance(max_length, int) and max_length > 0:
            encode_kwargs.update({"max_length": max_length, "truncation": True})
        losses = []
        device = torch.device(self.device)
        for start in range(0, len(test_dataset), batch_size):
            batch = test_dataset[start : start + batch_size]
            for text in batch:
                if not isinstance(text, str) or not text.strip():
                    continue
                token_ids = tokenizer.encode(text, **encode_kwargs)
                if len(token_ids) < 2:
                    continue
                inputs = torch.tensor(token_ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
                targets = torch.tensor(token_ids[1:], dtype=torch.long, device=device).unsqueeze(0)
                mask = torch.ones_like(inputs, dtype=torch.long, device=device)
                loss = model.compute_loss(inputs, targets, attention_mask=mask)
                losses.append(loss.detach().cpu().item())
        if not losses:
            raise ValueError("Trainer.evaluate_perplexity requires at least one valid sample to compute perplexity.")
        average_loss = sum(losses) / len(losses)
        return float(math.exp(average_loss))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_amp_dtype(self, device_type: str) -> torch.dtype:
        if self.amp_dtype == "bfloat16":
            return torch.bfloat16
        if self.amp_dtype == "float16":
            return torch.float16
        return torch.float16 if device_type == "cuda" else torch.bfloat16

    def _create_scheduler(self, optimizer) -> tuple[Optional[torch.optim.lr_scheduler.LRScheduler], bool]:
        """Instantiate the configured learning-rate scheduler.

        Returns a tuple of (scheduler, requires_metric) where ``requires_metric``
        indicates whether ``scheduler.step`` expects a validation metric.
        """

        scheduler_type = self.scheduler_type
        kwargs = dict(self.scheduler_kwargs)
        if scheduler_type == "none":
            return None, False

        if scheduler_type == "linear":
            total_iters = max(int(kwargs.pop("total_iters", self.num_epochs or 1)), 1)
            start_factor = kwargs.pop("start_factor", 1.0)
            default_end = 0.0 if self.num_epochs > 1 else 1.0
            end_factor = kwargs.pop("end_factor", default_end)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
                **kwargs,
            )
            return scheduler, False

        if scheduler_type == "cosine":
            t_max = max(int(kwargs.pop("t_max", self.num_epochs or 1)), 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **kwargs)
            return scheduler, False

        if scheduler_type == "cosine_warmup":
            warmup_iters = max(int(kwargs.pop("warmup_iters", self.warmup_steps)), 0)
            start_factor = kwargs.pop("start_factor", 0.1)
            if warmup_iters > 0:
                cosine_total = max(int(kwargs.pop("t_max", self.num_epochs - warmup_iters)), 1)
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warmup_iters,
                )
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_total,
                    **kwargs,
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_iters],
                )
            else:
                t_max = max(int(kwargs.pop("t_max", self.num_epochs or 1)), 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, **kwargs)
            return scheduler, False

        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            return scheduler, True

        raise RuntimeError(f"Unhandled scheduler type: {scheduler_type}")

    def _load_text_dataset(self, path: Path) -> List[str]:
        """Load newline-delimited text samples from ``path``."""
        with path.open("r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines()]
        dataset = [line for line in lines if line]
        if not dataset:
            raise ValueError("Trainer.load_dataset encountered an empty text dataset.")
        return dataset

    def _load_json_dataset(self, path: Path) -> List[str]:
        """Load text samples from JSON lists or list-of-dict files."""
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            if all(isinstance(item, str) for item in loaded):
                dataset = [item.strip() for item in loaded if item.strip()]
                if not dataset:
                    raise ValueError("Trainer.load_dataset found no usable text entries in the JSON list.")
                return dataset
            if all(isinstance(item, dict) for item in loaded):
                dataset = [str(item.get("text", "")).strip() for item in loaded if str(item.get("text", "")).strip()]
                if dataset:
                    return dataset
        raise ValueError("Trainer.load_dataset expects JSON files to contain a list of strings or dicts with a 'text' field.")

    def _load_csv_dataset(self, path: Path) -> List[str]:
        """Load text from a CSV file, defaulting to the first column if needed."""
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Trainer.load_dataset requires CSV files to include a header row.")
            text_field = "text" if "text" in reader.fieldnames else reader.fieldnames[0]
            dataset = [row[text_field].strip() for row in reader if row.get(text_field, "").strip()]
        if not dataset:
            raise ValueError("Trainer.load_dataset found no usable rows in the CSV file.")
        return dataset

    def _load_parquet_dataset(self, path: Path) -> List[str]:
        """Load text samples from a Parquet file or directory of shards."""
        try:
            import pyarrow.dataset as ds
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised in environments without pyarrow
            raise ImportError("Trainer.load_dataset requires 'pyarrow' to read parquet files.") from exc

        if path.is_dir():
            dataset = ds.dataset(str(path), format="parquet")
            table = dataset.to_table()
        else:
            table = pq.read_table(path)
        if table.num_columns == 0:
            raise ValueError("Trainer.load_dataset found no columns in the Parquet file.")
        column_name = "text" if "text" in table.column_names else table.column_names[0]
        column = table.column(column_name)
        values = []
        for item in column.to_pylist():
            if item is None:
                continue
            text = str(item).strip()
            if text:
                values.append(text)
        if not values:
            raise ValueError("Trainer.load_dataset found no usable rows in the Parquet file.")
        return values

