"""Module for the main language model."""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as f


class LanguageModel(nn.Module):
    """Decoder-only language model built around an encoder backbone."""

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer,
        model_type: str = "decoder-only",
    ) -> None:
        """Initialise the language model with its encoder and tokenizer.

        Args:
            encoder: Backbone module producing contextual embeddings. Must
                expose ``embedding_dim``.
            tokenizer: Tokenizer instance providing vocabulary metadata
                (``vocab_size`` etc.).
            model_type: Architecture descriptor for saved checkpoints. Only
                ``"decoder-only"`` is currently supported.

        Raises:
            ValueError: If required components are missing or ``model_type`` is
                unsupported.
        """
        super().__init__()
        if encoder is None:
            raise ValueError("LanguageModel requires an 'encoder' instance.")
        if tokenizer is None:
            raise ValueError("LanguageModel requires a 'tokenizer' instance.")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError("LanguageModel requires 'model_type' to be a non-empty string.")
        allowed_types = {"decoder-only"}
        if model_type not in allowed_types:
            raise ValueError("LanguageModel currently supports only 'decoder-only' model_type.")

        self.encoder = encoder
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.lm_head = nn.Linear(self.encoder.embedding_dim, self.tokenizer.vocab_size, bias=False)
        self.is_trained = False
        self.training_history: List[float] = []

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Return logits for the next-token prediction task.

        Args:
            input_ids: Input token IDs of shape ``(batch, seq_len)``.
            attention_mask: Optional mask where ``1`` marks valid tokens and
                ``0`` denotes padding.

        Returns:
            Tensor: Logits of shape ``(batch, seq_len, vocab_size)``.

        Raises:
            TypeError: If ``input_ids`` is not a tensor.
        """

        if not isinstance(input_ids, Tensor):
            raise TypeError("LanguageModel.forward expects 'input_ids' to be a torch.Tensor.")
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        hidden_states = self.encoder(input_ids, attention_mask)
        return self.lm_head(hidden_states)

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        *,
        do_sample: bool = True,
        num_beams: int = 1,
        device: Optional[torch.device] = None,
    ) -> str:
        """Generate text autoregressively from a prompt.

        Args:
            prompt: Initial text to condition generation on. Must be non-empty and
                tokenizable by this instance's tokenizer.
            max_length: Maximum number of tokens (including prompt and generated
                tokens) to produce.
            temperature: Softmax temperature applied to logits before sampling;
                higher values yield more diverse outputs.
            top_k: Keep only the ``top_k`` highest-probability tokens for
                sampling.
            top_p: Perform nucleus (top-p) sampling where the smallest set of
                tokens whose cumulative probability exceeds ``top_p`` is kept.
            do_sample: When ``True`` draw from the filtered distribution;
                otherwise take the argmax (greedy decoding).
            num_beams: Beam size for beam-search decoding. Only ``1`` is
                currently supported.
            device: Torch device to run generation on. Defaults to the model's
                first parameter device.

        Returns:
            Generated string with special tokens removed.

        Raises:
            ValueError: If any argument is malformed (e.g., empty prompt or
                invalid ranges).
            NotImplementedError: If ``num_beams`` is not equal to ``1``.
            RuntimeError: If the model has not been marked as trained.
        """

        if not isinstance(prompt, str) or not prompt:
            raise ValueError("LanguageModel.generate expects 'prompt' to be a non-empty string.")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("LanguageModel.generate expects 'max_length' to be a positive integer.")
        if not isinstance(temperature, (int, float)) or float(temperature) <= 0.0:
            raise ValueError("LanguageModel.generate expects 'temperature' to be a positive number.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("LanguageModel.generate expects 'top_k' to be a positive integer.")
        if not isinstance(top_p, (int, float)) or not 0.0 < float(top_p) <= 1.0:
            raise ValueError("LanguageModel.generate expects 'top_p' in the range (0.0, 1.0].")
        if not isinstance(do_sample, bool):
            raise ValueError("LanguageModel.generate expects 'do_sample' to be a boolean.")
        if num_beams != 1:
            raise NotImplementedError("Beam search generation is not yet supported.")
        if not getattr(self, "is_trained", False):
            raise RuntimeError("LanguageModel.generate requires the model to be trained before generation.")

        device = device or next(self.parameters()).device
        was_training = self.training
        self.eval()
        pad_id = self.tokenizer.word_to_id[self.tokenizer.pad_token]
        eos_id = self.tokenizer.word_to_id[self.tokenizer.eos_token]

        generated_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        generated_tensor = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            while generated_tensor.shape[1] < max_length:
                attention_mask = (generated_tensor != pad_id).long()
                logits = self.forward(generated_tensor, attention_mask)
                next_token_logits = logits[:, -1, :]
                next_token_logits = next_token_logits / float(temperature)
                filtered_logits = self._top_k_top_p_filter(next_token_logits, top_k=top_k, top_p=top_p)
                if torch.isinf(filtered_logits).all():
                    filtered_logits = next_token_logits
                if do_sample:
                    probabilities = f.softmax(filtered_logits, dim=-1)
                    next_token = torch.multinomial(probabilities, num_samples=1)
                else:
                    next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
                generated_tensor = torch.cat([generated_tensor, next_token], dim=1)
                if next_token.item() == eos_id:
                    break

        tokens = generated_tensor.squeeze(0).tolist()
        generated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if was_training:
            self.train()
        return generated_text

    def compute_loss(
        self,
        input_ids,
        target_ids,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute cross-entropy loss between logits and targets.

        Args:
            input_ids: Teacher-forced input sequence (tensor or iterable).
            target_ids: Expected outputs aligned with ``input_ids``.
            attention_mask: Optional mask weighting tokens during loss
                aggregation.

        Returns:
            Tensor: Scalar loss value.

        Raises:
            ValueError: If shapes mismatch or attention mask is invalid.
        """

        input_tensor = self._ensure_long_tensor(input_ids)
        target_tensor = self._ensure_long_tensor(target_ids)
        if input_tensor.shape != target_tensor.shape:
            raise ValueError("LanguageModel.compute_loss requires matching input and target shapes.")
        if attention_mask is not None:
            attention_mask = self._ensure_long_tensor(attention_mask)
            if attention_mask.shape != input_tensor.shape:
                raise ValueError("LanguageModel.compute_loss expects 'attention_mask' to match input shape.")

        logits = self.forward(input_tensor, attention_mask)
        vocab_size = logits.shape[-1]
        losses = f.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_tensor.reshape(-1),
            reduction="none",
        )
        if attention_mask is not None:
            mask = attention_mask.reshape(-1).float()
            if mask.numel() != losses.numel():
                raise ValueError("Attention mask must match the flattened token count.")
            valid_tokens = mask.sum().clamp_min(1.0)
            losses = (losses * mask).sum() / valid_tokens
        else:
            losses = losses.mean()
        return losses

    def save(self, path: str) -> None:
        """Persist model weights and tokenizer/encoder configuration.

        Args:
            path: Destination filepath for the serialized checkpoint.

        Raises:
            ValueError: If ``path`` is empty.
        """

        if not isinstance(path, str) or not path:
            raise ValueError("LanguageModel.save expects 'path' to be a non-empty string.")
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_type": self.model_type,
            "model_state_dict": self.state_dict(),
            "encoder_config": {
                "vocab_size": self.encoder.vocab_size,
                "embedding_dim": self.encoder.embedding_dim,
                "hidden_dim": self.encoder.hidden_dim,
                "num_layers": self.encoder.num_layers,
                "num_heads": self.encoder.num_heads,
                "dropout": self.encoder.dropout,
                "max_seq_length": self.encoder.max_seq_length,
            },
            "tokenizer_state": self._serialize_tokenizer(),
            "training_history": self.training_history,
            "is_trained": self.is_trained,
        }
        torch.save(state, path_obj)

    @classmethod
    def load(cls, path: str):
        """Load model weights and configuration from disk.

        Args:
            path: Path to a previously saved checkpoint.

        Returns:
            LanguageModel: Reconstructed model instance.

        Raises:
            ValueError: If ``path`` is empty or checkpoint metadata is missing.
            FileNotFoundError: If the checkpoint file does not exist.
        """

        if not isinstance(path, str) or not path:
            raise ValueError("LanguageModel.load expects 'path' to be a non-empty string.")
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"LanguageModel.load could not find saved model at: {path_obj}")

        state = torch.load(path_obj, map_location="cpu")
        from .encoder import Encoder

        tokenizer_state = state.get("tokenizer_state")
        if tokenizer_state is None:
            raise ValueError("Serialized model is missing tokenizer state.")
        tokenizer = cls._deserialize_tokenizer(tokenizer_state)

        encoder_config = state.get("encoder_config")
        if encoder_config is None:
            raise ValueError("Serialized model is missing encoder configuration.")
        encoder = Encoder(**encoder_config)

        model = cls(encoder=encoder, tokenizer=tokenizer, model_type=state.get("model_type", "decoder-only"))
        model.load_state_dict(state.get("model_state_dict", {}))
        model.training_history = state.get("training_history", [])
        model.is_trained = state.get("is_trained", False)
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_long_tensor(self, data) -> Tensor:
        """Convert data to a ``torch.long`` tensor.

        Args:
            data: Tensor or sequence convertible to a tensor.

        Returns:
            Tensor: Long tensor representation.

        Raises:
            TypeError: If ``data`` cannot be converted.
        """
        if isinstance(data, Tensor):
            return data.long()
        if isinstance(data, (list, tuple)):
            return torch.tensor(data, dtype=torch.long)
        raise TypeError("Expected tensor or sequence of integers.")

    def _serialize_tokenizer(self) -> Dict[str, object]:
        """Capture tokenizer configuration for checkpointing."""
        return {
            "vocab_size": self.tokenizer.vocab_size,
            "pad_token": self.tokenizer.pad_token,
            "unk_token": self.tokenizer.unk_token,
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "special_tokens": self.tokenizer.special_tokens,
            "word_to_id": self.tokenizer.word_to_id,
            "id_to_word": self.tokenizer.id_to_word,
            "is_fitted": self.tokenizer.is_fitted,
        }

    @staticmethod
    def _deserialize_tokenizer(state: Dict[str, object]):
        """Reconstruct a tokenizer instance from serialized state."""
        from .tokenizer import Tokenizer

        tokenizer = Tokenizer(
            vocab_size=state["vocab_size"],
            pad_token=state["pad_token"],
            unk_token=state["unk_token"],
            bos_token=state["bos_token"],
            eos_token=state["eos_token"],
        )
        tokenizer.special_tokens = state["special_tokens"]
        tokenizer.word_to_id = state["word_to_id"]
        tokenizer.id_to_word = state["id_to_word"]
        tokenizer.is_fitted = state.get("is_fitted", False)
        return tokenizer

    def _top_k_top_p_filter(self, logits: Tensor, top_k: int, top_p: float) -> Tensor:
        """Filter a distribution using top-k and nucleus (top-p) sampling."""

        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            threshold_values, _ = torch.topk(logits, top_k)
            min_threshold = threshold_values[..., -1, None]
            logits = torch.where(logits < min_threshold, torch.full_like(logits, float("-inf")), logits)

        if 0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            shifted_mask = sorted_mask[..., :-1].clone()
            sorted_mask[..., 1:] = shifted_mask
            sorted_mask[..., 0] = False
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(-1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        return logits
