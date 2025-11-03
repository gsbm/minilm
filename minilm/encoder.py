"""Module for encoding and embeddings."""

from typing import Optional, Sequence

import torch
from torch import Tensor, nn


class Encoder(nn.Module):
    """Transformer-based encoder for token sequences."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ) -> None:
        super().__init__()
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("Encoder requires a positive integer 'vocab_size'.")
        for name, value in {
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "max_seq_length": max_seq_length,
        }.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Encoder requires a positive integer '{name}'.")
        if not isinstance(dropout, (int, float)):
            raise ValueError("Encoder requires a numeric 'dropout'.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("Encoder 'dropout' must be in the range [0.0, 1.0).")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.max_seq_length = max_seq_length

        self.token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.position_embeddings = nn.Embedding(max_seq_length, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=float(dropout),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def encode(
        self,
        text: Optional[str] = None,
        token_ids: Optional[Sequence[int]] = None,
        *,
        return_embeddings: bool = True,
        attention_mask: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """Encode text or token IDs into contextual embeddings.

        Args:
            text: Raw string to encode. Mutually exclusive with ``token_ids``.
            token_ids: Pre-tokenised sequence of token indices. Mutually
                exclusive with ``text``.
            return_embeddings: Included for API parity; hidden states are
                returned regardless at present.
            attention_mask: Optional mask marking valid tokens (``1``) versus
                padding (``0``). Must match token sequence length when
                provided.
            device: Optional target device for the returned tensor(s).

        Returns:
            Tensor: Batched contextual embeddings of shape ``(1, seq_len,
            embedding_dim)``.

        Raises:
            ValueError: If neither or both of ``text``/``token_ids`` are
                provided, or the provided data is malformed.
        """

        tokens, mask = self._prepare_inputs(text=text, token_ids=token_ids, attention_mask=attention_mask, device=device)
        return self.forward(tokens, mask)

    def forward(self, token_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass returning contextual hidden states.

        Args:
            token_ids: Token indices of shape ``(batch, seq_len)`` or
                ``(seq_len,)``.
            attention_mask: Optional mask aligned with ``token_ids`` where ``0``
                indicates padding.

        Returns:
            Tensor: Contextual hidden states of shape ``(batch, seq_len,
            embedding_dim)``.

        Raises:
            TypeError: If ``token_ids`` is not a tensor.
            ValueError: If tensor ranks/shapes are invalid or token IDs fall
                outside the configured vocabulary/sequence length.
        """

        if not isinstance(token_ids, Tensor):
            raise TypeError("Encoder.forward expects 'token_ids' to be a torch.Tensor.")
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if token_ids.ndim != 2:
            raise ValueError("Encoder.forward expects 'token_ids' to have shape (batch, seq_len).")
        if attention_mask is not None:
            if not isinstance(attention_mask, Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=token_ids.device)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)
            if attention_mask.shape != token_ids.shape:
                raise ValueError("Encoder.forward requires 'attention_mask' to match token shape.")
        else:
            attention_mask = torch.ones_like(token_ids, dtype=torch.long)

        if torch.any(token_ids < 0):
            raise ValueError("Encoder.forward received negative token ids.")
        if torch.any(token_ids >= self.vocab_size):
            raise ValueError("Encoder.forward received token ids outside the vocabulary range.")

        batch_size, seq_length = token_ids.shape
        if seq_length > self.max_seq_length:
            raise ValueError("Encoder.forward received sequences longer than 'max_seq_length'.")

        positions = torch.arange(seq_length, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        token_embeds = self.token_embeddings(token_ids)
        position_embeds = self.position_embeddings(positions)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)

        key_padding_mask = attention_mask == 0
        return self.transformer(hidden_states, src_key_padding_mask=key_padding_mask)

    def get_embeddings(self, token_ids: Sequence[int], device: Optional[torch.device] = None) -> Tensor:
        """Return token embeddings without contextualisation.

        Args:
            token_ids: Sequence of token indices to embed.
            device: Optional device to allocate the output tensor on.

        Returns:
            Tensor: Lookup embeddings of shape ``(len(token_ids), embedding_dim)``.

        Raises:
            ValueError: If the sequence is empty, contains negative IDs, or IDs
                exceed the vocabulary size.
        """

        if not isinstance(token_ids, (list, tuple)) or not token_ids:
            raise ValueError("Encoder.get_embeddings expects 'token_ids' to be a non-empty sequence of integers.")
        invalid_ids = [idx for idx in token_ids if not isinstance(idx, int) or idx < 0]
        if invalid_ids:
            raise ValueError("Encoder.get_embeddings received invalid token IDs; expected non-negative integers.")
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        if torch.any(token_tensor >= self.vocab_size):
            raise ValueError("Encoder.get_embeddings received token ids outside the vocabulary range.")
        return self.token_embeddings(token_tensor)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _prepare_inputs(
        self,
        text: Optional[str],
        token_ids: Optional[Sequence[int]],
        attention_mask: Optional[Sequence[int]],
        device: Optional[torch.device],
    ) -> tuple[Tensor, Tensor]:
        if text is None and token_ids is None:
            raise ValueError("Encoder.encode requires either 'text' or 'token_ids'.")
        if text is not None and token_ids is not None:
            raise ValueError("Provide only one of 'text' or 'token_ids' to Encoder.encode.")
        if token_ids is None:
            if not isinstance(text, str):
                raise ValueError("Encoder.encode expects 'text' to be a string when 'token_ids' is not provided.")
            byte_vector = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
            token_tensor = byte_vector % self.vocab_size
        else:
            if not isinstance(token_ids, (list, tuple)) or not token_ids:
                raise ValueError("Encoder.encode expects 'token_ids' to be a non-empty sequence.")
            invalid_ids = [idx for idx in token_ids if not isinstance(idx, int) or idx < 0]
            if invalid_ids:
                raise ValueError("Encoder.encode received invalid token IDs; expected non-negative integers.")
            token_tensor = torch.tensor(token_ids, dtype=torch.long)

        if torch.any(token_tensor >= self.vocab_size):
            raise ValueError("Encoder.encode received token ids outside the vocabulary range.")

        if attention_mask is None:
            mask_tensor = torch.ones_like(token_tensor, dtype=torch.long)
        else:
            mask_tensor = torch.tensor(attention_mask, dtype=torch.long)
            if mask_tensor.shape != token_tensor.shape:
                raise ValueError("Encoder.encode expects 'attention_mask' to match token shape.")

        token_tensor = token_tensor.unsqueeze(0).to(device=device)
        mask_tensor = mask_tensor.unsqueeze(0).to(device=device)
        return token_tensor, mask_tensor
