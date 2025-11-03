"""Module for text tokenization."""

from collections import Counter
from typing import List, Optional


class Tokenizer:
    """Word-level tokenizer for preparing text corpora for language models."""

    def __init__(
        self,
        vocab_size: int = 10000,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS>",
        eos_token: str = "<EOS>",
    ):
        """Initialise a tokenizer with configurable vocabulary and special tokens.

        Args:
            vocab_size: Maximum vocabulary size, including special tokens.
            pad_token: Padding token used to equalise sequence lengths.
            unk_token: Token used for out-of-vocabulary words.
            bos_token: Special token inserted at the beginning of sequences.
            eos_token: Special token inserted at the end of sequences.

        Raises:
            ValueError: If ``vocab_size`` is not positive, special tokens are
                empty/duplicated, or cannot fit inside ``vocab_size``.
        """
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("Tokenizer requires a positive integer 'vocab_size'.")
        for name, token in {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
        }.items():
            if not isinstance(token, str) or not token:
                raise ValueError(f"Tokenizer requires '{name}' to be a non-empty string.")
        special_tokens = [pad_token, unk_token, bos_token, eos_token]
        if len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Tokenizer special tokens must be distinct.")
        if len(special_tokens) > vocab_size:
            raise ValueError("Tokenizer 'vocab_size' must accommodate all special tokens.")
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = special_tokens
        self.word_to_id = {}
        self.id_to_word = {}
        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Build the token vocabulary from training texts.

        Args:
            texts: Non-empty list of strings used to derive token frequencies.

        Raises:
            ValueError: If ``texts`` is empty, contains non-string entries, or
                strings that reduce to empty after stripping.
        """
        if not isinstance(texts, list) or not texts:
            raise ValueError("Tokenizer.fit expects 'texts' to be a non-empty list of strings.")
        invalid_texts = [text for text in texts if not isinstance(text, str) or not text.strip()]
        if invalid_texts:
            raise ValueError("Tokenizer.fit received invalid entries; expected non-empty strings.")
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common = [token for token, _ in counter.most_common(available_slots)]
        self.word_to_id = {token: idx for idx, token in enumerate(self.special_tokens)}
        next_id = len(self.word_to_id)
        for token in most_common:
            if token in self.word_to_id:
                continue
            if next_id >= self.vocab_size:
                break
            self.word_to_id[token] = next_id
            next_id += 1
        self.id_to_word = {idx: token for token, idx in self.word_to_id.items()}
        self.is_fitted = True

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """Convert text to token IDs with optional special tokens and padding.

        Args:
            text: Input string to convert.
            add_special_tokens: When ``True`` wrap the sequence with BOS/EOS
                tokens.
            max_length: Optional maximum sequence length after special tokens
                are added.
            padding: Whether to pad sequences shorter than ``max_length`` using
                the pad token.
            truncation: Whether to truncate sequences longer than
                ``max_length``.

        Returns:
            List[int]: Encoded token IDs.

        Raises:
            ValueError: If arguments are malformed or truncation is required but
                disabled.
            RuntimeError: If the tokenizer has not been fitted yet.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("Tokenizer.encode expects 'text' to be a non-empty string.")
        if not isinstance(add_special_tokens, bool):
            raise ValueError("Tokenizer.encode expects 'add_special_tokens' to be a boolean.")
        if max_length is not None:
            if not isinstance(max_length, int) or max_length <= 0:
                raise ValueError("Tokenizer.encode expects 'max_length' to be a positive integer when provided.")
        for name, flag in {"padding": padding, "truncation": truncation}.items():
            if not isinstance(flag, bool):
                raise ValueError(f"Tokenizer.encode expects '{name}' to be a boolean.")
        if not self.is_fitted:
            raise RuntimeError("Tokenizer.encode requires the tokenizer to be fitted before encoding.")
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        unk_id = self.word_to_id[self.unk_token]
        token_ids = [self.word_to_id.get(token, unk_id) for token in tokens]
        if max_length is not None and len(token_ids) > max_length:
            if truncation:
                token_ids = token_ids[:max_length]
            else:
                raise ValueError(
                    "Tokenizer.encode requires 'max_length' combined with 'truncation=True' when sequences exceed the limit.",
                )
        if padding:
            pad_id = self.word_to_id[self.pad_token]
            if max_length is None:
                max_length = len(token_ids)
            if len(token_ids) < max_length:
                token_ids = token_ids + [pad_id] * (max_length - len(token_ids))
        return token_ids

    def decode(
        self,
        token_ids: List[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back into a whitespace-delimited string.

        Args:
            token_ids: Sequence of token IDs to decode.
            skip_special_tokens: When ``True`` remove special tokens from the
                output.

        Returns:
            str: Decoded text.

        Raises:
            ValueError: If ``token_ids`` is empty/malformed or special-token
                handling flags are invalid.
            RuntimeError: If the tokenizer has not been fitted yet.
        """
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("Tokenizer.decode expects 'token_ids' to be a non-empty list of integers.")
        invalid_ids = [idx for idx in token_ids if not isinstance(idx, int) or idx < 0]
        if invalid_ids:
            raise ValueError("Tokenizer.decode received invalid token IDs; expected non-negative integers.")
        if not isinstance(skip_special_tokens, bool):
            raise ValueError("Tokenizer.decode expects 'skip_special_tokens' to be a boolean.")
        if not self.is_fitted:
            raise RuntimeError("Tokenizer.decode requires the tokenizer to be fitted before decoding.")
        specials = set(self.special_tokens) if skip_special_tokens else set()
        tokens = []
        for idx in token_ids:
            token = self.id_to_word.get(idx, self.unk_token)
            if skip_special_tokens and token in specials:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Split raw text into whitespace-delimited tokens.

        Args:
            text: Input string to split.

        Returns:
            List[str]: Tokens extracted from the input string.

        Raises:
            ValueError: If ``text`` is empty or not a string.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("Tokenizer.tokenize expects 'text' to be a non-empty string.")
        return text.strip().split()
