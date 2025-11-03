import pytest

from minilm import Tokenizer


@pytest.fixture()
def fitted_tokenizer():
    tokenizer = Tokenizer(vocab_size=16)
    tokenizer.fit(["hello world", "hello there"])
    return tokenizer


def test_fit_requires_non_empty_list():
    tokenizer = Tokenizer(vocab_size=8)
    with pytest.raises(ValueError):
        tokenizer.fit([])


def test_encode_requires_fit():
    tokenizer = Tokenizer(vocab_size=8)
    with pytest.raises(RuntimeError):
        tokenizer.encode("hello")


@pytest.mark.parametrize(
    "max_length,truncation",
    [
        (4, True),
        (3, True),
    ],
)
def test_encode_padding_and_truncation(fitted_tokenizer, max_length, truncation):
    token_ids = fitted_tokenizer.encode(
        "hello",
        add_special_tokens=True,
        max_length=max_length,
        padding=True,
        truncation=truncation,
    )
    assert len(token_ids) == max_length
    pad_id = fitted_tokenizer.word_to_id[fitted_tokenizer.pad_token]
    assert token_ids[-1] == pad_id or max_length == 3


def test_encode_without_truncation_raises(fitted_tokenizer):
    with pytest.raises(ValueError):
        fitted_tokenizer.encode(
            "hello world again",
            add_special_tokens=True,
            max_length=2,
            padding=False,
            truncation=False,
        )


def test_decode_skips_special_tokens(fitted_tokenizer):
    ids = fitted_tokenizer.encode("hello world", add_special_tokens=True)
    decoded = fitted_tokenizer.decode(ids, skip_special_tokens=True)
    assert decoded == "hello world"


def test_tokenize_requires_non_empty_string():
    tokenizer = Tokenizer(vocab_size=8)
    with pytest.raises(ValueError):
        tokenizer.tokenize("")
