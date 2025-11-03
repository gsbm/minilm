import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minilm import Trainer


def test_trainer_amp_requires_cuda_device():
    with pytest.raises(ValueError):
        Trainer(use_amp=True)


def test_resolve_amp_dtype_respects_override():
    trainer = Trainer(device="cuda", use_amp=False, amp_dtype="bfloat16")
    assert trainer._resolve_amp_dtype("cuda") is torch.bfloat16


def test_resolve_amp_dtype_defaults_to_float16_on_cuda():
    trainer = Trainer(device="cuda", use_amp=False, amp_dtype=None)
    assert trainer._resolve_amp_dtype("cuda") is torch.float16
