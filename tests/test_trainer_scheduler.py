import sys
from pathlib import Path

import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from minilm import Trainer


def build_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    return torch.optim.AdamW([parameter], lr=1e-3)


@pytest.mark.parametrize(
    "scheduler_type,expected_cls,requires_metric,kwargs",
    [
        ("linear", torch.optim.lr_scheduler.LinearLR, False, {}),
        ("cosine", torch.optim.lr_scheduler.CosineAnnealingLR, False, {"t_max": 3}),
        ("plateau", torch.optim.lr_scheduler.ReduceLROnPlateau, True, {"patience": 1}),
    ],
)
def test_scheduler_factory_known_types(scheduler_type, expected_cls, requires_metric, kwargs):
    trainer = Trainer(num_epochs=3, scheduler_type=scheduler_type, scheduler_kwargs=kwargs)
    optimizer = build_optimizer()

    scheduler, needs_metric = trainer._create_scheduler(optimizer)

    assert isinstance(scheduler, expected_cls)
    assert needs_metric is requires_metric


def test_scheduler_factory_cosine_warmup_returns_sequential():
    trainer = Trainer(num_epochs=4, warmup_steps=2, scheduler_type="cosine_warmup", scheduler_kwargs={"t_max": 4})
    optimizer = build_optimizer()

    scheduler, needs_metric = trainer._create_scheduler(optimizer)

    assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
    assert needs_metric is False
    assert len(scheduler._schedulers) == 2


def test_scheduler_factory_none_returns_none():
    trainer = Trainer(scheduler_type="none")
    optimizer = build_optimizer()

    scheduler, needs_metric = trainer._create_scheduler(optimizer)

    assert scheduler is None
    assert needs_metric is False


def test_scheduler_invalid_type_raises():
    with pytest.raises(ValueError):
        Trainer(scheduler_type="unknown")
