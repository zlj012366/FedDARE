from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import TensorDataset

from feddare.data import DatasetMeta
from feddare.metrics import _trigger_batch, evaluate_personalized_models


class ConstantModel(nn.Module):
    def forward(self, inputs):
        logits = torch.zeros(inputs.shape[0], 2, device=inputs.device)
        logits[:, 0] = 1
        return logits


def test_test_accuracy_is_client_macro_average() -> None:
    client_one = SimpleNamespace(
        client_id=0,
        model=ConstantModel(),
        test_data=TensorDataset(torch.zeros(1, 1, 4, 4), torch.zeros(1, dtype=torch.long)),
    )
    client_two = SimpleNamespace(
        client_id=1,
        model=ConstantModel(),
        test_data=TensorDataset(torch.zeros(3, 1, 4, 4), torch.ones(3, dtype=torch.long)),
    )
    meta = DatasetMeta("tiny", 1, 4, 2, (0.5,), (0.25,))
    result = evaluate_personalized_models(
        [client_one, client_two], set(), True, "none", 0, meta, 4, 0, torch.device("cpu")
    )
    assert result.test_accuracy == pytest.approx(50.0)
    assert result.clean_samples == 4


def test_trigger_size_matches_attack() -> None:
    inputs = torch.zeros(1, 1, 8, 8)
    upper = torch.tensor([2.0])
    scaling = _trigger_batch(inputs, upper, "scaling")
    dba = _trigger_batch(inputs, upper, "dba")
    assert scaling[:, :, -3:, -3:].eq(2).all()
    assert scaling.eq(2).sum().item() == 9
    assert dba[:, :, -4:, -4:].eq(2).all()
    assert dba.eq(2).sum().item() == 16
