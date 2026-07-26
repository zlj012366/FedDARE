import pytest

torch = pytest.importorskip("torch")

from feddare.behavior import BehaviorEvaluator
from feddare.config import BehaviorConfig
from feddare.data import DatasetMeta
from feddare.models import SemanticEncoder


def test_behavior_evaluation_runs_without_client_model_upload() -> None:
    encoder = SemanticEncoder(3, hidden_channels=4, first_kernel=3, second_kernel=3)
    meta = DatasetMeta(
        "tiny", 3, 8, 2, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    )
    config = BehaviorConfig(synthetic_size=2, virtual_steps=1, probe_pool_size=2)
    evaluator = BehaviorEvaluator(encoder, meta, 1, config, torch.device("cpu"))
    state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    result = evaluator.evaluate(0, state, state)
    assert result.deviation >= 0
    assert evaluator.knowledge[0].labels.tolist() == [0, 1]


def test_synthetic_labels_span_large_class_space() -> None:
    encoder = SemanticEncoder(3, hidden_channels=4, first_kernel=3, second_kernel=3)
    meta = DatasetMeta(
        "tiny100", 3, 8, 100, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    )
    config = BehaviorConfig(synthetic_size=10, virtual_steps=1, probe_pool_size=2)
    evaluator = BehaviorEvaluator(encoder, meta, 1, config, torch.device("cpu"))
    labels = evaluator.knowledge[0].labels
    assert labels[0].item() == 0
    assert labels[-1].item() == 99
    assert labels.unique().numel() == 10
