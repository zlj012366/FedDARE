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
    config = BehaviorConfig(synthetic_size=2, virtual_steps=1, task_head_pool_size=2)
    evaluator = BehaviorEvaluator(encoder, meta, 1, config, torch.device("cpu"))
    state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    result = evaluator.evaluate(0, state, state)
    assert result.deviation >= 0
    assert result.candidate_knowledge.labels.tolist() == [0, 1]
    assert evaluator.knowledge[0].labels.tolist() == [0, 1]
    assert all(not parameter.requires_grad for parameter in evaluator.task_head.parameters())
    assert torch.count_nonzero(evaluator.task_head.classifier.bias) == 0


def test_synthetic_labels_are_assigned_cyclically() -> None:
    encoder = SemanticEncoder(3, hidden_channels=4, first_kernel=3, second_kernel=3)
    meta = DatasetMeta(
        "tiny5", 3, 8, 5, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    )
    config = BehaviorConfig(synthetic_size=12, virtual_steps=1, task_head_pool_size=2)
    evaluator = BehaviorEvaluator(encoder, meta, 1, config, torch.device("cpu"))
    labels = evaluator.knowledge[0].labels
    assert labels.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert labels.unique().numel() == 5


def test_candidate_knowledge_requires_commit() -> None:
    encoder = SemanticEncoder(3, hidden_channels=4, first_kernel=3, second_kernel=3)
    meta = DatasetMeta(
        "tiny", 3, 8, 2, (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)
    )
    config = BehaviorConfig(synthetic_size=2, virtual_steps=1, task_head_pool_size=2)
    evaluator = BehaviorEvaluator(encoder, meta, 1, config, torch.device("cpu"))
    state = {name: value.detach().clone() for name, value in encoder.state_dict().items()}
    previous_inputs = evaluator.knowledge[0].inputs.clone()

    result = evaluator.evaluate(0, state, state)

    assert torch.equal(evaluator.knowledge[0].inputs, previous_inputs)
    evaluator.commit(0, result.candidate_knowledge)
    assert torch.equal(
        evaluator.knowledge[0].inputs, result.candidate_knowledge.inputs
    )
    assert evaluator.knowledge[0].labels.tolist() == [0, 1]
