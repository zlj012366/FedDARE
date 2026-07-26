import pytest

torch = pytest.importorskip("torch")

from feddare.models import SemanticEncoder


def test_cifar_encoder_matches_manuscript() -> None:
    encoder = SemanticEncoder(3)
    inputs = torch.randn(2, 3, 32, 32)
    assert encoder.trainable_parameter_count == 25_874
    assert encoder(inputs).shape == inputs.shape
    assert encoder.eca.channel_conv.weight.shape == (1, 1, 3)

