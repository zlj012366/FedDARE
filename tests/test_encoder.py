import pytest

torch = pytest.importorskip("torch")

from feddare.models import SemanticEncoder


def test_cifar_encoder_structure() -> None:
    encoder = SemanticEncoder(3)
    inputs = torch.randn(2, 3, 32, 32)
    assert encoder.trainable_parameter_count == 25_755
    assert encoder(inputs).shape == inputs.shape
    assert encoder.conv1.bias is None
    assert encoder.conv2.bias is None
    assert encoder.eca.channel_conv.weight.shape == (1, 1, 3)
