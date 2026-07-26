import pytest

torch = pytest.importorskip("torch")

from feddare.state import scale_update, weighted_average


def test_weighted_aggregation() -> None:
    first = {"weight": torch.tensor([1.0, 3.0])}
    second = {"weight": torch.tensor([5.0, 7.0])}
    result = weighted_average([first, second], [1, 3])
    assert torch.allclose(result["weight"], torch.tensor([4.0, 6.0]))


def test_scaling_changes_the_delta() -> None:
    global_state = {"weight": torch.tensor([10.0])}
    local_state = {"weight": torch.tensor([12.0])}
    assert scale_update(global_state, local_state, 4.0)["weight"].item() == 18.0

