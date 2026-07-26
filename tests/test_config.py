from pathlib import Path

from feddare.config import load_config


def test_smoke_config_contains_paper_defaults() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"
    config = load_config(str(path))
    assert config.training.semantic_weight == 0.5
    assert config.behavior.sam_radius == 0.05
    assert config.behavior.threshold == 0.08

