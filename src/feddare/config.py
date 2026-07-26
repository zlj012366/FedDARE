from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class DatasetConfig:
    name: str = "cifar10"
    root: str = "./data"
    download: bool = True
    iid: bool = False
    dirichlet_alpha: float = 0.5
    split: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    min_client_samples: int = 20
    synthetic_train_size: int = 1000
    synthetic_image_size: int = 32
    synthetic_num_classes: int = 10
    synthetic_channels: int = 3


@dataclass
class FederationConfig:
    num_clients: int = 50
    participation_rate: float = 0.2
    rounds: int = 100
    malicious_ratio: float = 0.0


@dataclass
class ModelConfig:
    family: str = "cnn"
    heterogeneous: bool = True
    cnn_variants: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    resnet_variants: List[int] = field(default_factory=lambda: [18, 34, 50])
    cifar_resnet_stem: bool = True


@dataclass
class TrainingConfig:
    local_epochs: int = 1
    batch_size: int = 64
    model_lr: float = 0.01
    encoder_lr: float = 0.01
    semantic_weight: float = 0.5
    momentum: float = 0.0
    weight_decay: float = 0.0
    num_workers: int = 0


@dataclass
class EncoderConfig:
    hidden_channels: int = 116
    first_kernel: int = 7
    second_kernel: int = 5
    eca_kernel: int = 3


@dataclass
class BehaviorConfig:
    enabled: bool = True
    synthetic_size: int = 10
    virtual_steps: int = 5
    sam_radius: float = 0.05
    virtual_lr: float = 0.01
    distill_lr: float = 0.01
    threshold: float = 0.08
    probe_pool_size: int = 4
    probe_seed: int = 9173
    warmup_rounds: int = 0


@dataclass
class AttackConfig:
    name: str = "none"
    target_label: int = 0
    poison_fraction: float = 0.5
    scale: float = 10.0
    noise_std: float = 1.0
    adaptive_ramp_rounds: int = 40
    adaptive_max_scale: float = 1.0
    adaptive_radius: float = 0.15
    adaptive_history: int = 3


@dataclass
class RuntimeConfig:
    seed: int = 2026
    device: str = "auto"
    deterministic: bool = True
    eval_every: int = 10
    output_dir: str = "./runs/feddare"
    evaluate_benign_only: bool = True
    save_checkpoint: bool = False
    save_client_models: bool = False


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    federation: FederationConfig = field(default_factory=FederationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        if self.dataset.name not in {"mnist", "cifar10", "cifar100", "synthetic"}:
            raise ValueError(f"Unsupported dataset: {self.dataset.name}")
        if len(self.dataset.split) != 3 or abs(sum(self.dataset.split) - 1.0) > 1e-8:
            raise ValueError("dataset.split must contain three ratios summing to 1")
        if any(value <= 0 for value in self.dataset.split):
            raise ValueError("dataset.split ratios must be positive")
        if self.dataset.dirichlet_alpha <= 0:
            raise ValueError("dataset.dirichlet_alpha must be positive")
        if self.dataset.min_client_samples < 3:
            raise ValueError("dataset.min_client_samples must be at least 3")
        if self.federation.num_clients < 1 or self.federation.rounds < 1:
            raise ValueError("num_clients and rounds must be positive")
        if not 0 < self.federation.participation_rate <= 1:
            raise ValueError("participation_rate must be in (0, 1]")
        if not 0 <= self.federation.malicious_ratio < 1:
            raise ValueError("malicious_ratio must be in [0, 1)")
        if self.model.family not in {"cnn", "resnet"}:
            raise ValueError("model.family must be 'cnn' or 'resnet'")
        if not self.model.cnn_variants or not self.model.resnet_variants:
            raise ValueError("model variant lists must not be empty")
        if self.training.local_epochs < 1 or self.training.batch_size < 1:
            raise ValueError("local_epochs and batch_size must be positive")
        if self.training.model_lr <= 0 or self.training.encoder_lr <= 0:
            raise ValueError("model and encoder learning rates must be positive")
        if not 0 <= self.training.semantic_weight <= 1:
            raise ValueError("semantic_weight must be in [0, 1]")
        if self.training.momentum < 0 or self.training.weight_decay < 0:
            raise ValueError("momentum and weight_decay must be non-negative")
        if self.training.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        for name, value in {
            "hidden_channels": self.encoder.hidden_channels,
            "first_kernel": self.encoder.first_kernel,
            "second_kernel": self.encoder.second_kernel,
            "eca_kernel": self.encoder.eca_kernel,
        }.items():
            if value < 1:
                raise ValueError(f"encoder.{name} must be positive")
        if any(kernel % 2 == 0 for kernel in (
            self.encoder.first_kernel,
            self.encoder.second_kernel,
            self.encoder.eca_kernel,
        )):
            raise ValueError("encoder kernels must be odd to preserve spatial shape")
        if self.behavior.synthetic_size < 1 or self.behavior.virtual_steps < 1:
            raise ValueError("synthetic_size and virtual_steps must be positive")
        if self.behavior.sam_radius < 0 or self.behavior.virtual_lr <= 0:
            raise ValueError("SAM radius must be non-negative and virtual_lr positive")
        if self.behavior.distill_lr <= 0 or self.behavior.threshold < 0:
            raise ValueError("distill_lr must be positive and threshold non-negative")
        if self.behavior.probe_pool_size < 1 or self.behavior.warmup_rounds < 0:
            raise ValueError("probe_pool_size must be positive and warmup non-negative")
        valid_attacks = {
            "none",
            "label_flip",
            "sign_flip",
            "scaling",
            "dba",
            "adaptive_scaling",
            "gaussian",
        }
        if self.attack.name not in valid_attacks:
            raise ValueError(
                f"Unsupported attack: {self.attack.name}; choose from {sorted(valid_attacks)}"
            )
        if not 0 <= self.attack.poison_fraction <= 1:
            raise ValueError("attack.poison_fraction must be in [0, 1]")
        if self.attack.scale < 0 or self.attack.noise_std < 0:
            raise ValueError("attack scale and noise_std must be non-negative")
        if self.attack.adaptive_ramp_rounds < 1 or self.attack.adaptive_history < 1:
            raise ValueError("adaptive_ramp_rounds and adaptive_history must be positive")
        if self.attack.adaptive_max_scale < 0 or self.attack.adaptive_radius < 0:
            raise ValueError("adaptive scale and radius must be non-negative")
        if self.runtime.eval_every < 1:
            raise ValueError("runtime.eval_every must be positive")
        if self.runtime.save_client_models and not self.runtime.save_checkpoint:
            raise ValueError("save_client_models requires save_checkpoint=true")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _section(cls: Any, raw: Dict[str, Any], name: str) -> Any:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise TypeError(f"Config section '{name}' must be a mapping")
    known = set(cls.__dataclass_fields__)
    unexpected = sorted(set(values) - known)
    if unexpected:
        raise ValueError(f"Unknown keys in '{name}': {unexpected}")
    return cls(**values)


def load_config(path: str) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("The YAML root must be a mapping")
    config = ExperimentConfig(
        dataset=_section(DatasetConfig, raw, "dataset"),
        federation=_section(FederationConfig, raw, "federation"),
        model=_section(ModelConfig, raw, "model"),
        training=_section(TrainingConfig, raw, "training"),
        encoder=_section(EncoderConfig, raw, "encoder"),
        behavior=_section(BehaviorConfig, raw, "behavior"),
        attack=_section(AttackConfig, raw, "attack"),
        runtime=_section(RuntimeConfig, raw, "runtime"),
    )
    config.validate()
    return config
