from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


class ECALayer(nn.Module):
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel size must be odd")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.channel_conv = nn.Conv1d(
            1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.pool(inputs).squeeze(-1).transpose(-1, -2)
        weights = self.channel_conv(weights).transpose(-1, -2).unsqueeze(-1)
        return inputs * weights.sigmoid()


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 116,
        first_kernel: int = 7,
        second_kernel: int = 5,
        eca_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels, first_kernel,
            stride=1, padding=first_kernel // 2,
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            hidden_channels, in_channels, second_kernel,
            stride=1, padding=second_kernel // 2,
        )
        self.eca = ECALayer(eca_kernel)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.eca(self.conv2(self.relu(self.conv1(inputs))))

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


CNN_SPECS: Dict[int, Dict[str, int]] = {
    1: {"conv2": 32, "fc1": 2000},
    2: {"conv2": 16, "fc1": 1000},
    3: {"conv2": 32, "fc1": 800},
    4: {"conv2": 32, "fc1": 500},
    5: {"conv2": 32, "fc1": 200},
}


class PaperCNN(nn.Module):
    def __init__(self, variant: int, in_channels: int, image_size: int, num_classes: int):
        super().__init__()
        if variant not in CNN_SPECS:
            raise ValueError(f"Unknown CNN variant: {variant}")
        spec = CNN_SPECS[variant]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, spec["conv2"], 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, in_channels, image_size, image_size)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, spec["fc1"]),
            nn.ReLU(inplace=True),
            nn.Linear(spec["fc1"], 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )
        self.variant = variant

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_resnet(depth: int, in_channels: int, num_classes: int, cifar_stem: bool):
    from torchvision import models

    factories = {18: models.resnet18, 34: models.resnet34, 50: models.resnet50}
    if depth not in factories:
        raise ValueError(f"Unknown ResNet depth: {depth}")
    model = factories[depth](weights=None, num_classes=num_classes)
    if cifar_stem:
        model.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    elif in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
    return model


def build_client_model(
    family: str,
    variant: int,
    in_channels: int,
    image_size: int,
    num_classes: int,
    cifar_resnet_stem: bool,
):
    if family == "cnn":
        return PaperCNN(variant, in_channels, image_size, num_classes)
    if family == "resnet":
        return build_resnet(variant, in_channels, num_classes, cifar_resnet_stem)
    raise ValueError(f"Unknown model family: {family}")


def balanced_assignments(
    num_clients: int, variants: List[int], heterogeneous: bool, seed: int
) -> List[int]:
    if not heterogeneous:
        return [variants[0]] * num_clients
    assignments = [variants[index % len(variants)] for index in range(num_clients)]
    order = torch.randperm(num_clients, generator=torch.Generator().manual_seed(seed)).tolist()
    return [assignments[index] for index in order]

