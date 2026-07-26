from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .config import BehaviorConfig
from .data import DatasetMeta
from .models import SemanticEncoder
from .state import TensorState, clone_state


class FrozenBehaviorProbe(nn.Module):
    def __init__(self, channels: int, classes: int, pool_size: int, seed: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.classifier = nn.Linear(channels * pool_size * pool_size, classes)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            nn.init.normal_(self.classifier.weight, 0.0, 0.05)
            nn.init.zeros_(self.classifier.bias)
        self.requires_grad_(False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(features).flatten(1))


@dataclass
class SyntheticKnowledge:
    inputs: torch.Tensor
    labels: torch.Tensor


@dataclass
class BehaviorResult:
    deviation: float
    virtual_state: dict


class BehaviorEvaluator:
    def __init__(
        self,
        encoder: SemanticEncoder,
        meta: DatasetMeta,
        num_clients: int,
        config: BehaviorConfig,
        device: torch.device,
    ) -> None:
        self.encoder = encoder.to(device)
        self.meta = meta
        self.config = config
        self.device = device
        self.probe = FrozenBehaviorProbe(
            meta.channels, meta.num_classes, config.probe_pool_size, config.probe_seed
        ).to(device)
        self.knowledge: Dict[int, SyntheticKnowledge] = {}
        lower = meta.normalized_lower.view(1, -1, 1, 1)
        upper = meta.normalized_upper.view(1, -1, 1, 1)
        for client_id in range(num_clients):
            generator = torch.Generator().manual_seed(config.probe_seed + 104729 * client_id)
            shape = (
                config.synthetic_size, meta.channels, meta.image_size, meta.image_size
            )
            inputs = lower + torch.rand(shape, generator=generator) * (upper - lower)
            if config.synthetic_size == 1:
                labels = torch.zeros(1, dtype=torch.long)
            else:
                labels = (
                    torch.linspace(0, meta.num_classes - 1, config.synthetic_size)
                    .round()
                    .long()
                )
            self.knowledge[client_id] = SyntheticKnowledge(inputs, labels)

    def _logits(
        self, params: Mapping[str, torch.Tensor], inputs: torch.Tensor
    ) -> torch.Tensor:
        from torch.func import functional_call

        return self.probe(functional_call(self.encoder, params, (inputs,)))

    def _virtual_update(self, global_state: TensorState, knowledge: SyntheticKnowledge):
        params = OrderedDict(
            (
                name,
                value.detach().to(self.device).clone().requires_grad_(True),
            )
            for name, value in global_state.items()
            if torch.is_floating_point(value)
        )
        inputs = knowledge.inputs.to(self.device)
        labels = knowledge.labels.to(self.device)
        for _ in range(self.config.virtual_steps):
            loss = F.cross_entropy(self._logits(params, inputs), labels)
            gradients = torch.autograd.grad(loss, tuple(params.values()))
            norm = torch.sqrt(
                sum(gradient.detach().float().square().sum() for gradient in gradients)
            )
            perturbation_scale = self.config.sam_radius / (norm + 1e-12)
            perturbed = OrderedDict(
                (name, parameter + perturbation_scale * gradient)
                for (name, parameter), gradient in zip(params.items(), gradients)
            )
            perturbed_loss = F.cross_entropy(self._logits(perturbed, inputs), labels)
            sam_gradients = torch.autograd.grad(
                perturbed_loss, tuple(perturbed.values())
            )
            params = OrderedDict(
                (
                    name,
                    (parameter - self.config.virtual_lr * gradient)
                    .detach()
                    .requires_grad_(True),
                )
                for (name, parameter), gradient in zip(params.items(), sam_gradients)
            )
        return OrderedDict((name, value.detach().cpu()) for name, value in params.items())

    @staticmethod
    def _normalized_squared_deviation(
        global_state: TensorState,
        virtual_state: TensorState,
        uploaded_state: TensorState,
    ) -> float:
        numerator = torch.zeros((), dtype=torch.float64)
        denominator = torch.zeros((), dtype=torch.float64)
        for name, global_value in global_state.items():
            if torch.is_floating_point(global_value):
                numerator += (
                    virtual_state[name].double() - uploaded_state[name].double()
                ).square().sum()
                denominator += global_value.double().square().sum()
        return float(numerator / (denominator + 1e-12))

    def _update_knowledge(
        self, client_id: int, virtual_state: TensorState, uploaded_state: TensorState
    ) -> None:
        knowledge = self.knowledge[client_id]
        inputs = knowledge.inputs.to(self.device).detach().requires_grad_(True)
        virtual = OrderedDict(
            (name, value.to(self.device)) for name, value in virtual_state.items()
        )
        uploaded = OrderedDict(
            (name, value.to(self.device))
            for name, value in uploaded_state.items()
            if torch.is_floating_point(value)
        )
        discrepancy = F.mse_loss(
            self._logits(virtual, inputs), self._logits(uploaded, inputs)
        )
        gradient = torch.autograd.grad(discrepancy, inputs)[0]
        updated = inputs - self.config.distill_lr * gradient
        lower = self.meta.normalized_lower.to(self.device).view(1, -1, 1, 1)
        upper = self.meta.normalized_upper.to(self.device).view(1, -1, 1, 1)
        updated = torch.maximum(torch.minimum(updated, upper), lower)
        self.knowledge[client_id] = SyntheticKnowledge(
            updated.detach().cpu(), knowledge.labels
        )

    def evaluate(
        self, client_id: int, global_state: TensorState, uploaded_state: TensorState
    ) -> BehaviorResult:
        virtual_state = self._virtual_update(global_state, self.knowledge[client_id])
        deviation = self._normalized_squared_deviation(
            global_state, virtual_state, uploaded_state
        )
        self._update_knowledge(client_id, virtual_state, uploaded_state)
        return BehaviorResult(deviation, clone_state(virtual_state))

