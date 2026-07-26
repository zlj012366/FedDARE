from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import TrainingConfig
from .models import SemanticEncoder
from .state import clone_state


@dataclass
class LocalUpdate:
    model_state: dict
    encoder_state: dict
    train_samples: int
    phase_i_loss: float
    phase_ii_loss: float


class FederatedClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: Dataset,
        validation_data: Dataset,
        test_data: Dataset,
        config: TrainingConfig,
    ) -> None:
        self.client_id = client_id
        self.model = model.cpu()
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.config = config

    def _loader(self, round_index: int, phase: int) -> DataLoader:
        seed = 1000003 * (round_index + 1) + 7919 * (self.client_id + 1) + phase
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            generator=torch.Generator().manual_seed(seed),
        )

    def update(
        self, global_encoder: SemanticEncoder, round_index: int, device: torch.device
    ) -> LocalUpdate:
        criterion = nn.CrossEntropyLoss()
        model = self.model.to(device)
        encoder = copy.deepcopy(global_encoder).to(device)

        model.train()
        encoder.eval()
        encoder.requires_grad_(False)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.model_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        phase_i_total = 0.0
        phase_i_batches = 0
        loader = self._loader(round_index, phase=1)
        for _ in range(self.config.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.no_grad():
                    semantic_inputs = encoder(inputs)
                semantic_loss = criterion(model(semantic_inputs), labels)
                raw_loss = criterion(model(inputs), labels)
                loss = (
                    self.config.semantic_weight * semantic_loss
                    + (1 - self.config.semantic_weight) * raw_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                phase_i_total += float(loss.detach())
                phase_i_batches += 1

        model.eval()
        model.requires_grad_(False)
        encoder.train()
        encoder.requires_grad_(True)
        optimizer = torch.optim.SGD(
            encoder.parameters(),
            lr=self.config.encoder_lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        phase_ii_total = 0.0
        phase_ii_batches = 0
        loader = self._loader(round_index, phase=2)
        for _ in range(self.config.local_epochs):
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = criterion(model(encoder(inputs)), labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                phase_ii_total += float(loss.detach())
                phase_ii_batches += 1

        model.requires_grad_(True)
        self.model = model.cpu()
        return LocalUpdate(
            clone_state(self.model.state_dict()),
            clone_state(encoder.state_dict()),
            len(self.train_data),
            phase_i_total / max(phase_i_batches, 1),
            phase_ii_total / max(phase_ii_batches, 1),
        )

