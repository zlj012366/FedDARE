from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class Evaluation:
    test_accuracy: float
    attack_success_rate: float
    evaluated_clients: int
    clean_samples: int
    triggered_samples: int


def _trigger_batch(
    inputs: torch.Tensor, upper_bound: torch.Tensor, attack_name: str
) -> torch.Tensor:
    triggered = inputs.clone()
    size = 4 if attack_name == "dba" else 3
    triggered[:, :, -size:, -size:] = upper_bound.view(1, -1, 1, 1)
    return triggered


@torch.no_grad()
def evaluate_personalized_models(
    clients,
    malicious_ids,
    benign_only,
    attack_name,
    target_label,
    meta,
    batch_size,
    num_workers,
    device,
) -> Evaluation:
    client_accuracies = []
    clean_samples = 0
    trigger_success = 0
    trigger_total = 0
    targeted = attack_name in {"scaling", "dba", "adaptive_scaling"}
    upper = meta.normalized_upper.to(device)
    for client in clients:
        if benign_only and client.client_id in malicious_ids:
            continue
        model = client.model.to(device).eval()
        loader = DataLoader(
            client.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        client_correct = 0
        client_total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            client_correct += int((model(inputs).argmax(1) == labels).sum())
            client_total += labels.numel()
            if targeted:
                eligible = labels != target_label
                if eligible.any():
                    predictions = model(
                        _trigger_batch(inputs[eligible], upper, attack_name)
                    ).argmax(1)
                    trigger_success += int((predictions == target_label).sum())
                    trigger_total += int(eligible.sum())
        if client_total:
            client_accuracies.append(100.0 * client_correct / client_total)
            clean_samples += client_total
        client.model = model.cpu()
    return Evaluation(
        sum(client_accuracies) / max(len(client_accuracies), 1),
        100.0 * trigger_success / trigger_total if trigger_total else float("nan"),
        len(client_accuracies),
        clean_samples,
        trigger_total,
    )
