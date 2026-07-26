from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

TensorState = Mapping[str, torch.Tensor]


def clone_state(state: TensorState, device: str = "cpu"):
    return OrderedDict(
        (name, value.detach().to(device).clone()) for name, value in state.items()
    )


def weighted_average(states: Sequence[TensorState], weights: Sequence[float]):
    if not states or len(states) != len(weights):
        raise ValueError("states and weights must have the same non-zero length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("aggregation weights must sum to a positive value")
    result = OrderedDict()
    for name in states[0]:
        reference = states[0][name]
        if not torch.is_floating_point(reference):
            result[name] = reference.clone()
            continue
        value = torch.zeros_like(reference)
        for state, weight in zip(states, weights):
            value.add_(state[name], alpha=float(weight) / total)
        result[name] = value
    return result


def scale_update(global_state: TensorState, client_state: TensorState, factor: float):
    return OrderedDict(
        (
            name,
            global_value + factor * (client_state[name] - global_value)
            if torch.is_floating_point(global_value)
            else client_state[name].clone(),
        )
        for name, global_value in global_state.items()
    )


def flatten_update(global_state: TensorState, client_state: TensorState) -> torch.Tensor:
    return torch.cat(
        [
            (client_state[name] - global_value).reshape(-1)
            for name, global_value in global_state.items()
            if torch.is_floating_point(global_value)
        ]
    )


def state_from_update_vector(global_state: TensorState, vector: torch.Tensor):
    offset = 0
    result = OrderedDict()
    for name, global_value in global_state.items():
        if not torch.is_floating_point(global_value):
            result[name] = global_value.clone()
            continue
        count = global_value.numel()
        result[name] = global_value + vector[offset : offset + count].reshape_as(global_value)
        offset += count
    if offset != vector.numel():
        raise ValueError("Update vector size does not match encoder state")
    return result

