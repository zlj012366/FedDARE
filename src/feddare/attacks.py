from __future__ import annotations

from collections import defaultdict

import torch

from .config import AttackConfig
from .state import (
    flatten_update,
    scale_update,
    state_from_update_vector,
)


class UploadAttack:
    def __init__(self, config: AttackConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.history = defaultdict(list)

    def _gaussian(self, client_id, round_index, global_state, local_state):
        update = flatten_update(global_state, local_state)
        generator = torch.Generator().manual_seed(
            self.seed + 1000003 * round_index + 9176 * client_id
        )
        noise = torch.randn(update.shape, generator=generator, dtype=update.dtype)
        corrupted = update + self.config.noise_std * update.std().clamp_min(1e-12) * noise
        return state_from_update_vector(global_state, corrupted)

    def _adaptive(self, client_id, round_index, global_state, local_state):
        progress = min(1.0, (round_index + 1) / max(self.config.adaptive_ramp_rounds, 1))
        coefficient = progress * self.config.adaptive_max_scale
        proposed = flatten_update(
            global_state, scale_update(global_state, local_state, coefficient)
        )
        history = self.history[client_id]
        if history:
            center = torch.stack(history[-self.config.adaptive_history :]).mean(dim=0)
            displacement = proposed - center
            allowed = self.config.adaptive_radius * center.norm().clamp_min(1e-12)
            if displacement.norm() > allowed:
                proposed = center + displacement * (allowed / displacement.norm())
        history.append(proposed.detach().clone())
        return state_from_update_vector(global_state, proposed)

    def apply(
        self, client_id, round_index, global_state, local_state, malicious: bool
    ):
        if not malicious or self.config.name in {"none", "label_flip"}:
            return dict(local_state)
        if self.config.name == "sign_flip":
            return scale_update(global_state, local_state, -self.config.scale)
        if self.config.name in {"scaling", "dba"}:
            return scale_update(global_state, local_state, self.config.scale)
        if self.config.name == "gaussian":
            return self._gaussian(client_id, round_index, global_state, local_state)
        if self.config.name == "adaptive_scaling":
            return self._adaptive(client_id, round_index, global_state, local_state)
        raise ValueError(f"Unsupported upload attack: {self.config.name}")

