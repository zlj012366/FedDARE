from __future__ import annotations

import copy
import csv
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .attacks import UploadAttack
from .behavior import BehaviorEvaluator
from .data import PoisonedDataset, build_federated_data
from .local import FederatedClient
from .metrics import evaluate_personalized_models
from .models import SemanticEncoder, balanced_assignments, build_client_model
from .state import clone_state, weighted_average


@dataclass
class RoundRecord:
    round: int
    selected: int
    retained: int
    malicious_selected: int
    malicious_retained: int
    malicious_filtered: int
    benign_selected: int
    benign_filtered: int
    detection_rate: float
    false_positive_rate: float
    mean_deviation: float
    max_deviation: float
    phase_i_loss: float
    phase_ii_loss: float
    test_accuracy: float
    attack_success_rate: float
    round_seconds: float


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


class FedDARERunner:
    def __init__(self, config) -> None:
        self.config = config
        set_seed(config.runtime.seed, config.runtime.deterministic)
        self.device = resolve_device(config.runtime.device)
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = build_federated_data(
            config.dataset, config.federation.num_clients, config.runtime.seed
        )
        self.malicious_ids = self._choose_malicious_clients()
        self.clients = self._build_clients()
        self.encoder = SemanticEncoder(
            self.data.meta.channels,
            config.encoder.hidden_channels,
            config.encoder.first_kernel,
            config.encoder.second_kernel,
            config.encoder.eca_kernel,
        ).cpu()
        self.behavior = (
            BehaviorEvaluator(
                copy.deepcopy(self.encoder),
                self.data.meta,
                config.federation.num_clients,
                config.behavior,
                self.device,
            )
            if config.behavior.enabled
            else None
        )
        self.attack = UploadAttack(config.attack, config.runtime.seed)
        self.records = []
        self._write_run_metadata()

    def _choose_malicious_clients(self):
        requested = (
            self.config.federation.num_clients
            * self.config.federation.malicious_ratio
        )
        count = int(math.floor(requested))
        if self.config.federation.malicious_ratio > 0 and count == 0:
            count = 1
        if not count:
            return set()
        rng = np.random.default_rng(self.config.runtime.seed + 23)
        return set(
            rng.choice(self.config.federation.num_clients, count, replace=False).tolist()
        )

    def _build_clients(self):
        model_config = self.config.model
        variants = (
            model_config.cnn_variants
            if model_config.family == "cnn"
            else model_config.resnet_variants
        )
        self.model_assignments = balanced_assignments(
            self.config.federation.num_clients,
            variants,
            model_config.heterogeneous,
            self.config.runtime.seed + 31,
        )
        data_attacks = {"label_flip", "scaling", "dba", "adaptive_scaling"}
        malicious_order = {
            client_id: index for index, client_id in enumerate(sorted(self.malicious_ids))
        }
        clients = []
        for client_id, split in enumerate(self.data.clients):
            train_data = split.train
            if self.config.attack.name in data_attacks and client_id in self.malicious_ids:
                piece = (
                    malicious_order[client_id] % 4
                    if self.config.attack.name == "dba"
                    else None
                )
                train_data = PoisonedDataset(
                    train_data,
                    self.config.attack.name,
                    self.data.meta.num_classes,
                    self.config.attack.target_label,
                    self.config.attack.poison_fraction,
                    self.data.meta.normalized_upper,
                    self.config.runtime.seed + client_id,
                    piece,
                )
            model = build_client_model(
                model_config.family,
                self.model_assignments[client_id],
                self.data.meta.channels,
                self.data.meta.image_size,
                self.data.meta.num_classes,
                model_config.cifar_resnet_stem,
            )
            clients.append(
                FederatedClient(
                    client_id,
                    model,
                    train_data,
                    split.validation,
                    split.test,
                    self.config.training,
                )
            )
        return clients

    def _environment_manifest(self):
        import torchvision

        cuda_name = None
        if self.device.type == "cuda":
            cuda_name = torch.cuda.get_device_name(self.device)
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(self.device),
            "gpu": cuda_name,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }

    def _partition_manifest(self):
        clients = []
        for client_id, split in enumerate(self.data.clients):
            train_data = self.clients[client_id].train_data
            poisoned_positions = (
                sorted(int(value) for value in train_data.poisoned)
                if isinstance(train_data, PoisonedDataset)
                else []
            )
            clients.append(
                {
                    "client_id": client_id,
                    "model_variant": self.model_assignments[client_id],
                    "malicious": client_id in self.malicious_ids,
                    "train_indices": [int(value) for value in split.train.indices],
                    "validation_indices": [
                        int(value) for value in split.validation.indices
                    ],
                    "test_indices": [int(value) for value in split.test.indices],
                    "poisoned_train_positions": poisoned_positions,
                }
            )
        return {
            "dataset": self.data.meta.name,
            "num_clients": self.config.federation.num_clients,
            "requested_malicious_ratio": self.config.federation.malicious_ratio,
            "malicious_count": len(self.malicious_ids),
            "effective_malicious_ratio": (
                len(self.malicious_ids) / self.config.federation.num_clients
            ),
            "clients": clients,
        }

    def _write_run_metadata(self) -> None:
        manifests = {
            "resolved_config.json": self.config.to_dict(),
            "environment.json": self._environment_manifest(),
            "partition_manifest.json": self._partition_manifest(),
        }
        for filename, payload in manifests.items():
            with (self.output_dir / filename).open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)

    def _select_clients(self, round_index):
        count = max(
            1,
            int(
                round(
                    self.config.federation.num_clients
                    * self.config.federation.participation_rate
                )
            ),
        )
        rng = np.random.default_rng(self.config.runtime.seed + 1009 * round_index)
        return sorted(
            rng.choice(self.config.federation.num_clients, count, replace=False).tolist()
        )

    def _evaluate(self):
        return asdict(
            evaluate_personalized_models(
                self.clients,
                self.malicious_ids,
                self.config.runtime.evaluate_benign_only,
                self.config.attack.name,
                self.config.attack.target_label,
                self.data.meta,
                self.config.training.batch_size,
                self.config.training.num_workers,
                self.device,
            )
        )

    def _write_records(self):
        if not self.records:
            return
        with (self.output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(asdict(self.records[0])))
            writer.writeheader()
            writer.writerows(asdict(record) for record in self.records)

    def _save_checkpoint(self, global_state) -> None:
        if not self.config.runtime.save_checkpoint:
            return
        payload = {
            "encoder": clone_state(global_state),
            "malicious_ids": sorted(self.malicious_ids),
            "model_assignments": self.model_assignments,
            "config": self.config.to_dict(),
        }
        if self.config.runtime.save_client_models:
            payload["client_models"] = {
                client.client_id: clone_state(client.model.state_dict())
                for client in self.clients
            }
        torch.save(payload, self.output_dir / "final_checkpoint.pt")

    def run(self):
        global_state = clone_state(self.encoder.state_dict())
        progress = tqdm(range(self.config.federation.rounds), desc="FedDARE rounds")
        for round_index in progress:
            started = time.perf_counter()
            selected = self._select_clients(round_index)
            uploads = []
            phase_i_losses = []
            phase_ii_losses = []
            for client_id in selected:
                local = self.clients[client_id].update(
                    self.encoder, round_index, self.device
                )
                uploaded = self.attack.apply(
                    client_id,
                    round_index,
                    global_state,
                    local.encoder_state,
                    client_id in self.malicious_ids,
                )
                deviation = (
                    self.behavior.evaluate(client_id, global_state, uploaded).deviation
                    if self.behavior
                    else 0.0
                )
                uploads.append(
                    {
                        "client_id": client_id,
                        "state": uploaded,
                        "samples": local.train_samples,
                        "deviation": deviation,
                    }
                )
                phase_i_losses.append(local.phase_i_loss)
                phase_ii_losses.append(local.phase_ii_loss)

            if not self.behavior or round_index < self.config.behavior.warmup_rounds:
                retained = uploads
            else:
                retained = [
                    upload
                    for upload in uploads
                    if upload["deviation"] <= self.config.behavior.threshold
                ]
            if retained:
                global_state = weighted_average(
                    [upload["state"] for upload in retained],
                    [upload["samples"] for upload in retained],
                )
                self.encoder.load_state_dict(global_state)

            should_evaluate = (
                (round_index + 1) % self.config.runtime.eval_every == 0
                or round_index + 1 == self.config.federation.rounds
            )
            evaluation = (
                self._evaluate()
                if should_evaluate
                else {"test_accuracy": float("nan"), "attack_success_rate": float("nan")}
            )
            retained_ids = {upload["client_id"] for upload in retained}
            selected_malicious = set(selected) & self.malicious_ids
            retained_malicious = retained_ids & self.malicious_ids
            malicious_selected = len(selected_malicious)
            malicious_retained = len(retained_malicious)
            malicious_filtered = malicious_selected - malicious_retained
            benign_selected = len(selected) - malicious_selected
            benign_retained = len(retained) - malicious_retained
            benign_filtered = benign_selected - benign_retained
            deviations = [upload["deviation"] for upload in uploads]
            record = RoundRecord(
                round_index + 1,
                len(selected),
                len(retained),
                malicious_selected,
                malicious_retained,
                malicious_filtered,
                benign_selected,
                benign_filtered,
                (
                    100.0 * malicious_filtered / malicious_selected
                    if malicious_selected
                    else float("nan")
                ),
                (
                    100.0 * benign_filtered / benign_selected
                    if benign_selected
                    else float("nan")
                ),
                float(np.mean(deviations)),
                float(np.max(deviations)),
                float(np.mean(phase_i_losses)),
                float(np.mean(phase_ii_losses)),
                float(evaluation["test_accuracy"]),
                float(evaluation["attack_success_rate"]),
                time.perf_counter() - started,
            )
            self.records.append(record)
            self._write_records()
            progress.set_postfix(
                retained=len(retained),
                tacc=(
                    f"{record.test_accuracy:.2f}"
                    if not np.isnan(record.test_accuracy)
                    else "-"
                ),
            )

        self._save_checkpoint(global_state)
        return self.records
