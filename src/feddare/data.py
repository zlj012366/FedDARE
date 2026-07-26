from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset

from .config import DatasetConfig


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    channels: int
    image_size: int
    num_classes: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]

    @property
    def normalized_lower(self) -> torch.Tensor:
        return torch.tensor(
            [(0.0 - mean) / std for mean, std in zip(self.mean, self.std)]
        )

    @property
    def normalized_upper(self) -> torch.Tensor:
        return torch.tensor(
            [(1.0 - mean) / std for mean, std in zip(self.mean, self.std)]
        )


@dataclass
class ClientSplit:
    train: Dataset
    validation: Dataset
    test: Dataset


@dataclass
class FederatedData:
    clients: List[ClientSplit]
    meta: DatasetMeta


DATASET_META = {
    "mnist": DatasetMeta("mnist", 1, 28, 10, (0.1307,), (0.3081,)),
    "cifar10": DatasetMeta(
        "cifar10", 3, 32, 10,
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616),
    ),
    "cifar100": DatasetMeta(
        "cifar100", 3, 32, 100,
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761),
    ),
}


def _load_torchvision_dataset(config: DatasetConfig):
    from torchvision import datasets, transforms

    meta = DATASET_META[config.name]
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(meta.mean, meta.std)]
    )
    factories = {
        "mnist": datasets.MNIST,
        "cifar10": datasets.CIFAR10,
        "cifar100": datasets.CIFAR100,
    }
    dataset = factories[config.name](
        root=config.root, train=True, transform=transform, download=config.download
    )
    return dataset, np.asarray(dataset.targets, dtype=np.int64), meta


def _load_synthetic_dataset(config: DatasetConfig, seed: int):
    meta = DatasetMeta(
        "synthetic",
        config.synthetic_channels,
        config.synthetic_image_size,
        config.synthetic_num_classes,
        tuple([0.5] * config.synthetic_channels),
        tuple([0.25] * config.synthetic_channels),
    )
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(
        0, meta.num_classes, (config.synthetic_train_size,), generator=generator
    )
    images = torch.randn(
        config.synthetic_train_size,
        meta.channels,
        meta.image_size,
        meta.image_size,
        generator=generator,
    ) * 0.15
    for index, label in enumerate(labels.tolist()):
        row = (label * 3) % max(1, meta.image_size - 4)
        column = (label * 5) % max(1, meta.image_size - 4)
        images[index, :, row : row + 4, column : column + 4] += 1.5
    return TensorDataset(images, labels), labels.numpy(), meta


def iid_partition(num_samples: int, num_clients: int, seed: int) -> List[np.ndarray]:
    indices = np.random.default_rng(seed).permutation(num_samples)
    return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]


def dirichlet_partition(
    targets: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_client_samples: int,
    max_attempts: int = 1000,
) -> List[np.ndarray]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        buckets: List[List[int]] = [[] for _ in range(num_clients)]
        for class_id in np.unique(targets):
            class_indices = np.flatnonzero(targets == class_id)
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            capacities = np.asarray(
                [len(bucket) < len(targets) / num_clients for bucket in buckets],
                dtype=np.float64,
            )
            proportions *= capacities
            proportions = (
                proportions / proportions.sum()
                if proportions.sum()
                else np.full(num_clients, 1 / num_clients)
            )
            cuts = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_id, shard in enumerate(np.split(class_indices, cuts)):
                buckets[client_id].extend(shard.tolist())
        if min(map(len, buckets)) >= min_client_samples:
            result = []
            for bucket in buckets:
                rng.shuffle(bucket)
                result.append(np.asarray(bucket, dtype=np.int64))
            return result
    raise RuntimeError(
        "Could not satisfy min_client_samples; lower it or reduce num_clients"
    )


def _three_way_split(
    indices: np.ndarray, ratios: Sequence[float], rng: np.random.Generator
):
    indices = indices.copy()
    rng.shuffle(indices)
    train_end = int(len(indices) * ratios[0])
    validation_end = train_end + int(len(indices) * ratios[1])
    if train_end == 0 or validation_end == train_end or validation_end == len(indices):
        raise RuntimeError(f"Cannot make a non-empty 8:1:1 split from {len(indices)} samples")
    return indices[:train_end], indices[train_end:validation_end], indices[validation_end:]


def build_federated_data(
    config: DatasetConfig, num_clients: int, seed: int
) -> FederatedData:
    if config.name == "synthetic":
        dataset, targets, meta = _load_synthetic_dataset(config, seed)
    else:
        dataset, targets, meta = _load_torchvision_dataset(config)
    partitions = (
        iid_partition(len(dataset), num_clients, seed)
        if config.iid
        else dirichlet_partition(
            targets, num_clients, config.dirichlet_alpha,
            seed, config.min_client_samples,
        )
    )
    rng = np.random.default_rng(seed + 1)
    clients = []
    for indices in partitions:
        train_ids, validation_ids, test_ids = _three_way_split(indices, config.split, rng)
        clients.append(
            ClientSplit(
                Subset(dataset, train_ids.tolist()),
                Subset(dataset, validation_ids.tolist()),
                Subset(dataset, test_ids.tolist()),
            )
        )
    return FederatedData(clients, meta)


class PoisonedDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        mode: str,
        num_classes: int,
        target_label: int,
        poison_fraction: float,
        upper_bound: torch.Tensor,
        seed: int,
        dba_piece: Optional[int] = None,
    ) -> None:
        self.base = base
        self.mode = mode
        self.num_classes = num_classes
        self.target_label = target_label
        self.upper_bound = upper_bound
        self.dba_piece = dba_piece
        count = min(int(round(len(base) * poison_fraction)), len(base))
        self.poisoned = set(
            np.random.default_rng(seed).choice(len(base), count, replace=False).tolist()
        )

    def __len__(self) -> int:
        return len(self.base)

    def _stamp(self, image: torch.Tensor) -> torch.Tensor:
        image = image.clone()
        height, width = image.shape[-2:]
        if self.dba_piece is None:
            points = [
                (height - 4 + row, width - 4 + column)
                for row in range(3) for column in range(3)
            ]
        else:
            base_row, base_column = [(0, 0), (0, 2), (2, 0), (2, 2)][
                self.dba_piece % 4
            ]
            points = [
                (height - 4 + base_row + row, width - 4 + base_column + column)
                for row in range(2) for column in range(2)
            ]
        for row, column in points:
            image[:, row, column] = self.upper_bound
        return image

    def __getitem__(self, index: int):
        image, label = self.base[index]
        label = int(label)
        if index not in self.poisoned:
            return image, label
        if self.mode == "label_flip":
            return image, self.num_classes - 1 - label
        if self.mode in {"scaling", "dba", "adaptive_scaling"}:
            return self._stamp(image), self.target_label
        return image, label

