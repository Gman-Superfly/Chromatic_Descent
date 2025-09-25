from __future__ import annotations

from typing import Tuple, List

import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as T


def build_transforms(dataset: str):
    if dataset == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        test_tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        return train_tf, test_tf
    raise ValueError(f"Unsupported dataset {dataset}")


def _get_cifar10_datasets(data_root: str, train_tf, test_tf):
    train_full = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    train_eval = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=test_tf)
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    return train_full, train_eval, test


def _stratified_indices(targets: List[int], per_class_counts: List[int], rng: np.random.Generator) -> List[int]:
    indices_per_class: List[List[int]] = [[] for _ in range(10)]
    for idx, y in enumerate(targets):
        indices_per_class[y].append(idx)
    selected: List[int] = []
    for cls in range(10):
        cls_indices = np.array(indices_per_class[cls], dtype=np.int64)
        rng.shuffle(cls_indices)
        take = min(per_class_counts[cls], cls_indices.shape[0])
        selected.extend(cls_indices[:take].tolist())
    rng.shuffle(selected)
    return selected


def _deterministic_train_val_indices(targets: List[int], fast: bool) -> Tuple[List[int], List[int]]:
    # Create a deterministic stratified split of the CIFAR-10 training set.
    # Validation (probe) size ~10% (5k) in full mode, ~1k in fast mode.
    rng = np.random.default_rng(12345)
    num_classes = 10
    n = len(targets)
    if fast:
        val_per_class = [100] * num_classes  # ~1000 total
        train_per_class = [409] * num_classes  # ~4090 total
        # Build indices per class then cap to desired counts
        val_idx = _stratified_indices(targets, val_per_class, rng)
        # Mask out validation indices
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        remaining = [i for i in range(n) if mask[i]]
        # Build per-class pools from remaining and sample train_per_class
        rem_targets = [targets[i] for i in remaining]
        train_idx_rel = _stratified_indices(rem_targets, train_per_class, rng)
        train_idx = [remaining[i] for i in train_idx_rel]
        return train_idx, val_idx
    else:
        # 5k validation (500 per class), rest for training
        val_per_class = [500] * num_classes
        val_idx = _stratified_indices(targets, val_per_class, rng)
        mask = np.ones(n, dtype=bool)
        mask[val_idx] = False
        train_idx = [i for i in range(n) if mask[i]]
        return train_idx, val_idx


def build_dataloaders(dataset: str, data_root: str, batch_size: int, fast: bool) -> Tuple[data.DataLoader, data.DataLoader]:
    train_tf, test_tf = build_transforms(dataset)
    if dataset != "cifar10":
        raise ValueError(f"Unsupported dataset {dataset}")

    train_full, train_eval, _ = _get_cifar10_datasets(data_root, train_tf, test_tf)
    # Deterministic stratified split into train/probe-val
    train_idx, val_idx = _deterministic_train_val_indices(train_eval.targets, fast)
    train_ds = data.Subset(train_full, train_idx)
    val_ds = data.Subset(train_eval, val_idx)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def get_probe_loader(dataset: str, data_root: str, batch_size: int) -> data.DataLoader:
    # Probe loader is the validation split carved out of the training set (no data leakage)
    train_tf, test_tf = build_transforms(dataset)
    if dataset != "cifar10":
        raise ValueError(f"Unsupported dataset {dataset}")
    _, train_eval, _ = _get_cifar10_datasets(data_root, train_tf, test_tf)
    _, val_idx = _deterministic_train_val_indices(train_eval.targets, fast=False)
    probe = data.Subset(train_eval, val_idx)
    return data.DataLoader(probe, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def get_test_loader(dataset: str, data_root: str, batch_size: int, fast: bool) -> data.DataLoader:
    _, test_tf = build_transforms(dataset)
    if dataset != "cifar10":
        raise ValueError(f"Unsupported dataset {dataset}")
    test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    if fast:
        test = data.Subset(test, list(range(2048)))
    return data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


