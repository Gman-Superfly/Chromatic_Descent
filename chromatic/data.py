from __future__ import annotations

from typing import Tuple

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


def build_dataloaders(dataset: str, data_root: str, batch_size: int, fast: bool) -> Tuple[data.DataLoader, data.DataLoader]:
    train_tf, test_tf = build_transforms(dataset)
    if dataset == "cifar10":
        train = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    if fast:
        # Use a small subset for quick iterations
        train = data.Subset(train, list(range(4096)))
        test = data.Subset(test, list(range(2048)))

    train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def get_probe_loader(dataset: str, data_root: str, batch_size: int) -> data.DataLoader:
    _, test_tf = build_transforms(dataset)
    if dataset == "cifar10":
        probe = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")
    # Fixed order, no shuffle
    return data.DataLoader(probe, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


