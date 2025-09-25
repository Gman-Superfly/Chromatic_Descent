from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from chromatic.data import build_dataloaders, get_probe_loader
from chromatic.models import SmallCIFARNet
from chromatic.utils import set_seed, select_device, save_checkpoint


@dataclass
class RunInfo:
    seed: int
    run_dir: str
    ckpt_path: str
    probe_logits_path: str
    val_acc: float


def compute_repel_loss(current_logits: torch.Tensor, anchor_logits: List[torch.Tensor]) -> torch.Tensor:
    if len(anchor_logits) == 0:
        return torch.zeros((), device=current_logits.device)
    # Mean-squared similarity (negative diversity). Encourage being different.
    losses = []
    for anchor in anchor_logits:
        # Normalize to reduce scale effects
        a = nn.functional.normalize(anchor, dim=-1)
        c = nn.functional.normalize(current_logits, dim=-1)
        losses.append((c - a).pow(2).mean())
    return torch.stack(losses).mean()


def train_one(
    seed: int,
    dataset: str,
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    device: str,
    fast: bool,
    repel: bool,
    repel_lambda: float,
    repel_warmup: int,
    runs_dir: Path,
    out_dir: Path,
    anchors_probe_logits: List[torch.Tensor],
) -> RunInfo:
    set_seed(seed)
    dev = select_device(device)

    run_dir = runs_dir / f"SEED_{seed:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(dataset, data_root, batch_size, fast)
    probe_loader = get_probe_loader(dataset, data_root, batch_size)

    model = SmallCIFARNet(num_classes=10).to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_ckpt = run_dir / "model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch}/{epochs}")
        for images, targets in pbar:
            images = images.to(dev)
            targets = targets.to(dev)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)

            if repel and epoch > repel_warmup and len(anchors_probe_logits) > 0:
                # compute current probe logits on a small batch to estimate divergence
                model.eval()
                with torch.no_grad():
                    probe_images, _ = next(iter(probe_loader))
                    probe_images = probe_images.to(dev)
                    current_probe_logits = model(probe_images)
                model.train()
                repel_loss = compute_repel_loss(current_probe_logits.detach(), [a.to(dev) for a in anchors_probe_logits])
                loss = loss + repel_lambda * repel_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(dev)
                targets = targets.to(dev)
                logits = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        acc = correct / max(1, total)
        if acc > best_acc:
            best_acc = acc
            save_checkpoint(model, path=str(best_ckpt))

    # Reload best
    model.load_state_dict(torch.load(best_ckpt, map_location=dev))

    # Export probe logits on the full probe set
    model.eval()
    probe_logits = []
    with torch.no_grad():
        for images, _ in probe_loader:
            images = images.to(dev)
            logits = model(images)
            probe_logits.append(logits.cpu())
    probe_logits = torch.cat(probe_logits, dim=0)

    probes_dir = Path(out_dir) / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probes_dir / f"probe_logits_seed{seed}.pt"
    torch.save(probe_logits, probe_path)

    return RunInfo(
        seed=seed,
        run_dir=str(run_dir),
        ckpt_path=str(best_ckpt),
        probe_logits_path=str(probe_path),
        val_acc=float(best_acc),
    )


def train_many(
    dataset: str,
    data_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    momentum: float,
    seeds: int,
    device: str,
    fast: bool,
    repel: bool,
    repel_lambda: float,
    repel_warmup: int,
    runs_dir: Path,
    out_dir: Path,
) -> List[Dict]:
    anchors: List[torch.Tensor] = []
    infos: List[RunInfo] = []
    for i in range(seeds):
        seed = 13 + i
        info = train_one(
            seed=seed,
            dataset=dataset,
            data_root=data_root,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            device=device,
            fast=fast,
            repel=repel,
            repel_lambda=repel_lambda,
            repel_warmup=repel_warmup,
            runs_dir=runs_dir,
            out_dir=out_dir,
            anchors_probe_logits=anchors,
        )
        infos.append(info)
        # Add the finished run's probe logits as an anchor for subsequent runs
        anchors.append(torch.load(info.probe_logits_path))

    # Serialize for summary
    return [info.__dict__ for info in infos]


