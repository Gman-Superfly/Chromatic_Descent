from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from chromatic.data import build_dataloaders
from chromatic.models import SmallCIFARNet
from chromatic.utils import select_device


def _eval_model(ckpt_path: str, dataset: str, data_root: str, device: str, fast: bool) -> float:
    dev = select_device(device)
    _, test_loader = build_dataloaders(dataset, data_root, batch_size=256, fast=fast)
    model = SmallCIFARNet(num_classes=10).to(dev)
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(dev)
            targets = targets.to(dev)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(1, total)


def _eval_ensemble(ckpt_paths: List[str], dataset: str, data_root: str, device: str, fast: bool) -> float:
    dev = select_device(device)
    _, test_loader = build_dataloaders(dataset, data_root, batch_size=256, fast=fast)
    models = []
    for p in ckpt_paths:
        m = SmallCIFARNet(num_classes=10).to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        models.append(m)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(dev)
            targets = targets.to(dev)
            logits_sum = None
            for m in models:
                logits = m(images)
                logits_sum = logits if logits_sum is None else logits_sum + logits
            preds = logits_sum.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / max(1, total)


def evaluate_ensembles(run_infos: List[Dict], out_dir: Path) -> Dict:
    dataset = "cifar10"
    data_root = "./data"
    device = "auto"
    fast = False

    ckpts = [r["ckpt_path"] for r in run_infos]
    seeds = [r["seed"] for r in run_infos]

    indiv = [_eval_model(p, dataset, data_root, device, fast) for p in ckpts]
    ens_all = _eval_ensemble(ckpts, dataset, data_root, device, fast)

    table = [(seeds[i], indiv[i]) for i in range(len(seeds))]
    table.append(("ensemble", ens_all))

    report_dir = Path(out_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ensemble.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tabulate(table, headers=["seed", "accuracy"], floatfmt=".4f"))

    return {"individual": dict(zip(seeds, indiv)), "ensemble_all": ens_all, "report": str(report_path)}


