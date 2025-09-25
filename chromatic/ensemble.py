from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tabulate import tabulate

from chromatic.data import build_dataloaders, get_test_loader
from chromatic.models import SmallCIFARNet
from chromatic.utils import select_device


def _eval_model(ckpt_path: str, dataset: str, data_root: str, device: str, fast: bool) -> float:
    dev = select_device(device)
    test_loader = get_test_loader(dataset, data_root, batch_size=256, fast=fast)
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
    test_loader = get_test_loader(dataset, data_root, batch_size=256, fast=fast)
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

    # Additional subsets: random-k, top-k by individual accuracy, and farthest-first by distance matrix
    report_dir = Path(out_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load distance matrix (prob_mse default) and implement farthest-first
    import numpy as np
    dist_path = Path(out_dir) / "embeddings" / "distance_matrix.npy"
    if dist_path.exists():
        dist = np.load(dist_path)
    else:
        dist = None

    def farthest_first_indices(k: int) -> List[int]:
        if dist is None:
            return list(range(min(k, len(ckpts))))
        n = dist.shape[0]
        if n == 0:
            return []
        selected = [int(np.argmax(np.sum(dist, axis=1)))]
        while len(selected) < min(k, n):
            rem = [i for i in range(n) if i not in selected]
            scores = []
            for i in rem:
                scores.append(np.min([dist[i, j] for j in selected]))
            selected.append(rem[int(np.argmax(scores))])
        return selected

    def topk_indices_by_individual(k: int) -> List[int]:
        order = np.argsort(indiv)[::-1]
        return [int(i) for i in order[:k]]

    def randomk_indices(k: int, seed: int = 0) -> List[int]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(ckpts))
        rng.shuffle(idx)
        return [int(i) for i in idx[:k]]

    subset_results = {}
    for k in [2, 3, 5, min(8, len(ckpts))]:
        if k < 2:
            continue
        ff_idx = farthest_first_indices(k)
        tk_idx = topk_indices_by_individual(k)
        rk_idx = randomk_indices(k, seed=42)
        subset_results[f"farthest_first_k{k}"] = _eval_ensemble([ckpts[i] for i in ff_idx], dataset, data_root, device, fast)
        subset_results[f"topk_k{k}"] = _eval_ensemble([ckpts[i] for i in tk_idx], dataset, data_root, device, fast)
        subset_results[f"random_k{k}"] = _eval_ensemble([ckpts[i] for i in rk_idx], dataset, data_root, device, fast)

    table = [(seeds[i], indiv[i]) for i in range(len(seeds))]
    table.append(("ensemble", ens_all))

    report_dir = Path(out_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ensemble.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(tabulate(table, headers=["seed", "accuracy"], floatfmt=".4f"))
        f.write("\n\n")
        f.write("subset ensembles (accuracy):\n")
        for k in [2, 3, 5, min(8, len(ckpts))]:
            if k < 2:
                continue
            f.write(f"farthest-first k={k}: {subset_results.get(f'farthest_first_k{k}', float('nan')):.4f}\n")
            f.write(f"top-k k={k}: {subset_results.get(f'topk_k{k}', float('nan')):.4f}\n")
            f.write(f"random k={k}: {subset_results.get(f'random_k{k}', float('nan')):.4f}\n")

    return {"individual": dict(zip(seeds, indiv)), "ensemble_all": ens_all, "subset_ensembles": subset_results, "report": str(report_path)}


