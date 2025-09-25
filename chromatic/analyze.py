from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

from chromatic.data import get_probe_loader


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    # probs: (N, C) softmax; labels: (N,)
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = probs.shape[0]
    for b in range(n_bins):
        l, r = bins[b], bins[b + 1]
        mask = (conf > l) & (conf <= r) if b > 0 else (conf >= l) & (conf <= r)
        if not np.any(mask):
            continue
        acc_bin = np.mean((preds[mask] == labels[mask]).astype(np.float32))
        conf_bin = np.mean(conf[mask])
        w = np.mean(mask.astype(np.float32))
        ece += w * abs(acc_bin - conf_bin)
    return float(ece)


def load_probe_logits(artifacts_dir: Path) -> Tuple[List[int], List[np.ndarray]]:
    probe_dir = artifacts_dir / "probes"
    paths = sorted(probe_dir.glob("probe_logits_seed*.pt"))
    seeds = []
    logits = []
    for p in paths:
        name = p.stem
        seed = int(name.split("seed")[-1])
        seeds.append(seed)
        logits.append(torch.load(p, map_location="cpu").numpy())
    return seeds, logits


def compute_confusions(logits_list: List[np.ndarray], labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    confs = []
    for logits in logits_list:
        preds = logits.argmax(axis=1)
        cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
        # row-normalize (per true class)
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sums
        confs.append(cm)
    return np.stack(confs, axis=0)  # (S, C, C)


def analyze(out_dir: Path = Path("artifacts"), data_root: Path = Path("./data")) -> Dict:
    seeds, logits_list = load_probe_logits(out_dir)
    n_models = len(logits_list)
    if n_models < 2:
        raise RuntimeError("Need at least two probe files to analyze")

    # Load labels in the same order as probe set
    loader = get_probe_loader("cifar10", str(data_root), batch_size=256)
    ys = []
    for _, targets in loader:
        ys.append(targets.numpy())
    labels = np.concatenate(ys, axis=0)

    # Metrics per model
    avg_conf = []
    eces = []
    accs = []
    logit_norm = []
    for logits in logits_list:
        probs = softmax_numpy(logits)
        avg_conf.append(float(probs.max(axis=1).mean()))
        eces.append(expected_calibration_error(probs, labels, n_bins=15))
        preds = probs.argmax(axis=1)
        accs.append(float((preds == labels).mean()))
        logit_norm.append(float(np.linalg.norm(logits, axis=1).mean()))

    # Confusion PCA
    confs = compute_confusions(logits_list, labels, num_classes=10)
    conf_vecs = confs.reshape(n_models, -1)
    pca = PCA(n_components=min(5, n_models))
    pc = pca.fit_transform(conf_vecs)
    pc1 = pc[:, 0]

    # MDS (classical) from saved distance matrix
    dist = np.load(out_dir / "embeddings" / "distance_matrix.npy")
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Coordinates
    pos_eig = np.clip(eigvals, a_min=0.0, a_max=None)
    coords = eigvecs * np.sqrt(pos_eig + 1e-12)
    mds1 = coords[:, 0]
    mds2 = coords[:, 1] if coords.shape[1] > 1 else np.zeros_like(mds1)

    # Correlations
    def corr(a, b):
        r, p = pearsonr(np.asarray(a), np.asarray(b))
        return float(r), float(p)

    r1_conf, p1_conf = corr(mds1, avg_conf)
    r1_ece, p1_ece = corr(mds1, eces)
    r2_pc1, p2_pc1 = corr(mds2, pc1)
    r2_conf, p2_conf = corr(mds2, avg_conf)

    # Explained variance by top components
    total = float(pos_eig.sum() + 1e-12)
    evr1 = float(pos_eig[0] / total)
    evr2 = float(pos_eig[1] / total) if len(pos_eig) > 1 else 0.0

    report = {
        "seeds": seeds,
        "metrics": {
            "accuracy": accs,
            "avg_confidence": avg_conf,
            "ece": eces,
            "logit_norm": logit_norm,
            "confusion_pc1": pc1.tolist(),
        },
        "mds": {
            "eigvals": pos_eig.tolist(),
            "explained_var_top2": [evr1, evr2],
            "coords_top2": np.stack([mds1, mds2], axis=1).tolist(),
        },
        "correlations": {
            "mds1_vs_avg_conf": [r1_conf, p1_conf],
            "mds1_vs_ece": [r1_ece, p1_ece],
            "mds2_vs_confusion_pc1": [r2_pc1, p2_pc1],
            "mds2_vs_avg_conf": [r2_conf, p2_conf],
        },
    }

    out_path = Path(out_dir) / "reports" / "analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({
        "explained_var_top2": [evr1, evr2],
        "mds1_vs_avg_conf": [r1_conf, p1_conf],
        "mds1_vs_ece": [r1_ece, p1_ece],
        "mds2_vs_confusion_pc1": [r2_pc1, p2_pc1],
    }))

    return report


if __name__ == "__main__":
    analyze()


