from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from chromatic.data import get_probe_loader
from chromatic.utils import calibrate_logits_to_probs


def _load_logits(paths: List[str]) -> List[torch.Tensor]:
    return [torch.load(p, map_location="cpu") for p in paths]


def _softmax_numpy(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def _symmetric_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    d = 0.5 * (np.sum(p * (np.log(p) - np.log(q)), axis=1) + np.sum(q * (np.log(q) - np.log(p)), axis=1))
    return float(np.mean(d))


def function_space_distance_matrix(logits: List[torch.Tensor], labels: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    # Compute several metrics; choose prob MSE as the default for embedding
    logits_np = [t.numpy() for t in logits]
    # Per-model temperature scaling
    probs_list = []
    temps = []
    for log in logits_np:
        probs, T = calibrate_logits_to_probs(log, labels)
        probs_list.append(probs)
        temps.append(T)
    n = len(logits_np)
    d_logit = np.zeros((n, n), dtype=np.float64)
    d_prob = np.zeros((n, n), dtype=np.float64)
    d_symkl = np.zeros((n, n), dtype=np.float64)
    d_disagree = np.zeros((n, n), dtype=np.float64)
    preds = [p.argmax(axis=1) for p in probs_list]
    for i in range(n):
        for j in range(i + 1, n):
            d_logit[i, j] = d_logit[j, i] = float(np.mean((logits_np[i] - logits_np[j]) ** 2))
            d_prob[i, j] = d_prob[j, i] = float(np.mean((probs_list[i] - probs_list[j]) ** 2))
            d_symkl[i, j] = d_symkl[j, i] = _symmetric_kl(probs_list[i], probs_list[j])
            d_disagree[i, j] = d_disagree[j, i] = float(np.mean(preds[i] != preds[j]))
    return d_prob, {"logit_mse": d_logit, "prob_mse": d_prob, "sym_kl": d_symkl, "zero_one": d_disagree}


def build_embeddings(run_infos: List[Dict], out_dir: Path) -> Dict:
    probe_paths = [r["probe_logits_path"] for r in run_infos]
    logits = _load_logits(probe_paths)
    # Load labels in probe order
    # We rely on analyze to already compute labels; to keep this module self-contained, fetch labels via data loader
    from chromatic.data import get_probe_loader
    loader = get_probe_loader("cifar10", "./data", batch_size=256)
    ys = []
    for _, targets in loader:
        ys.append(targets.numpy())
    labels = np.concatenate(ys, axis=0)

    dist, metrics_all = function_space_distance_matrix(logits, labels)

    # Convert to similarity for TSNE via perplexity handling; TSNE works on features, so we embed with MDS-like trick
    # Here we use a simple spectral trick by double-centering the squared distances to get an inner-product matrix and then TSNE on that.
    # For small N, TSNE directly on distances is okay if we embed a pseudo-feature via classical MDS pre-step.
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    # Keep top components
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    k = min(8, n)
    X = eigvecs[:, idx[:k]] * np.sqrt(np.maximum(eigvals[idx[:k]], 0.0))

    # Robust small-N handling: for very few samples, skip t-SNE and use classical MDS (top-2 components)
    if n <= 3:
        if X.shape[1] < 2:
            # pad with zeros if rank-1 or rank-0
            pad = np.zeros((n, 2 - X.shape[1]))
            emb = np.concatenate([X, pad], axis=1)
        else:
            emb = X[:, :2]
    else:
        perp = float(min(30, n - 1))
        if perp >= n:
            perp = float(n - 1) - 1e-6
        perp = max(2.0, perp)
        tsne = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0)
        emb = tsne.fit_transform(X)

    emb_dir = Path(out_dir) / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(emb_dir / "distance_matrix.npy", dist)
    # Also save alternative metrics for downstream analysis
    for name, mat in metrics_all.items():
        np.save(emb_dir / f"distance_{name}.npy", mat)
    np.save(emb_dir / "embedding_tsne.npy", emb)

    # Plot
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=emb[:, 0], y=emb[:, 1])
    for i, r in enumerate(run_infos):
        plt.text(emb[i, 0], emb[i, 1], str(r["seed"]))
    plt.title("Chromatic (function-space) embedding")
    plt.tight_layout()
    plt.savefig(emb_dir / "embedding_tsne.png", dpi=160)
    plt.close()

    return {"distance_matrix": str(emb_dir / "distance_matrix.npy"), "embedding": str(emb_dir / "embedding_tsne.npy"), "plot": str(emb_dir / "embedding_tsne.png")}


