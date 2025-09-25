from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def _load_logits(paths: List[str]) -> List[torch.Tensor]:
    return [torch.load(p, map_location="cpu") for p in paths]


def function_space_distance_matrix(logits: List[torch.Tensor]) -> np.ndarray:
    n = len(logits)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            # Mean squared difference of logits on the probe set
            d = torch.mean((logits[i] - logits[j]) ** 2).item()
            mat[i, j] = mat[j, i] = d
    return mat


def build_embeddings(run_infos: List[Dict], out_dir: Path) -> Dict:
    probe_paths = [r["probe_logits_path"] for r in run_infos]
    logits = _load_logits(probe_paths)
    dist = function_space_distance_matrix(logits)

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


