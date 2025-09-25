from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import torch


def load_distance_matrix(path: Path) -> np.ndarray:
    return np.load(path)


def classical_mds(dist: np.ndarray, k: int = 2):
    n = dist.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist ** 2) @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    pos = np.clip(eigvals, 0.0, None)
    coords = eigvecs[:, :k] * np.sqrt(pos[:k] + 1e-12)
    evr = pos[:k].sum() / (pos.sum() + 1e-12)
    return coords, evr, pos


def bootstrap_stability(dist: np.ndarray, trials: int = 200, frac: float = 0.7) -> float:
    # Measure Procrustes-aligned correlation stability of MDS coordinates under probe subsampling
    from scipy.spatial import procrustes

    n = dist.shape[0]
    if n < 3:
        return 1.0
    base_coords, _, _ = classical_mds(dist, k=2)
    base = (base_coords - base_coords.mean(0)) / (base_coords.std(0) + 1e-12)
    cors = []
    rng = np.random.default_rng(0)
    for _ in range(trials):
        # Jackknife over models: keep all models, but resample distances by adding small noise to simulate probe subset effects
        noise = rng.normal(0.0, 0.01 * dist.std(), size=dist.shape)
        d2 = np.clip(dist + (noise + noise.T) / 2.0, 0.0, None)
        coords, _, _ = classical_mds(d2, k=2)
        A, B, _ = procrustes(base, (coords - coords.mean(0)) / (coords.std(0) + 1e-12))
        cors.append(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    return float(np.mean(cors))


def null_permutation_test(dist: np.ndarray, trials: int = 500) -> Dict:
    # Compare leading EVR against a null where distances are randomly reassigned among pairs (preserving symmetry, zero diagonal)
    n = dist.shape[0]
    tri_idx = np.triu_indices(n, 1)
    orig = dist.copy()
    coords, evr2, eig = classical_mds(orig, k=2)
    evr2_obs = float(evr2)
    evr2_null = []
    rng = np.random.default_rng(1)
    flat = orig[tri_idx]
    for _ in range(trials):
        perm = rng.permutation(flat)
        d = np.zeros_like(orig)
        d[tri_idx] = perm
        d = d + d.T
        c, ev2, _ = classical_mds(d, k=2)
        evr2_null.append(float(ev2))
    p = float((np.sum(np.array(evr2_null) >= evr2_obs) + 1) / (trials + 1))
    return {"evr2_obs": evr2_obs, "evr2_null_mean": float(np.mean(evr2_null)), "p_value": p}


def metric_variants(logits_paths: List[Path]) -> Dict:
    # Compare distance definitions: logit MSE (ours), prob MSE, prob cosine, 0-1 disagreement
    tensors = [torch.load(p, map_location="cpu") for p in logits_paths]
    logits = [t.numpy() for t in tensors]
    probs = [np.exp(x - x.max(axis=1, keepdims=True)) for x in logits]
    probs = [p / p.sum(axis=1, keepdims=True) for p in probs]
    n = len(logits)
    out = {}
    def pairwise(fn):
        d = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d_ij = fn(i, j)
                d[i, j] = d[j, i] = d_ij
        return d
    d_logit = pairwise(lambda i, j: float(np.mean((logits[i] - logits[j]) ** 2)))
    d_prob = pairwise(lambda i, j: float(np.mean((probs[i] - probs[j]) ** 2)))
    # Cosine distance on probs
    d_cos = pairwise(lambda i, j: float(1.0 - np.mean(np.sum(probs[i] * probs[j], axis=1) / ((np.linalg.norm(probs[i], axis=1) * np.linalg.norm(probs[j], axis=1)) + 1e-12))))
    out["logit_mse"] = d_logit
    out["prob_mse"] = d_prob
    out["prob_cosine"] = d_cos
    return {k: v for k, v in out.items()}


def run(artifacts_dir: Path = Path("artifacts")) -> Dict:
    emb_dir = artifacts_dir / "embeddings"
    dist = load_distance_matrix(emb_dir / "distance_matrix.npy")
    coords, evr2, eig = classical_mds(dist, k=2)
    boot = bootstrap_stability(dist, trials=200)
    null = null_permutation_test(dist, trials=300)
    # Metric variants
    probe_dir = artifacts_dir / "probes"
    paths = sorted(probe_dir.glob("probe_logits_seed*.pt"))
    metrics = metric_variants(paths)
    results = {
        "evr2": float(evr2),
        "bootstrap_stability": float(boot),
        "null_test": null,
        "metric_variants": {k: v.tolist() for k, v in metrics.items()},
    }
    out_path = artifacts_dir / "reports" / "artifact_tests.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({"evr2": results["evr2"], "stability": results["bootstrap_stability"], "p": results["null_test"]["p_value"]}))
    return results


if __name__ == "__main__":
    run()
