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


def bootstrap_stability_from_probes(logit_paths: List[Path], labels: np.ndarray, trials: int = 200, frac: float = 0.7) -> float:
    # Recompute distance matrices from subsampled probe examples and measure coordinate stability
    from scipy.spatial import procrustes

    tensors = [torch.load(p, map_location="cpu") for p in logit_paths]
    logits = [t.numpy() for t in tensors]
    n = len(logits)
    if n < 3:
        return 1.0

    # Base distance using prob MSE with temperature calibration
    from chromatic.utils import calibrate_logits_to_probs
    probs_list = [calibrate_logits_to_probs(x, labels)[0] for x in logits]

    def prob_mse(probs):
        d = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = float(np.mean((probs[i] - probs[j]) ** 2))
        return d

    base_dist = prob_mse(probs_list)
    base_coords, _, _ = classical_mds(base_dist, k=2)
    base = (base_coords - base_coords.mean(0)) / (base_coords.std(0) + 1e-12)

    rng = np.random.default_rng(0)
    cors = []
    N = labels.shape[0]
    for _ in range(trials):
        m = max(2, int(frac * N))
        idx = rng.choice(N, size=m, replace=False)
        probs_sub = [p[idx] for p in probs_list]
        d2 = prob_mse(probs_sub)
        coords, _, _ = classical_mds(d2, k=2)
        A, B, _ = procrustes(base, (coords - coords.mean(0)) / (coords.std(0) + 1e-12))
        cors.append(np.corrcoef(A.ravel(), B.ravel())[0, 1])
    return float(np.mean(cors))


def null_permutation_test_examples(logit_paths: List[Path], labels: np.ndarray, trials: int = 300) -> Dict:
    # Build a null by permuting per-example logits assignments across models (breaking structure while preserving marginals)
    tensors = [torch.load(p, map_location="cpu") for p in logit_paths]
    logits = [t.numpy() for t in tensors]
    n = len(logits)
    N = logits[0].shape[0]
    from chromatic.utils import calibrate_logits_to_probs

    def prob_mse_from_logits(logs):
        probs = [calibrate_logits_to_probs(x, labels)[0] for x in logs]
        d = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d[i, j] = d[j, i] = float(np.mean((probs[i] - probs[j]) ** 2))
        return d

    orig_dist = prob_mse_from_logits(logits)
    _, evr2_obs, _ = classical_mds(orig_dist, k=2)

    rng = np.random.default_rng(1)
    evr2_null = []
    for _ in range(trials):
        perm = rng.permutation(N)
        logs_perm = [x[perm] for x in logits]
        d = prob_mse_from_logits(logs_perm)
        _, ev2, _ = classical_mds(d, k=2)
        evr2_null.append(float(ev2))
    p = float((np.sum(np.array(evr2_null) >= evr2_obs) + 1) / (trials + 1))
    return {"evr2_obs": float(evr2_obs), "evr2_null_mean": float(np.mean(evr2_null)), "p_value": p}


def metric_variants(logits_paths: List[Path], labels: np.ndarray) -> Dict:
    # Compare distance definitions: logit MSE, prob MSE (calibrated), prob cosine, 0-1 disagreement, symmetric KL
    tensors = [torch.load(p, map_location="cpu") for p in logits_paths]
    logits = [t.numpy() for t in tensors]
    from chromatic.utils import calibrate_logits_to_probs
    probs = [calibrate_logits_to_probs(x, labels)[0] for x in logits]
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
    d_cos = pairwise(lambda i, j: float(1.0 - np.mean(np.sum(probs[i] * probs[j], axis=1) / ((np.linalg.norm(probs[i], axis=1) * np.linalg.norm(probs[j], axis=1)) + 1e-12))))
    preds = [np.argmax(p, axis=1) for p in probs]
    d_dis = pairwise(lambda i, j: float(np.mean(preds[i] != preds[j])))
    def symkl(i, j):
        p = np.clip(probs[i], 1e-12, 1.0)
        q = np.clip(probs[j], 1e-12, 1.0)
        d = 0.5 * (np.sum(p * (np.log(p) - np.log(q)), axis=1) + np.sum(q * (np.log(q) - np.log(p)), axis=1))
        return float(np.mean(d))
    d_skl = pairwise(symkl)
    out["logit_mse"] = d_logit
    out["prob_mse"] = d_prob
    out["prob_cosine"] = d_cos
    out["zero_one"] = d_dis
    out["sym_kl"] = d_skl
    return {k: v for k, v in out.items()}


def run(artifacts_dir: Path = Path("artifacts")) -> Dict:
    emb_dir = artifacts_dir / "embeddings"
    dist = load_distance_matrix(emb_dir / "distance_matrix.npy")
    # Load labels in probe order
    from chromatic.data import get_probe_loader
    loader = get_probe_loader("cifar10", "./data", batch_size=256)
    ys = []
    for _, targets in loader:
        ys.append(targets.numpy())
    labels = np.concatenate(ys, axis=0)

    coords, evr2, eig = classical_mds(dist, k=2)
    # Bootstrap via probe subsampling
    probe_dir = artifacts_dir / "probes"
    paths = sorted(probe_dir.glob("probe_logits_seed*.pt"))
    boot = bootstrap_stability_from_probes(paths, labels, trials=200, frac=0.7)
    # Null via example permutation
    null = null_permutation_test_examples(paths, labels, trials=300)
    # Metric variants
    metrics = metric_variants(paths, labels)
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
