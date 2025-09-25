Chromatic Descent Experiment
============================

This repository contains a minimal, reproducible experiment to explore "chromatic descent": the idea that multiple high-performing minima occupy a structured space when measured in function-space, analogous to a higher-dimensional color space.

Features
--------
- Multi-seed training of a small CNN on CIFAR-10 (with CPU/GPU support)
- Probe-set logits export for function-space comparisons
- Function-space distance matrix and 2D embeddings (t-SNE/UMAP optional)
- Optional repelled descent term to diversify solutions
- Ensemble evaluation and summaries

Quickstart (uv)
---------------
1. Create a virtual environment with Python 3.12.1 and install dependencies using `uv`:

```
uv venv --python 3.12.1
uv pip install -r requirements.txt
```

2. Run a quick smoke test (few epochs, 3 seeds):

```
uv run python chromatic/run.py --dataset cifar10 --epochs 2 --seeds 3 --fast --device auto
```

Artifacts
---------
- Runs and checkpoints under `runs/SEED_XXXX/`
- Probe logits under `artifacts/probes/`
- Distance matrix/embedding plots under `artifacts/embeddings/`
- Ensemble report under `artifacts/reports/`

Notes
-----
- Metrics prioritize function-space comparisons (logit MSE on a held-out probe set). Parameter-space metrics can be added but are symmetry-sensitive.
- For UMAP, install optional extras: `pip install umap-learn`.

GPU note
--------
- CPU-only installs will work out of the box. For NVIDIA GPU wheels, you can install with the PyTorch CUDA index (example for CUDA 12.1):

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Preliminary findings (CPU, CIFAR‑10, fast subset)
-------------------------------------------------
- Setup: 12 independent seeds, 5 epochs each (fast mode). We analyze function-space distances computed from probe-set logits.
- Accuracy: individuals ≈ 0.42–0.45; 12‑member ensemble ≈ 0.454.
- Structure: classical MDS on distance matrix shows low‑rank geometry.
  - Top‑2 explained variance ratio ≈ 0.506 (half the signal in two components).
  - Bootstrap stability of the 2D layout ≈ 0.99996 (very stable to perturbations).
  - Null permutation test p ≈ 0.0498 (unlikely under random distances, borderline at 5%).
- Interpretation: the structure appears real (not a plotting artefact). Its semantic drivers (e.g., temperature vs. confusion mode) are weak in this short run; we will revisit after longer training and recipe diversity.

Reproduce analysis
------------------
```
uv run python -m chromatic.run --dataset cifar10 --epochs 5 --seeds 12 --fast --device cpu
uv run python -m chromatic.artifact_tests
uv run python -m chromatic.analyze
```


