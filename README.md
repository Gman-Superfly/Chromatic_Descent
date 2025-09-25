Chromatic Descent Experiment
============================

This repository contains a minimal, reproducible experiment to explore "chromatic descent": the idea that multiple high-performing minima occupy a structured space when measured in function-space, analogous to a higher-dimensional color space.

Features
--------
- Multi-seed training of a small CNN on CIFAR-10 (CPU/GPU)
- Deterministic train / probe-val / test split (no leakage)
- Probe-set logits export for function-space comparisons
- Calibration-aware distances (per-model temperature scaling): prob MSE (default), symmetric KL, 0-1 disagreement, logit MSE
- Classical MDS embedding (default) and saved t-SNE view; saved distance matrices under `artifacts/embeddings/`
- Optional repelled descent term to diversify solutions (seed order randomized; probe batches cycled)
- Ensemble evaluation: full ensemble plus subset baselines (random-k, top-k by single accuracy, farthest-first by function-space distance)

Motivation
----------
- Validate the chromatic descent hypothesis: do good minima form a structured “palette” in function space, rather than isolated points?
- Distinguish real geometry from visualization artefacts via bootstrap stability, null permutations, and metric variants.
- Quantify diversity and utility: relate pairwise function differences to ensemble gains and robustness.
- Provide controllable knobs (repulsion, recipe/augmentation diversity) to intentionally widen or traverse the solution gamut, fast on CPU.

Knowns vs unknowns
------------------
- Known: many good minima exist; ensembles help when errors differ; parameter-space geometry is symmetry-sensitive; calibration matters.
- Unknowns we test: does the solution set exhibit low-rank structure in function space? does increased distance correlate with ensemble gains and/or robustness? can simple repulsion reliably widen the palette without hurting singles?

### How we define and measure “function-space”
- Function-space is the model’s behavior on a shared probe‑validation set: for each model we collect its outputs on the same ordered examples.
- We calibrate each model’s logits with per‑model temperature scaling on the probe‑val labels, then compute pairwise distances between models by averaging per‑example differences.
- Default distance: mean squared error on calibrated probabilities. Also computed: symmetric KL, 0‑1 prediction disagreement, and (for reference) raw logit MSE.
- Distances are averaged over all probe‑val examples to form an S×S distance matrix for S models. No parameter/weight distances are used.

### What “structured space” properties we test
- Dimensionality/low‑rank: classical MDS eigen‑spectrum and top‑k explained variance ratio (EVR). If a few components explain most variance, geometry is low‑rank.
- Stability (non‑viz artefact): bootstrap by subsampling probe examples, recomputing distances and MDS, and measuring Procrustes‑aligned coordinate correlation.
- Null baselines: example‑level permutation null (shuffle example assignments across models), compare observed top‑2 EVR to a null distribution (p‑value).
- Semantic axes: correlations of MDS axes with average confidence, ECE, and the first principal component of per‑class confusion matrices.
- Not (yet) tested: formal clustering metrics (e.g., silhouette) and connectivity/mode‑connectivity; these are natural next steps.

### Color‑space analogy, mathematically
- It’s an embedding of behavioral distances: we double‑center the squared distance matrix to get a Gram matrix, take its top eigenvectors/eigenvalues (classical MDS), and use those coordinates as a low‑D “palette” where Euclidean distances approximate function‑space distances.
- TSNE is provided as an optional view, but we use classical MDS for interpretability and to tie coordinates to the spectrum (EVR).

### What we observe so far (from the included artifacts)
- Dimensionality: top‑2 EVR ≈ 0.33 and 0.18 (sum ≈ 0.51), i.e., about half the variance captured by two axes on the calibrated‑probability distance.
- Stability: bootstrap (probe subsampling) ≈ 0.9999 Procrustes correlation, indicating a very stable 2D layout to probe resampling.
- Null: example‑permutation p ≈ 0.296 for combined top‑2 EVR (not significant under this stricter null in the short run).
- Ensembles: best single test accuracy ≈ 0.6411; 12‑member ensemble ≈ 0.6495 (≈ +0.8 pts). Subset baselines are reported (random‑k, top‑k, farthest‑first).

In short: we measure function‑space via calibrated outputs on a held‑out probe set; we test for low‑rank structure, stability, and non‑artefactuality using spectral methods and resampling/nulls. The “color space” is a concrete classical‑MDS embedding of those distances. Current results show a stable, low‑rank layout with modest ensemble gains, but the strict null isn’t yet rejected—consistent with short training; more epochs/recipes typically sharpen both structure and utility.

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

3. Optional: enable repelled descent (diversify solutions):

```
uv run python chromatic/run.py --dataset cifar10 --epochs 5 --seeds 6 --repel --repel_lambda 0.1 --repel_warmup 2 --device auto
```

4. Artefact checks and analysis:

```
uv run python -m chromatic.artifact_tests
uv run python -m chromatic.analyze
```

Artifacts
---------
- Runs and checkpoints under `runs/SEED_XXXX/`
- Probe logits under `artifacts/probes/`
- Distance matrices under `artifacts/embeddings/`:
  - `distance_matrix.npy` (default: calibrated prob MSE)
  - `distance_prob_mse.npy`, `distance_sym_kl.npy`, `distance_logit_mse.npy`, `distance_zero_one.npy`
- 2D embedding arrays and plot under `artifacts/embeddings/`
- Reports under `artifacts/reports/` (`ensemble.txt`, `artifact_tests.json`, `analysis.json`)

Notes
-----
- Metrics prioritize function-space comparisons on the held-out probe-val set. Distances default to calibrated probability MSE; we also compute symmetric KL, 0-1 disagreement, and logit MSE for comparison. Parameter-space metrics can be added but are symmetry-sensitive.
- For UMAP, install optional extras: `pip install umap-learn`.

GPU note
--------
- CPU-only installs will work out of the box. For NVIDIA GPU wheels, you can install with the PyTorch CUDA index (example for CUDA 12.1):

```
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Method (what the pipeline does)
-------------------------------
1) Train many independent models (different seeds). Optional repulsion nudges later runs away from earlier ones in function space.
2) Export probe-val logits for each model (fixed order, no shuffle).
3) Temperature-scale each model on probe-val to mitigate logit scale; compute distances (prob MSE default, plus variants).
4) Build embeddings via classical MDS (default) and save a t‑SNE view.
5) Run artefact checks: probe‑subsample bootstrap stability; example‑permutation null; metric variants.
6) Evaluate ensembles: full set and subset baselines (random‑k, top‑k, farthest‑first by distance).

Sample results (from included artifacts)
---------------------------------------
- Setup: 12 independent seeds, short training. These are example numbers; your exact results will vary.
- Embedding geometry (calibrated prob‑MSE distances):
  - Top‑2 EVR (classical MDS): ~0.33, 0.18 (sum ~0.51)
  - Bootstrap stability (probe subsampling): ~0.9999
  - Example‑permutation null p‑value: ~0.296 (not significant vs null in this quick run)
- Test accuracy (single vs ensemble):
  - Best single: ~0.6411; 12‑member ensemble: ~0.6495 (≈ +0.8 pts)
- Interpretation: the 2D structure is highly stable to probe resampling, but with the stricter example‑permutation null it is not statistically distinct in this small/short run. Ensembles improve modestly; stronger training or recipe diversity typically sharpens both geometry and ensemble gains.

Reproduce analysis
------------------
```
uv run python -m chromatic.run --dataset cifar10 --epochs 5 --seeds 12 --fast --device cpu
uv run python -m chromatic.artifact_tests
uv run python -m chromatic.analyze
```

Limitations and next steps
--------------------------
- The strict null can be hard to reject at small N; running more seeds/epochs or adding recipe diversity (augmentations, optimizer, mild arch variants) generally increases distances and the top‑k EVR.
- Explore correlations between diversity and ensemble gains; try farthest‑first subsets vs top‑k and random‑k.
- Evaluate robustness/shift (e.g., light corruptions) and calibration; add OOD probes.

License
-------
MIT


