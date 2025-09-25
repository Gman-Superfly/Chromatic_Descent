from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def calibrate_logits_to_probs(logits: np.ndarray, labels: np.ndarray, max_iters: int = 200, lr: float = 0.05) -> tuple[np.ndarray, float]:
    """
    Temperature scale logits to minimize NLL on provided labels, returning calibrated probabilities and temperature.
    """
    device = torch.device("cpu")
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)

    log_T = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([log_T], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(max_iters):
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T) + 1e-8
        loss = criterion(logits_t / T, labels_t)
        loss.backward()
        optimizer.step()

    T_final = float(torch.exp(log_T).item())
    probs = F.softmax(logits_t / T_final, dim=1).cpu().numpy()
    return probs, T_final

