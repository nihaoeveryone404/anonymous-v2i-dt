"""Common evaluation metrics."""

from __future__ import annotations

import numpy as np


def rmse(prediction, target) -> float:
    error = np.asarray(prediction) - np.asarray(target)
    return float(np.sqrt(np.mean(error**2)))


def mae(prediction, target) -> float:
    return float(np.mean(np.abs(np.asarray(prediction) - np.asarray(target))))


def spearman(prediction, target) -> float:
    x = np.asarray(prediction).reshape(-1)
    y = np.asarray(target).reshape(-1)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def jain_fairness(values, eps: float = 1e-12) -> float:
    x = np.asarray(values, dtype=float).reshape(-1)
    return float((x.sum() ** 2) / (max(x.size, 1) * np.square(x).sum() + eps))
