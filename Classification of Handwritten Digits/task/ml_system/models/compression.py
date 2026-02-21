from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PruningResult:
    x_train: np.ndarray
    x_test: np.ndarray
    stats: Dict[str, float]


def weight_prune_features(x_train: np.ndarray, x_test: np.ndarray, level: float) -> PruningResult:
    threshold = np.quantile(np.abs(x_train), level)
    x_train_pruned = x_train.copy()
    x_test_pruned = x_test.copy()
    x_train_pruned[np.abs(x_train_pruned) < threshold] = 0.0
    x_test_pruned[np.abs(x_test_pruned) < threshold] = 0.0
    sparsity = float((x_train_pruned == 0.0).mean())
    return PruningResult(x_train_pruned, x_test_pruned, {"weight_threshold": float(threshold), "sparsity": sparsity})


def neuron_prune_features(x_train: np.ndarray, x_test: np.ndarray, level: float) -> PruningResult:
    variances = x_train.var(axis=0)
    cutoff = np.quantile(variances, level)
    keep_mask = variances >= cutoff
    x_train_pruned = x_train[:, keep_mask]
    x_test_pruned = x_test[:, keep_mask]
    sparsity = 1.0 - float(keep_mask.mean())
    return PruningResult(x_train_pruned, x_test_pruned, {"variance_cutoff": float(cutoff), "feature_pruned_ratio": sparsity})
