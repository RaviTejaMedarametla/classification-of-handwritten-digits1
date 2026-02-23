from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


@dataclass
class PruningResult:
    x_train: np.ndarray
    x_test: np.ndarray
    stats: Dict[str, float]


@dataclass
class ModelCompressionResult:
    model: object
    x_train: np.ndarray
    y_train: np.ndarray
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


def knn_prototype_reduction(x_train: np.ndarray, y_train: np.ndarray, reduction_level: float) -> ModelCompressionResult:
    if reduction_level <= 0:
        return ModelCompressionResult(None, x_train, y_train, {"prototype_kept_ratio": 1.0, "prototype_removed_ratio": 0.0})

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(x_train)
    distances, _ = nn.kneighbors(x_train)
    redundancy_score = distances[:, 1]
    remove_n = int(len(x_train) * min(max(reduction_level, 0.0), 0.95))
    remove_idx = np.argsort(redundancy_score)[:remove_n]
    keep_mask = np.ones(len(x_train), dtype=bool)
    keep_mask[remove_idx] = False

    kept_ratio = float(keep_mask.mean())
    return ModelCompressionResult(
        None,
        x_train[keep_mask],
        y_train[keep_mask],
        {
            "prototype_kept_ratio": kept_ratio,
            "prototype_removed_ratio": 1.0 - kept_ratio,
            "median_redundancy_distance": float(np.median(redundancy_score)),
        },
    )


def prune_random_forest(
    rf_model: RandomForestClassifier,
    max_depth: Optional[int],
    ccp_alpha: float,
    min_samples_leaf: int,
) -> RandomForestClassifier:
    params = rf_model.get_params()
    params.update(
        {
            "max_depth": max_depth,
            "ccp_alpha": max(0.0, float(ccp_alpha)),
            "min_samples_leaf": max(1, int(min_samples_leaf)),
        }
    )
    return clone(rf_model).set_params(**params)
