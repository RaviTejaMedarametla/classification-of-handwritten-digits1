import numpy as np

from compression.compression import knn_prototype_reduction, neuron_prune_features, weight_prune_features


def test_weight_prune_increases_sparsity():
    rng = np.random.default_rng(1)
    x_train = rng.normal(size=(20, 10)).astype(np.float32)
    x_test = rng.normal(size=(5, 10)).astype(np.float32)
    res = weight_prune_features(x_train, x_test, level=0.5)
    assert res.stats["sparsity"] > 0


def test_neuron_prune_reduces_features():
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(20, 10)).astype(np.float32)
    x_test = rng.normal(size=(5, 10)).astype(np.float32)
    res = neuron_prune_features(x_train, x_test, level=0.3)
    assert res.x_train.shape[1] <= x_train.shape[1]


def test_knn_prototype_reduction_reduces_rows():
    rng = np.random.default_rng(3)
    x_train = rng.normal(size=(50, 8)).astype(np.float32)
    y_train = rng.integers(0, 2, size=50)
    res = knn_prototype_reduction(x_train, y_train, reduction_level=0.4)
    assert len(res.x_train) < len(x_train)
