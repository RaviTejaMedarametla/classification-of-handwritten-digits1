import numpy as np

from compression.compression import neuron_prune_features, weight_prune_features


def test_weight_prune_increases_zero_ratio():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 10)).astype(np.float32)
    x_test = rng.normal(size=(20, 10)).astype(np.float32)
    baseline = float((x_train == 0.0).mean())
    out = weight_prune_features(x_train, x_test, level=0.5)
    assert out.stats["sparsity"] >= baseline


def test_neuron_prune_reduces_feature_dimension():
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(50, 10)).astype(np.float32)
    x_test = rng.normal(size=(20, 10)).astype(np.float32)
    out = neuron_prune_features(x_train, x_test, level=0.3)
    assert out.x_train.shape[1] <= x_train.shape[1]
    assert 0.0 <= out.stats["feature_pruned_ratio"] <= 1.0
