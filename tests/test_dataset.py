from research.config import SystemConfig
from research.core.data.dataset import load_dataset, split_and_normalize


def test_load_digits_dataset_shape():
    cfg = SystemConfig(dataset="digits", sample_size=200, seed=40)
    x, y, meta = load_dataset(cfg)
    assert x.shape[0] == 200
    assert y.shape[0] == 200
    assert meta["dataset"] == "digits"


def test_split_and_normalize_shapes():
    cfg = SystemConfig(dataset="digits", sample_size=200, seed=40)
    x, y, _ = load_dataset(cfg)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, cfg)
    assert len(x_train) + len(x_test) == 200
    assert len(y_train) + len(y_test) == 200
