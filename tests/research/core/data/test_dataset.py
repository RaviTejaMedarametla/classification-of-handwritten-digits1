from research.config import SystemConfig
from research.core.data.dataset import load_dataset


def test_digits_checksum_is_deterministic():
    cfg = SystemConfig(dataset="digits", sample_size=100, seed=40)
    _, _, m1 = load_dataset(cfg)
    _, _, m2 = load_dataset(cfg)
    assert m1["checksum_sha256"] == m2["checksum_sha256"]
    assert len(m1["checksum_sha256"]) == 64


def test_mnist_or_fallback_behavior():
    cfg = SystemConfig(dataset="mnist", sample_size=10)
    try:
        _, _, meta = load_dataset(cfg)
        assert meta["dataset"] == "mnist"
    except RuntimeError:
        assert True
