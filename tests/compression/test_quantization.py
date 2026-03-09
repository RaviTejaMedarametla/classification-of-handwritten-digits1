import numpy as np

from compression.quantization import quantize_dataset


def test_quantization_modes():
    x_train = np.array([[1.0, -1.0], [0.5, -0.5]], dtype=np.float32)
    x_test = np.array([[0.1, -0.1]], dtype=np.float32)
    for mode in ("float32", "float16", "int8_sim"):
        q = quantize_dataset(x_train, x_test, mode)
        assert q.x_train.shape == x_train.shape
        assert "memory_train_mb" in q.metrics
