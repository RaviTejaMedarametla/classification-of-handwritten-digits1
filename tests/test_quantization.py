import numpy as np
import pytest

from compression.quantization import quantize_dataset


@pytest.mark.parametrize("mode", ["float32", "float16", "int8_sim"])
def test_quantization_modes(mode: str):
    x_train = np.random.rand(32, 8).astype(np.float32)
    x_test = np.random.rand(16, 8).astype(np.float32)
    out = quantize_dataset(x_train, x_test, mode)
    assert out.x_train.shape == x_train.shape
    assert out.x_test.shape == x_test.shape
    assert out.metrics["mode"] == mode
