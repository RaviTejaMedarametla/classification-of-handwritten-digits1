from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class QuantizedData:
    x_train: np.ndarray
    x_test: np.ndarray
    metrics: Dict[str, float]


def quantize_dataset(x_train: np.ndarray, x_test: np.ndarray, mode: str) -> QuantizedData:
    if mode == "float32":
        q_train, q_test = x_train.astype(np.float32), x_test.astype(np.float32)
    elif mode == "float16":
        q_train, q_test = x_train.astype(np.float16), x_test.astype(np.float16)
    elif mode == "int8_sim":
        scale = np.max(np.abs(x_train)) / 127.0 + 1e-8
        q_train = np.round(x_train / scale).astype(np.int8).astype(np.float32) * scale
        q_test = np.round(x_test / scale).astype(np.int8).astype(np.float32) * scale
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    mem_train = q_train.nbytes / (1024 ** 2)
    return QuantizedData(q_train, q_test, {"memory_train_mb": float(mem_train), "mode": mode})
