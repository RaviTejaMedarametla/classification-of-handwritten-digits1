from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from ml_system.config import SystemConfig
from ml_system.utils.metrics import save_json


def load_mnist(config: SystemConfig):
    try:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x = x_train.reshape(x_train.shape[0], 28 * 28).astype(np.float32)
        y = y_train
    except Exception:
        ds = load_digits()
        x = ds.data.astype(np.float32)
        y = ds.target

    n = min(config.sample_size, len(x))
    return x[:n], y[:n]


def split_and_normalize(x: np.ndarray, y: np.ndarray, config: SystemConfig) -> Tuple[np.ndarray, ...]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.seed, stratify=y
    )
    if config.normalize:
        normalizer = Normalizer()
        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)
    return x_train, x_test, y_train, y_test


def dataset_fingerprint(x: np.ndarray, y: np.ndarray) -> Dict:
    return {
        "shape": list(x.shape),
        "labels": sorted(list(map(int, np.unique(y)))),
        "dtype": str(x.dtype),
    }


def save_dataset_metadata(config: SystemConfig, x: np.ndarray, y: np.ndarray) -> Path:
    path = config.artifacts_dir / "dataset_metadata.json"
    payload = {"config": asdict(config), "fingerprint": dataset_fingerprint(x, y)}
    payload["config"]["artifacts_dir"] = str(config.artifacts_dir)
    save_json(payload, path)
    return path
