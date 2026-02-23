import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import sklearn
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

from research.config import SystemConfig
from research.core.utils.metrics import save_json


def _dataset_checksum(x: np.ndarray, y: np.ndarray) -> str:
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(x).tobytes())
    hasher.update(np.ascontiguousarray(y).tobytes())
    return hasher.hexdigest()


def load_dataset(config: SystemConfig):
    if config.dataset == "mnist":
        try:
            (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
            x = x_train.reshape(x_train.shape[0], 28 * 28).astype(np.float32)
            y = y_train.astype(np.int64)
            version = tf.__version__
        except Exception as exc:
            if config.fail_fast_dataset:
                raise RuntimeError("MNIST download/load failed with fail-fast enabled") from exc
            raise RuntimeError(
                "MNIST download/load failed. Use --dataset digits for offline runs or --fail-fast-dataset for strict mode."
            ) from exc
    elif config.dataset == "digits":
        ds = load_digits()
        x = ds.data.astype(np.float32)
        y = ds.target.astype(np.int64)
        version = sklearn.__version__
    else:
        raise ValueError(f"Unsupported dataset={config.dataset}")

    n = min(config.sample_size, len(x))
    x, y = x[:n], y[:n]
    metadata = {
        "dataset": config.dataset,
        "version": version,
        "checksum_sha256": _dataset_checksum(x, y),
        "source_samples": int(len(x)),
    }
    return x, y, metadata


def load_mnist(config: SystemConfig):
    x, y, _ = load_dataset(config)
    return x, y


def split_and_normalize(x: np.ndarray, y: np.ndarray, config: SystemConfig) -> Tuple[np.ndarray, ...]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.test_size, random_state=config.seed, stratify=y
    )
    if config.normalize:
        normalizer = Normalizer()
        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)
    return x_train, x_test, y_train, y_test


def dataset_fingerprint(x: np.ndarray, y: np.ndarray, metadata: Dict) -> Dict:
    return {
        "shape": list(x.shape),
        "labels": sorted(list(map(int, np.unique(y)))),
        "dtype": str(x.dtype),
        "dataset": metadata["dataset"],
        "version": metadata["version"],
        "checksum_sha256": metadata["checksum_sha256"],
    }


def save_dataset_metadata(config: SystemConfig, x: np.ndarray, y: np.ndarray, metadata: Dict) -> Path:
    path = config.artifacts_dir / "dataset_metadata.json"
    payload = {"config": asdict(config), "fingerprint": dataset_fingerprint(x, y, metadata)}
    payload["config"]["artifacts_dir"] = str(config.artifacts_dir)
    save_json(payload, path)
    return path
