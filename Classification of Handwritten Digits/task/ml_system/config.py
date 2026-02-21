from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class SystemConfig:
    seed: int = 40
    sample_size: int = 6000
    test_size: float = 0.3
    normalize: bool = True
    artifacts_dir: Path = Path("artifacts")
    confidence_z: float = 1.96


@dataclass
class TrainingConfig:
    model_name: str = "knn"
    knn_neighbors: int = 4
    knn_weights: str = "distance"
    knn_algorithm: str = "auto"
    rf_n_estimators: int = 500
    rf_max_features: str = "sqrt"
    rf_class_weight: str = "balanced"


@dataclass
class CompressionConfig:
    weight_pruning_level: float = 0.2
    neuron_pruning_level: float = 0.1


@dataclass
class QuantizationConfig:
    modes: Tuple[str, ...] = ("float32", "float16", "int8_sim")


@dataclass
class DeploymentConfig:
    onnx_opset: int = 12
    batch_size: int = 64


@dataclass
class HardwareSimConfig:
    memory_budget_mb: int = 8
    compute_scale: float = 1.0
    auto_batch_sizes: Tuple[int, ...] = field(default_factory=lambda: (1, 8, 32, 64, 128))
