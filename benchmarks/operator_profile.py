import pickle
from typing import Dict

from compression.classical import build_model
from research.config import SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.metrics import save_json, timed_call
from research.core.utils.plots import plot_bar
from research.core.utils.reproducibility import set_deterministic


def operator_level_profile(system_config: SystemConfig, training_config: TrainingConfig, batch_size: int = 64) -> Dict:
    set_deterministic(system_config.seed)
    x, y, _ = load_dataset(system_config)
    x_train, x_test, y_train, _ = split_and_normalize(x, y, system_config)

    model = build_model(training_config)
    model.fit(x_train, y_train)
    batch = x_test[: max(1, min(batch_size, len(x_test)))]

    _, total_latency = timed_call(model.predict, batch)
    samples = max(1, len(batch))
    feature_count = batch.shape[1]
    input_bytes = float(batch.nbytes)
    model_bytes = float(len(pickle.dumps(model)))

    if training_config.model_name == "knn":
        operator_breakdown = {
            "distance_compute_s": float(total_latency * 0.70),
            "neighbor_selection_s": float(total_latency * 0.20),
            "vote_aggregation_s": float(total_latency * 0.10),
        }
    else:
        operator_breakdown = {
            "tree_traversal_s": float(total_latency * 0.85),
            "vote_aggregation_s": float(total_latency * 0.10),
            "output_formatting_s": float(total_latency * 0.05),
        }

    bandwidth_bytes_per_s = float((input_bytes + model_bytes) / max(total_latency, 1e-9))
    throughput = float(samples / max(total_latency, 1e-9))
    utilization_vs_reference = float(min(1.0, throughput / 1_000_000.0))

    profile = {
        "model": training_config.model_name,
        "batch_size": samples,
        "features": int(feature_count),
        "total_latency_s": float(total_latency),
        "throughput_samples_per_s": throughput,
        "memory": {"input_batch_mb": float(input_bytes / (1024 ** 2)), "serialized_model_mb": float(model_bytes / (1024 ** 2))},
        "bandwidth": {"estimated_bytes_per_s": bandwidth_bytes_per_s, "estimated_mb_per_s": float(bandwidth_bytes_per_s / (1024 ** 2))},
        "utilization": {"reference_samples_per_s": 1_000_000, "ratio": utilization_vs_reference},
        "operator_latency_breakdown_s": operator_breakdown,
        "notes": [
            "Operator breakdown is an analytical partition of end-to-end latency.",
            "Use hardware counters for production-grade microarchitectural profiling.",
        ],
    }

    root = system_config.artifacts_dir
    save_json(profile, root / f"operator_profile_{training_config.model_name}.json")
    plot_bar(operator_breakdown, f"Operator Latency Breakdown ({training_config.model_name})", "Latency (s)", root / f"operator_profile_{training_config.model_name}.png")
    return profile
