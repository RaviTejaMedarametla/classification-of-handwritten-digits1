from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import accuracy_score

from compression.classical import build_model
from compression.compression import neuron_prune_features, weight_prune_features
from research.config import HardwareSimConfig, SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.metrics import estimate_energy_joules, memory_mb, save_json, timed_call
from research.core.utils.plots import plot_curve
from research.core.utils.reproducibility import set_deterministic

from benchmarks.repeated_benchmark import summary


def _effective_batch_size(feature_count: int, hw: HardwareSimConfig) -> int:
    bytes_per_sample = max(feature_count, 1) * 4
    budget_bytes = hw.memory_budget_mb * 1024 ** 2
    memory_cap = max(1, int(budget_bytes / bytes_per_sample))
    compute_cap = max(1, int(memory_cap * hw.compute_scale))
    return min(memory_cap, compute_cap)


def _predict_batched(model, x: np.ndarray, batch_size: int):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(model.predict(x[i:i + batch_size]))
    return np.concatenate(out)


def _apply_pruning(x_train: np.ndarray, x_test: np.ndarray, pruning_type: str, level: float):
    if pruning_type == "weight":
        return weight_prune_features(x_train, x_test, level)
    if pruning_type == "neuron":
        return neuron_prune_features(x_train, x_test, level)
    raise ValueError(f"Unsupported pruning_type={pruning_type}")


def pruning_hardware_experiment(
    system_config: SystemConfig,
    training_config: TrainingConfig,
    hw: HardwareSimConfig,
    sparsity_levels: Sequence[float],
    pruning_type: str = "weight",
    runs: int = 3,
) -> Dict:
    rows = []
    for level in sparsity_levels:
        level_acc = []
        level_lat = []
        level_thr = []
        level_energy = []
        level_mem = []
        level_model_mem = []
        level_baseline_sparsity = []
        level_added_sparsity = []
        achieved = float(level)

        for run in range(runs):
            set_deterministic(system_config.seed + run)
            x, y, _ = load_dataset(system_config)
            x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)
            baseline_sparsity = float((x_train == 0.0).mean())
            pruned = _apply_pruning(x_train, x_test, pruning_type=pruning_type, level=level)
            achieved = float(pruned.stats.get("sparsity", pruned.stats.get("feature_pruned_ratio", level)))
            added_sparsity = max(0.0, achieved - baseline_sparsity)

            model = build_model(training_config)
            model.fit(pruned.x_train, y_train)

            batch_size = _effective_batch_size(pruned.x_test.shape[1], hw)
            preds, lat = timed_call(_predict_batched, model, pruned.x_test, batch_size)
            acc = float(accuracy_score(y_test, preds))

            level_acc.append(acc)
            level_lat.append(float(lat))
            thr = float(len(y_test) / max(lat, 1e-9))
            level_thr.append(thr)
            level_energy.append(estimate_energy_joules(lat / max(len(y_test), 1)))
            level_mem.append(float(memory_mb(pruned.x_test[:batch_size])))
            level_model_mem.append(float(len(str(model)) / (1024 ** 2)))
            level_baseline_sparsity.append(baseline_sparsity)
            level_added_sparsity.append(added_sparsity)

        rows.append(
            {
                "target_sparsity_level": float(level),
                "baseline_sparsity": summary(level_baseline_sparsity, system_config.confidence_z),
                "achieved_sparsity": achieved,
                "added_sparsity": summary(level_added_sparsity, system_config.confidence_z),
                "accuracy": summary(level_acc, system_config.confidence_z),
                "latency_s": summary(level_lat, system_config.confidence_z),
                "throughput_samples_per_s": summary(level_thr, system_config.confidence_z),
                "energy_per_inference_j": summary(level_energy, system_config.confidence_z),
                "eval_memory_mb": summary(level_mem, system_config.confidence_z),
                "model_memory_mb": summary(level_model_mem, system_config.confidence_z),
                "effective_batch_size": _effective_batch_size(pruned.x_test.shape[1], hw),
            }
        )

    report = {
        "summary_version": "v2",
        "model": training_config.model_name,
        "pruning_type": pruning_type,
        "hardware_constraints": {
            "memory_budget_mb": hw.memory_budget_mb,
            "compute_scale": hw.compute_scale,
        },
        "runs": runs,
        "levels": rows,
    }

    root = system_config.artifacts_dir
    save_json(report, root / f"pruning_efficiency_{training_config.model_name}_{pruning_type}.json")

    sparsity_x = [item["added_sparsity"]["mean"] for item in rows]
    plot_curve(sparsity_x, [item["accuracy"]["mean"] for item in rows], "Sparsity vs Accuracy", "Added Sparsity", "Accuracy", root / f"sparsity_vs_accuracy_{training_config.model_name}_{pruning_type}.png")
    plot_curve(sparsity_x, [item["latency_s"]["mean"] for item in rows], "Sparsity vs Latency", "Added Sparsity", "Latency (s)", root / f"sparsity_vs_latency_{training_config.model_name}_{pruning_type}.png")
    plot_curve(sparsity_x, [item["energy_per_inference_j"]["mean"] for item in rows], "Sparsity vs Energy", "Added Sparsity", "Energy per inference (J)", root / f"sparsity_vs_energy_{training_config.model_name}_{pruning_type}.png")
    return report
