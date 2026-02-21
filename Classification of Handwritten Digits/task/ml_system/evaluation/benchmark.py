from typing import Dict, List, Sequence
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from ml_system.config import HardwareSimConfig, SystemConfig, TrainingConfig
from ml_system.data.dataset import load_mnist, split_and_normalize
from ml_system.models.classical import build_model
from ml_system.models.compression import neuron_prune_features, weight_prune_features
from ml_system.utils.metrics import confidence_interval, estimate_energy_joules, save_json, timed_call
from ml_system.utils.plots import plot_curve
from ml_system.utils.reproducibility import set_deterministic


def repeated_benchmark(system_config: SystemConfig, training_config: TrainingConfig, runs: int = 5) -> Dict:
    accs: List[float] = []
    infs: List[float] = []
    trains: List[float] = []
    for i in range(runs):
        set_deterministic(system_config.seed + i)
        x, y = load_mnist(system_config)
        x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)
        model = build_model(training_config)
        _, train_t = timed_call(model.fit, x_train, y_train)
        preds, infer_t = timed_call(model.predict, x_test)
        accs.append(float(accuracy_score(y_test, preds)))
        infs.append(float(infer_t))
        trains.append(float(train_t))

    acc_mean, acc_std, acc_ci = confidence_interval(accs)
    inf_mean, inf_std, inf_ci = confidence_interval(infs)
    train_mean, train_std, train_ci = confidence_interval(trains)

    stats = {
        "accuracy": {"mean": acc_mean, "std": acc_std, "ci95": acc_ci},
        "inference_latency_s": {"mean": inf_mean, "std": inf_std, "ci95": inf_ci},
        "training_time_s": {"mean": train_mean, "std": train_std, "ci95": train_ci},
        "throughput_samples_per_s": 1800 / max(inf_mean, 1e-9),
        "energy_per_inference_j": estimate_energy_joules(inf_mean / 1800),
    }
    save_json(stats, system_config.artifacts_dir / f"benchmark_{training_config.model_name}.json")
    return stats


def hardware_simulation(system_config: SystemConfig, training_config: TrainingConfig, hw: HardwareSimConfig):
    set_deterministic(system_config.seed)
    x, y = load_mnist(system_config)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)

    model = build_model(training_config)
    model.fit(x_train, y_train)

    resource = []
    accuracy = []
    for batch in hw.auto_batch_sizes:
        effective_batch = max(1, int(batch * hw.compute_scale))
        if (effective_batch * x_test.shape[1] * 4) / (1024 ** 2) > hw.memory_budget_mb:
            effective_batch = max(1, int(hw.memory_budget_mb * (1024 ** 2) / (x_test.shape[1] * 4)))
        subset = x_test[:effective_batch]
        preds = model.predict(subset)
        acc = accuracy_score(y_test[:effective_batch], preds)
        resource.append(effective_batch)
        accuracy.append(float(acc))

    plot_curve(
        resource,
        accuracy,
        "Resource vs Accuracy",
        "Effective Batch Size",
        "Accuracy",
        system_config.artifacts_dir / f"resource_accuracy_{training_config.model_name}.png",
    )
    save_json(
        {"effective_batch_sizes": resource, "accuracy": accuracy},
        system_config.artifacts_dir / f"hardware_sim_{training_config.model_name}.json",
    )
    return resource, accuracy


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

        for run in range(runs):
            set_deterministic(system_config.seed + run)
            x, y = load_mnist(system_config)
            x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)
            baseline_sparsity = float((x_train == 0.0).mean())
            pruned = _apply_pruning(x_train, x_test, pruning_type=pruning_type, level=level)
            achieved_sparsity = float(pruned.stats.get("sparsity", pruned.stats.get("feature_pruned_ratio", level)))
            added_sparsity = max(0.0, achieved_sparsity - baseline_sparsity)

            model = build_model(training_config)
            model.fit(pruned.x_train, y_train)

            batch_size = _effective_batch_size(pruned.x_test.shape[1], hw)
            eval_count = min(len(pruned.x_test), batch_size * 10)
            eval_x = pruned.x_test[:eval_count]
            eval_y = y_test[:eval_count]

            preds, latency_s = timed_call(_predict_batched, model, eval_x, batch_size)
            accuracy = float(accuracy_score(eval_y, preds))
            throughput = float(len(eval_x) / max(latency_s, 1e-9))
            energy = float(estimate_energy_joules(latency_s / max(len(eval_x), 1)))
            dataset_memory_mb = float(eval_x.nbytes / (1024 ** 2))
            model_memory_mb = float(len(pickle.dumps(model)) / (1024 ** 2))

            level_acc.append(accuracy)
            level_lat.append(float(latency_s))
            level_thr.append(throughput)
            level_energy.append(energy)
            level_mem.append(dataset_memory_mb)
            level_model_mem.append(model_memory_mb)
            level_baseline_sparsity.append(baseline_sparsity)
            level_added_sparsity.append(added_sparsity)

        acc_mean, acc_std, acc_ci = confidence_interval(level_acc, system_config.confidence_z)
        lat_mean, lat_std, lat_ci = confidence_interval(level_lat, system_config.confidence_z)
        thr_mean, thr_std, thr_ci = confidence_interval(level_thr, system_config.confidence_z)
        energy_mean, energy_std, energy_ci = confidence_interval(level_energy, system_config.confidence_z)
        mem_mean, mem_std, mem_ci = confidence_interval(level_mem, system_config.confidence_z)
        model_mem_mean, model_mem_std, model_mem_ci = confidence_interval(level_model_mem, system_config.confidence_z)
        baseline_mean, baseline_std, baseline_ci = confidence_interval(level_baseline_sparsity, system_config.confidence_z)
        added_mean, added_std, added_ci = confidence_interval(level_added_sparsity, system_config.confidence_z)

        rows.append(
            {
                "target_sparsity_level": float(level),
                "baseline_sparsity": {"mean": baseline_mean, "std": baseline_std, "ci95": baseline_ci},
                "achieved_sparsity": float(pruned.stats.get("sparsity", pruned.stats.get("feature_pruned_ratio", level))),
                "added_sparsity": {"mean": added_mean, "std": added_std, "ci95": added_ci},
                "accuracy": {"mean": acc_mean, "std": acc_std, "ci95": acc_ci},
                "latency_s": {"mean": lat_mean, "std": lat_std, "ci95": lat_ci},
                "throughput_samples_per_s": {"mean": thr_mean, "std": thr_std, "ci95": thr_ci},
                "energy_per_inference_j": {"mean": energy_mean, "std": energy_std, "ci95": energy_ci},
                "eval_memory_mb": {"mean": mem_mean, "std": mem_std, "ci95": mem_ci},
                "model_memory_mb": {"mean": model_mem_mean, "std": model_mem_std, "ci95": model_mem_ci},
                "effective_batch_size": _effective_batch_size(pruned.x_test.shape[1], hw),
            }
        )

    report = {
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
    plot_curve(
        sparsity_x,
        [item["accuracy"]["mean"] for item in rows],
        "Sparsity vs Accuracy",
        "Added Sparsity",
        "Accuracy",
        root / f"sparsity_vs_accuracy_{training_config.model_name}_{pruning_type}.png",
    )
    plot_curve(
        sparsity_x,
        [item["latency_s"]["mean"] for item in rows],
        "Sparsity vs Latency",
        "Added Sparsity",
        "Latency (s)",
        root / f"sparsity_vs_latency_{training_config.model_name}_{pruning_type}.png",
    )
    plot_curve(
        sparsity_x,
        [item["energy_per_inference_j"]["mean"] for item in rows],
        "Sparsity vs Energy",
        "Added Sparsity",
        "Energy per inference (J)",
        root / f"sparsity_vs_energy_{training_config.model_name}_{pruning_type}.png",
    )
    return report
