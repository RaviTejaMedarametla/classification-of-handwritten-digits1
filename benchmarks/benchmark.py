from typing import Dict, List, Sequence
import pickle

import numpy as np
from sklearn.metrics import accuracy_score

from research.config import HardwareSimConfig, SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from compression.classical import build_model
from compression.compression import (
    knn_prototype_reduction,
    neuron_prune_features,
    prune_random_forest,
    weight_prune_features,
)
from research.core.utils.metrics import confidence_interval, estimate_energy_joules, save_json, timed_call
from research.core.utils.plots import plot_bar, plot_curve
from research.core.utils.reproducibility import set_deterministic


def _summary(values: Sequence[float], z: float) -> Dict[str, float]:
    mean, std, ci = confidence_interval(values, z)
    return {"mean": mean, "std": std, "ci95": ci}


def repeated_benchmark(system_config: SystemConfig, training_config: TrainingConfig, runs: int = 5) -> Dict:
    accs: List[float] = []
    infs: List[float] = []
    trains: List[float] = []
    sample_counts: List[int] = []
    seeds: List[int] = []

    for i in range(runs):
        seed = system_config.seed + i
        seeds.append(seed)
        set_deterministic(seed)
        x, y, _ = load_dataset(system_config)
        x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)
        model = build_model(training_config)
        _, train_t = timed_call(model.fit, x_train, y_train)
        preds, infer_t = timed_call(model.predict, x_test)

        accs.append(float(accuracy_score(y_test, preds)))
        infs.append(float(infer_t))
        trains.append(float(train_t))
        sample_counts.append(int(len(x_test)))

    inf_mean = _summary(infs, system_config.confidence_z)["mean"]
    avg_samples = int(np.mean(sample_counts)) if sample_counts else 0
    stats = {
        "summary_version": "v2",
        "model": training_config.model_name,
        "runs": int(runs),
        "seed_schedule": seeds,
        "sample_count_eval": avg_samples,
        "accuracy": _summary(accs, system_config.confidence_z),
        "inference_latency_s": _summary(infs, system_config.confidence_z),
        "training_time_s": _summary(trains, system_config.confidence_z),
        "throughput_samples_per_s": float(avg_samples / max(inf_mean, 1e-9)),
        "energy_per_inference_j": estimate_energy_joules(inf_mean / max(avg_samples, 1)),
    }
    save_json(stats, system_config.artifacts_dir / f"benchmark_{training_config.model_name}.json")
    return stats


def hardware_simulation(system_config: SystemConfig, training_config: TrainingConfig, hw: HardwareSimConfig):
    set_deterministic(system_config.seed)
    x, y, _ = load_dataset(system_config)
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
            x, y, _ = load_dataset(system_config)
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

        rows.append(
            {
                "target_sparsity_level": float(level),
                "baseline_sparsity": _summary(level_baseline_sparsity, system_config.confidence_z),
                "achieved_sparsity": float(pruned.stats.get("sparsity", pruned.stats.get("feature_pruned_ratio", level))),
                "added_sparsity": _summary(level_added_sparsity, system_config.confidence_z),
                "accuracy": _summary(level_acc, system_config.confidence_z),
                "latency_s": _summary(level_lat, system_config.confidence_z),
                "throughput_samples_per_s": _summary(level_thr, system_config.confidence_z),
                "energy_per_inference_j": _summary(level_energy, system_config.confidence_z),
                "eval_memory_mb": _summary(level_mem, system_config.confidence_z),
                "model_memory_mb": _summary(level_model_mem, system_config.confidence_z),
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


def model_level_compression_experiment(
    system_config: SystemConfig,
    training_config: TrainingConfig,
    runs: int = 3,
    levels: Sequence[float] = (0.0, 0.2, 0.4, 0.6),
) -> Dict:
    rows = []
    for level in levels:
        accs: List[float] = []
        lats: List[float] = []
        model_sizes: List[float] = []
        compression_ratios: List[float] = []

        for run in range(runs):
            set_deterministic(system_config.seed + run)
            x, y, _ = load_dataset(system_config)
            x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)

            base = build_model(training_config)
            base.fit(x_train, y_train)
            base_model_size = max(float(len(pickle.dumps(base)) / (1024 ** 2)), 1e-12)

            if training_config.model_name == "knn":
                reduced = knn_prototype_reduction(x_train, y_train, reduction_level=level)
                model = build_model(training_config)
                model.fit(reduced.x_train, reduced.y_train)
            elif training_config.model_name == "rf":
                max_depth = None if level <= 0 else max(2, int(20 * (1.0 - level)))
                ccp_alpha = 0.001 * level
                min_samples_leaf = 1 + int(9 * level)
                model = prune_random_forest(base, max_depth=max_depth, ccp_alpha=ccp_alpha, min_samples_leaf=min_samples_leaf)
                model.fit(x_train, y_train)
            else:
                raise ValueError(f"Unsupported model_name={training_config.model_name}")

            preds, lat = timed_call(model.predict, x_test)
            accs.append(float(accuracy_score(y_test, preds)))
            lats.append(float(lat))
            model_size = float(len(pickle.dumps(model)) / (1024 ** 2))
            model_sizes.append(model_size)
            compression_ratios.append(1.0 - (model_size / base_model_size))

        rows.append(
            {
                "target_level": float(level),
                "accuracy": _summary(accs, system_config.confidence_z),
                "latency_s": _summary(lats, system_config.confidence_z),
                "model_size_mb": _summary(model_sizes, system_config.confidence_z),
                "compression_ratio": _summary(compression_ratios, system_config.confidence_z),
            }
        )

    report = {
        "summary_version": "v2",
        "model": training_config.model_name,
        "runs": runs,
        "levels": rows,
    }
    root = system_config.artifacts_dir
    save_json(report, root / f"model_level_compression_{training_config.model_name}.json")

    x_comp = [item["compression_ratio"]["mean"] for item in rows]
    plot_curve(
        x_comp,
        [item["accuracy"]["mean"] for item in rows],
        "Model Compression Ratio vs Accuracy",
        "Compression Ratio",
        "Accuracy",
        root / f"compression_ratio_vs_accuracy_{training_config.model_name}.png",
    )
    plot_curve(
        x_comp,
        [item["latency_s"]["mean"] for item in rows],
        "Model Compression Ratio vs Latency",
        "Compression Ratio",
        "Latency (s)",
        root / f"compression_ratio_vs_latency_{training_config.model_name}.png",
    )
    return report


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
        "memory": {
            "input_batch_mb": float(input_bytes / (1024 ** 2)),
            "serialized_model_mb": float(model_bytes / (1024 ** 2)),
        },
        "bandwidth": {
            "estimated_bytes_per_s": bandwidth_bytes_per_s,
            "estimated_mb_per_s": float(bandwidth_bytes_per_s / (1024 ** 2)),
        },
        "utilization": {
            "reference_samples_per_s": 1_000_000,
            "ratio": utilization_vs_reference,
        },
        "operator_latency_breakdown_s": operator_breakdown,
        "notes": [
            "Operator breakdown is an analytical partition of end-to-end latency.",
            "Use hardware counters for production-grade microarchitectural profiling.",
        ],
    }

    root = system_config.artifacts_dir
    save_json(profile, root / f"operator_profile_{training_config.model_name}.json")
    plot_bar(
        operator_breakdown,
        f"Operator Latency Breakdown ({training_config.model_name})",
        "Latency (s)",
        root / f"operator_profile_{training_config.model_name}.png",
    )
    return profile
