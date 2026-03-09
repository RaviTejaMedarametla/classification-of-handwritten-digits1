import pickle
from typing import Dict, List, Sequence

from sklearn.metrics import accuracy_score

from benchmarks.repeated_benchmark import summary
from compression.classical import build_model
from compression.compression import knn_prototype_reduction, prune_random_forest
from research.config import SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.metrics import save_json, timed_call
from research.core.utils.plots import plot_curve
from research.core.utils.reproducibility import set_deterministic


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
                model = prune_random_forest(base, max_depth=max_depth, ccp_alpha=0.001 * level, min_samples_leaf=1 + int(9 * level))
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
                "accuracy": summary(accs, system_config.confidence_z),
                "latency_s": summary(lats, system_config.confidence_z),
                "model_size_mb": summary(model_sizes, system_config.confidence_z),
                "compression_ratio": summary(compression_ratios, system_config.confidence_z),
            }
        )

    report = {"summary_version": "v2", "model": training_config.model_name, "runs": runs, "levels": rows}
    root = system_config.artifacts_dir
    save_json(report, root / f"model_level_compression_{training_config.model_name}.json")

    x_comp = [item["compression_ratio"]["mean"] for item in rows]
    plot_curve(x_comp, [item["accuracy"]["mean"] for item in rows], "Model Compression Ratio vs Accuracy", "Compression Ratio", "Accuracy", root / f"compression_ratio_vs_accuracy_{training_config.model_name}.png")
    plot_curve(x_comp, [item["latency_s"]["mean"] for item in rows], "Model Compression Ratio vs Latency", "Compression Ratio", "Latency (s)", root / f"compression_ratio_vs_latency_{training_config.model_name}.png")
    return report
