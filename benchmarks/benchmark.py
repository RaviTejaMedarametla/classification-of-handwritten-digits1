"""Backward-compatible benchmark facade."""

from benchmarks.compression_benchmark import model_level_compression_experiment
from benchmarks.operator_profile import operator_level_profile
from benchmarks.pruning_benchmark import pruning_hardware_experiment
from benchmarks.repeated_benchmark import repeated_benchmark
from research.config import HardwareSimConfig, SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.metrics import save_json
from research.core.utils.plots import plot_curve
from research.core.utils.reproducibility import set_deterministic
from compression.classical import build_model
from sklearn.metrics import accuracy_score


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
