from dataclasses import asdict
from typing import Dict

from sklearn.metrics import accuracy_score

from ml_system.config import CompressionConfig, QuantizationConfig, SystemConfig, TrainingConfig
from ml_system.data.dataset import load_dataset, save_dataset_metadata, split_and_normalize
from ml_system.models.classical import build_model
from ml_system.models.compression import neuron_prune_features, weight_prune_features
from ml_system.models.quantization import quantize_dataset
from ml_system.utils.metrics import save_json, timed_call
from ml_system.utils.plots import plot_bar
from ml_system.utils.reproducibility import set_deterministic


def train_once(system_config: SystemConfig, training_config: TrainingConfig) -> Dict:
    set_deterministic(system_config.seed)
    x, y, dataset_meta = load_dataset(system_config)
    save_dataset_metadata(system_config, x, y, dataset_meta)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)

    model = build_model(training_config)
    _, train_time = timed_call(model.fit, x_train, y_train)
    preds, infer_time = timed_call(model.predict, x_test)
    acc = accuracy_score(y_test, preds)

    metrics = {
        "model": training_config.model_name,
        "accuracy": float(acc),
        "train_time_s": float(train_time),
        "infer_time_s": float(infer_time),
    }
    save_json(metrics, system_config.artifacts_dir / f"train_metrics_{training_config.model_name}.json")
    return {"metrics": metrics, "model": model, "data": (x_train, x_test, y_train, y_test)}


def run_compression(system_config: SystemConfig, training_config: TrainingConfig, compression_config: CompressionConfig):
    base = train_once(system_config, training_config)
    x_train, x_test, y_train, y_test = base["data"]

    weight_data = weight_prune_features(x_train, x_test, compression_config.weight_pruning_level)
    neuron_data = neuron_prune_features(x_train, x_test, compression_config.neuron_pruning_level)

    records = {}
    for tag, data in [("weight", weight_data), ("neuron", neuron_data)]:
        model = build_model(training_config)
        model.fit(data.x_train, y_train)
        acc = accuracy_score(y_test, model.predict(data.x_test))
        records[tag] = {"accuracy": float(acc), **data.stats}

    save_json(records, system_config.artifacts_dir / f"compression_{training_config.model_name}.json")
    plot_bar(
        {k: v["accuracy"] for k, v in records.items()},
        "Compression Accuracy",
        "Accuracy",
        system_config.artifacts_dir / f"compression_accuracy_{training_config.model_name}.png",
    )
    return records


def run_quantization(system_config: SystemConfig, training_config: TrainingConfig, quant_config: QuantizationConfig):
    set_deterministic(system_config.seed)
    x, y, _ = load_dataset(system_config)
    x_train, x_test, y_train, y_test = split_and_normalize(x, y, system_config)

    out = {}
    for mode in quant_config.modes:
        q = quantize_dataset(x_train, x_test, mode)
        model = build_model(training_config)
        _, train_t = timed_call(model.fit, q.x_train, y_train)
        preds, infer_t = timed_call(model.predict, q.x_test)
        out[mode] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "latency_s": float(infer_t),
            "train_time_s": float(train_t),
            **q.metrics,
        }

    save_json(out, system_config.artifacts_dir / f"quantization_{training_config.model_name}.json")
    plot_bar(
        {k: v["accuracy"] for k, v in out.items()},
        "Quantization Accuracy by Mode",
        "Accuracy",
        system_config.artifacts_dir / f"quantization_accuracy_{training_config.model_name}.png",
    )
    return out


def export_run_metadata(system_config: SystemConfig, training_config: TrainingConfig):
    payload = {
        "system": {**asdict(system_config), "artifacts_dir": str(system_config.artifacts_dir)},
        "training": asdict(training_config),
    }
    save_json(payload, system_config.artifacts_dir / "run_config.json")
