from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import accuracy_score

from compression.classical import build_model
from research.config import SystemConfig, TrainingConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.metrics import confidence_interval, estimate_energy_joules, save_json, timed_call
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
