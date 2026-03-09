"""Run deterministic end-to-end experiments and persist CSV/plot artifacts in results/."""

from pathlib import Path

import pandas as pd

from benchmarks.benchmark import (
    model_level_compression_experiment,
    operator_level_profile,
    pruning_hardware_experiment,
    repeated_benchmark,
)
from deployment.onnx_pipeline import export_to_onnx, validate_onnx
from research.config import HardwareSimConfig, SystemConfig, TrainingConfig
from research.core.training.trainer import train_once
from research.core.utils.reproducibility import set_deterministic


def main(non_output_mode: bool = False):
    set_deterministic(40)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    sys_cfg = SystemConfig(seed=40, sample_size=6000, dataset="digits", artifacts_dir=results_dir)
    tr_cfg = TrainingConfig(model_name="knn", rf_random_state=40)
    hw_cfg = HardwareSimConfig(memory_budget_mb=8, compute_scale=1.0)

    repeat = repeated_benchmark(sys_cfg, tr_cfg, runs=3)
    pruning = pruning_hardware_experiment(sys_cfg, tr_cfg, hw_cfg, sparsity_levels=(0.0, 0.2), runs=2)
    compression = model_level_compression_experiment(sys_cfg, tr_cfg, runs=2, levels=(0.0, 0.2))
    profile = operator_level_profile(sys_cfg, tr_cfg, batch_size=32)

    trained = train_once(sys_cfg, tr_cfg)
    _, x_test, _, _ = trained["data"]
    model = trained["model"]
    onnx_path = export_to_onnx(model, x_test[:1], results_dir / "knn.onnx")
    onnx_report = validate_onnx(onnx_path, model, x_test, min_agreement=0.98)

    df = pd.DataFrame(
        [
            {"experiment": "repeated", "accuracy_mean": repeat["accuracy"]["mean"]},
            {"experiment": "operator_profile", "accuracy_mean": None, "throughput": profile["throughput_samples_per_s"]},
            {"experiment": "onnx_parity", "agreement": onnx_report["agreement"]},
        ]
    )
    if not non_output_mode:
        df.to_csv(results_dir / "summary_table.csv", index=False)
        pd.DataFrame(pruning["levels"]).to_csv(results_dir / "pruning_levels.csv", index=False)
        pd.DataFrame(compression["levels"]).to_csv(results_dir / "compression_levels.csv", index=False)
    return {
        "repeat": repeat,
        "pruning": pruning,
        "compression": compression,
        "profile": profile,
        "onnx": onnx_report,
    }


if __name__ == "__main__":
    main()
