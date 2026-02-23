import argparse

from research.config import HardwareSimConfig, SystemConfig, TrainingConfig
from benchmarks.benchmark import (
    hardware_simulation,
    model_level_compression_experiment,
    pruning_hardware_experiment,
    repeated_benchmark,
)


def _parse_levels(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark models")
    parser.add_argument("--model", choices=["knn", "rf"], default="knn")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    parser.add_argument("--fail-fast-dataset", action="store_true")
    parser.add_argument("--memory-budget-mb", type=int, default=8)
    parser.add_argument("--compute-scale", type=float, default=1.0)
    parser.add_argument("--run-pruning-efficiency", action="store_true")
    parser.add_argument("--run-model-compression", action="store_true")
    parser.add_argument("--pruning-type", choices=["weight", "neuron"], default="weight")
    parser.add_argument("--sparsity-levels", type=str, default="0.0,0.2,0.4,0.6,0.8")
    parser.add_argument("--compression-levels", type=str, default="0.0,0.2,0.4,0.6")
    args = parser.parse_args()

    sys_cfg = SystemConfig(
        sample_size=args.sample_size,
        dataset=args.dataset,
        fail_fast_dataset=args.fail_fast_dataset,
    )
    tr_cfg = TrainingConfig(model_name=args.model)
    stats = repeated_benchmark(sys_cfg, tr_cfg, runs=args.runs)
    hw_cfg = HardwareSimConfig(memory_budget_mb=args.memory_budget_mb, compute_scale=args.compute_scale)
    resource, acc = hardware_simulation(sys_cfg, tr_cfg, hw_cfg)
    print(f"benchmark={stats}")
    print(f"hardware_sim={{'batch_sizes': {resource}, 'accuracy': {acc}}}")

    if args.run_pruning_efficiency:
        report = pruning_hardware_experiment(
            sys_cfg,
            tr_cfg,
            hw_cfg,
            sparsity_levels=_parse_levels(args.sparsity_levels),
            pruning_type=args.pruning_type,
            runs=args.runs,
        )
        print(f"pruning_efficiency={report}")

    if args.run_model_compression:
        report = model_level_compression_experiment(
            sys_cfg,
            tr_cfg,
            runs=args.runs,
            levels=_parse_levels(args.compression_levels),
        )
        print(f"model_level_compression={report}")


if __name__ == "__main__":
    main()
