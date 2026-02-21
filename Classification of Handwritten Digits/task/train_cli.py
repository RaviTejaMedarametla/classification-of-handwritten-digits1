import argparse

from ml_system.config import CompressionConfig, QuantizationConfig, SystemConfig, TrainingConfig
from ml_system.training.trainer import export_run_metadata, run_compression, run_quantization, train_once


def main():
    parser = argparse.ArgumentParser(description="Train handwritten digit classifiers")
    parser.add_argument("--model", choices=["knn", "rf"], default="knn")
    parser.add_argument("--sample-size", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="mnist")
    parser.add_argument("--fail-fast-dataset", action="store_true")
    parser.add_argument("--run-compression", action="store_true")
    parser.add_argument("--run-quantization", action="store_true")
    parser.add_argument("--weight-prune", type=float, default=0.2)
    parser.add_argument("--neuron-prune", type=float, default=0.1)
    args = parser.parse_args()

    sys_cfg = SystemConfig(
        seed=args.seed,
        sample_size=args.sample_size,
        dataset=args.dataset,
        fail_fast_dataset=args.fail_fast_dataset,
    )
    tr_cfg = TrainingConfig(model_name=args.model)
    export_run_metadata(sys_cfg, tr_cfg)
    base = train_once(sys_cfg, tr_cfg)
    print(f"model={args.model} accuracy={base['metrics']['accuracy']:.6f}")

    if args.run_compression:
        comp = CompressionConfig(weight_pruning_level=args.weight_prune, neuron_pruning_level=args.neuron_prune)
        out = run_compression(sys_cfg, tr_cfg, comp)
        print(f"compression={out}")

    if args.run_quantization:
        out = run_quantization(sys_cfg, tr_cfg, QuantizationConfig())
        print(f"quantization={out}")


if __name__ == "__main__":
    main()
