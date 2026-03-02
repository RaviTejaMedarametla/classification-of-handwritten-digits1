# Handwritten Digit Classification: Compression, Benchmarking, and Deployment

This repository contains a compact engineering workflow for training, compressing, evaluating, and deploying handwritten-digit classifiers under resource constraints.

The codebase centers on two baseline model families:
- k-nearest neighbors (KNN)
- random forest (RF)

It provides command-line workflows for training, benchmarking, compression experiments, and ONNX export with parity validation.

## Design Goals

- Keep model behavior measurable under constrained memory and compute settings.
- Compare compression interventions with repeatable protocols.
- Preserve deterministic runs through explicit seeding and artifact capture.
- Provide deployment checks that fail fast when model parity regresses.

## Repository Layout

- `research/`: configuration, data loading, training pipeline, and CLI entrypoints.
- `compression/`: feature pruning, prototype reduction, quantization simulation, and model builders.
- `benchmarks/`: repeated experiments, statistical summaries, and hardware-constrained sweeps.
- `deployment/`: ONNX export, runtime parity checks, and batched CPU inference.
- `docs/`: architecture notes, failure modes, constraints, and hardware profiling guidance.
- `artifacts/`: generated reports and plots (created at runtime).

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m research.train_cli --model knn --dataset digits --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --runs 5 --run-model-compression --memory-budget-mb 8 --compute-scale 0.75
python -m research.infer_cli --model knn --dataset digits --batch-size 64 --export-onnx --onnx-min-agreement 0.98
```

## Workflow Summary

1. **Training**: train one model with deterministic split and persist run metadata.
2. **Compression**: evaluate pruning- and prototype-based reductions with accuracy deltas.
3. **Benchmarking**: run repeated experiments with confidence intervals and hardware constraints.
4. **Deployment**: export to ONNX and enforce agreement thresholds before accepting outputs.

## Performance and Hardware Scope

The project reports latency, throughput, model size, and estimated energy metrics for CPU-oriented inference. Hardware simulation is approximate and intended for ranking trade-offs, not replacing cycle-accurate profiling.

For detailed assumptions, bottlenecks, and profiling methodology, see:
- `docs/system_design.md`
- `docs/hardware_profiling.md`

## Assumptions and Limitations

- Default experiments prioritize deterministic comparability over absolute peak performance.
- MNIST fetch requires network access unless the dataset is cached locally.
- Hardware simulation uses memory-budget and compute-scaling abstractions; it does not model cache behavior at instruction level.
- Quantization support is simulation-oriented (`float16` and `int8_sim`) rather than backend-specific kernel quantization.

## Reproducibility

- Set explicit seeds through CLI options.
- Prefer `--dataset digits` for fully offline runs.
- Keep CLI flags and generated JSON artifacts together for each experiment.
- Re-run with identical flags and compare stored reports under `artifacts/`.
