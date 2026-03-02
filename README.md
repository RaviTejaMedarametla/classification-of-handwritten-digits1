# Streaming Digit Analytics: Hardware-Aware Classification Pipeline

This repository provides a production-style analytics workflow for low-latency handwritten digit classification. It is organized for repeatable experimentation, model compression, and CPU-first deployment validation.

## Engineering Scope

- Streaming-oriented inference flow from trained model to batched runtime execution.
- Hardware-aware evaluation under bounded memory and compute budgets.
- Deterministic experiment controls for comparable runs across environments.
- ONNX export plus parity validation for deployment readiness checks.

## Repository Layout

- `research/` — configuration, data loading, training/evaluation CLIs, and orchestration.
- `compression/` — feature pruning, prototype reduction, quantization simulation, model builders.
- `benchmarks/` — repeated benchmark runs, confidence intervals, hardware profiling outputs.
- `deployment/` — ONNX conversion and runtime parity validation.
- `docs/` — design decisions, operational constraints, and profiling guidance.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m research.train_cli --model knn --dataset digits --seed 40 --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --seed 40 --runs 5 --run-model-compression --run-operator-profile
python -m research.infer_cli --model knn --dataset digits --seed 40 --batch-size 64 --export-onnx --onnx-min-agreement 0.98
```

## Operational Notes

- Use `--dataset digits` for offline and CI-stable execution.
- Use explicit `--seed` values to preserve deterministic comparisons.
- Treat hardware simulation as comparative ranking, not cycle-accurate modeling.
- Review generated JSON artifacts in `artifacts/` for auditability.

## Assumptions and Limits

- The pipeline targets Python 3.10+ and CPU execution by default.
- MNIST loading requires TensorFlow when `--dataset mnist` is selected.
- Operator-level profile metrics are estimated partitions of end-to-end latency.
