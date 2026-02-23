# Model Compression and Deployment for Resource-Constrained Systems

This repository is an independent research prototype focused on **model compression and deployment for resource-constrained systems**. The project studies how compression strategies affect predictive quality and runtime behavior, with emphasis on practical edge inference constraints.

## Research Focus

- **Model compression** across feature-space pruning and model-level reduction.
- **Prototype reduction** for k-nearest neighbors to shrink memory footprint.
- **Hardware-aware benchmarking** under constrained memory and scaled compute budgets.
- **ONNX deployment** with parity validation against native inference.
- **Accuracy–latency–memory trade-offs** reported with statistical summaries.

## Repository Structure

- `research/` — core experiment orchestration, dataset handling, reproducibility controls, and experiment entrypoints.
- `compression/` — compression and quantization methods plus baseline model constructors.
- `deployment/` — ONNX export, validation, and batched inference utilities.
- `benchmarks/` — repeated evaluation, hardware simulation, and compression benchmarking.
- `artifacts/` — generated metrics, plots, and deployment outputs.

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m research.train_cli --model knn --dataset digits --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --runs 5 --run-model-compression --memory-budget-mb 8 --compute-scale 0.75
python -m research.infer_cli --model knn --dataset digits --batch-size 64 --export-onnx --onnx-min-agreement 0.98
```

## Comparison Methodology

Compression studies use controlled comparisons where only one factor changes at a time:

1. Train a baseline model and record accuracy, latency, and serialized model size.
2. Apply a single compression intervention (feature pruning, prototype reduction, or tree-pruning policy).
3. Re-run the same evaluation protocol and compute deltas.
4. Sweep compression levels and build trade-off curves.

This design isolates compression effects and avoids conflating algorithm, data split, and hardware-budget changes.

## Statistical Rigor

Evaluation routines support repeated runs with deterministic seed schedules and report:

- mean,
- standard deviation,
- 95% confidence interval.

These statistics are logged for accuracy, inference latency, training time, throughput, model size, and estimated energy per inference where applicable.

## Deployment Pipeline

The deployment flow is designed for edge portability:

1. Train compressed or baseline scikit-learn model.
2. Export to ONNX.
3. Validate ONNX predictions against native predictions.
4. Enforce a minimum parity threshold before accepting deployment.
5. Save deployment report JSON for auditability.

## Research-Style Evaluation

Suggested experiments:

- **Compression sweep:** measure how aggressive compression changes accuracy and latency.
- **Hardware-budget sweep:** vary memory budget and compute scale to estimate robust operating points.
- **Precision comparison:** benchmark float32, float16, and int8-simulated pathways.
- **Model family comparison:** compare KNN prototype reduction against Random Forest pruning.

Primary outputs in `artifacts/` include benchmark reports, compression reports, ONNX parity reports, and trade-off plots.

## Reproducibility Instructions

- Fix seeds via configuration (`SystemConfig.seed`) and run repeated experiments.
- Keep dataset choice explicit (`--dataset mnist` or `--dataset digits`).
- Persist run configuration and dataset metadata fingerprints in artifacts.
- Re-run the same command lines and compare generated JSON reports.

## Hardware Relevance and Semiconductor Angle

This project is motivated by practical constraints in semiconductor-backed edge platforms:

- limited DRAM/SRAM budgets,
- strict energy envelopes,
- latency requirements for near-sensor inference,
- deployment portability across CPU-class accelerators via ONNX.

The benchmarking utilities intentionally expose memory-pressure and compute-scaling knobs to emulate the design-space exploration commonly used in hardware-software co-optimization workflows.
