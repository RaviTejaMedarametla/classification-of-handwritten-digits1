# Handwritten Digit Classification: Compression, Benchmarking, and Deployment

## Overview

This repository implements an engineering-focused machine learning workflow for handwritten digit classification, with explicit support for compression experiments, benchmark analysis, and deployment validation under resource constraints.

It is part of a broader portfolio focused on hardware-aware machine learning, edge AI optimization, deterministic ML pipelines, and production ML systems.

## System Architecture

The project is organized into modular packages that separate experimentation concerns:

- `research/`: configuration, dataset utilities, analysis utilities, and CLI entry points for training/evaluation/inference.
- `compression/`: feature pruning, prototype reduction, quantization simulation, and model construction helpers.
- `benchmarks/`: repeated experiment execution, confidence interval summaries, and hardware-constrained benchmarking flows.
- `deployment/`: ONNX export pipeline, parity checks, and batch CPU inference utilities.
- `docs/`: system design notes and hardware profiling guidance.
- `artifacts/`: generated runtime outputs (reports, plots, and serialized model artifacts).

## Features

- Deterministic training and evaluation controls through configurable seeding.
- Compression-oriented experimentation for size/performance trade-off analysis.
- Benchmark routines with repeatable runs and summary statistics.
- Deployment verification with ONNX export and agreement checks.
- Hardware-aware reporting for latency, throughput, and memory-oriented constraints.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Example command-line workflows:

```bash
python -m research.train_cli --model knn --dataset digits --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --runs 5 --run-model-compression --memory-budget-mb 8 --compute-scale 0.75
python -m research.infer_cli --model knn --dataset digits --batch-size 64 --export-onnx --onnx-min-agreement 0.98
```

For implementation and profiling details, see:

- `docs/system_design.md`
- `docs/hardware_profiling.md`

## Reproducibility

- Use explicit seed options in CLI commands for deterministic runs.
- Prefer `--dataset digits` for fully offline execution.
- Preserve command flags and generated JSON reports for each run.
- Re-run with identical parameters to compare outputs under `artifacts/`.

## Related Projects

This repository is part of a larger AI systems engineering portfolio:

- `neural-network-systems`
- `digit-classification-benchmark`
- `edge-ai-model-optimization`
- `hospital-analytics-pipeline`
- `nba-data-engineering`
- `ai-systems-ml-platform`
