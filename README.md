# Hardware-Aware Handwritten Digit Classification Pipeline

A reproducible machine learning system for compression-aware handwritten digit classification, benchmarking, and deployment validation.

## Overview

This repository implements a research-oriented engineering pipeline for handwritten digit classification using modular components for training, evaluation, compression studies, and deployment checks. The system is designed to support controlled experimentation across model families while preserving traceability from data processing to runtime artifacts.

A central objective is to assess model quality together with resource behavior under practical constraints such as memory budgets, compute scaling, and inference latency. This hardware-aware perspective is important for transitioning machine learning models from development environments to edge or production settings where computational resources are limited and reproducibility requirements are strict.

## Project Motivation

Machine learning performance is often reported primarily through accuracy metrics, while deployment constraints are addressed late in the development cycle. This project is motivated by the need to evaluate classification models jointly on predictive quality and system-level efficiency.

The repository therefore emphasizes deterministic experimentation, compression-oriented analysis, and benchmark repeatability. This enables rigorous comparison of design choices when targeting resource-constrained environments and improves confidence in downstream deployment decisions.

## System Architecture

- **Data Pipeline**  
  Dataset access and preprocessing utilities are organized within the research package to provide consistent inputs for training, evaluation, and inference workflows.

- **Model Training**  
  CLI-driven training routines support controlled model fitting and parameterized experiments for handwritten digit classification.

- **Model Compression**  
  Compression modules provide feature pruning, prototype reduction, and quantization simulation to study efficiency/accuracy trade-offs.

- **Hardware-Aware Evaluation**  
  Benchmark utilities execute repeated runs, summarize statistics, and support memory- and compute-constrained evaluation scenarios.

- **Inference and Deployment Validation**  
  Deployment tooling includes ONNX export and prediction parity checks to verify that exported artifacts align with native model behavior.

## Repository Structure

- **`research/`**  
  Core research workflow components, including configuration helpers, dataset utilities, and CLI entry points for train/evaluate/infer.

- **`compression/`**  
  Compression and model-reduction logic used for efficiency experiments.

- **`benchmarks/`**  
  Benchmark orchestration, repeated-run execution, and summary reporting utilities.

- **`deployment/`**  
  Export and deployment validation tooling, including ONNX conversion and inference checks.

- **`docs/`**  
  Design and profiling documentation for system behavior and hardware-oriented analysis.

- **`artifacts/`**  
  Generated experiment outputs such as reports, plots, and serialized model artifacts.

## Features

- Deterministic machine learning workflows via configurable run settings and seed control.
- Compression-aware experimentation for model size/performance trade-off analysis.
- Hardware-aware benchmarking with repeated evaluations and statistical summaries.
- Deployment validation through ONNX export and agreement testing.
- CLI-oriented execution for reproducible end-to-end experiments.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Usage

Example experiment commands:

```bash
python -m research.train_cli --model knn --dataset digits --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --runs 5 --run-model-compression --memory-budget-mb 8 --compute-scale 0.75
python -m research.infer_cli --model knn --dataset digits --batch-size 64 --export-onnx --onnx-min-agreement 0.98
```

Additional implementation details are available in:

- `docs/system_design.md`
- `docs/hardware_profiling.md`

## Reproducibility

Reproducible experimentation is supported through:

- **Configuration-driven execution** via parameterized CLI options.
- **Deterministic seeds** to control stochastic components across repeated runs.
- **Experiment artifacts** stored under `artifacts/` for traceable comparison of outputs, metrics, and reports.

For strict replication, rerun commands with identical flags and dataset settings.

## Related Projects

This repository is part of a broader portfolio on:

- hardware-aware machine learning
- edge AI optimization
- deterministic ML pipelines
- production ML systems

Related repositories:

- `neural-network-from-scratch`
- `classification-of-handwritten-digits1`
- `edge-ai-hardware-optimization`
- `data-analysis-for-hospitals`
- `nba-data-preprocessing`
- `Data-Science-AI-Portfolio`

## Future Work

Potential extensions include:

- deployment benchmarking on embedded and low-power edge devices,
- expanded compression methods for more aggressive memory reduction,
- enhanced benchmarking frameworks for deeper latency/throughput profiling across hardware classes.

## License

This project is distributed under the terms specified in the `LICENSE` file.
