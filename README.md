# Hardware-Aware Handwritten Digit Classification Pipeline

Reproducible handwritten-digit classification with compression studies, benchmark suites, ONNX parity checks, and profiler-backed deployment measurements.

## Install

```bash
python -m pip install -r requirements-lock.txt
```

> `requirements-lock.txt` is the reproducible lock file. `requirements.txt` is pinned and mirrored.

## Project Structure

- `research/`: training/eval/inference CLIs and core ML pipeline.
- `compression/`: pruning, prototype reduction, and quantization simulation.
- `benchmarks/`: benchmark modules (`repeated_benchmark.py`, `pruning_benchmark.py`, `compression_benchmark.py`, `operator_profile.py`) with facade `benchmark.py`.
- `deployment/`: ONNX export/validation and measured ONNX profiling.
- `results/`: experiment runners and statistical analysis scripts.
- `docs/experimental_protocol.md`: scientific protocol and reporting guidance.

## Reproduce Core Runs

```bash
python -m research.train_cli --model knn --dataset digits --run-compression --run-quantization
python -m research.evaluate_cli --model rf --dataset digits --runs 5 --run-model-compression --run-pruning-efficiency
python -m research.infer_cli --model knn --dataset digits --export-onnx --onnx-min-agreement 0.98
python -m results.run_all_experiments
```

## Measured vs Simulated Metrics

- **Simulated**: pruning/hardware scaling effects and analytical operator breakdown from benchmark modules.
- **Measured**: real CPU inference latency with ONNXRuntime:

```bash
python -m deployment.profile_onnx --onnx-path results/knn.onnx --feature-dim 64 --iterations 100 --warmup 10
```

## Statistical Comparison

```bash
python -m results.statistical_comparison --a path/to/model_a_outputs.npy --b path/to/model_b_outputs.npy
```

## Additional Documentation

- `docs/system_design.md`
- `docs/hardware_profiling.md`
- `docs/experimental_protocol.md`
