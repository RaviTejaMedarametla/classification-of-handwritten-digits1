# Hardware-Aware Handwritten Digit Classification Pipeline

CLI-driven ML pipeline for handwritten digit classification with reproducible training, compression experiments, benchmark reporting, and ONNX deployment validation.

## Reproducibility Lock-In

- Dependencies are pinned in `requirements.txt` and mirrored in `requirements-lock.txt`.
- All scripts support explicit seeds.
- Canonical evidence workflow:

```bash
python -m pip install -r requirements.txt
python results/run_all_experiments.py
```

Expected outputs:
- `artifacts/benchmark_*.json`, `artifacts/compression_*.json`, `artifacts/quantization_*.json`
- `artifacts/onnx_report_*.json`, `results/onnx_profile_knn.json`
- `results/statistical_comparison.json`

## Project Layout

- `research/`: CLI entrypoints and core training/data utilities.
- `compression/`: feature pruning, prototype reduction, quantization.
- `benchmarks/`: split benchmark modules:
  - `repeated_benchmark.py`
  - `pruning_benchmark.py`
  - `compression_benchmark.py`
  - `operator_profile.py`
  - `benchmark.py` (backward-compatible facade)
- `deployment/`: ONNX export, parity checks, and measured ONNX runtime profiler.
- `results/`: experiment orchestration and statistical comparison scripts.
- `docs/experimental_protocol.md`: scientific protocol and measured vs simulated metric framing.

## Main Commands

```bash
# Train baseline
python -m research.train_cli --model knn --dataset digits --sample-size 1200 --seed 40

# Benchmark + compression studies
python -m research.evaluate_cli --model knn --dataset digits --sample-size 1200 --runs 3 \
  --run-pruning-efficiency --run-model-compression --run-operator-profile

# ONNX export and parity validation
python -m research.infer_cli --model knn --dataset digits --seed 40 --export-onnx --onnx-min-agreement 0.98

# Measured ONNXRuntime CPU profiling
python deployment/profile_onnx.py --onnx-model artifacts/knn.onnx --dataset digits --iterations 100 --warmup 20

# Statistical comparison (Welch t-test)
python results/statistical_comparison.py --baseline artifacts/benchmark_knn.json --candidate artifacts/benchmark_rf.json
```

## Testing

```bash
pytest
```

CI runs unit tests with coverage plus smoke CLI checks.

## Scientific Framing

- **Measured metrics**: ONNX Runtime latency from `deployment/profile_onnx.py`.
- **Simulated/analytical metrics**: energy estimates and operator-level partitions in benchmark reports.

See `docs/experimental_protocol.md` for protocol details.
