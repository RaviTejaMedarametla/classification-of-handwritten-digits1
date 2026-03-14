# Classification of Handwritten Digits

A research‑oriented repository implementing multiple classifiers from scratch (k‑NN, SVM, MLP, CNN) and evaluating them under hardware‑aware constraints (latency, memory, energy). The project includes feature extraction (HOG, PCA), hyperparameter tuning, interpretability (saliency maps), and comprehensive benchmarking.

## Features

- **Classifiers implemented from scratch** – k‑Nearest Neighbors, SVM (with simplified SMO), Multi‑Layer Perceptron, and a small CNN.
- **Feature extraction** – HOG, PCA, and standard scaling.
- **Hardware‑aware metrics** – inference latency, memory footprint, and energy proxy.
- **Model compression** – pruning and quantization experiments.
- **ONNX export** – convert trained models to ONNX and validate parity.
- **Reproducibility** – deterministic seeds, configuration via dataclasses, and experiment logging.
- **CLI tools** – train, evaluate, and infer from the command line.
- **Extensive test suite** – ensures correctness of all components.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python -m research.train_cli --model knn --dataset digits
```

Evaluate a trained model:
```bash
python -m research.evaluate_cli --model knn --dataset digits --runs 5
```

Run inference (optionally export to ONNX):
```bash
python -m research.infer_cli --model knn --dataset digits --export-onnx
```

Run all experiments and generate summary tables:
```bash
python results/run_all_experiments.py
```

## Project Structure

- `research/` – core implementation: configuration, data loading, training, evaluation, CLI.
- `benchmarks/` – scripts for repeated benchmarking, pruning, compression, and operator profiling.
- `compression/` – utilities for pruning and quantization.
- `deployment/` – ONNX export and inference.
- `results/` – experiment orchestration and statistical comparison.
- `tests/` – unit and integration tests.

## License

This project is released under the MIT License. See `LICENSE` for details.
