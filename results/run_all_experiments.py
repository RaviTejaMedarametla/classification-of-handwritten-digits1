"""Run reproducible baseline, compression, and ONNX parity experiments.

Usage:
    python results/run_all_experiments.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def run_cmd(args: list[str]) -> None:
    full_cmd = [sys.executable, "-m", *args]
    print("$", " ".join(full_cmd))
    subprocess.run(full_cmd, cwd=ROOT, check=True)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    common = ["--dataset", "digits", "--seed", "40", "--sample-size", "1200"]

    # Baseline + compression + quantization evidence.
    run_cmd(["research.train_cli", "--model", "knn", *common, "--run-compression", "--run-quantization"])
    run_cmd(["research.train_cli", "--model", "rf", *common, "--run-compression", "--run-quantization"])

    # Repeated benchmark and hardware-aware reports.
    run_cmd(
        [
            "research.evaluate_cli",
            "--model",
            "knn",
            "--dataset",
            "digits",
            "--seed",
            "40",
            "--sample-size",
            "1200",
            "--runs",
            "3",
            "--run-pruning-efficiency",
            "--run-model-compression",
            "--run-operator-profile",
        ]
    )

    # ONNX export/parity (measured), then explicit ONNX runtime profile.
    run_cmd(
        [
            "research.infer_cli",
            "--model",
            "knn",
            "--dataset",
            "digits",
            "--seed",
            "40",
            "--export-onnx",
            "--onnx-min-agreement",
            "0.98",
        ]
    )

    subprocess.run(
        [
            sys.executable,
            "deployment/profile_onnx.py",
            "--onnx-model",
            "artifacts/knn.onnx",
            "--dataset",
            "digits",
            "--seed",
            "40",
            "--iterations",
            "100",
            "--warmup",
            "20",
            "--output",
            "results/onnx_profile_knn.json",
        ],
        cwd=ROOT,
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            "results/statistical_comparison.py",
            "--baseline",
            "artifacts/benchmark_knn.json",
            "--candidate",
            "artifacts/benchmark_rf.json",
            "--output",
            "results/statistical_comparison.json",
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
