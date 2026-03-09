import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from research.core.utils.metrics import save_json


def profile_onnx(onnx_path: Path, feature_dim: int, warmup: int = 10, iterations: int = 100, batch_size: int = 1):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.random.default_rng(42).normal(size=(batch_size, feature_dim)).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {input_name: sample})

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    report = {
        "onnx_path": str(onnx_path),
        "batch_size": batch_size,
        "feature_dim": feature_dim,
        "warmup_iterations": warmup,
        "iterations": iterations,
        "mean_latency_ms": statistics.mean(latencies_ms),
        "std_latency_ms": statistics.pstdev(latencies_ms),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
    }
    out_path = Path("results") / f"onnx_latency_profile_{onnx_path.stem}.json"
    save_json(report, out_path)
    return report, out_path


def main():
    parser = argparse.ArgumentParser(description="Profile ONNXRuntime CPU inference latency")
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--feature-dim", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    report, out_path = profile_onnx(args.onnx_path, args.feature_dim, args.warmup, args.iterations, args.batch_size)
    print(f"saved_profile={out_path}")
    print(report)


if __name__ == "__main__":
    main()
