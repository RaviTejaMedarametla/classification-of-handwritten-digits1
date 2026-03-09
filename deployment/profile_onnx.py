from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from research.config import SystemConfig
from research.core.data.dataset import load_dataset, split_and_normalize
from research.core.utils.reproducibility import set_deterministic


def profile_model(model_path: Path, x_test: np.ndarray, warmup: int, iterations: int) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    sample = x_test.astype(np.float32)
    for _ in range(warmup):
        _ = session.run(None, {input_name: sample})

    latencies_ms = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = session.run(None, {input_name: sample})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.array(latencies_ms, dtype=float)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "samples": int(len(sample)),
        "latency_ms": {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        },
        "throughput_samples_per_s": float((len(sample) * 1000.0) / max(float(arr.mean()), 1e-9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile ONNX model latency with ONNX Runtime")
    parser.add_argument("--onnx-model", type=Path, required=True)
    parser.add_argument("--dataset", choices=["mnist", "digits"], default="digits")
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results/onnx_profile.json"))
    args = parser.parse_args()

    cfg = SystemConfig(dataset=args.dataset, seed=args.seed, sample_size=args.sample_size)
    set_deterministic(cfg.seed)
    x, y, _ = load_dataset(cfg)
    _, x_test, _, _ = split_and_normalize(x, y, cfg)

    profile = profile_model(args.onnx_model, x_test, args.warmup, args.iterations)
    profile["model_path"] = str(args.onnx_model)
    profile["dataset"] = args.dataset

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2))
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
