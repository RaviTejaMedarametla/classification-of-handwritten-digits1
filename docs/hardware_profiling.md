# Hardware Profiling Guide for Real-Time Analytics

## Objective

Provide reproducible latency, memory, and bandwidth diagnostics for constrained inference pipelines.

## Operator-Level Mapping

For non-neural estimators, profiling is tracked at operator granularity:

- **KNN:** distance compute, neighbor selection, vote aggregation.
- **RF:** tree traversal, vote aggregation, output formatting.

## Metrics to Capture

- End-to-end latency (`total_latency_s`).
- Throughput (`throughput_samples_per_s`).
- Input and model memory footprint (`memory.*`).
- Estimated bandwidth (`bandwidth.*`).
- Utilization proxy against a fixed reference (`utilization.ratio`).

## Quantization and Precision

Supported modes (`float32`, `float16`, `int8_sim`) are useful for trade-off studies:

- memory footprint reduction,
- potential latency changes,
- possible accuracy drift.

`int8_sim` is an approximation and not a replacement for backend-native integer kernels.

## Edge Deployment Checks

For constrained scenarios, validate:

1. smallest stable batch size under memory budget,
2. accuracy tolerance under compression,
3. ONNX parity threshold pass rates,
4. repeatability across identical seed schedules.

## Reproducibility Checklist

- Record full CLI command.
- Persist seed, dataset, and config metadata.
- Compare confidence intervals across repeated runs.
- Store raw JSON reports and generated plots.
