# Hardware Awareness, Profiling, and Measurement Guidance

## Scope

This project includes lightweight profiling primitives intended to compare compression and deployment configurations under constrained scenarios.

## Layer-wise / Operator-wise Mapping

Traditional layer-wise breakdown is less direct for KNN and RF than for neural networks. The repository uses operator-level approximations:

- **KNN operators:** distance computation, neighborhood selection, vote aggregation.
- **RF operators:** per-tree traversal, node comparison, vote aggregation.

Each operator can be profiled by timing prediction on fixed-size batches and normalizing by sample count.

## Latency and Memory Breakdown

For each benchmark level, collect:

- end-to-end latency (seconds),
- throughput (samples/second),
- evaluation-batch memory (MB),
- serialized model memory (MB).

These quantities are available in generated benchmarking reports and can be used to build a latency-memory frontier.

## Bandwidth and Utilization Estimates

Bandwidth can be approximated with:

- `bytes_moved / latency` where `bytes_moved` includes input batch and model reads.

Utilization is approximated relative to a reference throughput:

- `observed_throughput / peak_reference_throughput`.

These are comparative heuristics, not hardware counters.

## Precision and Quantization Trade-offs

Quantization modes (`float32`, `float16`, `int8_sim`) allow comparing:

- memory footprint changes,
- latency sensitivity,
- accuracy degradation risk.

`int8_sim` emulates quantization effects in preprocessing space and does not guarantee parity with backend-specific integer kernels.

## Edge and Constrained Scenarios

Use low memory budgets and reduced compute scales to emulate edge constraints. Recommended checks:

- minimum batch size that preserves acceptable throughput,
- accuracy drift at high sparsity/compression,
- ONNX parity under constrained inference settings.

## Reproducible Reporting

When reporting results:

1. record exact CLI command,
2. record seed and dataset choice,
3. include confidence intervals across repeated runs,
4. retain generated JSON and plot artifacts.
