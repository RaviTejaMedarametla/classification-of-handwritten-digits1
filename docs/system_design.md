# System Design and Operational Trade-offs

## Architecture Overview

The pipeline is organized as a linear flow with modular experiment stages:

1. Dataset loading and fingerprinting.
2. Deterministic split and normalization.
3. Model training and baseline evaluation.
4. Optional compression and quantization experiments.
5. Optional hardware-constrained benchmark sweeps.
6. Optional ONNX export and prediction-parity validation.

This structure keeps each stage independently testable while allowing end-to-end execution from CLI entrypoints.

## Design Motivations

- **Minimal moving parts:** scikit-learn baselines and NumPy preprocessing keep dependencies predictable.
- **Controlled comparisons:** each experiment mutates one factor (e.g., pruning level) while preserving split and seed policies.
- **Auditability:** metrics and metadata are stored as JSON artifacts.

## Architectural Trade-offs

- **Pros:** low implementation complexity, short time-to-run, clear artifact outputs.
- **Cons:** CPU-centric assumptions and simplified hardware abstractions can under-represent accelerator behavior.
- **Pros:** deterministic split and seed policy improves reproducibility.
- **Cons:** deterministic settings can mask variance that appears in uncontrolled production settings.

## Failure Modes and Bottlenecks

- Dataset loading can fail on MNIST when network/cache is unavailable.
- KNN inference cost scales with reference set size and can dominate latency.
- RF memory footprint can increase rapidly with large ensemble sizes.
- ONNX parity can fail when conversion paths alter numeric behavior.

## Scalability Considerations

- Increasing sample size affects training and inference differently across model families.
- KNN scales memory with dataset retention; prototype reduction helps but may reduce class-boundary fidelity.
- RF scales with number and depth of trees; pruning depth lowers latency but may reduce accuracy.
- Artifact growth should be managed if running large sweep grids repeatedly.

## Deployment Challenges

- Reproducing local latency on target hardware requires platform-specific runtime tuning.
- Batch-size tuning can improve throughput but may exceed edge memory budgets.
- Parity thresholds should be selected per deployment risk tolerance and model sensitivity.

## Assumptions

- Input features are numeric and can be normalized without domain loss.
- CPU inference is the primary baseline for comparison.
- Performance estimates are relative indicators across configurations.

## Limitations

- No distributed training/inference orchestration.
- No automatic artifact retention policy.
- No cycle-level hardware simulator integration.
