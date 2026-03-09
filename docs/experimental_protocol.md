# Experimental Protocol

This protocol distinguishes **simulated** efficiency metrics from **measured** latency metrics.

## Baselines
- Baseline models: KNN and RandomForest using identical train/test splits and seed schedule.
- Dataset: `digits` for offline reproducibility.
- Seeds: fixed base seed + deterministic increments for repeated runs.

## Repeated runs and confidence intervals
- Repeat each benchmark run at least `n=3` times (default `n=5`).
- Report mean, sample standard deviation, and 95% confidence interval using normal approximation.

## Compression experiments
- Evaluate feature-level pruning and model-level compression on predefined levels.
- For each level, record accuracy and latency summaries and compare against baseline.

## Statistical significance testing
- Use a two-sample Welch t-test on repeated-run accuracy summaries.
- Script: `results/statistical_comparison.py`.
- Primary output: p-value and mean difference.

## Measured vs simulated metrics
- **Measured**: ONNX Runtime CPU latency from `deployment/profile_onnx.py`.
- **Simulated/analytical**: energy and operator-level decomposition from benchmark scripts.
- Any report must label these categories explicitly.
