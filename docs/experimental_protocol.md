# Experimental Protocol

## Baselines
- **KNN**: distance-weighted baseline for strong non-parametric performance.
- **Random Forest (RF)**: tree ensemble baseline with robust tabular behavior.
- **Simple MLP**: recommended extension baseline for neural network comparison.

## Repeated Runs and Confidence Intervals
- All experiments should be run with fixed seeds and repeated runs (`n>=5` when feasible).
- Mean, standard deviation, and 95% confidence intervals are reported from repeated runs.

## Statistical Significance
- Use Welch's t-test for output/value comparisons when variances may differ.
- Script: `python -m results.statistical_comparison --a <a.npy> --b <b.npy>`.

## Hardware and Simulation Assumptions
- Simulated metrics (energy, hardware scaling, operator partitioning) are analytical and based on explicit assumptions in code.
- Measured metrics (ONNXRuntime CPU latency from `deployment/profile_onnx.py`) are empirical and should be reported separately.
- Record CPU model, RAM, Python version, and dependency lock when publishing numbers.
