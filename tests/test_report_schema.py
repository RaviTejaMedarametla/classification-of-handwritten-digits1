from research.config import SystemConfig, TrainingConfig
from benchmarks.repeated_benchmark import repeated_benchmark


def test_repeated_benchmark_schema():
    sys_cfg = SystemConfig(dataset="digits", sample_size=300, seed=40)
    tr_cfg = TrainingConfig(model_name="knn")
    report = repeated_benchmark(sys_cfg, tr_cfg, runs=2)

    required = {
        "summary_version",
        "model",
        "runs",
        "seed_schedule",
        "sample_count_eval",
        "accuracy",
        "inference_latency_s",
        "training_time_s",
        "throughput_samples_per_s",
        "energy_per_inference_j",
    }
    assert required.issubset(set(report.keys()))
    assert report["summary_version"] == "v2"
